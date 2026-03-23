import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from deepsc_preprocess import build_dataloaders, load_preprocessed, preprocess_and_save_europarl


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_repo_data(config: dict, root: Path):
    dataset_cfg = config["dataset"]
    special_tokens = dataset_cfg["special_tokens"]

    raw_text_dir = root / dataset_cfg["raw_text_dir"]
    train_pkl = root / dataset_cfg["train_pickle"]
    test_pkl = root / dataset_cfg["test_pickle"]
    vocab_json = root / dataset_cfg["vocab_json"]

    required_paths = [train_pkl, test_pkl, vocab_json]
    missing_outputs = [p for p in required_paths if not p.exists()]

    if missing_outputs:
        if not dataset_cfg.get("preprocess_if_missing", False):
            missing_names = ", ".join(str(path.relative_to(root)) for path in missing_outputs)
            raise FileNotFoundError(
                f"Missing preprocessed dataset artifacts: {missing_names}. "
                "Enable dataset.preprocess_if_missing or generate files first."
            )

        if not raw_text_dir.exists():
            raise FileNotFoundError(
                f"Raw Europarl text folder not found: {raw_text_dir}. "
                "Set dataset.raw_text_dir to the folder containing .txt files."
            )

        stats = preprocess_and_save_europarl(
            input_dir=raw_text_dir,
            train_pkl_path=train_pkl,
            test_pkl_path=test_pkl,
            vocab_json_path=vocab_json,
            special_tokens=special_tokens,
            min_sentence_len=int(dataset_cfg["min_sentence_len"]),
            max_len=int(dataset_cfg["max_len"]),
            min_freq=int(dataset_cfg["min_freq"]),
            punct_to_keep=dataset_cfg.get("punct_to_keep", [";", ","]),
            punct_to_remove=dataset_cfg.get("punct_to_remove", ["?", "."]),
        )
        print("Preprocessed Europarl dataset:", stats)

    train_data, test_data, stoi, itos = load_preprocessed(
        train_pkl_path=train_pkl,
        test_pkl_path=test_pkl,
        vocab_json_path=vocab_json,
    )

    max_seq_len = max(
        max(len(sample) for sample in train_data),
        max(len(sample) for sample in test_data),
    )

    pad_id = stoi[special_tokens[0]]
    loaders = build_dataloaders(
        train_data=train_data,
        test_data=test_data,
        pad_id=pad_id,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    token_ids = {
        "pad": stoi[special_tokens[0]],
        "start": stoi[special_tokens[1]],
        "end": stoi[special_tokens[2]],
        "unk": stoi[special_tokens[3]],
    }
    return loaders, stoi, itos, pad_id, token_ids, max_seq_len


def snr_to_noise_std(snr_db: float) -> float:
    snr_linear = 10 ** (snr_db / 10)
    return 1.0 / math.sqrt(2.0 * snr_linear)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def power_normalize(x: torch.Tensor) -> torch.Tensor:
    power = torch.mean(x * x).sqrt()
    if power > 1:
        return x / power
    return x


def add_awgn(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    return x + torch.normal(0.0, noise_std, size=x.shape).to(x.device)


class ChannelEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, channel_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channel_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelDecoder(nn.Module):
    def __init__(self, channel_dim: int, d_model: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(channel_dim, d_model)
        self.linear2 = nn.Linear(d_model, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        return self.layernorm(x1 + x5)


class DeepSCBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_enc_layers: int,
        num_dec_layers: int,
        dim_feedforward: int,
        channel_dim: int,
        channel_hidden: int,
        channel_decoder_hidden: int,
        max_len: int,
        pad_id: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.semantic_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        self.channel_encoder = ChannelEncoder(d_model, channel_hidden, channel_dim)
        self.channel_decoder = ChannelDecoder(channel_dim, d_model, channel_decoder_hidden)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.semantic_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        self.output_layer = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def _encode_channel(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, noise_std: float):
        emb = self.embedding(input_ids) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        src_pad = attention_mask == 0
        semantic_features = self.semantic_encoder(emb, src_key_padding_mask=src_pad)
        channel_symbols = power_normalize(self.channel_encoder(semantic_features))
        noisy = add_awgn(channel_symbols, noise_std)
        memory = self.channel_decoder(noisy)
        return memory, src_pad

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
        memory, src_pad = self._encode_channel(input_ids, attention_mask, noise_std)
        dec_emb = self.embedding(input_ids) * math.sqrt(self.d_model)
        dec_emb = self.pos_encoder(dec_emb)
        causal_mask = self._causal_mask(input_ids.size(1), input_ids.device)
        dec_out = self.semantic_decoder(
            dec_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=src_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.output_layer(dec_out)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        noise_std: float,
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        self.eval()
        memory, src_pad = self._encode_channel(input_ids, attention_mask, noise_std)
        generated = torch.full((input_ids.size(0), 1), sos_id, dtype=torch.long, device=input_ids.device)

        for _ in range(self.max_len - 1):
            dec_emb = self.embedding(generated) * math.sqrt(self.d_model)
            dec_emb = self.pos_encoder(dec_emb)
            causal_mask = self._causal_mask(generated.size(1), generated.device)
            dec_out = self.semantic_decoder(
                dec_emb,
                memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=src_pad,
            )
            next_logits = self.output_layer(dec_out[:, -1, :])
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return generated


def decode_tokens(ids, itos, special_ids, eos_id, unk_token):
    toks = []
    for tid in ids:
        if tid == eos_id:
            break
        if tid in special_ids:
            continue
        toks.append(itos.get(tid, unk_token))
    return toks


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int, criterion) -> torch.Tensor:
    ntokens = logits.size(-1)
    loss = criterion(logits.contiguous().view(-1, ntokens), targets.contiguous().view(-1))
    mask = (targets.contiguous().view(-1) != pad_id).type_as(loss)
    loss = loss * mask
    return loss.mean()


def bleu_scores(refs, hyps, bleu_mode: str, report_metrics: list[str]):
    metric_weights = {
        "bleu1": (1.0, 0, 0, 0),
        "bleu2": (0.5, 0.5, 0, 0),
        "bleu3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }

    if bleu_mode == "sentence":
        out = {}
        for metric in report_metrics:
            w = metric_weights[metric]
            vals = [sentence_bleu([r], h, weights=w) for r, h in zip(refs, hyps)]
            out[metric] = float(sum(vals) / len(vals)) if vals else 0.0
        return out

    references = [[r] for r in refs]
    out = {}
    for metric in report_metrics:
        out[metric] = float(corpus_bleu(references, hyps, weights=metric_weights[metric]))
    return out


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    pad_id,
    snr_mode,
    snr_min_db,
    snr_max_db,
):
    model.train()
    total = 0.0
    epoch_noise_std = None
    if snr_mode == "sampled_per_epoch":
        epoch_noise_std = snr_to_noise_std(random.uniform(snr_min_db, snr_max_db))

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if snr_mode == "sampled_per_epoch":
            noise_std = epoch_noise_std
        else:
            noise_std = snr_to_noise_std(random.uniform(snr_min_db, snr_max_db))

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, noise_std=noise_std)
        loss = masked_ce_loss(
            logits[:, :-1],
            input_ids[:, 1:],
            pad_id,
            criterion,
        )
        loss.backward()
        optimizer.step()

        total += loss.item() * input_ids.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_ce(model, loader, criterion, device, pad_id, snr_db):
    model.eval()
    total = 0.0
    noise_std = snr_to_noise_std(snr_db)
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask, noise_std=noise_std)
        loss = masked_ce_loss(
            logits[:, :-1],
            input_ids[:, 1:],
            pad_id,
            criterion,
        )
        total += loss.item() * input_ids.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_bleu_snr(model, loader, snr_values, token_ids, itos, device, bleu_mode, report_metrics, unk_token):
    sos_id = token_ids["start"]
    eos_id = token_ids["end"]
    special_ids = {token_ids["pad"], token_ids["start"], token_ids["end"], token_ids["unk"]}

    rows = []
    for snr in snr_values:
        noise_std = snr_to_noise_std(snr)
        refs, hyps = [], []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            for i in range(input_ids.size(0)):
                inp = input_ids[i : i + 1]
                msk = attention_mask[i : i + 1]
                out_ids = model.generate(inp, msk, noise_std=noise_std, sos_id=sos_id, eos_id=eos_id)
                ref = decode_tokens(input_ids[i].cpu().tolist(), itos, special_ids, eos_id, unk_token)
                hyp = decode_tokens(out_ids.squeeze(0).cpu().tolist(), itos, special_ids, eos_id, unk_token)
                if not ref:
                    continue
                refs.append(ref)
                hyps.append(hyp if hyp else ["the"])

        scores = bleu_scores(refs, hyps, bleu_mode=bleu_mode, report_metrics=report_metrics)
        rows.append({"snr_db": snr, **scores})
        summary = " ".join([f"{m.upper()}={scores[m]:.4f}" for m in report_metrics])
        print(f"SNR {snr:>3} dB | {summary}")

    return pd.DataFrame(rows)


def plot_bleu(df: pd.DataFrame, out_path: Path, report_metrics: list[str]) -> None:
    plt.figure(figsize=(7, 5))
    markers = ["o", "s", "^", "d"]
    for idx, metric in enumerate(report_metrics):
        plt.plot(df["snr_db"], df[metric], marker=markers[idx % len(markers)], label=metric.upper())
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLEU")
    plt.title("DeepSC Replication: BLEU vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "deepsc_replication_config.json")

    seed = int(config["dataset"]["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, stoi, itos, pad_id, token_ids, max_seq_len = load_repo_data(config, root)
    print(
        f"Loaded repo-style data. Train={len(loaders['train'].dataset)} "
        f"Test={len(loaders['test'].dataset)} MaxSeqLen={max_seq_len}"
    )

    model_max_len = max(int(config["dataset"]["max_len"]), int(max_seq_len))

    model_cfg = config["model"]
    model = DeepSCBaseline(
        vocab_size=len(stoi),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_enc_layers=int(model_cfg["num_enc_layers"]),
        num_dec_layers=int(model_cfg["num_dec_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        channel_dim=int(model_cfg["channel_dim"]),
        channel_hidden=int(model_cfg["channel_hidden"]),
        channel_decoder_hidden=int(model_cfg["channel_decoder_hidden"]),
        max_len=model_max_len,
        pad_id=pad_id,
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"].get("betas", [0.9, 0.98])),
        eps=float(config["training"].get("eps", 1e-8)),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_val = float("inf")
    ckpt_path = root / config["artifacts"]["checkpoint"]

    for epoch in range(1, int(config["training"]["num_epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            pad_id,
            config["channel"]["train_snr_mode"],
            float(config["channel"]["train_snr_min_db"]),
            float(config["channel"]["train_snr_max_db"]),
        )

        val_loss = evaluate_ce(
            model,
            loaders["validation"],
            criterion,
            device,
            pad_id,
            snr_db=6.0,
        )

        print(f"Epoch {epoch:>2}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best checkpoint: {ckpt_path.name}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    eval_split = config["evaluation"]["eval_split"]
    snr_values = config["channel"]["eval_snr_db"]
    report_metrics = config["evaluation"]["report_metrics"]
    df = evaluate_bleu_snr(
        model,
        loaders[eval_split],
        snr_values,
        token_ids,
        itos,
        device,
        bleu_mode=config["evaluation"]["bleu_mode"],
        report_metrics=report_metrics,
        unk_token=config["dataset"]["special_tokens"][3],
    )

    csv_path = root / config["artifacts"]["metrics_csv"]
    df.to_csv(csv_path, index=False)

    payload = {
        "model": "deepsc_replication_baseline",
        "seed": seed,
        "checkpoint": ckpt_path.name,
        "snr_values": snr_values,
        "bleu_by_snr": {
            str(int(row["snr_db"])): {metric: float(row[metric]) for metric in report_metrics}
            for _, row in df.iterrows()
        },
    }

    json_path = root / config["artifacts"]["metrics_json"]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_path = root / config["artifacts"]["plot_png"]
    plot_bleu(df, plot_path, report_metrics)

    print(f"Saved: {csv_path.name}")
    print(f"Saved: {json_path.name}")
    print(f"Saved: {plot_path.name}")


if __name__ == "__main__":
    main()
