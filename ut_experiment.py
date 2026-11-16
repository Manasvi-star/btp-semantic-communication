"""
Train and evaluate semantic communication models under AWGN / Rayleigh fading.

Models (``--model``):
    deepsc   fixed-depth Transformer (existing DeepSC baseline)
    ut       Universal Transformer + ACT (Zhou et al., arXiv:2108.09119)
    ut_snr   UT + ACT with SNR-aware halting (our extension)

Metrics per SNR:
    BLEU-1..4                      word-level accuracy
    SBERT similarity               meaning-level accuracy (seq2seq-SC metric)
    avg encoder / decoder steps    compute used (ACT); fixed for DeepSC
    ms per sentence                wall-clock decoding time

Outputs (prefix = <model>_<channel>, e.g. ut_awgn):
    <prefix>_best.pt                   checkpoint
    <prefix>_results_latest.csv        all metrics per SNR
    <prefix>_results_latest.json       includes "bleu_by_snr" (same format as
                                       the other *_results_latest.json files)
    <prefix>_examples.csv              per-sentence examples with BLEU & SBERT
    <prefix>_ood_results.csv           (optional) results on unseen text
    <prefix>_bleu_sbert_snr.png        BLEU + SBERT vs SNR
    <prefix>_steps_snr.png             ACT steps vs SNR (UT models)

Examples:
    python ut_experiment.py --model deepsc --channel awgn
    python ut_experiment.py --model ut     --channel awgn
    python ut_experiment.py --model ut_snr --channel rayleigh
    python ut_experiment.py --model ut --channel awgn --skip_train --ood_text flickr_captions.txt
"""

import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader

from deepsc_preprocess import SeqDataset, collate_batch, normalize_text, tokenize
from deepsc_replication import bleu_scores, decode_tokens, load_repo_data, masked_ce_loss, set_seed
from semantic_metrics import SBERTScorer
from ut_model import DeepSCWrapper, UTSemCom

# NLTK warns on every sentence with no higher-order n-gram overlap; that is
# expected at low SNR and floods the console.
warnings.filterwarnings("ignore", module="nltk.translate.bleu_score")

ALL_BLEU = ["bleu1", "bleu2", "bleu3", "bleu4"]

# Used only if deepsc_replication_config.json is not found. Mirrors the
# defaults documented in README.md so all models share the same data setup.
DEFAULT_CONFIG = {
    "dataset": {
        "seed": 42,
        "raw_text_dir": "europarl/text",
        "train_pickle": "europarl/train_data.pkl",
        "test_pickle": "europarl/test_data.pkl",
        "vocab_json": "europarl/vocab.json",
        "preprocess_if_missing": True,
        "special_tokens": ["<PAD>", "<START>", "<END>", "<UNK>"],
        "min_sentence_len": 4,
        "max_len": 30,
        "min_freq": 1,
        "punct_to_keep": [";", ","],
        "punct_to_remove": ["?", "."],
    },
    "model": {
        "d_model": 128,
        "nhead": 8,
        "num_enc_layers": 4,
        "num_dec_layers": 4,
        "dim_feedforward": 512,
        "channel_dim": 16,
        "channel_hidden": 256,
        "channel_decoder_hidden": 512,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 20,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
        "betas": [0.9, 0.98],
        "eps": 1e-8,
        "num_workers": 0,
    },
    "channel": {
        "train_snr_min_db": 5,
        "train_snr_max_db": 10,
        "eval_snr_db": [0, 3, 6, 9, 12, 15, 18],
    },
    "evaluation": {"bleu_mode": "sentence", "eval_split": "test"},
}


def deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["deepsc", "ut", "ut_snr"], default="ut")
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--config", default="deepsc_replication_config.json",
                   help="Shared data/training config. Falls back to built-in defaults if missing.")
    p.add_argument("--epochs", type=int, default=None, help="Override training.num_epochs")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train_snr_min", type=float, default=None)
    p.add_argument("--train_snr_max", type=float, default=None)
    p.add_argument("--eval_snrs", type=float, nargs="+", default=None)

    # DeepSC depth (paper's baseline uses 3 layers; repo default is 4)
    p.add_argument("--num_layers", type=int, default=None, help="DeepSC encoder/decoder layers")

    # UT / ACT hyper-parameters (paper: m = 5, T = 0.9)
    p.add_argument("--max_steps", type=int, default=5, help="Maximum UT recurrent steps m")
    p.add_argument("--act_threshold", type=float, default=0.9, help="ACT halting threshold T")
    p.add_argument("--ponder_weight", type=float, default=0.01,
                   help="Weight on ponder cost rho in L = L_CE + w * rho (paper uses w = 1)")

    p.add_argument("--max_train_sentences", type=int, default=None, help="Subsample training set (quick runs)")
    p.add_argument("--max_eval_sentences", type=int, default=2000, help="Test sentences used per SNR point")
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--val_snr", type=float, default=6.0, help="SNR used for checkpoint selection")
    p.add_argument("--example_snrs", type=float, nargs="+", default=[0, 6, 12])
    p.add_argument("--num_examples", type=int, default=200, help="Saved example sentences per SNR")

    p.add_argument("--ood_text", default=None,
                   help="Optional .txt file (one sentence per line) of unseen-domain text for generalisation test")
    p.add_argument("--no_sbert", action="store_true", help="Skip SBERT semantic similarity")
    p.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--skip_train", action="store_true", help="Load existing checkpoint and only evaluate")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_every", type=int, default=200)
    return p.parse_args()


def build_config(args, root: Path) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = root / args.config
    if cfg_path.exists():
        deep_update(config, json.loads(cfg_path.read_text(encoding="utf-8")))
        print(f"Loaded config: {cfg_path.name}")
    else:
        print(f"Config {cfg_path.name} not found, using built-in defaults.")

    tr, ch = config["training"], config["channel"]
    if args.epochs is not None:
        tr["num_epochs"] = args.epochs
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size
    if args.lr is not None:
        tr["learning_rate"] = args.lr
    if args.train_snr_min is not None:
        ch["train_snr_min_db"] = args.train_snr_min
    if args.train_snr_max is not None:
        ch["train_snr_max_db"] = args.train_snr_max
    if args.eval_snrs is not None:
        ch["eval_snr_db"] = args.eval_snrs
    if args.num_layers is not None:
        config["model"]["num_enc_layers"] = args.num_layers
        config["model"]["num_dec_layers"] = args.num_layers
    if args.seed is not None:
        config["dataset"]["seed"] = args.seed
    return config


def build_model(args, config, vocab_size, pad_id, max_len):
    m = config["model"]
    common = dict(
        vocab_size=vocab_size,
        d_model=int(m["d_model"]),
        nhead=int(m["nhead"]),
        dim_feedforward=int(m["dim_feedforward"]),
        channel_dim=int(m["channel_dim"]),
        channel_hidden=int(m["channel_hidden"]),
        channel_decoder_hidden=int(m["channel_decoder_hidden"]),
        max_len=max_len,
        pad_id=pad_id,
        dropout=float(m["dropout"]),
    )
    if args.model == "deepsc":
        return DeepSCWrapper(
            num_enc_layers=int(m["num_enc_layers"]),
            num_dec_layers=int(m["num_dec_layers"]),
            **common,
        )
    return UTSemCom(
        max_steps=args.max_steps,
        act_threshold=args.act_threshold,
        snr_aware_halting=(args.model == "ut_snr"),
        **common,
    )


def make_loader(samples, pad_id, batch_size, shuffle=False):
    return DataLoader(
        SeqDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
    )


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, pad_id, args, snr_range, epoch):
    model.train()
    total_loss, total_ce, total_ponder, n_seen = 0.0, 0.0, 0.0, 0
    for step, batch in enumerate(loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        snr_db = random.uniform(*snr_range)  # new SNR for every batch

        optimizer.zero_grad()
        logits, aux = model(input_ids, attention_mask, snr_db=snr_db, channel=args.channel)
        ce = masked_ce_loss(logits[:, :-1], input_ids[:, 1:], pad_id, criterion)
        loss = ce + args.ponder_weight * aux["ponder"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bsz = input_ids.size(0)
        n_seen += bsz
        total_loss += loss.item() * bsz
        total_ce += ce.item() * bsz
        total_ponder += float(aux["ponder"]) * bsz
        if args.log_every and step % args.log_every == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} loss={total_loss / n_seen:.4f} "
                  f"ce={total_ce / n_seen:.4f} ponder={total_ponder / n_seen:.3f}")
    return total_loss / n_seen, total_ce / n_seen, total_ponder / n_seen


@torch.no_grad()
def evaluate_ce(model, loader, criterion, device, pad_id, snr_db, channel):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits, _ = model(input_ids, attention_mask, snr_db=snr_db, channel=channel)
        total += masked_ce_loss(logits[:, :-1], input_ids[:, 1:], pad_id, criterion).item() * input_ids.size(0)
        n += input_ids.size(0)
    return total / max(n, 1)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def run_snr_point(model, loader, snr_db, channel, token_ids, itos, unk_token, device):
    """Greedy-decode every sentence at one SNR. Returns refs, hyps and per-sentence stats."""
    special_ids = {token_ids["pad"], token_ids["start"], token_ids["end"], token_ids["unk"]}
    eos_id = token_ids["end"]
    refs, hyps, enc_steps, dec_steps = [], [], [], []
    elapsed, n_sent = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_ids, stats = model.generate(
            input_ids, attention_mask, snr_db=snr_db,
            sos_id=token_ids["start"], eos_id=eos_id, channel=channel,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        n_sent += input_ids.size(0)

        for i in range(input_ids.size(0)):
            ref = decode_tokens(input_ids[i].cpu().tolist(), itos, special_ids, eos_id, unk_token)
            if not ref:
                continue
            hyp = decode_tokens(out_ids[i].cpu().tolist(), itos, special_ids, eos_id, unk_token)
            refs.append(ref)
            hyps.append(hyp if hyp else ["the"])
            enc_steps.append(float(stats["enc_steps"][i]))
            dec_steps.append(float(stats["dec_steps"][i]))
    return refs, hyps, enc_steps, dec_steps, 1000.0 * elapsed / max(n_sent, 1)


def evaluate_over_snr(model, loader, snr_values, args, config, token_ids, itos, device, sbert, tag=""):
    unk_token = config["dataset"]["special_tokens"][3]
    bleu_mode = config["evaluation"].get("bleu_mode", "sentence")
    smooth = SmoothingFunction().method1
    rows, examples = [], []

    for snr in snr_values:
        refs, hyps, enc_steps, dec_steps, ms = run_snr_point(
            model, loader, snr, args.channel, token_ids, itos, unk_token, device
        )
        scores = bleu_scores(refs, hyps, bleu_mode=bleu_mode, report_metrics=ALL_BLEU)

        ref_txt = [" ".join(r) for r in refs]
        hyp_txt = [" ".join(h) for h in hyps]
        sims = sbert.pairwise(ref_txt, hyp_txt) if sbert is not None else [math.nan] * len(refs)
        valid_sims = [s for s in sims if not math.isnan(s)]
        sbert_mean = sum(valid_sims) / len(valid_sims) if valid_sims else math.nan
        exact = sum(r == h for r, h in zip(refs, hyps)) / max(len(refs), 1)

        enc_avg = sum(enc_steps) / len(enc_steps)
        dec_avg = sum(dec_steps) / len(dec_steps)
        row = {
            "snr_db": snr,
            **scores,
            "sbert": sbert_mean,
            "exact_match": exact,
            "enc_steps": enc_avg,
            "dec_steps": dec_avg,
            "layer_passes": enc_avg + dec_avg,
            "ms_per_sentence": ms,
            "n_sentences": len(refs),
        }
        rows.append(row)
        print(f"{tag}SNR {snr:>5} dB | " + " ".join(f"{m.upper()}={scores[m]:.4f}" for m in ALL_BLEU)
              + f" SBERT={sbert_mean:.4f} enc_steps={enc_avg:.2f} dec_steps={dec_avg:.2f} {ms:.2f} ms/sent")

        if snr in args.example_snrs:
            for i in range(min(args.num_examples, len(refs))):
                examples.append({
                    "snr_db": snr,
                    "reference": ref_txt[i],
                    "received": hyp_txt[i],
                    "bleu4": sentence_bleu([refs[i]], hyps[i], smoothing_function=smooth),
                    "bleu1": sentence_bleu([refs[i]], hyps[i], weights=(1, 0, 0, 0)),
                    "sbert": sims[i],
                    "enc_steps": enc_steps[i],
                    "dec_steps": dec_steps[i],
                    "n_words": len(refs[i]),
                })
    return pd.DataFrame(rows), pd.DataFrame(examples)


def load_ood_samples(path: Path, config, stoi, max_len):
    """Unseen-domain sentences mapped with the training vocabulary (OOV -> <UNK>)."""
    ds = config["dataset"]
    pad_t, start_t, end_t, unk_t = ds["special_tokens"]
    unk_id = stoi[unk_t]
    samples, n_tok, n_unk = [], 0, 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = normalize_text(line)
        n_words = len(text.split())
        if not text or not (int(ds["min_sentence_len"]) < n_words < int(ds["max_len"])):
            continue
        toks = tokenize(text, start_t, end_t, ds.get("punct_to_keep", [";", ","]), ds.get("punct_to_remove", ["?", "."]))
        toks = toks[:max_len]
        ids = [stoi.get(t, unk_id) for t in toks]
        n_tok += len(ids) - 2
        n_unk += sum(i == unk_id for i in ids)
        samples.append(ids)
    return samples, (n_unk / n_tok if n_tok else 0.0)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_results(df, prefix, title, out_dir: Path, is_ut: bool):
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, marker in zip(ALL_BLEU, ["o", "s", "^", "d"]):
        ax.plot(df["snr_db"], df[metric], marker=marker, label=metric.upper())
    if df["sbert"].notna().any():
        ax.plot(df["snr_db"], df["sbert"], marker="*", linestyle="--", color="black", label="SBERT")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{title}: BLEU / SBERT vs SNR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_bleu_sbert_snr.png", dpi=150)
    plt.close(fig)

    if is_ut:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(df["snr_db"], df["enc_steps"], marker="o", label="Encoder steps")
        ax.plot(df["snr_db"], df["dec_steps"], marker="s", label="Decoder steps")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Average ACT steps per token")
        ax.set_title(f"{title}: adaptive computation vs SNR")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_steps_snr.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = build_config(args, root)
    set_seed(int(config["dataset"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix = f"{args.model}_{args.channel}"
    title = {"deepsc": "DeepSC", "ut": "UT + ACT", "ut_snr": "UT + SNR-aware ACT"}[args.model] + f" ({args.channel.upper()})"
    print(f"Run: {prefix} | device={device}")

    loaders, stoi, itos, pad_id, token_ids, max_seq_len = load_repo_data(config, root)
    max_len = max(int(config["dataset"]["max_len"]), int(max_seq_len))
    bs = int(config["training"]["batch_size"])

    train_samples = list(loaders["train"].dataset.encoded_samples)
    if args.max_train_sentences:
        random.shuffle(train_samples)
        train_samples = train_samples[: args.max_train_sentences]
    test_samples = list(loaders[config["evaluation"].get("eval_split", "test")].dataset.encoded_samples)
    eval_samples = test_samples[: args.max_eval_sentences] if args.max_eval_sentences else test_samples

    train_loader = make_loader(train_samples, pad_id, bs, shuffle=True)
    val_loader = make_loader(eval_samples, pad_id, args.eval_batch_size)
    print(f"Train={len(train_samples)} Eval={len(eval_samples)} Vocab={len(stoi)} MaxLen={max_len}")

    model = build_model(args, config, len(stoi), pad_id, max_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    ckpt_path = root / f"{prefix}_best.pt"
    criterion = nn.CrossEntropyLoss(reduction="none")
    history = []

    if not args.skip_train:
        tr = config["training"]
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(tr["learning_rate"]), betas=tuple(tr.get("betas", [0.9, 0.98])),
            eps=float(tr.get("eps", 1e-8)), weight_decay=float(tr.get("weight_decay", 0.0)),
        )
        snr_range = (float(config["channel"]["train_snr_min_db"]), float(config["channel"]["train_snr_max_db"]))
        best_val = float("inf")
        for epoch in range(1, int(tr["num_epochs"]) + 1):
            t0 = time.time()
            loss, ce, ponder = train_one_epoch(model, train_loader, optimizer, criterion, device, pad_id, args, snr_range, epoch)
            val = evaluate_ce(model, val_loader, criterion, device, pad_id, args.val_snr, args.channel)
            history.append({"epoch": epoch, "train_loss": loss, "train_ce": ce, "ponder": ponder, "val_ce": val})
            print(f"Epoch {epoch:>2}: loss={loss:.4f} ce={ce:.4f} ponder={ponder:.3f} "
                  f"val_ce@{args.val_snr:g}dB={val:.4f} ({time.time() - t0:.0f}s)")
            if val < best_val:
                best_val = val
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> saved {ckpt_path.name}")
        pd.DataFrame(history).to_csv(root / f"{prefix}_train_history.csv", index=False)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint {ckpt_path.name}; train first (drop --skip_train).")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    sbert = None if args.no_sbert else SBERTScorer(args.sbert_model, device=str(device))
    snr_values = [float(s) for s in config["channel"]["eval_snr_db"]]

    df, examples = evaluate_over_snr(model, val_loader, snr_values, args, config, token_ids, itos, device, sbert)
    df.insert(0, "model", prefix)
    df.to_csv(root / f"{prefix}_results_latest.csv", index=False)
    if not examples.empty:
        examples.to_csv(root / f"{prefix}_examples.csv", index=False)

    payload = {
        "model": prefix,
        "architecture": args.model,
        "channel": args.channel,
        "parameters": n_params,
        "seed": int(config["dataset"]["seed"]),
        "checkpoint": ckpt_path.name,
        "act": None if args.model == "deepsc" else {
            "max_steps": args.max_steps, "threshold": args.act_threshold, "ponder_weight": args.ponder_weight,
        },
        "snr_values": snr_values,
        "bleu_by_snr": {
            str(int(r["snr_db"])): {m: float(r[m]) for m in ALL_BLEU} for _, r in df.iterrows()
        },
        "metrics_by_snr": {
            str(int(r["snr_db"])): {
                k: (None if isinstance(r[k], float) and math.isnan(r[k]) else float(r[k]))
                for k in ALL_BLEU + ["sbert", "exact_match", "enc_steps", "dec_steps", "layer_passes", "ms_per_sentence"]
            }
            for _, r in df.iterrows()
        },
    }

    if args.ood_text:
        ood_path = Path(args.ood_text)
        ood_samples, unk_rate = load_ood_samples(ood_path, config, stoi, max_len)
        if args.max_eval_sentences:
            ood_samples = ood_samples[: args.max_eval_sentences]
        print(f"OOD set {ood_path.name}: {len(ood_samples)} sentences, UNK rate {unk_rate:.2%}")
        if ood_samples:
            ood_loader = make_loader(ood_samples, pad_id, args.eval_batch_size)
            ood_args = argparse.Namespace(**{**vars(args), "example_snrs": []})
            ood_df, _ = evaluate_over_snr(model, ood_loader, snr_values, ood_args, config, token_ids, itos, device, sbert, tag="[OOD] ")
            ood_df.insert(0, "model", prefix)
            ood_df["unk_rate"] = unk_rate
            ood_df.to_csv(root / f"{prefix}_ood_results.csv", index=False)
            payload["ood"] = {"file": ood_path.name, "unk_rate": unk_rate, "n_sentences": len(ood_samples)}

    with (root / f"{prefix}_results_latest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_results(df, prefix, title, root, is_ut=args.model != "deepsc")
    print(f"Saved results with prefix '{prefix}_*' in {root}")


if __name__ == "__main__":
    main()
