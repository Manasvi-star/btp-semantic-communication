import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu

import deepsc_replication as ds


def save_bleu_plot(df: pd.DataFrame, out_path: Path, metrics: list[str]) -> None:
    plt.figure(figsize=(7, 5))
    markers = ["o", "s", "^", "d"]
    for idx, metric in enumerate(metrics):
        plt.plot(df["snr_db"], df[metric], marker=markers[idx % len(markers)], label=metric.upper())

    plt.xlabel("SNR (dB)")
    plt.ylabel("BLEU")
    plt.title("DeepSC Eval: BLEU1-4 vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_model(config: dict, stoi: dict, pad_id: int, max_seq_len: int, device: torch.device):
    model_cfg = config["model"]
    model_max_len = max(int(config["dataset"]["max_len"]), int(max_seq_len))
    model = ds.DeepSCBaseline(
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
    return model


@torch.no_grad()
def benchmark_generation_speed(model, loader, token_ids, device, snr_db: float, max_samples: int = 20) -> float:
    sos_id = token_ids["start"]
    eos_id = token_ids["end"]
    noise_std = ds.snr_to_noise_std(snr_db)

    processed = 0
    t0 = time.perf_counter()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        for i in range(input_ids.size(0)):
            inp = input_ids[i : i + 1]
            msk = attention_mask[i : i + 1]
            _ = model.generate(inp, msk, noise_std=noise_std, sos_id=sos_id, eos_id=eos_id)
            processed += 1
            if processed >= max_samples:
                elapsed = time.perf_counter() - t0
                return elapsed / max(1, processed)

    elapsed = time.perf_counter() - t0
    return elapsed / max(1, processed)


@torch.no_grad()
def sampled_reconstruction(model, loader, snr_values, token_ids, itos, device, num_samples, seed, unk_token):
    special_ids = {token_ids["pad"], token_ids["start"], token_ids["end"], token_ids["unk"]}
    data = loader.dataset.encoded_samples

    rng = random.Random(seed)
    selected = rng.sample(range(len(data)), k=min(num_samples, len(data)))

    rows = []
    for i, sample_idx in enumerate(selected, start=1):
        encoded_ids = data[sample_idx]
        inp = torch.tensor(encoded_ids, dtype=torch.long, device=device).unsqueeze(0)
        msk = torch.ones_like(inp)

        ref_tokens = ds.decode_tokens(encoded_ids, itos, special_ids, token_ids["end"], unk_token)
        reference = " ".join(ref_tokens)

        for snr in snr_values:
            noise_std = ds.snr_to_noise_std(float(snr))
            out_ids = model.generate(inp, msk, noise_std=noise_std, sos_id=token_ids["start"], eos_id=token_ids["end"])
            hyp_tokens = ds.decode_tokens(out_ids.squeeze(0).tolist(), itos, special_ids, token_ids["end"], unk_token)
            hypothesis = " ".join(hyp_tokens) if hyp_tokens else "<empty>"

            hyp_for_bleu = hyp_tokens if hyp_tokens else ["the"]
            bleu1 = sentence_bleu([ref_tokens], hyp_for_bleu, weights=(1.0, 0, 0, 0)) if ref_tokens else 0.0
            bleu2 = sentence_bleu([ref_tokens], hyp_for_bleu, weights=(0.5, 0.5, 0, 0)) if ref_tokens else 0.0
            bleu3 = sentence_bleu([ref_tokens], hyp_for_bleu, weights=(1 / 3, 1 / 3, 1 / 3, 0)) if ref_tokens else 0.0
            bleu4 = sentence_bleu([ref_tokens], hyp_for_bleu, weights=(0.25, 0.25, 0.25, 0.25)) if ref_tokens else 0.0

            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "snr_db": float(snr),
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "bleu1": float(bleu1),
                    "bleu2": float(bleu2),
                    "bleu3": float(bleu3),
                    "bleu4": float(bleu4),
                }
            )

        if i % 5 == 0 or i == len(selected):
            print(f"Reconstruction progress: {i}/{len(selected)} samples")

    return pd.DataFrame(rows).sort_values(["sample_idx", "snr_db"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepSC checkpoint (BLEU1-4) and run sampled reconstructions.")
    parser.add_argument("--num-recon-samples", type=int, default=20, help="Number of reconstruction samples (recommended 15-20).")
    parser.add_argument("--benchmark-samples", type=int, default=20, help="Samples used for runtime estimate benchmark.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = ds.load_config(root / "deepsc_replication_config.json")

    seed = int(config["dataset"]["seed"])
    ds.set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, stoi, itos, pad_id, token_ids, max_seq_len = ds.load_repo_data(config, root)
    eval_split = config["evaluation"]["eval_split"]
    eval_loader = loaders[eval_split]
    snr_values = config["channel"]["eval_snr_db"]

    model = build_model(config, stoi, pad_id, max_seq_len, device)
    ckpt_path = root / config["artifacts"]["checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path.name}")

    sec_per_sample = benchmark_generation_speed(
        model=model,
        loader=eval_loader,
        token_ids=token_ids,
        device=device,
        snr_db=float(snr_values[0]),
        max_samples=max(1, int(args.benchmark_samples)),
    )

    n_eval = len(eval_loader.dataset)
    n_snr = len(snr_values)
    est_bleu_sec = sec_per_sample * n_eval * n_snr
    est_recon_sec = sec_per_sample * min(int(args.num_recon_samples), n_eval) * n_snr
    print(
        "Estimated runtime | "
        f"BLEU1-4 eval: ~{est_bleu_sec / 60:.1f} min, "
        f"sampled reconstruction: ~{est_recon_sec / 60:.1f} min"
    )

    report_metrics = ["bleu1", "bleu2", "bleu3", "bleu4"]
    print("\nPhase 1/2: BLEU2/3/4 (and BLEU1) evaluation across SNRs")
    t_bleu = time.perf_counter()
    bleu_df = ds.evaluate_bleu_snr(
        model,
        eval_loader,
        snr_values,
        token_ids,
        itos,
        device,
        bleu_mode=config["evaluation"]["bleu_mode"],
        report_metrics=report_metrics,
        unk_token=config["dataset"]["special_tokens"][3],
    )
    bleu_elapsed = time.perf_counter() - t_bleu

    bleu_csv = root / "deepsc_results_all_bleu_latest.csv"
    bleu_json = root / "deepsc_results_all_bleu_latest.json"
    bleu_png = root / "deepsc_results_all_bleu_latest.png"
    bleu_df.to_csv(bleu_csv, index=False)

    payload = {
        "model": "deepsc_replication_baseline",
        "seed": seed,
        "checkpoint": ckpt_path.name,
        "snr_values": snr_values,
        "bleu_by_snr": {
            str(int(row["snr_db"])): {metric: float(row[metric]) for metric in report_metrics}
            for _, row in bleu_df.iterrows()
        },
    }
    with bleu_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_bleu_plot(bleu_df, bleu_png, report_metrics)

    print("\nPhase 2/2: Reconstruction for selected samples across all SNRs")
    t_recon = time.perf_counter()
    recon_df = sampled_reconstruction(
        model=model,
        loader=eval_loader,
        snr_values=snr_values,
        token_ids=token_ids,
        itos=itos,
        device=device,
        num_samples=max(1, int(args.num_recon_samples)),
        seed=seed,
        unk_token=config["dataset"]["special_tokens"][3],
    )
    recon_elapsed = time.perf_counter() - t_recon

    recon_csv = root / "deepsc_reconstruction_samples_all_snrs.csv"
    recon_df.to_csv(recon_csv, index=False)

    print(f"Saved: {bleu_csv.name}")
    print(f"Saved: {bleu_json.name}")
    print(f"Saved: {bleu_png.name}")
    print(f"Saved: {recon_csv.name}")
    print(
        "Actual runtime | "
        f"BLEU1-4 eval: {bleu_elapsed / 60:.1f} min, "
        f"sampled reconstruction: {recon_elapsed / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
