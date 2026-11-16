"""
Combine the outputs of ut_experiment.py runs into report-ready plots/tables.

Reads every available <model>_<channel>_results_latest.csv for
model in {deepsc, ut, ut_snr} and channel in {awgn, rayleigh}, then writes:

    ut_compare_bleu4_sbert.png     BLEU-4 and SBERT vs SNR, one panel per channel
    ut_compare_bleu_all.png        BLEU-1..4 vs SNR for every model/channel
    ut_compare_steps.png           layer passes (compute) vs SNR
    ut_compare_summary.csv         every metric, every model, every SNR
    ut_compare_compute.csv         compute saving of UT models vs DeepSC
    ut_compare_semantic_wins.csv   sentences with low BLEU but high SBERT
                                   (meaning kept, words changed)
    ut_compare_ood.csv             unseen-text results (if any *_ood_results.csv)

Usage:
    python compare_ut_results.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = {"deepsc": "DeepSC (fixed depth)", "ut": "UT + ACT", "ut_snr": "UT + SNR-aware ACT"}
CHANNELS = ["awgn", "rayleigh"]
STYLE = {"deepsc": ("tab:gray", "o"), "ut": ("tab:blue", "s"), "ut_snr": ("tab:orange", "^")}


def load(kind: str = "results_latest") -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for channel in CHANNELS:
            path = ROOT / f"{model}_{channel}_{kind}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["arch"], df["channel"] = model, channel
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_channel_plot(df, metrics, ylabel, out_name, ylim=None):
    channels = [c for c in CHANNELS if c in set(df["channel"])]
    fig, axes = plt.subplots(len(metrics), len(channels), figsize=(6 * len(channels), 4.2 * len(metrics)), squeeze=False)
    for r, metric in enumerate(metrics):
        for c, channel in enumerate(channels):
            ax = axes[r][c]
            sub = df[df["channel"] == channel]
            for arch, label in MODELS.items():
                d = sub[sub["arch"] == arch].sort_values("snr_db")
                if d.empty or d[metric].isna().all():
                    continue
                color, marker = STYLE[arch]
                ax.plot(d["snr_db"], d[metric], marker=marker, color=color, label=label)
            ax.set_title(f"{metric.upper()} - {channel.upper()}")
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel if isinstance(ylabel, str) else ylabel[r])
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / out_name, dpi=150)
    plt.close(fig)
    print(f"Saved {out_name}")


def main():
    df = load()
    if df.empty:
        print("No *_results_latest.csv files from ut_experiment.py found. Run ut_experiment.py first.")
        return

    cols = ["arch", "channel", "snr_db", "bleu1", "bleu2", "bleu3", "bleu4", "sbert", "exact_match",
            "enc_steps", "dec_steps", "layer_passes", "ms_per_sentence"]
    summary = df[cols].sort_values(["channel", "snr_db", "arch"])
    summary.to_csv(ROOT / "ut_compare_summary.csv", index=False)
    print("Saved ut_compare_summary.csv")

    has_sbert = df["sbert"].notna().any()
    per_channel_plot(df, ["bleu4", "sbert"] if has_sbert else ["bleu4"], "Score", "ut_compare_bleu4_sbert.png", (0, 1.02))
    per_channel_plot(df, ["bleu1", "bleu2", "bleu3", "bleu4"], "BLEU", "ut_compare_bleu_all.png", (0, 1.02))
    per_channel_plot(df, ["layer_passes", "ms_per_sentence"], ["Avg layer passes (enc + dec)", "ms / sentence"],
                     "ut_compare_steps.png")

    # Compute saving of UT models relative to DeepSC, at every SNR
    rows = []
    for channel in CHANNELS:
        base = df[(df["arch"] == "deepsc") & (df["channel"] == channel)].set_index("snr_db")
        if base.empty:
            continue
        for arch in ["ut", "ut_snr"]:
            d = df[(df["arch"] == arch) & (df["channel"] == channel)].set_index("snr_db")
            for snr in d.index.intersection(base.index):
                rows.append({
                    "channel": channel, "arch": arch, "snr_db": snr,
                    "bleu4": d.at[snr, "bleu4"], "bleu4_deepsc": base.at[snr, "bleu4"],
                    "bleu4_gain": d.at[snr, "bleu4"] - base.at[snr, "bleu4"],
                    "sbert_gain": d.at[snr, "sbert"] - base.at[snr, "sbert"],
                    "layer_passes": d.at[snr, "layer_passes"],
                    "layer_passes_deepsc": base.at[snr, "layer_passes"],
                    "compute_saving_pct": 100 * (1 - d.at[snr, "layer_passes"] / base.at[snr, "layer_passes"]),
                    "time_ratio_vs_deepsc": d.at[snr, "ms_per_sentence"] / base.at[snr, "ms_per_sentence"],
                })
    if rows:
        pd.DataFrame(rows).to_csv(ROOT / "ut_compare_compute.csv", index=False)
        print("Saved ut_compare_compute.csv")

    # Sentences where words changed but meaning survived
    ex_frames = []
    for model in MODELS:
        for channel in CHANNELS:
            p = ROOT / f"{model}_{channel}_examples.csv"
            if p.exists():
                e = pd.read_csv(p)
                e["arch"], e["channel"] = model, channel
                ex_frames.append(e)
    if ex_frames:
        ex = pd.concat(ex_frames, ignore_index=True)
        if ex["sbert"].notna().any():
            wins = ex[(ex["bleu4"] < 0.3) & (ex["sbert"] > 0.8)].sort_values("sbert", ascending=False)
            wins.to_csv(ROOT / "ut_compare_semantic_wins.csv", index=False)
            print(f"Saved ut_compare_semantic_wins.csv ({len(wins)} sentences with BLEU-4<0.3 but SBERT>0.8)")

            ex["n_words_bin"] = pd.cut(ex["n_words"], [0, 8, 15, 22, 40], labels=["<=8", "9-15", "16-22", "23+"])
            ut_ex = ex[ex["arch"] != "deepsc"]
            if not ut_ex.empty:
                by_len = ut_ex.groupby(["arch", "channel", "snr_db", "n_words_bin"], observed=True)[["enc_steps", "dec_steps", "bleu4"]].mean()
                by_len.to_csv(ROOT / "ut_compare_steps_by_length.csv")
                print("Saved ut_compare_steps_by_length.csv")

    ood = load("ood_results")
    if not ood.empty:
        ood[[c for c in cols + ["unk_rate"] if c in ood.columns]].to_csv(ROOT / "ut_compare_ood.csv", index=False)
        print("Saved ut_compare_ood.csv")

    pivot = summary.pivot_table(index=["channel", "snr_db"], columns="arch", values="bleu4")
    print("\nBLEU-4 by SNR:\n", pivot.round(4).to_string())


if __name__ == "__main__":
    main()
