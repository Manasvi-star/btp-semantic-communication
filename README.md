# Semantic Communication Over Noisy Channels

This repository contains a full project comparing three approaches for sentence transmission under channel noise:

- Traditional dense autoencoder baseline
- Semantic model pipeline (notebook-driven)
- DeepSC-style Transformer replication pipeline

All approaches are evaluated with BLEU metrics across SNR sweeps and saved as CSV/JSON/plot artifacts for analysis.

## Project Scope

The project studies how well natural-language meaning survives channel corruption (AWGN) after encoding, transmission, and decoding.

Main goals:

- Build and train multiple semantic communication models on Europarl text
- Compare reconstruction quality across SNR values
- Produce consistent artifacts for report writing and model comparison

## Repository Layout

Top-level:

- `README.md`: Project overview and runbook (this file)
- `btp_env/`: Main implementation, notebooks, checkpoints, and results

Core files inside `btp_env/`:

- `data_prep.py`: Shared text cleaning, vocabulary, encoding, dataloaders
- `traditional_model.ipynb`: Traditional autoencoder training and evaluation
- `semantic_model.ipynb`: Semantic/Transformer notebook experiments
- `deepsc_replication.py`: Scripted DeepSC baseline training and evaluation
- `deepsc_replication_config.json`: Frozen config for DeepSC replication
- `deepsc_preprocess.py`: Repo-style Europarl preprocessing for replication mode
- `deepsc_eval_and_reconstruct.py`: DeepSC reconstruction/evaluation helper script
- `compare_results.ipynb`: Cross-model metric comparison and plotting

Common artifacts produced in `btp_env/`:

- Checkpoints: `traditional_best.pt`, `semantic_best.pt`, `semantic_best_v5.pt`, `deepsc_best.pt`
- Metrics: `*_results_latest.csv`, `*_results_latest.json`, `*_results_all_bleu_latest.csv`
- Plots: `*_bleu_snr*.png`, `compare_all_models_bleu1234_vs_snr.png`

## Model Families

### 1) Traditional Baseline (Dense Autoencoder)

Implemented in `traditional_model.ipynb`.

High-level flow:

1. Token IDs -> learnable `nn.Embedding`
2. Masked mean pooling across sequence
3. Dense encoder to latent vector
4. AWGN noise added in latent space
5. Dense decoder reconstructs per-token representations
6. Linear projection to vocabulary logits

Notes:

- This is not Transformer-based.
- Word embeddings are learned jointly during end-to-end training.

### 2) Semantic Model (Notebook Pipeline)

Implemented in `semantic_model.ipynb`.

Highlights:

- Transformer-based semantic communication architecture
- AWGN-based channel simulation
- BLEU-focused training/evaluation workflows
- Additional analysis artifacts for reconstructions and SNR behavior

### 3) DeepSC Replication Baseline (Script Pipeline)

Implemented in `deepsc_replication.py`, configured by `deepsc_replication_config.json`.

Architecture in replication script:

- Token embedding + positional encoding
- Transformer semantic encoder
- Channel encoder -> AWGN -> channel decoder
- Transformer semantic decoder
- Output projection to vocabulary logits

Default config snapshot:

- Dataset: local Europarl text from `europarl/text`
- Sequence length: 30
- `d_model=128`, `nhead=8`, `enc_layers=4`, `dec_layers=4`
- Channel bottleneck: 16
- Train SNR sampled in [5, 10] dB
- Eval SNRs: [0, 3, 6, 9, 12, 15, 18]

## Data Pipeline

### Shared Data Pipeline (`data_prep.py`)

- Cleaning and tokenization for Europarl (or WikiText fallback)
- Vocabulary construction with special tokens:
	- `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Fixed-length encoding
- Train/validation/test loaders and attention masks

Important defaults:

- `MAX_LEN = 30`
- `MAX_VOCAB_SIZE = 20000`
- `MIN_FREQ = 1`

### Replication Data Pipeline (`deepsc_preprocess.py`)

- Repo-aligned normalization/tokenization style
- Special tokens:
	- `<PAD>`, `<START>`, `<END>`, `<UNK>`
- Saved artifacts:
	- `europarl/train_data.pkl`
	- `europarl/test_data.pkl`
	- `europarl/vocab.json`

## Setup

## 1) Prerequisites

- Windows or Linux
- Python 3.10+ recommended
- Local Europarl text files in `btp_env/europarl/text`

## 2) Environment

From repository root:

```powershell
cd btp_env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you use the committed environment already present under `btp_env`, activate that environment instead.

## 3) Install dependencies

```powershell
pip install torch datasets nltk pandas matplotlib sentence-transformers
```

If NLTK token/metric resources are missing during runs, install them once inside Python.

## Run Workflows

### A) Traditional Baseline Notebook

Open and run:

- `btp_env/traditional_model.ipynb`

Recommended order inside notebook:

1. Model definition and sanity check
2. Europarl load and statistics
3. Dataloader creation
4. Training loop and checkpoint save
5. BLEU evaluation and SNR sweep
6. Qualitative reconstruction analysis

### B) Semantic Notebook

Open and run:

- `btp_env/semantic_model.ipynb`

This notebook contains semantic model training/evaluation flows and saves semantic checkpoints/results.

### C) DeepSC Replication Script

From `btp_env`:

```powershell
python deepsc_replication.py
```

This script will:

1. Load config (`deepsc_replication_config.json`)
2. Build/load preprocessed Europarl artifacts
3. Train DeepSC baseline
4. Save best checkpoint
5. Run SNR evaluation
6. Save CSV/JSON/plot outputs

## Evaluation Artifacts

You will typically use these files for reporting:

- Traditional:
	- `traditional_results_latest.csv`
	- `traditional_results_latest.json`
	- `traditional_bleu_snr_latest.png`
- Semantic:
	- `semantic_results_latest.csv`
	- `semantic_results_latest.json`
	- `semantic_results_all_bleu_latest.csv`
	- `semantic_bleu1234_snr.png`
- DeepSC replication:
	- `deepsc_results_latest.csv`
	- `deepsc_results_latest.json`
	- `deepsc_bleu_snr.png`
- Cross-model comparison:
	- `compare_all_models_bleu1234_long.csv`
	- `compare_all_models_bleu1234_winners.csv`
	- `compare_all_models_bleu1234_gains.csv`
	- `compare_all_models_bleu1234_vs_snr.png`

## Reproducibility Notes

- Seeds are set in replication scripts.
- SNR handling is explicit and config-driven in `deepsc_replication_config.json`.
- Checkpoints and metrics are written with stable filenames for downstream comparison.
- `deepsc_replication_roadmap.md` documents parity expectations versus reference DeepSC behavior.

## Troubleshooting

- Missing dataset files:
	- Confirm text files exist under `btp_env/europarl/text`.
- CUDA not detected:
	- The code falls back to CPU automatically.
- Notebook import issues:
	- Ensure the selected kernel uses the same environment where dependencies were installed.
- Missing checkpoints on evaluation cells:
	- Run training cells/scripts first, then rerun evaluation cells.

## Phase 2: Adaptive Universal Transformer (UT + ACT)

Based on Zhou et al., *"Semantic Communication with Adaptive Universal Transformer"* (arXiv:2108.09119),
with the SBERT semantic-similarity metric from Lee et al., *"seq2seq-SC"* (arXiv:2210.15237).

### What is new compared to Phase 1

| | Phase 1 | Phase 2 |
|---|---|---|
| Semantic encoder/decoder | Fixed-depth Transformer (4 layers) | One shared Transformer block applied recurrently, depth chosen per token by ACT (max `m = 5`, threshold `T = 0.9`) |
| Channel | AWGN | AWGN **and Rayleigh block fading** (perfect CSI, zero-forcing) |
| Metrics | BLEU-1..4 | BLEU-1..4 **+ SBERT semantic similarity**, exact match |
| Compute analysis | none | average ACT steps per token, layer passes, ms / sentence |

Our extensions beyond the paper:

1. **SNR-aware halting** (`--model ut_snr`): the ACT halting unit also receives the channel SNR, so the
   model can choose how many steps to spend based on channel quality. In the original paper the halting
   unit never sees the SNR.
2. **Compute-vs-accuracy study**: layer passes and decode time are compared against fixed-depth DeepSC
   (`ut_compare_compute.csv`).
3. **Generalisation to unseen text** (`--ood_text file.txt`): evaluate on sentences from a different domain
   (e.g. Flickr captions), reporting the out-of-vocabulary rate.
4. **BLEU vs SBERT analysis**: `ut_compare_semantic_wins.csv` lists sentences whose words changed
   (low BLEU) but whose meaning survived (high SBERT).

### New files

- `ut_model.py`: ACT halting (paper Algorithm 1), `UTSemCom` model, AWGN/Rayleigh channels, `DeepSCWrapper`
  (existing DeepSC with the same interface + Rayleigh support)
- `semantic_metrics.py`: SBERT cosine similarity (falls back to NaN if the model cannot be loaded)
- `ut_experiment.py`: train + evaluate one model on one channel
- `compare_ut_results.py`: combines all runs into report plots/tables
- `requirements.txt`

`ut_experiment.py` reads the same `deepsc_replication_config.json` (data paths, `d_model`, batch size,
learning rate, SNR ranges) so all models use identical data. If that file is missing it uses built-in
defaults that match this README.

### Run

```powershell
pip install -r requirements.txt

# Baseline, paper model, and our extension, on both channels
python ut_experiment.py --model deepsc --channel awgn
python ut_experiment.py --model ut     --channel awgn
python ut_experiment.py --model ut_snr --channel awgn
python ut_experiment.py --model deepsc --channel rayleigh
python ut_experiment.py --model ut     --channel rayleigh
python ut_experiment.py --model ut_snr --channel rayleigh

# Optional: unseen-domain test on an existing checkpoint
python ut_experiment.py --model ut --channel awgn --skip_train --ood_text flickr_captions.txt

# Combine everything
python compare_ut_results.py
```

Useful flags: `--epochs`, `--max_train_sentences 100000` (quick runs), `--max_eval_sentences 2000`,
`--num_layers 3` (DeepSC depth used in the paper), `--max_steps`, `--act_threshold`, `--ponder_weight`,
`--train_snr_min/--train_snr_max`, `--eval_snrs`, `--no_sbert`.

### Outputs (prefix `<model>_<channel>`, e.g. `ut_snr_rayleigh`)

- `<prefix>_best.pt`, `<prefix>_train_history.csv`
- `<prefix>_results_latest.csv/.json`: all metrics per SNR. The JSON keeps the `bleu_by_snr` format, so
  these files can also be added to `compare_results.ipynb`.
- `<prefix>_examples.csv`: per-sentence reference / received text with BLEU, SBERT and ACT steps
- `<prefix>_bleu_sbert_snr.png`, `<prefix>_steps_snr.png`
- `ut_compare_*.png/.csv` from `compare_ut_results.py`

### Implementation notes

- **Ponder cost weight.** The paper writes `L = L_CE + rho`. With `m = 5`, an unweighted `rho` (about 2 to 5)
  can be much larger than the cross-entropy and push the model toward halting too early, so we use `L = L_CE + w * rho`
  with `w = 0.01` by default (`--ponder_weight 1.0` reproduces the paper exactly).
- **Learning rate.** The paper reports 1e-6, which is very slow to train; we use the shared config's rate.
- **Baseline depth.** The paper's Transformer baseline uses 3 layers; Phase 1 DeepSC uses 4. Use
  `--num_layers 3` for a paper-matched comparison, and state which one is reported.
- **Rayleigh channel.** Channel symbols are grouped into I/Q pairs (`channel_dim = 16` gives 8 complex symbols
  per word). One gain `h ~ CN(0,1)` is drawn per sentence and the receiver equalises with perfect CSI.
- **Training SNR.** A new SNR is drawn uniformly for every batch from the configured range. For Rayleigh a
  wider range (e.g. `--train_snr_min 0 --train_snr_max 20`) usually helps.
- **Decoding** is batched greedy decoding for all three models, so timings are comparable.

## Next Extensions

- Add a `requirements.txt` or `environment.yml` for exact dependency pinning
- Add script wrappers for notebook-equivalent training runs
- Add unified experiment registry (single table for all model runs)

