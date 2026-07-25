# LES-wrapper (ppiDCE): Learning Efficiency Score Evaluation

## Overview

The **LES-wrapper** automates evaluation of model trainability across ppiDCE
training checkpoints. It runs inference on PRS (Positive Reference Set) and RRS
(Random Reference Set) sequence-pair files at each per-epoch checkpoint, computes
ROC metrics, and derives integrated learning-efficiency scores.

This is the ppiDCE port of the LES-wrapper family
([ppiGPLM](https://github.com/kouroshSA/ppiGPLM),
[ppiBTEP](https://github.com/kouroshSA/ppiBTEP),
[ppiYYD](https://github.com/kouroshSA/ppiYYD)). The evaluation logic (PRS/RRS →
`prob_1` → ROC → AUC/Best-F1 → LES) is identical; only the model-specific glue
differs. **Its outputs match ppiGPLM's `LES-wrapper_v2.py`** — see
[Differences from the other wrappers](#differences-from-the-other-wrappers).

## What is LES?

LES (Learning Efficiency Score) is the **area under the metric-vs-epoch curve**.
Unlike metrics that capture only final performance, LES summarizes the entire
learning trajectory:

- **LES-AUC**: Area under the AUC trajectory curve
- **LES-F1**: Area under the Best-F1 trajectory curve

Epochs are normalized to `[0, 1]` before integration, so LES values are
comparable across runs of different length. Higher LES indicates faster, more
consistent learning across training.

> **v2 change (adopted here):** the optimal-F1 **threshold** metric is *not*
> reported. For non-discriminating controls the best-F1 threshold collapses
> toward 0 ("predict everything positive"), so it added noise. Dropped
> throughout: `trajectory_Threshold`, `LES-Threshold`, the `Best_F1_Threshold`
> summary column, the manifest `Threshold` entry, and the threshold panel of the
> combined figure.

## Workflow

For each checkpoint the wrapper:

1. Runs `inference_ppiDCE.py` on the PRS and RRS files
2. Extracts the positive-class probability (`prob_1`) for every pair
3. Combines PRS and RRS probabilities into a single file for ROC analysis
4. Draws a per-checkpoint probability-distribution plot (PRS vs RRS violins)
5. Computes AUC and Best-F1 and renders the ROC curve
6. Aggregates per-checkpoint results into a summary table
7. Plots metric trajectories and probability-distribution summaries across epochs
8. Computes LES for AUC and Best-F1

## Installation

Use the same `esm` environment as training/inference:

```bash
conda activate esm
pip install -r requirements.txt   # numpy, scikit-learn, matplotlib, pandas, torch, transformers
```

## Basic Usage

```bash
python LES-wrapper.py \
    --checkpoint_dir ROC_Checkpoints \
    --prs_file MED4_PRS-RRS/PRS-V3-1.csv \
    --rrs_file MED4_PRS-RRS/RRS-V3-1.csv \
    --output_dir LES_results_MED4 \
    --include_final
```

`--include_final` additionally evaluates `ppiDCE_final.pth` (plotted after the
last numbered epoch; it is excluded from the LES integral so it does not distort
the area).

## Input File Format

PRS and RRS files are CSVs read by `inference_ppiDCE.py` — only the first two
columns (`seq1`, `seq2`) are used; any third `label` column is ignored. The
wrapper assigns labels itself: every PRS pair is positive (1), every RRS pair is
negative (0).

> **Header note:** `inference_ppiDCE.py` reads input with pandas' default
> behavior, which treats the **first row as a header**. Give each PRS/RRS file a
> `seq1,seq2` header row; otherwise the first sequence pair is silently consumed
> as the header and dropped from the analysis. The shipped `MED4_PRS-RRS/PRS-V3-1.csv` /
> `MED4_PRS-RRS/RRS-V3-1.csv` are headerless — add a header line if you need all 100 pairs
> scored.

## Common Patterns

### Selecting specific checkpoints

```bash
# Only epochs 5, 10, 15, 20
--checkpoint_pattern "ppiDCE_epoch[51]*0.pth"

# Every epoch (default)
--checkpoint_pattern "ppiDCE_epoch*.pth"
```

### Re-computing metrics without re-running inference

```bash
python LES-wrapper.py ... --skip_inference
```

reuses the existing `*_probabilities.csv` files. Add `--no_plots` to skip the
trajectory / distribution figures when you only need the summary CSV.

### CPU / inference tuning

```bash
--device cpu          # run on CPU (slow)
--batch_size 8        # larger batches if GPU memory allows
--max_length 1024     # must match the value used at training time
--model_config facebook/esm1b_t33_650M_UR50S   # tokenizer/config source
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | *(required)* | Directory containing `ppiDCE_epoch*.pth` |
| `--prs_file` | *(required)* | Positive Reference Set CSV (`seq1,seq2[,label]`) |
| `--rrs_file` | *(required)* | Random Reference Set CSV (`seq1,seq2[,label]`) |
| `--output_dir` | `LES_results` | Directory for all output files |
| `--checkpoint_pattern` | `ppiDCE_epoch*.pth` | Glob to select per-epoch checkpoints |
| `--include_final` | False | Also evaluate `ppiDCE_final.pth` |
| `--inference_script` | *(auto)* | Path to `inference_ppiDCE.py` (defaults alongside this wrapper) |
| `--model_config` | `facebook/esm1b_t33_650M_UR50S` | ESM tokenizer/config source |
| `--batch_size` | `4` | Inference batch size |
| `--max_length` | `1024` | Max total tokens (seq1+seq2+special) |
| `--device` | `cuda` | Inference device (`cuda` or `cpu`) |
| `--skip_inference` | False | Reuse existing probability CSVs |
| `--no_plots` | False | Skip trajectory / distribution plots |
| `--color_threshold` | False | Color the ROC curve by decision threshold and add a colorbar |

> The old `--plot_format` switch is gone: PNG **and** vector PDF are now always
> written (matching `LES-wrapper_v2.py`).

## Figures

All PNGs are written at **publication quality** (600 dpi, tight bounding box,
enlarged fonts, heavier axis lines), and every PNG has a companion vector
**`.pdf`** at the same path. Individual ROC plots annotate **AUC and Best F1**;
by default the ROC curve is a single color with no threshold colorbar — pass
`--color_threshold` to render the threshold-colored curve.

The probability-distribution figures show `prob_1` for PRS (blue, positives) vs
RRS (red, negatives) as violins + jittered points, y-axis fixed to `[0, 1]`. A
discriminating model keeps PRS high and RRS low.

## Output Structure

```
LES_results_MED4/
├── epoch_1/
│   ├── PRS_epoch1_probabilities.csv        # full inference_ppiDCE.py output
│   ├── RRS_epoch1_probabilities.csv
│   ├── combined_probabilities_epoch1.csv   # PRS,RRS prob_1 columns for ROC
│   ├── prob_dist_epoch1.png / .pdf         # PRS-vs-RRS distribution
│   └── ROC_epoch1.png / .pdf
├── epoch_2/ ...
├── epoch_final/ ...                        # only with --include_final
├── trajectory_AUC.png / .pdf
├── trajectory_F1.png / .pdf
├── trajectory_combined.png / .pdf          # 1x2 AUC + Best-F1 (no threshold panel)
├── summary_prob_distributions.png / .pdf           # one panel per checkpoint
├── summary_prob_distributions_combined.png / .pdf  # all PRS then all RRS, one axes
├── summary_table.csv
├── manifest.json
└── README.md                               # legend for the analysis-level plots
```

`summary_table.csv` has columns `checkpoint, epoch, AUC, Best_F1, PRS_samples,
RRS_samples` plus a final LES row. `manifest.json` records run metadata
(timestamp, inputs, model config, per-checkpoint results, and LES scores).

> **Single-checkpoint runs.** With only one matching checkpoint the wrapper does
> the per-checkpoint analysis (probabilities, ROC, distribution plot) but skips
> LES, the trajectory plots, and the distribution summaries — these need ≥ 2
> checkpoints.

## Differences from the other wrappers

| Aspect | ppiGPLM `v2` | ppiBTEP | ppiDCE (this wrapper) |
|--------|--------------|---------|-----------------------|
| Checkpoints | `ckpt_*.pt` (iterations) | `ppiBTPE_epoch_*.pth` | `ppiDCE_epoch*.pth` (+ `ppiDCE_final.pth`) |
| Trajectory x-axis | iteration | epoch | **epoch** |
| Inference engine | `sample_fasta…3f.py` | `inference_ppiBTPE_2GPU.py` (requires `--num_layers`) | `inference_ppiDCE.py` (`--model_path`/`--model_config`/`--output_file`) |
| Positive-class prob | `Probability_of_1` → `[-2]` | `Probability_Friends` (2nd-to-last) | `prob_1` read **by name** (**last** column) |
| Output shape | v2 (no threshold, PDFs, prob-dist plots, README) | v2 | **v2** (matches ppiGPLM) |

Two model-specific notes for ppiDCE:

- **Score meaning.** `prob_1` is a genuine 2-class softmax over `{0, 1}`, so
  `prob_0 + prob_1 = 1` exactly — unlike ppiGPLM's whole-vocabulary language-model
  softmax where `P(1)` and `P(0)` need not sum to 1. The `README.md` written into
  each output dir explains this.
- **Column position.** The positive-class probability is the column literally
  named `prob_1`, which is the **last** column (falling back to the last column if
  the header is absent) — the opposite end from ppiBTEP's `Probability_Friends`.
