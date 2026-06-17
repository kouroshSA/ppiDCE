# LES-wrapper (ppiDCE): Learning Efficiency Score Evaluation

## Overview

The **LES-wrapper** automates evaluation of model trainability across ppiDCE
training checkpoints. It runs inference on PRS (Positive Reference Set) and RRS
(Random Reference Set) sequence-pair files at each per-epoch checkpoint, computes
ROC metrics, and derives integrated learning-efficiency scores.

This is the ppiDCE port of the [ppiGPLM LES-wrapper](https://github.com/kouroshSA/ppiGPLM).
The evaluation logic (PRS/RRS → ROC → AUC/Best-F1/threshold → LES) is identical;
only the model-specific glue differs (see [Differences from ppiGPLM](#differences-from-ppigplm)).

## What is LES?

LES (Learning Efficiency Score) is the **area under the metric-vs-epoch curve**.
Unlike metrics that capture only final performance, LES summarizes the entire
learning trajectory:

- **LES-AUC**: Area under the AUC trajectory curve
- **LES-F1**: Area under the Best-F1 trajectory curve
- **LES-Threshold**: Area under the optimal-threshold trajectory curve

Epochs are normalized to `[0, 1]` before integration, so LES values are
comparable across runs of different length. Higher LES indicates faster, more
consistent learning across training.

## Workflow

For each checkpoint the wrapper:

1. Runs `inference_ppiDCE.py` on the PRS and RRS files
2. Extracts the positive-class probability (`prob_1`) for every pair
3. Combines PRS and RRS probabilities into a single file for ROC analysis
4. Computes AUC, Best-F1, and the optimal threshold
5. Generates a publication-quality ROC curve plot (600 dpi PNG + vector PDF;
   plain single-color by default, threshold-colored with `--color_threshold`)
6. Aggregates per-checkpoint results into a summary table
7. Plots metric trajectories across epochs
8. Computes LES for each metric

## Installation

Use the same `esm` environment as training/inference:

```bash
conda activate esm
pip install -r requirements.txt   # already includes scikit-learn, matplotlib, numpy
```

## Basic Usage

```bash
python LES-wrapper.py \
    --checkpoint_dir ROC_Checkpoints \
    --prs_file MED4_PRS_100.csv \
    --rrs_file MED4_RRS_100.csv \
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
> as the header and dropped from the analysis. The shipped `MED4_PRS_100.csv` /
> `MED4_RRS_100.csv` are headerless — add a header line if you need all 100 pairs
> scored.

## Common Patterns

### Selecting specific checkpoints

```bash
# Only epochs 5, 10, 15, 20
python LES-wrapper.py \
    --checkpoint_dir ROC_Checkpoints \
    --prs_file MED4_PRS_100.csv \
    --rrs_file MED4_RRS_100.csv \
    --output_dir LES_results \
    --checkpoint_pattern "ppiDCE_epoch[51]*0.pth"

# Every epoch (default)
--checkpoint_pattern "ppiDCE_epoch*.pth"
```

### Skipping inference (re-computing metrics only)

If inference already ran and you only want to recompute metrics/plots:

```bash
python LES-wrapper.py \
    --checkpoint_dir ROC_Checkpoints \
    --prs_file MED4_PRS_100.csv \
    --rrs_file MED4_RRS_100.csv \
    --output_dir LES_results_MED4 \
    --skip_inference
```

### CPU / inference tuning

```bash
--device cpu          # run on CPU (slow for the 12-layer model)
--batch_size 8        # larger batches if GPU memory allows
--max_length 1024     # must match the value used at training time
--model_config facebook/esm1b_t33_650M_UR50S   # tokenizer/config source
```

Use `--no_plots` to skip trajectory figures when you only need the summary CSV.

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
| `--no_plots` | False | Skip trajectory plots |
| `--color_threshold` | False | Color the ROC curve by decision threshold and add a colorbar (default: plain single-color curve) |
| `--plot_format` | `both` | Plot output format(s): `png` (600 dpi raster), `pdf` (vector), or `both` |

## Output Structure

```
LES_results_MED4/
├── epoch_1/
│   ├── PRS_epoch1_probabilities.csv      # full inference_ppiDCE.py output
│   ├── RRS_epoch1_probabilities.csv
│   ├── combined_probabilities_epoch1.csv # PRS,RRS prob_1 columns for ROC
│   └── ROC_epoch1.png / ROC_epoch1.pdf    # 600 dpi PNG + vector PDF
├── epoch_2/ ...
├── epoch_final/ ...                       # only with --include_final
├── trajectory_AUC.png  / .pdf
├── trajectory_F1.png   / .pdf
├── trajectory_Threshold.png / .pdf
├── trajectory_combined.png  / .pdf
├── summary_table.csv
└── manifest.json
```
(PDF/PNG per the `--plot_format` setting; `both` by default.)

`summary_table.csv` contains one row per checkpoint plus a final row with the
integrated LES values. `manifest.json` records full run metadata (timestamp,
inputs, model config, per-checkpoint results, and LES scores).

## Differences from ppiGPLM

| Aspect | ppiGPLM | ppiDCE (this wrapper) |
|--------|---------|-----------------------|
| Checkpoints | `ckpt_*.pt` (training iterations) | `ppiDCE_epoch*.pth` + `ppiDCE_final.pth` (epochs) |
| Trajectory x-axis | iteration | epoch |
| Inference engine | `sample_fasta3.3..._hope_v3.py` (`--model_dir`/`--ckpt_name`/`--output_prefix`) | `inference_ppiDCE.py` (`--model_path`/`--model_config`/`--output_file`) |
| Inference output | `Prompt, Probability_of_1, Probability_of_0` → `prob_1` at column `[-2]` (prompt commas push probs to the end) | `seq1, seq2, [label,] pred_label, prob_0, prob_1` → `prob_1` read **by name** (last column) |
| HOPE/Titan/`--vanilla` flags | present | removed — not applicable to the ESM transformer |

The probability-extraction routine looks up the column literally named `prob_1`
(falling back to the last column if absent), which is the concrete adjustment for
ppiDCE's output format.
