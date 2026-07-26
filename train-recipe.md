# ppiDCE train-recipe — from-scratch V3-1 run with per-epoch ROC/F1 + LES

Adapted from the [ppiYYD train-recipe](https://github.com/kouroshSA/ppiYYD/blob/main/train-recipe.md)
for ppiDCE. ppiDCE is a single dual **cross-encoder** (not a two-tower model),
so the tower/pair-op/decoder-fusion knobs in the ppiYYD recipe don't apply here
— everything else (warmup+cosine LR, per-epoch PRS/RRS metrics, a final
`LES-wrapper.py` pass) carries over directly.

One model per replicate: model *k* trains on
`MED4_V3_Trains/depleted_training_set-V3-k.csv` and is evaluated on the matched
`MED4_PRS-RRS/PRS-V3-k.csv` / `RRS-V3-k.csv`. This repo currently ships the
driver for replicate 1 (`run_V3-1.sh`); other replicates follow the same
recipe — copy the script and point `REP`/`TRAIN_SRC` at replicate *k*.

---

## TL;DR — one command

```bash
# from the repo root, detached (a multi-hour GPU job):
./run_V3-1.sh > results/dce_V3-1_scratch6L_ml1024/run.log 2>&1 &
tail -f results/dce_V3-1_scratch6L_ml1024/run.log
```

`run_V3-1.sh` does the whole thing: split → from-scratch training with
per-epoch ROC/F1 → a final `LES-wrapper.py` pass. Rebuild only the LES analysis
from existing checkpoints with `LES_ONLY=1 ./run_V3-1.sh`.

---

## Fixed training configuration (the schedule we use)

Matches the ppiYYD warmup+cosine recipe, with ppiDCE's own architecture (single
cross-encoder, `[CLS] Seq_A [SEP] Seq_B [EOS]`) in place of YYD's two towers:

| Setting | Value |
|---|---|
| LR schedule | `warmup_cosine` — **peak 2e-5**, **floor (`--min_lr`) 2e-6**, **warmup 10 %** of steps (`--warmup_ratio 0.1`) |
| Architecture | from scratch, `--num_layers 6`, no ESM-1b pretrained weights |
| Epochs / batch | `--epochs 10`, `--batch_size 4` |
| Max length | `--max_length 1024` |
| Val split | 10 % stratified, **sampled but not removed from train** (`make_split.py`, no `--deplete`) |

> **Checkpoint selection:** pick the best epoch on **PRS/RRS AUC**, not val
> loss — the val set is in-distribution and the two can disagree.

---

## Data

- **Training:** `MED4_V3_Trains/depleted_training_set-V3-{1..10}.csv` — 3 columns
  `seq1,seq2,label`, headerless, ~23.6k rows (5 904 positives / ~17.7k negatives).
  Combined pair length (`len(seq1)+len(seq2)`) has median ~399, mean ~410,
  max ~1017 — `max_length=1024` fits essentially every pair uncut;
  `max_length=512` would silently truncate ~30 % of them (and ~39 % of PRS-V3-1).
  Note the file itself has substantial duplication (~38 % of rows repeat), so
  even a disjoint split will show some train/val row overlap.
- **Reference sets:** `MED4_PRS-RRS/PRS-V3-{1..10}.csv` and `RRS-V3-{1..10}.csv` —
  2 columns `seq1,seq2`, 100 pairs each. Replicate *k*'s PRS/RRS is matched to its
  training set *k*.

### Train/val split

`make_split.py` carves a 10 %, label-stratified val set from the replicate CSV
with a fixed seed (42). Two modes:

- *(default, no flag, used by this recipe)* — val rows are **sampled but stay
  in** train.
- **`--deplete`** — val rows are **removed** from train, giving the standard
  disjoint split.

**Why the default (non-depleting) is recommended here:** this is a *small*
training set — 23.6k rows, only ~14.5k of them unique. Carving out a real 10 %
holdout costs meaningful training signal for a val set that's redundant anyway
(it's in-distribution, and checkpoint selection is done on PRS/RRS AUC, not val
loss — the two can disagree). Depleting is worth it on a large dataset where
10 % is not scarce; on this scale it isn't. Use `--deplete` if you specifically
want a disjoint holdout for some other purpose.

Either way, the split is written **with** a `seq1,seq2,label` header —
`PPICrossDataset` reads with `pd.read_csv` (inferred header) and would
otherwise silently drop the first data row.

```bash
python make_split.py \
    --input MED4_V3_Trains/depleted_training_set-V3-1.csv \
    --output_dir results/dce_V3-1_scratch6L_ml1024/data \
    --val_frac 0.1 --seed 42
```

---

## What the run does (and where it lands)

```
results/dce_V3-1_scratch6L_ml1024/
├── data/{train,val}.csv                    # split (make_split.py)
├── ppiDCE_epoch{1..10}.pth, ppiDCE_final.pth
├── eval/
│   ├── metrics_by_epoch.csv                # AUC, Best-F1 per epoch
│   ├── epoch{N}_PRS-RRS_probabilities.csv  # probabilities behind each ROC figure
│   └── roc_epoch{N}.png                    # per-epoch ROC + Best-F1 figure
└── LES/                                    # LES-wrapper.py output
    ├── summary_table.csv                   # per-epoch AUC/Best-F1 + a final LES row
    ├── epoch_{N}/{PRS,RRS}_epoch{N}_probabilities.csv, ROC/dist plots
    ├── trajectory_*.png/.pdf, summary_prob_distributions*.png/.pdf
    └── README.md
```

Two layers of "results after each epoch", both built directly into
`train_ppiDCE.py` (unlike ppiYYD, which uses a separate `eval_auc_f1.py`):

1. **during training** — `--eval_prs`/`--eval_rrs` scores the 100 PRS + 100 RRS
   pairs after every epoch (one extra forward pass, no gradient) and appends
   AUC / Best-F1 to `eval/metrics_by_epoch.csv`; `--roc_script` (on by default)
   renders `eval/roc_epoch{N}.png` from those same probabilities — no
   additional inference cost. The optimal-F1 *threshold* is deliberately not
   reported anywhere in this pipeline — for a non/weakly-discriminating
   checkpoint it collapses toward the minimum score ("predict everything
   positive"), a degenerate diagnostic rather than a meaningful decision
   boundary.
2. **at the end of the run** — `LES-wrapper.py` re-scores every saved
   checkpoint via `inference_ppiDCE.py` and produces the full LES analysis:
   ROC-AUC / Best-F1 trajectories (with LES = area under each trajectory),
   PRS-vs-RRS probability-distribution violins, and a `summary_table.csv`.

`inference_ppiDCE.py`'s output columns (`seq1, seq2, Prediction,
Probability_Friends, Probability_Enemies`) match the ppiYYD / ppiBTEP
convention — see [LES-wrapper.md](LES-wrapper.md).

---

## Prompt for Claude Code

> Train the ppiDCE V3-1 replicate from scratch with the warmup+cosine LR
> schedule (peak 2e-5, floor 2e-6, 10 % warmup), 6 layers, max_length 1024, 10
> epochs, batch 4, seed 42, val split sampled but kept in train (small
> training set — don't deplete).
>
> 1. Split `MED4_V3_Trains/depleted_training_set-V3-1.csv` 90/10 with
>    `make_split.py` (no `--deplete`).
> 2. Train `train_ppiDCE.py` on it with the config above, writing checkpoints
>    to `results/dce_V3-1_scratch6L_ml1024/`, with `--eval_prs
>    MED4_PRS-RRS/PRS-V3-1.csv --eval_rrs MED4_PRS-RRS/RRS-V3-1.csv --eval_dir
>    results/dce_V3-1_scratch6L_ml1024/eval` for per-epoch AUC/Best-F1/ROC.
>    Report `metrics_by_epoch.csv` after the run.
> 3. Run `LES-wrapper.py --checkpoint_dir results/dce_V3-1_scratch6L_ml1024
>    --prs_file MED4_PRS-RRS/PRS-V3-1.csv --rrs_file MED4_PRS-RRS/RRS-V3-1.csv
>    --output_dir results/dce_V3-1_scratch6L_ml1024/LES --num_layers 6
>    --max_length 1024 --include_final`.
>
> This is exactly what `./run_V3-1.sh` automates — you may just run it
> (detached) and report `metrics_by_epoch.csv` and the LES trajectory figures.
> Use the `esm2` conda python and export `HF_HUB_OFFLINE=1 MPLBACKEND=Agg`.
> Pick the best epoch on PRS/RRS AUC, not val loss.

---

## Runtime & practicalities

- **Long, single-GPU job.** 10 epochs on ~23.6k training pairs at
  `max_length 1024`, plus 22 LES-wrapper inference passes (11 checkpoints ×
  PRS + RRS) at the end. Run detached (`nohup … &` or tmux). Expect roughly
  2x the per-epoch time of a `max_length 512` run on the same GPU (longer
  sequences cost more in attention) — plan for several hours.
- **Env:** `esm2` conda env; `HF_HUB_OFFLINE=1`, `MPLBACKEND=Agg`. Override the
  interpreter with `PY=/path/to/python ./run_V3-1.sh`.
- Outputs under `results/` are gitignored — commit figures deliberately if you
  want them in the repo.

## Files

- `run_V3-1.sh` — the driver (split → train → per-epoch ROC/F1 → LES).
- `make_split.py` — deterministic, label-stratified train/val split.
- `train_ppiDCE.py` — training loop, warmup+cosine LR, per-epoch PRS/RRS eval + ROC figure.
- `roc_analysis_color_threshold_F1e.py` — standalone ROC + Best-F1 plot from a probability CSV.
- `inference_ppiDCE.py` — batch inference (`seq1, seq2, Prediction, Probability_Friends, Probability_Enemies`).
- `LES-wrapper.py` / `LES-wrapper.md` — end-of-run LES analysis across all checkpoints.
