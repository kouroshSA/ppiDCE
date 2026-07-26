# ppiDCE train-recipe — 10-replicate V3 campaign with per-epoch ROC/F1 + dual LES

Adapted from the [ppiYYD train-recipe](https://github.com/kouroshSA/ppiYYD/blob/main/train-recipe.md)
for ppiDCE. ppiDCE is a single dual **cross-encoder** (not a two-tower model),
so the tower/pair-op/decoder-fusion knobs in the ppiYYD recipe don't apply here
— everything else (warmup+cosine LR, per-epoch PRS/RRS metrics, a final
`LES-wrapper.py` pass) carries over directly.

One model per replicate: model *k* trains on
`MED4_V3_Trains/depleted_training_set-V3-k.csv` and is evaluated on the matched
`MED4_PRS-RRS/PRS-V3-k.csv` / `RRS-V3-k.csv`. `run_replicate.sh REP=V3-k` drives
one replicate; `run_all_replicates.sh` drives several in sequence.

---

## TL;DR — one command

```bash
# a single replicate, detached (a multi-hour GPU job):
REP=V3-2 ./run_replicate.sh > results/dce_V3-2_scratch12L_ml1024/run.log 2>&1 &

# the whole campaign (replicates 2..10; V3-1 is done), detached:
nohup ./run_all_replicates.sh > results/run_all_replicates.log 2>&1 &
tail -f results/run_all_replicates.log
```

Each replicate: split → from-scratch training with per-epoch ROC/F1 → two
`LES-wrapper.py` passes (with and without homodimers — see below). Rebuild
just the LES analyses for one replicate with `LES_ONLY=1 REP=V3-k
./run_replicate.sh`. `run_all_replicates.sh` is resumable — a replicate whose
`LES_no_homodimers/summary_table.csv` already exists is skipped, so re-running
it after a kill/restart doesn't redo finished work.

---

## Fixed training configuration (the schedule we use)

Matches the ppiYYD warmup+cosine recipe, with ppiDCE's own architecture (single
cross-encoder, `[CLS] Seq_A [SEP] Seq_B [EOS]`) in place of YYD's two towers:

| Setting | Value |
|---|---|
| LR schedule | `warmup_cosine` — **peak 2e-5**, **floor (`--min_lr`) 2e-6**, **warmup 10 %** of steps (`--warmup_ratio 0.1`) |
| Architecture | from scratch, `--num_layers 12`, no ESM-1b pretrained weights |
| Epochs / batch | `--epochs 10`, `--batch_size 2` |
| Max length | `--max_length 1024` |
| Val split | 10 % stratified, **sampled but not removed from train** (`make_split.py`, no `--deplete`) |

> **Checkpoint selection:** pick the best epoch on **PRS/RRS AUC**, not val
> loss — the val set is in-distribution and the two can disagree.

> **12 layers / batch 2, not 6 / 4:** the first V3-1 run used 6 layers, batch
> 4; this was raised to 12 layers (more capacity) with batch 2 (to fit the
> larger model) for the 10-replicate campaign. `NUM_LAYERS=`/`BATCH_SIZE=`
> override either independently — see [Runtime & practicalities](#runtime--practicalities)
> for the resulting per-epoch cost.

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

### Homodimers in PRS (checked across all 10 replicates, 2026-07-26)

| | PRS homodimers | RRS homodimers |
|---|---|---|
| V3-1 | 32/100 | 0/100 |
| V3-2 | 30/100 | 0/100 |
| V3-3 | 26/100 | 0/100 |
| V3-4 | 28/100 | 0/100 |
| V3-5 | 25/100 | 0/100 |
| V3-6 | 25/100 | 0/100 |
| V3-7 | 32/100 | 0/100 |
| V3-8 | 24/100 | 0/100 |
| V3-9 | 22/100 | 0/100 |
| V3-10 | 25/100 | 0/100 |

Every PRS set is 22-32 % homodimer pairs (`seq1 == seq2`); no RRS set has any.
On V3-1's 12-layer run, homodimer PRS pairs scored consistently much closer to
the RRS (non-interacting) baseline than heterodimer PRS pairs did at every
epoch (e.g. epoch 1: homodimer mean 0.367, heterodimer mean 0.570, RRS mean
0.282) — the model treats them as the hard case, and they drag the combined
AUC/Best-F1 down. [`MED4_PRS-RRS_no_homodimers/`](MED4_PRS-RRS_no_homodimers)
is the pre-built, homodimer-depleted counterpart to `MED4_PRS-RRS/` (same
headerless `seq1,seq2` format, regeneratable with its own
`make_no_homodimers.py`); `run_replicate.sh` runs `LES-wrapper.py` against
both directories automatically (see below).

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
    --output_dir results/dce_V3-1_scratch12L_ml1024/data \
    --val_frac 0.1 --seed 42
```

---

## What each replicate's run does (and where it lands)

```
results/dce_V3-k_scratch12L_ml1024/
├── data/{train,val}.csv                    # split (make_split.py)
├── ppiDCE_epoch{1..10}.pth, ppiDCE_final.pth
├── eval/
│   ├── metrics_by_epoch.csv                # AUC, Best-F1 per epoch (full PRS/RRS, homodimers included)
│   ├── epoch{N}_PRS-RRS_probabilities.csv  # probabilities behind each ROC figure
│   └── roc_epoch{N}.png                    # per-epoch ROC + Best-F1 figure
├── LES/                                    # LES-wrapper.py — full PRS/RRS (homodimers included)
│   ├── summary_table.csv                   # per-epoch AUC/Best-F1 + a final LES row
│   ├── epoch_{N}/{PRS,RRS}_epoch{N}_probabilities.csv, ROC/dist plots
│   ├── trajectory_*.png/.pdf, summary_prob_distributions*.png/.pdf
│   └── README.md
└── LES_no_homodimers/                      # LES-wrapper.py — homodimer PRS/RRS pairs excluded
    └── (same structure as LES/)
```

Two layers of "results after each epoch", both built directly into
`train_ppiDCE.py` (unlike ppiYYD, which uses a separate `eval_auc_f1.py`):

1. **during training** — `--eval_prs`/`--eval_rrs` scores the 100 PRS + 100 RRS
   pairs after every epoch (one extra forward pass, no gradient) and appends
   AUC / Best-F1 to `eval/metrics_by_epoch.csv`; `--roc_script` (on by default)
   renders `eval/roc_epoch{N}.png` from those same probabilities — no
   additional inference cost. This pass always uses the **full** PRS/RRS
   (homodimers included) — it's a fast in-training signal, not the final word.
   The optimal-F1 *threshold* is deliberately not reported anywhere in this
   pipeline — for a non/weakly-discriminating checkpoint it collapses toward
   the minimum score ("predict everything positive"), a degenerate diagnostic
   rather than a meaningful decision boundary.
2. **at the end of the run** — `LES-wrapper.py` re-scores every saved
   checkpoint via `inference_ppiDCE.py` and produces the full LES analysis
   (ROC-AUC / Best-F1 trajectories, LES = area under each trajectory,
   PRS-vs-RRS probability-distribution violins, `summary_table.csv`) **twice**:
   once on `MED4_PRS-RRS/` (`LES/`), once on the pre-built homodimer-depleted
   `MED4_PRS-RRS_no_homodimers/` (`LES_no_homodimers/`).

`inference_ppiDCE.py`'s output columns (`seq1, seq2, Prediction,
Probability_Friends, Probability_Enemies`) match the ppiYYD / ppiBTEP
convention — see [LES-wrapper.md](LES-wrapper.md).

### V3-1 results (12 layers, batch 2, max_length 1024, 10 epochs)

| | LES-AUC | LES-F1 |
|---|---|---|
| Full PRS/RRS (`LES/`) | 0.7197 | 0.6967 |
| No homodimers (`LES_no_homodimers/`) | **0.7881** | 0.7051 |

Per-epoch AUC on the full set hovered in a flat 0.69-0.72 band for all 10
epochs with no clear trend; excluding homodimers pushed it to a similarly flat
but consistently higher 0.76-0.80 band at *every* epoch (not just early
ones) — a stable ~0.07-0.08 AUC gap across the whole run, not something that
opens or closes over training. Best-F1 moves up slightly too (0.697 -> 0.705).
Interpretation: the model's genuine heterodimer discrimination is
meaningfully better than the combined metric shows; homodimer pairs are
consistently the harder case it hasn't learned to separate from RRS.

---

## Prompt for Claude Code

> Train ppiDCE replicate `V3-k` from scratch with the warmup+cosine LR
> schedule (peak 2e-5, floor 2e-6, 10 % warmup), 12 layers, max_length 1024, 10
> epochs, batch 2, seed 42, val split sampled but kept in train (small
> training set — don't deplete).
>
> 1. Split `MED4_V3_Trains/depleted_training_set-V3-k.csv` 90/10 with
>    `make_split.py` (no `--deplete`).
> 2. Train `train_ppiDCE.py` on it with the config above, writing checkpoints
>    to `results/dce_V3-k_scratch12L_ml1024/`, with `--eval_prs
>    MED4_PRS-RRS/PRS-V3-k.csv --eval_rrs MED4_PRS-RRS/RRS-V3-k.csv --eval_dir
>    results/dce_V3-k_scratch12L_ml1024/eval` for per-epoch AUC/Best-F1/ROC.
>    Report `metrics_by_epoch.csv` after the run.
> 3. Run `LES-wrapper.py` on the full `MED4_PRS-RRS/PRS-V3-k.csv` /
>    `RRS-V3-k.csv` into `results/dce_V3-k_scratch12L_ml1024/LES`, and again on
>    the pre-built `MED4_PRS-RRS_no_homodimers/PRS-V3-k.csv` /
>    `RRS-V3-k.csv` into `results/dce_V3-k_scratch12L_ml1024/LES_no_homodimers`
>    (`--num_layers 12 --max_length 1024 --include_final` both times).
>
> This is exactly what `REP=V3-k ./run_replicate.sh` automates (or
> `./run_all_replicates.sh` for several replicates in sequence) — you may just
> run it (detached) and report `metrics_by_epoch.csv` and both LES trajectory
> figures. Use the `esm2` conda python and export `HF_HUB_OFFLINE=1
> MPLBACKEND=Agg`. Pick the best epoch on PRS/RRS AUC, not val loss.

---

## Runtime & practicalities

- **Long, single-GPU job, x10.** 10 epochs on ~23.6k training pairs at
  `max_length 1024` with a 12-layer model and batch 2 — on an RTX 5070 Ti,
  ~4.6 it/s over 11 781 steps/epoch, i.e. **~42 min/epoch, ~7 hours of
  training per replicate**, plus two LES-wrapper passes (22 inference calls
  each) at the end. Nine remaining replicates (V3-2..V3-10) is roughly
  **2.5-3 days of continuous GPU time** run back-to-back via
  `run_all_replicates.sh`.
- **Env:** `esm2` conda env; `HF_HUB_OFFLINE=1`, `MPLBACKEND=Agg`. Override the
  interpreter with `PY=/path/to/python ./run_replicate.sh`.
- Outputs under `results/` are gitignored — commit figures deliberately if you
  want them in the repo.

## Files

- `run_replicate.sh` — the per-replicate driver (split → train → per-epoch ROC/F1 → LES ×2). Formerly `run_V3-1.sh`.
- `run_all_replicates.sh` — sequential, resumable driver across several replicates.
- `make_split.py` — deterministic, label-stratified train/val split.
- `train_ppiDCE.py` — training loop, warmup+cosine LR, per-epoch PRS/RRS eval + ROC figure.
- `roc_analysis_color_threshold_F1e.py` — standalone ROC + Best-F1 plot from a probability CSV.
- `MED4_PRS-RRS_no_homodimers/` — homodimer-depleted PRS/RRS counterpart to `MED4_PRS-RRS/`; regenerate with its own `make_no_homodimers.py`.
- `inference_ppiDCE.py` — batch inference (`seq1, seq2, Prediction, Probability_Friends, Probability_Enemies`).
- `LES-wrapper.py` / `LES-wrapper.md` — end-of-run LES analysis across all checkpoints.
