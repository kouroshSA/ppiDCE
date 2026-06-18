# Revised LES-wrapper & inference scripts — change log

This note documents the revisions made to the **LES-wrapper** (Learning
Efficiency Score evaluation) and its associated inference scripts across the
three MED4 tri-model components — **ppiGPLM**, **ppiDCE**, and **ppiBTEP** —
during the ROC / LES analysis (with and without the random-50-aa N-/C-terminal
extensions).

All three wrappers share the same evaluation core (PRS/RRS → ROC → AUC,
Best-F1, optimal-F1 threshold → LES = area under the metric-vs-epoch/iteration
curve). The changes below were applied consistently; model-specific glue is
noted where it differs.

---

## 0. Definitions

Each checkpoint is scored on two labeled sets of protein pairs: the **PRS**
(Positive Reference Set — known/true interactions, label = 1) and the **RRS**
(Random Reference Set — random pairs treated as non-interactions, label = 0).
The model outputs, for each pair, a probability `p` that the pair interacts
(the positive-class score). A pair is **predicted positive** when
`p >= threshold`. From the predictions at a given threshold:

- **TP** = true positives (PRS pairs correctly predicted positive)
- **FP** = false positives (RRS pairs predicted positive)
- **FN** = false negatives (PRS pairs predicted negative)
- **Precision** = TP / (TP + FP) — of the pairs called interacting, the fraction that truly are.
- **Recall** (= TPR, sensitivity) = TP / (TP + FN) — of the true interactions, the fraction recovered.
- **FPR** (false-positive rate) = FP / (FP + TN) — of the true non-interactions, the fraction wrongly called interacting.

### ROC and ROC-AUC

The **ROC curve** (Receiver Operating Characteristic) plots **TPR (recall)** on
the y-axis against **FPR** on the x-axis as the decision `threshold` is swept
across all possible values. Each threshold gives one (FPR, TPR) point; together
they trace the curve.

**ROC-AUC** (Area Under the ROC Curve) is the area under that curve, a single
number in **[0, 1]** summarizing ranking quality **independent of any chosen
threshold**:

- **1.0** = perfect separation (every PRS pair scores above every RRS pair);
- **0.5** = no better than random;
- equivalently, AUC is the probability that a randomly chosen PRS pair receives
  a higher score than a randomly chosen RRS pair.

Computed here as `auc(fpr, tpr)` over the ROC curve from `roc_curve(labels, p)`.

### F1 and Best-F1

The **F1 score** is the harmonic mean of precision and recall at a *specific*
threshold:

```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

It rewards a threshold that is good at *both* finding true interactions
(recall) and not over-calling (precision); it is high only when both are high.
Unlike AUC, F1 depends on the chosen threshold.

**Best-F1** is the **maximum F1 achievable over all possible thresholds** — i.e.
the F1 of the single best operating point for that checkpoint. It answers “at
its best single cutoff, how well does this model trade off precision and
recall?”

### Best-F1 threshold (optimal-F1 threshold)

The **Best-F1 threshold** is the probability cutoff `threshold*` at which that
maximum F1 is attained — the value of `threshold` that maximizes F1:

```
threshold* = argmax_threshold  F1(threshold)
```

Applying the rule `p >= threshold*` to classify pairs reproduces the reported
Best-F1. It is the recommended operating cutoff if one had to pick a single
decision boundary for screening. (When several thresholds tie at the maximum
F1, the wrapper reports the **highest / most stringent** one.)

> In this analysis AUC and Best-F1 are the primary trainability metrics
> (summarized over training as the **LES**, the area under the metric-vs-epoch
> curve), while the Best-F1 threshold characterizes *where* the best operating
> point sits.

---

## 1. Metric correctness — Best-F1 threshold (all three wrappers)

**This is the most important change for the analysis.**

`run_roc_analysis_internal()` computes ROC-AUC, the Best-F1, and the
F1-optimal threshold at each checkpoint.

- **ROC / AUC** — unchanged; was already correct
  (`roc_curve(labels, probs)` + `auc(fpr, tpr)`, with PRS = 1, RRS = 0, and
  `probs` = positive-class probability).

- **Best-F1 and its threshold — search made exhaustive.**
  The original code searched for the best F1 only over the thresholds returned
  by `roc_curve(...)`. `roc_curve` defaults to `drop_intermediate=True`, which
  **prunes collinear ROC vertices** (on the real PRS/RRS data this collapsed
  201 unique scores down to 26 candidate thresholds). The search now scans
  **every unique score** (`np.unique(probs)`), so the true F1-optimal cutoff is
  always considered and the reported threshold is unambiguous.

  Additional hardening in the same block:
  - `f1_score(..., zero_division=0)` — no undefined-precision warnings/surprises.
  - Explicit, deterministic tie-break: among thresholds that achieve the same
    maximum F1, the **highest (most stringent)** threshold is kept.
  - Decision rule made explicit: **`prob >= threshold` ⇒ predicted positive**
    (consistent with `roc_curve` and the screening pipeline).

  *Why it is safe / why numbers did not change:* F1 is a ratio of two linear
  functions of (TP, FP) along any straight ROC segment, hence monotonic there,
  so its maximum always lands on a *retained* vertex. The pruned search was
  therefore already finding the correct optimum — but that relied on a subtle
  argument. The exhaustive search makes correctness obvious rather than
  incidental.

  *Verification:* identical metrics on the real ppiGPLM ckpt data
  (AUC = 0.980296, Best-F1 = 0.947368, threshold = 0.294209) and **0 mismatches
  across 2,000 random tie-heavy datasets** versus an independent brute-force
  reference; the reported threshold reproduces the reported F1 in every case.

> **Note for downstream use:** the reported threshold is the cutoff for
> `prob >= threshold`. If a screening step applies a strict `>` instead, there
> is an epsilon mismatch only at exact ties — keep the comparison consistent.

---

## 2. Publication-quality figures

- **Resolution raised to 600 dpi** (was 300 in ppiDCE/ppiBTEP and 300→600 in
  ppiGPLM). Controlled by the single `PUB_DPI = 600` constant; a
  `set_publication_style()` helper applies tight bounding box, larger fonts, and
  heavier axis lines to every figure (ROC curves and the AUC / F1 / threshold
  trajectory plots).

- **Vector PDF output.**
  - **ppiGPLM**: every PNG is now accompanied by a vector PDF (`ROC_*.pdf`,
    `trajectory_*.pdf`).
  - **ppiDCE**: a `--plot_format {png,pdf,both}` option was added (default
    `both`) — publication-quality PNG (raster), PDF (vector), or both.
  - **ppiBTEP**: 600-dpi PNG (PDF not added here; can be ported on request).

## 3. Optional threshold-colored ROC curve

The ROC plot default is now a **clean single-color curve** (no colorbar).
Pass **`--color_threshold`** to colour the curve by decision threshold and add a
threshold colorbar (the previous always-on behavior). The optimal threshold is
always computed and recorded in `summary_table.csv` / `manifest.json`; it is no
longer forced onto the plot. (All three wrappers expose `--color_threshold`.)

---

## 4. ppiGPLM — de-HOPE'd inference path + new sampler `…3f.py`

ppiGPLM is a **vanilla GPT-2** model (no HOPE / Titan / CMS). The wrapper and
sampler were cleaned up accordingly.

### `sample_fasta3.3_softmax_error_handling3f.py` (new)

Created from the older vanilla `…3e.py` to be driven by the wrapper, which
evaluates many checkpoints in one directory:

1. **Added `--model_dir` and `--ckpt_name` arguments** — any checkpoint file
   (e.g. `ckpt_1000.pt`) in any directory can be selected. `3e` hardcoded
   `out/ckpt.pt`.
2. **`meta.pkl` fallback** — the checkpoint's stored dataset name can be a stale
   nanoGPT default (`shakespeare_char`) whose `meta.pkl` is absent; the script
   now falls back to `data/MED4_char/meta.pkl` (the ppiGPLM character vocab).
3. Fully vanilla — imports only `model.py` (`GPTConfig`, `GPT`); no HOPE module.

The previous `…3e.py` was removed from the repo (superseded by `3f`); the
HOPE sampler (`…3e_hope_v3.py`) is no longer referenced.

### ppiGPLM `LES-wrapper.py`

- Points at `sample_fasta3.3_softmax_error_handling3f.py`.
- **All HOPE/Titan/CMS options removed** (`--vanilla`, `--use_titan_in_forward`,
  `--enable_surprise_updates`, `--surprise_update_in_eval`, `--adapt_mode`,
  `--adapt_steps`, `--memory_state_in`, all `--teach_*`) along with the code that
  forwarded them.

---

## 5. Model-specific inference glue (unchanged behavior, for reference)

- **ppiGPLM**: vanilla GPT next-token softmax over `0`/`1`; prompts wrapped with
  `<ps1>`, `<ps2>`, `<l1..3>`. Positive-class prob = `Probability_of_1`.
- **ppiDCE**: `inference_ppiDCE.py`; positive-class prob = `prob_1` (last
  column). x-axis = training **epoch**.
- **ppiBTEP**: `inference_ppiBTPE_2GPU.py`; **requires `--num_layers 12`** to
  rebuild the SiameseBTPE config; positive-class ("friends") prob =
  `Probability_Friends` (second-to-last column). x-axis = training **epoch**.

---

## 6. Repository / Hugging Face notes

- The per-epoch / per-iteration ROC-trajectory checkpoints were uploaded to each
  model's Hugging Face repo under a **`ROC-Checkpoints/`** folder:
  - ppiGPLM — 33 checkpoints (`ckpt_250…8000` + final `ckpt.pt`)
  - ppiDCE — 15 checkpoints (`ppiDCE_epoch1…15.pth`)
  - ppiBTEP — 30 checkpoints (`ppiBTPE_epoch_1…30.pth`)
- **`.gitignore` caveat (important if re-uploading):** the Hugging Face Hub
  honors `.gitignore` server-side. These repos ignore `*.pth`/`*.pt` (only
  `checkpoints/*` un-ignored), so uploads to `ROC-Checkpoints/` are **silently
  dropped** unless the repo's `.gitignore` first un-ignores
  `!ROC-Checkpoints/*.pth` (and `.pt`). This negation was added to each HF repo.
  The GitHub repos still ignore `*.pth`/`*.pt` (checkpoints live only on HF).
- Each GitHub repo has the wrapper + docs and a `ROC-Checkpoints/README.md`
  giving the training parameters (mirroring the screening-checkpoint `.md`) and
  the Hugging Face download link.

---

## 7. Files changed (summary)

| Repo | File | Change |
|------|------|--------|
| ppiGPLM | `sample_fasta3.3_softmax_error_handling3f.py` | **new** vanilla sampler (`--model_dir`/`--ckpt_name`, meta fallback) |
| ppiGPLM | `sample_fasta3.3_softmax_error_handling3e.py` | **removed** (superseded by `3f`) |
| ppiGPLM | `LES-wrapper.py` | de-HOPE'd; points to `3f`; 600 dpi + PDF; exhaustive Best-F1 |
| ppiDCE | `LES-wrapper.py` | `--color_threshold`, `--plot_format`, 600 dpi; exhaustive Best-F1 |
| ppiBTEP | `LES-wrapper.py` | 600 dpi; exhaustive Best-F1 (already had `--color_threshold`) |
| all | `LES-wrapper.md` / `ROC-Checkpoints/README.md` | docs updated to match |

*All metric changes verified to produce numerically identical AUC / Best-F1 /
threshold on the real data — the revisions improve correctness guarantees and
output quality without altering the reported results.*
