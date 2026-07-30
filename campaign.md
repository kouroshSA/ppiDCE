# ppiDCE V3 campaign — LES, composite & ensemble analyses

End-to-end recipe for the ppiDCE V3 evaluation campaign: train the 10 MCCV
replicates, run LES against every reference-set condition, and build the
cross-replicate composite / ensemble figures. See `train-recipe.md` and
`inference-recipe.md` for the per-run details this ties together.

Config throughout: **from-scratch, `--num_layers 12`, `--max_length 1024`**
(the V3 checkpoints; the LES-wrapper now defaults to both).

## 0. Reference sets

All PRS/RRS conditions are in `V3_PRS-RRS/` (2-column `SEQ1,SEQ2`):
`PRS-RRS/` (regular), `PRS-RRS_no_homodimers/`, `PRS-RRS_homodimers_only/`,
`random_controls/`. Training sets are not committed.

## 1. Train + dual LES (full and homodimer-depleted)

```bash
./run_all_replicates.sh            # replicates 2..10 (V3-1 done separately)
```

Per replicate `run_replicate.sh` does a 10-epoch from-scratch training (per-epoch
ROC/Best-F1) then two LES-wrapper passes — full PRS/RRS and the
homodimer-depleted set — writing `results/dce_V3-k_scratch12L_ml1024/{LES,LES_no_homodimers}/`.
Resumable (skips replicates whose `LES_no_homodimers/summary_table.csv` exists).

## 2. Homodimers-only LES (evaluation only)

```bash
./run_homodimers_only_les.sh       # -> Results_Homodimers_only/dce_V3-k_.../LES_homodimers_only/
```

PRS = self-pairs only, paired with the full RRS (there are no RRS homodimers to
keep — see `V3_PRS-RRS/PRS-RRS_homodimers_only/NOTE_RRS_homodimers.md`).

## 3. Random-control LES (evaluation only) — AUC/F1 excluded

```bash
./run_random_controls_les.sh       # ps1 / ps2 / ps1-ps2  -> Results_random_controls/.../LES_{cond}_random/
```

These sets have **no true positives** (both PRS and RRS are random pairs), so the
LES-wrapper **auto-skips ROC-AUC / Best-F1 / LES** (filenames contain `random`);
only the PRS-vs-RRS probability-distribution **violins** and the raw probability
CSVs are produced. A model relying on genuine pair information should show the two
random distributions collapse together. See `inference-recipe.md`.

## 4. Composite / ensemble across replicates

Two scripts, per condition, over the per-replicate LES output dirs:

- **`make_composite_les.py`** — across-replicate mean AUC/Best-F1 **trajectories**
  (± SD), ensemble LES, and pooled PRS-vs-RRS **violins**.
- **`composite_roc_dce.py`** — vertically-averaged **composite ROC** (per-epoch,
  all-epoch overlay with SD bands, AUC-vs-epoch).

Both expect the replicates' per-condition LES dirs presented as `LES_V3-{k}` under
one parent, so stage symlinks per condition and run them (skip `composite_roc_dce.py`
for the random controls — no ROC there):

```bash
PY=/home/ksa/anaconda3/envs/gpt/bin/python      # numpy/matplotlib/sklearn env
stage=$(mktemp -d)
for k in 1 2 3 ... ; do ln -s <path>/dce_V3-$k_.../LES_<condition> "$stage/LES_V3-$k"; done
$PY make_composite_les.py  --parent "$stage" --run_glob 'LES_V3-*' --out composite/<condition> --pos_col Probability_Friends
$PY composite_roc_dce.py   --parent "$stage" --models <N> --model-name ppiDCE --out composite/<condition>/ROC
```

## Summary of outputs

| step | script(s) | output |
|---|---|---|
| train + dual LES | `run_all_replicates.sh` → `run_replicate.sh` | `results/.../LES`, `LES_no_homodimers` |
| homodimers-only | `run_homodimers_only_les.sh` | `Results_Homodimers_only/.../LES_homodimers_only` |
| random controls | `run_random_controls_les.sh` | `Results_random_controls/.../LES_{cond}_random` (violins only) |
| composite/ensemble | `make_composite_les.py`, `composite_roc_dce.py` | trajectories, violins, composite ROC per condition |
