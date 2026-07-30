# Random-substituted PRS / RRS controls in ppiDCE / ppiBTEP format

The random-substituted PRS / RRS control sets, re-encoded into the ppiDCE /
ppiBTEP two-column format (`SEQ1,SEQ2`, no tags, no label) — the same conversion
applied to the real PRS / RRS sets in the parent folder.

Generated 2026-07-24 by `convert_controls_to_dce_btep.py` (in this folder).

## Source

`/home/ksa/Dropbox/LES_and_V3_Datasets/OOF_set_v2/V2_PRS_RRS_random_controls/`

Each source row is the standard 5-field PRS/RRS layout
`<ps1>,SEQ1,<ps2>,SEQ2,<`; conversion keeps the two sequence columns only.

## Files (60 total)

10 replicates × {PRS, RRS} × 3 randomization schemes, 100 pairs each:

- `*_ps1_random.csv`     — partner 1 sequence randomly substituted
- `*_ps2_random.csv`     — partner 2 sequence randomly substituted
- `*_ps1-ps2_random.csv` — both partners randomly substituted

e.g. `PRS-V3-1_ps1_random.csv`, `RRS-V3-10_ps1-ps2_random.csv`. Filenames match
the source apart from the `-V2-` -> `-V3-` retag applied on 2026-07-24 (so they
line up with the renamed PRS/RRS sets one level up); the upstream source folder
still uses the original `-V2-` names. See that folder's README / `manifest.csv`
for how the substitutions were built.

## Format & integrity

`SEQ1,SEQ2`, two columns, headerless — matching the real PRS/RRS conversion one
level up. Sequences are copied verbatim; verified that per-file row counts (100)
and the `(SEQ1, SEQ2)` pairs are byte-identical to the source.

## Regenerating

```bash
/home/ksa/anaconda3/envs/gpt/bin/python convert_controls_to_dce_btep.py
```

Reads from the V2 controls source folder and writes into this one. See the
parent `../README.md` for the overall format definition and the header/label
assumptions.
