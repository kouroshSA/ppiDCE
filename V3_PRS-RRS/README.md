# V3 PRS / RRS reference sets

The complete set of V3 (MCCV replicate) PRS/RRS **evaluation** reference sets, in
the shared **ppiDCE / ppiBTEP / ppiYYD 2-column format** (`SEQ1,SEQ2`, headerless).
Ten replicates (V3-1 … V3-10). Training datasets are intentionally **not**
included here.

## Contents

| folder | what it is |
|---|---|
| `PRS-RRS/` | regular PRS (positives) / RRS (random negatives), **100 pairs** each — `PRS-V3-{1..10}.csv`, `RRS-V3-{1..10}.csv` |
| `PRS-RRS_no_homodimers/` | PRS with homodimers (`SEQ1==SEQ2`) removed → **68–78 pairs**; RRS unchanged (100) |
| `PRS-RRS_homodimers_only/` | PRS with **only** the homodimers kept → **22–32 pairs**; RRS = full 100 |
| `random_controls/` | random-substituted controls: `{ps1,ps2,ps1-ps2}_random` × {PRS,RRS} × V3-{1..10}, 100 pairs each |

`PRS-RRS_no_homodimers/` and `PRS-RRS_homodimers_only/` are a disjoint partition
of `PRS-RRS/` (heterotypic vs. self-pair positives), both against the full RRS —
see each folder's `NOTE_RRS_homodimers.md`. The RRS never contains homodimers, so
it is the full 100-pair set in every case.

Format: headerless `SEQ1,SEQ2`. Interaction is order-invariant, so pair order is
not meaningful. The 5-column ppiGPLM-format encoding of exactly the same data is
in the ppiGPLM repo's `V3_PRS-RRS/`.
