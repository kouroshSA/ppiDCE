# Note — the RRS contains no homodimers

The random reference sets (`RRS-V3-{1..10}`) contain **no homodimers** — no pair
has `SEQ1 == SEQ2`. This was verified: 0 homodimers across all 10 replicates.
Homodimers (self-pairs) occur **only in the positive sets (PRS)**, 22–32 per
replicate.

Consequence for the homodimer-related folders (both carry the **full** RRS):

- **`PRS-RRS_no_homodimers/`** — only the PRS is depleted (self-pairs dropped).
  The RRS is the full 100-pair set, unchanged, because there is nothing to remove.

- **`PRS-RRS_homodimers_only/`** — only the PRS is filtered to homodimers. The RRS
  is the full 100-pair set, copied unchanged: a "homodimers-only" RRS would be
  **empty**, so the full random set is retained instead, keeping this a usable
  positives-vs-negatives evaluation set (homodimer positives vs random negatives).

In short: every RRS here is the same full 100-pair random reference set; only the
PRS differs between the `full` / `no_homodimers` / `homodimers_only` views.
