# MED4 PRS / RRS — homodimer-depleted

Homodimer-depleted versions of the V3 positive (PRS) and random (RRS) reference
sets in [`../MED4_PRS-RRS/`](../MED4_PRS-RRS). Same 2-column, headerless
`SEQ1,SEQ2` format (shared by ppiYYD / ppiDCE / ppiBTEP); sequences and pairing
are copied verbatim.

## What "homodimer-depleted" means

A **homodimer** is a reference pair whose two sequences are identical
(`seq1 == seq2`) — a self-interaction that is trivially detectable from the pair
alone. The positive sets carry 22–32 such self-pairs per replicate; the random
sets carry none. Evaluating on the full PRS can therefore flatter a score
without reflecting genuine interaction understanding — especially for Siamese /
two-tower encoders, where an identical-input shortcut is easy to exploit.

Depleting the homodimers isolates the **heterotypic** case (the two partners
differ), giving the cleaner discrimination measurement.

## Files (20)

- `PRS-V3-{1..10}.csv` — positives, homodimers removed → **68–78 pairs** each
  (down from 100).
- `RRS-V3-{1..10}.csv` — randoms, **100 pairs** each, **unchanged** (they never
  contained homodimers).

Row order is preserved relative to the source; the depleted PRS is exactly the
non-depleted PRS with the `seq1 == seq2` rows dropped.

| replicate | PRS (depleted) | homodimers removed |
|---|---:|---:|
| V3-1 | 68 | 32 |
| V3-2 | 70 | 30 |
| V3-3 | 74 | 26 |
| V3-4 | 72 | 28 |
| V3-5 | 75 | 25 |
| V3-6 | 75 | 25 |
| V3-7 | 68 | 32 |
| V3-8 | 76 | 24 |
| V3-9 | 78 | 22 |
| V3-10 | 75 | 25 |

## Regenerating

Fully reproducible from the sibling folder — no external data needed:

```bash
python make_no_homodimers.py
```

It reads `../MED4_PRS-RRS/*-V3-*.csv`, drops the `seq1 == seq2` rows from each,
and writes the result here (RRS files come out identical to the source).
