#!/usr/bin/env python
"""Regenerate the homodimer-depleted PRS/RRS sets in this folder from the
non-depleted sibling `../MED4_PRS-RRS/`.

A *homodimer* is a reference pair whose two sequences are identical
(`seq1 == seq2`) — a self-interaction that is trivially detectable and is absent
from the random reference sets. Depletion drops those rows from each PRS so the
evaluation isolates *heterotypic* interactions; the RRS sets contain no
homodimers and are copied unchanged.

Row order is preserved. Files are the same 2-column headerless `SEQ1,SEQ2`
format as the source. Run from anywhere:

    python make_no_homodimers.py
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "MED4_PRS-RRS")


def load(path):
    with open(path) as f:
        return [r for r in csv.reader(f) if r]


def main():
    os.makedirs(HERE, exist_ok=True)
    print(f"source : {SRC}\noutput : {HERE}\n")
    for path in sorted(glob.glob(os.path.join(SRC, "*-V3-*.csv"))):
        name = os.path.basename(path)
        rows = load(path)
        kept = [r for r in rows if r[0].strip() != r[1].strip()]
        with open(os.path.join(HERE, name), "w", newline="") as fout:
            csv.writer(fout).writerows(kept)
        dropped = len(rows) - len(kept)
        print(f"  {name:15s} kept={len(kept):4d}  dropped_homodimers={dropped}")


if __name__ == "__main__":
    main()
