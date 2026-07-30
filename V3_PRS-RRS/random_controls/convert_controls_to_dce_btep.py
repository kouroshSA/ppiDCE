#!/usr/bin/env python
"""
Convert the V2 random-substituted PRS / RRS controls into ppiDCE / ppiBTEP
format (two sequence columns, no tags, no label).

Source rows are the standard PRS/RRS 5-field layout
`<ps1>,SEQ1,<ps2>,SEQ2,<` (empty trailing label); output keeps the two
sequence columns only, exactly as done for the non-randomized PRS/RRS sets one
level up. Sequences are copied verbatim; the only filename change is the
`-V2-` -> `-V3-` retag, matching the renamed PRS/RRS sets one level up (the
upstream source folder still uses the original `-V2-` names).

    python convert_controls_to_dce_btep.py
"""

import csv
import glob
import os

SRC = "/home/ksa/Dropbox/LES_and_V3_Datasets/OOF_set_v2/V2_PRS_RRS_random_controls"
OUT = os.path.dirname(os.path.abspath(__file__))


def convert_pairs(path, out_path):
    n = 0
    with open(path) as fin, open(out_path, "w", newline="") as fout:
        w = csv.writer(fout)
        for i, row in enumerate(csv.reader(fin), 1):
            if len(row) != 5:
                raise ValueError(f"{path}:{i}: expected 5 fields, got {len(row)}")
            ps1, seq1, ps2, seq2, _ = (c.strip() for c in row)
            if ps1 != "<ps1>" or ps2 != "<ps2>":
                raise ValueError(f"{path}:{i}: unexpected tags {ps1!r} {ps2!r}")
            w.writerow([seq1, seq2])
            n += 1
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"source : {SRC}\noutput : {OUT}\n")
    files = sorted(glob.glob(os.path.join(SRC, "*_random.csv")))
    for path in files:
        name = os.path.basename(path).replace("-V2-", "-V3-")
        out = os.path.join(OUT, name)
        n = convert_pairs(path, out)
        print(f"  {name:35s} rows={n:5d}")
    print(f"\nconverted {len(files)} control files")


if __name__ == "__main__":
    main()
