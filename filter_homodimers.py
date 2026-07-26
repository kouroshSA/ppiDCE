#!/usr/bin/env python3
"""
filter_homodimers.py — Drop homodimer pairs (seq1 == seq2) from a PRS/RRS
reference CSV, for a "no homodimers" variant of the LES analysis.

Across all 10 MED4 V3 replicates, PRS-V3-{1..10}.csv is 22-32% homodimers
while RRS-V3-{1..10}.csv has none (checked 2026-07-26). Homodimer PRS pairs
score consistently closer to the RRS (non-interacting) baseline than
heterodimer PRS pairs do, dragging down AUC/Best-F1 for a metric that's
otherwise meant to measure heterodimer discrimination — see
LES-wrapper.md / train-recipe.md for the numbers this produced on V3-1.

Usage:
    python filter_homodimers.py --input MED4_PRS-RRS/PRS-V3-1.csv \\
        --output results/dce_V3-1_.../PRS-V3-1_no-homodimers.csv
"""
import argparse

import pandas as pd


def sniff_header(path):
    first = pd.read_csv(path, header=None, nrows=1)
    return 0 if str(first.iloc[0, 0]).strip().lower() in (
        'seq1', 'sequence1', 'seq_a', 'protein1') else None


def main():
    ap = argparse.ArgumentParser(description='Drop homodimer (seq1==seq2) rows from a reference CSV.')
    ap.add_argument('--input', required=True, help='PRS/RRS CSV (seq1,seq2[,label]).')
    ap.add_argument('--output', required=True, help='Path to write the filtered CSV (with header).')
    args = ap.parse_args()

    df = pd.read_csv(args.input, header=sniff_header(args.input))
    cols = ['seq1', 'seq2', 'label'][:len(df.columns)]
    df.columns = cols

    homo = df.seq1 == df.seq2
    filtered = df.loc[~homo]
    filtered.to_csv(args.output, index=False)

    print(f"{args.input}: {homo.sum()}/{len(df)} homodimer pairs dropped, "
          f"{len(filtered)} written to {args.output}")


if __name__ == '__main__':
    main()
