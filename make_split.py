#!/usr/bin/env python3
"""
make_split.py — Deterministic, label-stratified train/val split of a
`depleted_training_set-V3-{k}.csv` (or any `seq1,seq2,label` CSV) for ppiDCE.

The V3 training sets ship as one file per replicate with no validation split, so
this makes one explicitly rather than leaving it implicit. The split is
stratified on the label and reproducible from --seed.

Two modes:

  * default (`--keep_val_in_train`, on by default) — the validation rows are
    *sampled* from the training set but **not removed** from it. Train stays the
    full replicate; val is an in-distribution monitor of fit, not a holdout.
    The real holdout is the matched PRS/RRS reference set, which is what
    checkpoints are selected on.
  * `--deplete` — the classic disjoint split: val rows are carved out of train.

Output is written **with** a `seq1,seq2,label` header row, because ppiDCE's
`PPICrossDataset` reads with `pd.read_csv(path)` (header inferred) and would
otherwise silently consume the first data row as column names.

Example
-------
    python make_split.py \\
        --input MED4_V3_Trains/depleted_training_set-V3-1.csv \\
        --output_dir results/dce_V3-1/data \\
        --val_frac 0.1 --seed 42
"""
import argparse
import os
import random

import pandas as pd


def sniff_header(path):
    """Return 0 if the first row looks like a header, else None (headerless)."""
    first = pd.read_csv(path, header=None, nrows=1)
    return 0 if str(first.iloc[0, 0]).strip().lower() in (
        'seq1', 'sequence1', 'seq_a', 'protein1') else None


def main():
    ap = argparse.ArgumentParser(description='Stratified train/val split for ppiDCE.')
    ap.add_argument('--input', required=True, help='Training CSV (seq1, seq2, label).')
    ap.add_argument('--output_dir', required=True, help='Directory for train.csv / val.csv.')
    ap.add_argument('--val_frac', type=float, default=0.1, help='Validation fraction (default 0.1).')
    ap.add_argument('--seed', type=int, default=42, help='Random seed.')
    ap.add_argument('--deplete', action='store_true',
                    help='Remove the val rows from train (disjoint split). Default is to '
                         'keep them in train — val is an in-distribution fit monitor.')
    args = ap.parse_args()

    if not 0 < args.val_frac < 1:
        raise SystemExit(f"--val_frac must be in (0, 1), got {args.val_frac}")

    df = pd.read_csv(args.input, header=sniff_header(args.input))
    if len(df.columns) != 3:
        raise SystemExit(f"{args.input}: expected 3 columns (seq1, seq2, label), "
                         f"got {len(df.columns)}")
    df.columns = ['seq1', 'seq2', 'label']

    rng = random.Random(args.seed)
    val_idx = []
    for label, group in df.groupby('label'):
        idx = list(group.index)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * args.val_frac))
        val_idx.extend(idx[:n_val])
        print(f"label {label}: {len(idx)} rows -> {n_val} val")

    val_set = set(val_idx)
    val = df.loc[sorted(val_set)]
    train = df if not args.deplete else df.loc[[i for i in df.index if i not in val_set]]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, 'train.csv')
    val_path = os.path.join(args.output_dir, 'val.csv')
    train.to_csv(train_path, index=False, header=True)
    val.to_csv(val_path, index=False, header=True)

    mode = 'DEPLETED (val removed from train)' if args.deplete \
        else 'NON-DEPLETED (val rows retained in train)'
    print(f"\nwrote {train_path} ({len(train)} rows)")
    print(f"wrote {val_path} ({len(val)} rows)")
    print(f"mode: {mode}")
    print(f"source: {args.input}  seed={args.seed}  val_frac={args.val_frac}")


if __name__ == '__main__':
    main()
