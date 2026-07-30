#!/usr/bin/env python
"""Vertical-averaged composite ROC across N ppiYYD models, per epoch + overview.

For each training epoch, every model's ROC curve (PRS = positives, RRS =
negatives) is computed from its `epoch_<e>/combined_probabilities_epoch<e>.csv`
(col 0 = PRS P(interaction), col 1 = RRS P(interaction); ragged columns/blank
cells allowed). Each curve's TPR is linearly interpolated onto a common 501-point
FPR grid and the curves are **vertically averaged** (mean TPR at each FPR), with a
mean +/- 1 SD band showing between-model spread — the same construction as the
ppiGPLM composite ROC, on ppiYYD's epoch axis.

Outer product: per-epoch composite figure + mean-curve CSV, an all-epochs overlay
coloured by epoch, and an AUC-vs-epoch composite trajectory.

  python composite_roc_ppiyyd.py --parent <dir with LES_V3-1..N/> \
      --models 3 --out <outdir>

Per-model dirs are `--folder-template` with {k} (default 'LES_V3-{k}'); the
per-epoch file is `--file-template` with {it} (default the combined_probabilities
path above). Requires numpy / matplotlib / scikit-learn (the `gpt` env has them).
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import roc_curve, auc

PUB_DPI = 600
GRID = np.linspace(0.0, 1.0, 501)
MEAN_C = "#1f4e9c"
CMAP = plt.cm.viridis

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 14,
    'axes.labelsize': 13, 'axes.linewidth': 1.1, 'savefig.bbox': 'tight',
})


def read_combined(path):
    prs, rrs = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) >= 1 and row[0].strip() != "":
                prs.append(float(row[0]))
            if len(row) >= 2 and row[1].strip() != "":
                rrs.append(float(row[1]))
    return np.array(prs), np.array(rrs)


def best_f1(y, s):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(yy.sum(), 1)
    with np.errstate(invalid="ignore"):
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(np.nanmax(f1))


def model_roc(prs, rrs):
    y = np.r_[np.ones(len(prs)), np.zeros(len(rrs))]
    s = np.r_[prs, rrs]
    fpr, tpr, _ = roc_curve(y, s)
    tpr_i = np.interp(GRID, fpr, tpr); tpr_i[0] = 0.0; tpr_i[-1] = 1.0
    return tpr_i, auc(fpr, tpr), best_f1(y, s)


def composite_at(parent, ftmpl, filetmpl, models, ep):
    curves, aucs, f1s = [], [], []
    for k in range(1, models + 1):
        path = os.path.join(parent, ftmpl.format(k=k), filetmpl.format(it=ep))
        if not os.path.isfile(path):
            print(f"  [WARN] missing: {path}")
            continue
        prs, rrs = read_combined(path)
        tpr_i, a, f1 = model_roc(prs, rrs)
        curves.append(tpr_i); aucs.append(a); f1s.append(f1)
    return np.array(curves), np.array(aucs), np.array(f1s)


def per_epoch_figure(ep, curves, aucs, out, n_models, model_name):
    mean = curves.mean(axis=0); sd = curves.std(axis=0, ddof=0)
    lo = np.clip(mean - sd, 0, 1); hi = np.clip(mean + sd, 0, 1)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for c in curves:
        ax.plot(GRID, c, color=MEAN_C, lw=0.8, alpha=0.30)
    ax.fill_between(GRID, lo, hi, color=MEAN_C, alpha=0.20,
                    label=f"±1 SD (n = {n_models} models)")
    ax.plot(GRID, mean, color=MEAN_C, lw=2.6, label="Vertical mean ROC")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Composite ROC — {n_models}-model {model_name} ensemble, epoch {ep}\n"
                 f"AUC = {aucs.mean():.4f} ± {aucs.std(ddof=0):.4f} "
                 f"(mean ± SD, n = {n_models})")
    ax.grid(True, ls="--", alpha=0.4); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"composite_ROC_epoch{ep}.{ext}"), dpi=PUB_DPI)
    plt.close(fig)
    with open(os.path.join(out, f"composite_ROC_epoch{ep}_mean_curve.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["FPR", "TPR_mean", "TPR_sd"])
        for x, m, s in zip(GRID, mean, sd):
            w.writerow([f"{x:.5f}", f"{m:.6f}", f"{s:.6f}"])
    return mean, sd


def overview(epochs, means, sds, auc_means, auc_sds, out, n_models, model_name):
    # all-epochs mean-ROC overlay, coloured by epoch (lines only)
    norm = Normalize(min(epochs), max(epochs))
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for e, m in zip(epochs, means):
        ax.plot(GRID, m, color=CMAP(norm(e)), lw=2)
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Mean ROC of the {n_models}-model {model_name} ensemble\nacross training epochs")
    ax.grid(True, ls="--", alpha=0.4); ax.set_axisbelow(True)
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Training epoch")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"composite_ROC_all_epochs.{ext}"), dpi=PUB_DPI)
    plt.close(fig)

    # same overlay, but with each epoch's between-model ±1 SD band
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for e, m, s in zip(epochs, means, sds):
        c = CMAP(norm(e))
        ax.fill_between(GRID, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                        color=c, alpha=0.12, linewidth=0)
        ax.plot(GRID, m, color=c, lw=1.8)
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Mean ROC ±1 SD of the {n_models}-model {model_name} ensemble\n"
                 f"across training epochs (bands = between-model SD)")
    ax.grid(True, ls="--", alpha=0.4); ax.set_axisbelow(True)
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Training epoch")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"composite_ROC_all_epochs_bands.{ext}"), dpi=PUB_DPI)
    plt.close(fig)

    # AUC vs epoch (composite mean +/- SD)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.fill_between(epochs, np.clip(auc_means - auc_sds, 0, 1),
                    np.clip(auc_means + auc_sds, 0, 1), color=MEAN_C, alpha=0.20,
                    label=f"±1 SD (n = {n_models} models)")
    ax.plot(epochs, auc_means, "o-", color=MEAN_C, lw=2.4, markersize=6,
            label="Ensemble mean AUC")
    for e, m in zip(epochs, auc_means):
        ax.annotate(f"{m:.3f}", (e, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="0.15")
    ax.set_xlabel("Training epoch"); ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.0); ax.set_xticks(list(epochs))
    ax.set_title(f"Composite ROC-AUC across training — {n_models}-model {model_name} ensemble")
    ax.grid(True, ls="--", alpha=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"composite_ROC_AUC_vs_epoch.{ext}"), dpi=PUB_DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--models", type=int, default=3)
    ap.add_argument("--folder-template", default="LES_V3-{k}")
    ap.add_argument("--file-template",
                    default="epoch_{it}/combined_probabilities_epoch{it}.csv")
    ap.add_argument("--epochs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="ppiYYD")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    epochs, means, sds, auc_m, auc_s = [], [], [], [], []
    rows = []
    for ep in a.epochs:
        curves, aucs, f1s = composite_at(a.parent, a.folder_template,
                                         a.file_template, a.models, ep)
        if len(curves) == 0:
            continue
        m, s = per_epoch_figure(ep, curves, aucs, a.out, len(curves), a.model_name)
        epochs.append(ep); means.append(m); sds.append(s)
        auc_m.append(aucs.mean()); auc_s.append(aucs.std(ddof=0))
        rows.append((ep, len(curves), aucs.mean(), aucs.std(ddof=0),
                     f1s.mean(), f1s.std(ddof=0)))
        print(f"  epoch {ep:>2}: composite AUC = {aucs.mean():.4f} ± "
              f"{aucs.std(ddof=0):.4f}  (n={len(curves)})")

    epochs = np.array(epochs, float)
    overview(epochs, means, sds, np.array(auc_m), np.array(auc_s), a.out, a.models, a.model_name)
    with open(os.path.join(a.out, "composite_ROC_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "n_models", "AUC_mean", "AUC_sd", "BestF1_mean", "BestF1_sd"])
        for r in rows:
            w.writerow([int(r[0]), r[1]] + [f"{v:.6f}" for v in r[2:]])
    print(f"DONE -> {a.out}")


if __name__ == "__main__":
    main()
