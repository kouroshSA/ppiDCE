#!/usr/bin/env python3
"""
make_composite_roc.py — Composite ROC-curve ensemble analysis for ppiDCE,
matching the labeling/structure of ppiGPLM's ensemble composites
(ppiGPLM_Ensemble_Composites_V3_6k), ported to ppiDCE's per-*epoch*
checkpoints (ppiGPLM is iteration-based; ppiDCE trains for 10 epochs).

For each training epoch, and across the N replicate models (V3-1..V3-N):
  - composite_ROC_epoch{N}.png/.pdf        - single-panel mean ROC (+/-1 SD
                                              band, individual model curves,
                                              chance line)
  - composite_ROC_epoch{N}_panel.png/.pdf  - two-panel: [mean ROC + pooled
                                              ROC] | [per-model AUC box+strip
                                              vs chance]
  - composite_ROC_epoch{N}_mean_curve.csv  - FPR grid, TPR mean/sd/min/max
  - composite_ROC_epoch{N}_per_model.csv   - per-model AUC/Best-F1 at this epoch
  - composite_ROC_epoch{N}_stats.txt       - text summary
  - _run_log.txt                            - same text (matches ppiGPLM's
                                              convention of keeping a captured
                                              log alongside the stats file)
Each is produced twice: raw and Gaussian-smoothed (display-only smoothing;
every reported number is from the raw data), in epoch{N}_ROCs/ and
epoch{N}_ROCs_smoothed/.

Aggregated across all epochs (written directly under --out):
  - all_checkpoints_summary.csv
  - AUC_vs_epoch.png/.pdf                   - ensemble mean+-SD AUC vs epoch,
                                              jittered per-model points
  - composite_ROC_all_checkpoints.png/.pdf  - all epochs' mean ROC overlaid,
                                              colored by epoch
  - composite_ROC_checkpoint_grid.png/.pdf  - small-multiples grid, one panel
                                              per epoch
  - composite_native_PRS-RRS_violins.png/.pdf - PRS vs RRS P(interaction),
    pooled across all replicates, per epoch (matches make_composite_les.py's
    violin; this script's own copy for structural parity with the ppiGPLM
    output tree, which keeps the violin at the same level as ROC/).

For conditions with no numbered-epoch data worth trending (e.g. the
random-substituted controls, which ppiGPLM's own tree also treats as
violins-only), pass --violins_only to skip the ROC/trajectory outputs.

Example
-------
    python make_composite_roc.py --parent results --les_subdir LES \\
        --out Ensemble_Composites_V3/regular

    python make_composite_roc.py --parent Results_random_controls \\
        --les_subdir LES_ps1_random --out Ensemble_Composites_V3/ps1_random \\
        --violins_only
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from scipy import stats as spstats
from sklearn.metrics import roc_curve, auc as sk_auc, f1_score

PUB_DPI = 600
MODEL_C = "#9ecae1"   # light blue, individual-model ROC curves
MEAN_C = "#1f4e9c"    # dark navy, mean/ensemble curves
POOLED_C = "#b2182b"  # dashed crimson, pooled-data curve
PRS_C, RRS_C = "#2166ac", "#b2182b"
GRID = np.linspace(0.0, 1.0, 501)
SMOOTH_SIGMA = 5  # FPR-grid points (~0.01 FPR), matches the reference example

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# -----------------------------------------------------------------------------
def _replicate_num(path):
    m = re.search(r'V3-(\d+)', path)
    return int(m.group(1)) if m else 0


def find_run_dirs(parent, replicate_glob, les_subdir):
    candidates = glob.glob(os.path.join(parent, replicate_glob))
    dirs = [os.path.join(d, les_subdir) for d in candidates
            if os.path.isdir(os.path.join(d, les_subdir))
            and os.path.exists(os.path.join(d, les_subdir, 'summary_table.csv'))]
    if not dirs:
        raise SystemExit(f"No replicate dirs with {les_subdir}/summary_table.csv "
                         f"match '{replicate_glob}' under {parent}")
    dirs.sort(key=_replicate_num)
    return dirs


def replicate_name(run_dir):
    m = re.search(r'V3-(\d+)', run_dir)
    return f"V3-{m.group(1)}" if m else os.path.basename(os.path.dirname(run_dir))


def read_pos_probs(path, pos_col='Probability_Friends'):
    vals = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r, None)
        idx = None
        if header is not None:
            for i, c in enumerate(header):
                if c.strip().lower() == pos_col.lower():
                    idx = i
                    break
        if idx is None:
            idx = -2
        for row in r:
            if not row:
                continue
            try:
                vals.append(float(row[idx]))
            except (ValueError, IndexError):
                pass
    return vals


def read_epochs(run_dir):
    """Numbered epochs (int) with a summary_table.csv row, sorted."""
    out = []
    with open(os.path.join(run_dir, 'summary_table.csv')) as fh:
        for row in csv.DictReader(fh):
            ckpt = (row.get('checkpoint') or '').strip()
            ep = (row.get('epoch') or '').strip()
            if ckpt.startswith('ppiDCE_epoch') and ep.isdigit():
                out.append(int(ep))
    return sorted(out)


def read_bestf1(run_dir, epoch):
    with open(os.path.join(run_dir, 'summary_table.csv')) as fh:
        for row in csv.DictReader(fh):
            if (row.get('epoch') or '').strip() == str(epoch):
                return float(row['Best_F1'])
    raise KeyError(f"epoch {epoch} not found in {run_dir}/summary_table.csv")


# -----------------------------------------------------------------------------
def gaussian_smooth(y, sigma):
    return gaussian_filter1d(y, sigma=sigma, mode='nearest')


def best_f1_exhaustive(probs, labels):
    best = -1.0
    for t in np.unique(probs):
        f = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f >= best:
            best = f
    return best


def load_epoch(run_dirs, epoch, pos_col):
    """Per-model curves/AUC/Best-F1 + pooled probs/labels, for one epoch."""
    models, tprs, aucs, bestf1s, n_prs, n_rrs = [], [], [], [], [], []
    pooled_probs, pooled_labels = [], []
    for d in run_dirs:
        prs = read_pos_probs(os.path.join(d, f"epoch_{epoch}", f"PRS_epoch{epoch}_probabilities.csv"), pos_col)
        rrs = read_pos_probs(os.path.join(d, f"epoch_{epoch}", f"RRS_epoch{epoch}_probabilities.csv"), pos_col)
        probs = np.array(prs + rrs, dtype=float)
        labels = np.array([1] * len(prs) + [0] * len(rrs))
        fpr, tpr, _ = roc_curve(labels, probs)
        a = float(sk_auc(fpr, tpr))
        tpr_grid = np.interp(GRID, fpr, tpr)
        models.append(replicate_name(d))
        tprs.append(tpr_grid)
        aucs.append(a)
        bestf1s.append(read_bestf1(d, epoch))
        n_prs.append(len(prs))
        n_rrs.append(len(rrs))
        pooled_probs.extend(probs.tolist())
        pooled_labels.extend(labels.tolist())
    return {
        'models': models, 'tprs': np.array(tprs), 'aucs': np.array(aucs),
        'bestf1s': np.array(bestf1s), 'n_prs': n_prs, 'n_rrs': n_rrs,
        'pooled_probs': np.array(pooled_probs), 'pooled_labels': np.array(pooled_labels),
    }


# -----------------------------------------------------------------------------
def write_stats(path, data, epoch, out_dir_label, smoothed, ensemble_label):
    models, aucs, bestf1s = data['models'], data['aucs'], data['bestf1s']
    n = len(models)
    lines = []
    for m, npr, nrr, a, bf in zip(models, data['n_prs'], data['n_rrs'], aucs, bestf1s):
        lines.append(f"{m:>6}  n_PRS={npr:4d}  n_RRS={nrr:4d}  AUC={a:.4f}  bestF1={bf:.4f}")
    prs_range = f"{min(data['n_prs'])}" if len(set(data['n_prs'])) == 1 else f"{min(data['n_prs'])}-{max(data['n_prs'])}"
    rrs_range = f"{min(data['n_rrs'])}" if len(set(data['n_rrs'])) == 1 else f"{min(data['n_rrs'])}-{max(data['n_rrs'])}"

    mean, sd = aucs.mean(), aucs.std(ddof=0)
    sem = aucs.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    tcrit = spstats.t.ppf(0.975, df=n - 1) if n > 1 else float('nan')
    ci_lo, ci_hi = mean - tcrit * sem, mean + tcrit * sem

    pooled_fpr, pooled_tpr, _ = roc_curve(data['pooled_labels'], data['pooled_probs'])
    pooled_auc = float(sk_auc(pooled_fpr, pooled_tpr))
    pooled_f1 = best_f1_exhaustive(data['pooled_probs'], data['pooled_labels'])

    smooth_line = (f"figure curves: Gaussian-smoothed for display, sigma = {SMOOTH_SIGMA} "
                   f"FPR-grid points (~{SMOOTH_SIGMA / (len(GRID) - 1):.4f} FPR). All numbers "
                   f"above are from the raw data.") if smoothed else "figure curves: raw (unsmoothed)"

    text = (
        "\n".join(lines) + "\n\n"
        f"Composite ROC — {ensemble_label} — epoch {epoch}\n"
        f"models (n = {n}): {', '.join(models)}\n"
        f"positives per model (PRS): {prs_range}    negatives per model (RRS): {rrs_range}\n\n"
        f"AUC  mean = {mean:.4f}   SD = {sd:.4f}   SEM = {sem:.4f}\n"
        f"AUC  median = {np.median(aucs):.4f}   min = {aucs.min():.4f}   max = {aucs.max():.4f}\n"
        f"AUC  95% CI (t, df={n - 1}) = [{ci_lo:.4f}, {ci_hi:.4f}]\n\n"
        f"Best-F1  mean = {bestf1s.mean():.4f}   SD = {bestf1s.std(ddof=0):.4f}   "
        f"min = {bestf1s.min():.4f}   max = {bestf1s.max():.4f}\n\n"
        f"Pooled data ({sum(data['n_prs'])} PRS + {sum(data['n_rrs'])} RRS): "
        f"AUC = {pooled_auc:.4f}   Best F1 = {pooled_f1:.4f}\n\n"
        f"{smooth_line}\n"
    )
    with open(path, 'w') as f:
        f.write(text)
    return text, pooled_fpr, pooled_tpr, pooled_auc


def write_run_log(path, stats_text, out_dir):
    with open(path, 'w') as f:
        f.write(stats_text + f"\nWrote figures and tables to {out_dir}\n")


# -----------------------------------------------------------------------------
def plot_single(grid, tprs, mean_auc, sd_auc, epoch, ensemble_label, out_base, smoothed):
    plot_tprs = np.array([gaussian_smooth(t, SMOOTH_SIGMA) for t in tprs]) if smoothed else tprs
    mean_tpr = plot_tprs.mean(axis=0)
    sd_tpr = plot_tprs.std(axis=0, ddof=0)
    lo, hi = np.clip(mean_tpr - sd_tpr, 0, 1), np.clip(mean_tpr + sd_tpr, 0, 1)
    n = len(tprs)

    fig, ax = plt.subplots(figsize=(8, 8))
    for t in plot_tprs:
        ax.plot(grid, t, color=MODEL_C, lw=1.0, alpha=0.7)
    ax.fill_between(grid, lo, hi, color=MEAN_C, alpha=0.20)
    ax.plot(grid, mean_tpr, color=MEAN_C, lw=3, label=f"Mean ROC (AUC = {mean_auc:.3f} ± {sd_auc:.3f})")
    ax.fill_between([], [], color=MEAN_C, alpha=0.20, label="± 1 SD across models")
    ax.plot([], [], color=MODEL_C, lw=1.2, label=f"Individual models (n = {n})")
    ax.plot([0, 1], [0, 1], color='gray', ls='--', label='Chance')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_title(f"Composite ROC — {ensemble_label} (epoch {epoch})")
    ax.legend(loc='lower right')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f"{out_base}.{ext}", dpi=PUB_DPI)
    plt.close(fig)


def plot_panel(grid, tprs, aucs, mean_auc, sd_auc, pooled_fpr, pooled_tpr, pooled_auc,
               epoch, ensemble_label, out_base, smoothed):
    plot_tprs = np.array([gaussian_smooth(t, SMOOTH_SIGMA) for t in tprs]) if smoothed else tprs
    mean_tpr = plot_tprs.mean(axis=0)
    sd_tpr = plot_tprs.std(axis=0, ddof=0)
    lo, hi = np.clip(mean_tpr - sd_tpr, 0, 1), np.clip(mean_tpr + sd_tpr, 0, 1)
    n = len(tprs)
    pooled_tpr_grid = np.interp(grid, pooled_fpr, pooled_tpr)
    if smoothed:
        pooled_tpr_grid = gaussian_smooth(pooled_tpr_grid, SMOOTH_SIGMA)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    ax = axes[0]
    for t in plot_tprs:
        ax.plot(grid, t, color=MODEL_C, lw=1.0, alpha=0.7)
    ax.fill_between(grid, lo, hi, color=MEAN_C, alpha=0.20)
    ax.plot(grid, mean_tpr, color=MEAN_C, lw=3, label="Mean ROC")
    ax.fill_between([], [], color=MEAN_C, alpha=0.20, label="± 1 SD across models")
    ax.plot(grid, pooled_tpr_grid, color=POOLED_C, lw=2.2, ls='--',
            label=f"Pooled data (AUC = {pooled_auc:.3f})")
    ax.plot([], [], color=MODEL_C, lw=1.2, label=f"Individual models (n = {n})")
    ax.plot([0, 1], [0, 1], color='gray', ls='--', label='Chance')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_title(f"Composite ROC — {ensemble_label} (epoch {epoch})")
    ax.legend(loc='lower right', fontsize=10)

    ax2 = axes[1]
    bp = ax2.boxplot([aucs], positions=[1], widths=0.5, showfliers=False,
                     patch_artist=True, medianprops=dict(color=MEAN_C, lw=2))
    for b in bp['boxes']:
        b.set_facecolor(MODEL_C); b.set_alpha(0.5)
    rng = np.random.default_rng(0)
    jitter = 1 + rng.uniform(-0.08, 0.08, size=n)
    ax2.scatter(jitter, aucs, color=MEAN_C, s=28, zorder=3)
    ax2.errorbar([1.6], [mean_auc], yerr=[sd_auc], fmt='o', color='black',
                markersize=7, capsize=5, elinewidth=1.4)
    ax2.text(1.72, mean_auc, "mean ± SD", va='center', fontsize=11)
    ax2.axhline(0.5, color='gray', ls='--', lw=1.2)
    ax2.set_xlim(0.5, 2.3)
    y0 = min(0.45, float(aucs.min()) - 0.05)
    ax2.set_ylim(y0, 1.0)
    ax2.set_xticks([])
    ax2.set_ylabel('AUC')
    ax2.set_title(f"Per-model AUC\n(mean {mean_auc:.3f} ± {sd_auc:.3f} SD)")

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f"{out_base}_panel.{ext}", dpi=PUB_DPI)
    plt.close(fig)


def write_curve_csvs(out_dir, epoch, grid, tprs, data, smoothed):
    plot_tprs = np.array([gaussian_smooth(t, SMOOTH_SIGMA) for t in tprs]) if smoothed else tprs
    mean_tpr = plot_tprs.mean(axis=0)
    sd_tpr = plot_tprs.std(axis=0, ddof=0)
    with open(os.path.join(out_dir, f"composite_ROC_epoch{epoch}_mean_curve.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["FPR", "TPR_mean", "TPR_sd", "TPR_mean_minus_sd", "TPR_mean_plus_sd", "TPR_min", "TPR_max"])
        for i, fp in enumerate(grid):
            w.writerow([f"{fp:.4f}", f"{mean_tpr[i]:.6f}", f"{sd_tpr[i]:.6f}",
                        f"{max(0.0, mean_tpr[i] - sd_tpr[i]):.6f}", f"{min(1.0, mean_tpr[i] + sd_tpr[i]):.6f}",
                        f"{plot_tprs[:, i].min():.6f}", f"{plot_tprs[:, i].max():.6f}"])
    with open(os.path.join(out_dir, f"composite_ROC_epoch{epoch}_per_model.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["model", "epoch", "AUC", "Best_F1", "PRS_samples", "RRS_samples"])
        for m, a, bf, npr, nrr in zip(data['models'], data['aucs'], data['bestf1s'], data['n_prs'], data['n_rrs']):
            w.writerow([m, epoch, f"{a:.6f}", f"{bf:.6f}", npr, nrr])


def per_epoch_roc(run_dirs, epoch, pos_col, out_root, ensemble_label, smoothed):
    data = load_epoch(run_dirs, epoch, pos_col)
    sub = f"epoch{epoch}_ROCs" + ("_smoothed" if smoothed else "")
    out_dir = os.path.join(out_root, sub)
    os.makedirs(out_dir, exist_ok=True)

    mean_auc, sd_auc = data['aucs'].mean(), data['aucs'].std(ddof=0)
    base = os.path.join(out_dir, f"composite_ROC_epoch{epoch}")
    plot_single(GRID, data['tprs'], mean_auc, sd_auc, epoch, ensemble_label, base, smoothed)

    stats_text, pooled_fpr, pooled_tpr, pooled_auc = write_stats(
        os.path.join(out_dir, f"composite_ROC_epoch{epoch}_stats.txt"),
        data, epoch, sub, smoothed, ensemble_label)
    write_run_log(os.path.join(out_dir, "_run_log.txt"), stats_text, out_dir)

    plot_panel(GRID, data['tprs'], data['aucs'], mean_auc, sd_auc, pooled_fpr, pooled_tpr,
              pooled_auc, epoch, ensemble_label, base, smoothed)
    write_curve_csvs(out_dir, epoch, GRID, data['tprs'], data, smoothed)

    print(f"  epoch {epoch}{' (smoothed)' if smoothed else ''}: "
          f"AUC = {mean_auc:.4f} ± {sd_auc:.4f}  (pooled {pooled_auc:.4f})")
    return {'epoch': epoch, 'aucs': data['aucs'], 'bestf1s': data['bestf1s'],
            'tprs_raw': data['tprs']}


# -----------------------------------------------------------------------------
def plot_auc_vs_epoch(per_epoch, ensemble_label, out_dir):
    epochs = [r['epoch'] for r in per_epoch]
    means = np.array([r['aucs'].mean() for r in per_epoch])
    sds = np.array([r['aucs'].std(ddof=0) for r in per_epoch])
    n = len(per_epoch[0]['aucs'])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    lo, hi = np.clip(means - sds, 0, 1), np.clip(means + sds, 0, 1)
    ax.fill_between(epochs, lo, hi, color=MEAN_C, alpha=0.20)
    ax.plot(epochs, means, 'o-', color=MEAN_C, lw=2.5, markersize=9, zorder=3)
    rng = np.random.default_rng(0)
    for r in per_epoch:
        jitter = r['epoch'] + rng.uniform(-0.08, 0.08, size=len(r['aucs']))
        ax.scatter(jitter, r['aucs'], color=MEAN_C, s=16, alpha=0.45, zorder=2)
    ax.axhline(0.5, color='gray', ls='--', lw=1.2)
    ax.text(max(epochs), 0.49, "chance", ha='right', va='top', color='0.4', fontsize=12)
    ax.set_xlabel('Training epoch'); ax.set_ylabel('AUC')
    ax.set_xticks(epochs)
    ax.set_ylim(0, 1)
    ax.set_title(f"Ensemble AUC across training (mean ± 1 SD, n = {n} models)")
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f"AUC_vs_epoch.{ext}"), dpi=PUB_DPI)
    plt.close(fig)


def _epoch_cmap(n):
    cmap = LinearSegmentedColormap.from_list("epoch_cmap", ["#9ecae1", "#08306b", "#00441b"])
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plot_all_checkpoints(per_epoch, ensemble_label, out_dir):
    epochs = [r['epoch'] for r in per_epoch]
    colors = _epoch_cmap(len(epochs))
    fig, ax = plt.subplots(figsize=(9, 8))
    for r, c in zip(per_epoch, colors):
        mean_auc = r['aucs'].mean()
        mean_tpr = r['tprs_raw'].mean(axis=0)
        sd_tpr = r['tprs_raw'].std(axis=0, ddof=0)
        lo, hi = np.clip(mean_tpr - sd_tpr, 0, 1), np.clip(mean_tpr + sd_tpr, 0, 1)
        ax.fill_between(GRID, lo, hi, color=c, alpha=0.15)
        ax.plot(GRID, mean_tpr, color=c, lw=2.2,
                label=f"{r['epoch']} (AUC {mean_auc:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', ls='--')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_title(f"Mean ROC of the {len(per_epoch[0]['aucs'])}-model V3 ensemble\nacross training epochs")
    ax.text(0.03, 0.97, f"shading: ± 1 SD across the {len(per_epoch[0]['aucs'])} models",
            color='0.4', fontsize=10, ha='left', va='top')
    ax.legend(title="epoch (mean AUC)", loc='lower right', ncol=2, fontsize=9, title_fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f"composite_ROC_all_checkpoints.{ext}"), dpi=PUB_DPI)
    plt.close(fig)


def plot_checkpoint_grid(per_epoch, ensemble_label, out_dir):
    n_ep = len(per_epoch)
    ncols = 5
    nrows = int(np.ceil(n_ep / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, r in enumerate(per_epoch):
        ax = axes[i // ncols][i % ncols]
        mean_auc, sd_auc = r['aucs'].mean(), r['aucs'].std(ddof=0)
        mean_tpr = r['tprs_raw'].mean(axis=0)
        sd_tpr = r['tprs_raw'].std(axis=0, ddof=0)
        lo, hi = np.clip(mean_tpr - sd_tpr, 0, 1), np.clip(mean_tpr + sd_tpr, 0, 1)
        ax.fill_between(GRID, lo, hi, color=MEAN_C, alpha=0.20)
        ax.plot(GRID, mean_tpr, color=MEAN_C, lw=2)
        ax.plot([0, 1], [0, 1], color='gray', ls='--', lw=1)
        ax.set_title(f"epoch {r['epoch']}", fontsize=12)
        ax.text(0.97, 0.05, f"AUC {mean_auc:.3f} ± {sd_auc:.3f}", ha='right', va='bottom',
                fontsize=10, transform=ax.transAxes)
    for j in range(n_ep, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle(f"Mean ROC ± 1 SD of the {len(per_epoch[0]['aucs'])}-model V3 ensemble, per epoch", fontsize=16)
    fig.supxlabel('False positive rate')
    fig.supylabel('True positive rate')
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.96])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f"composite_ROC_checkpoint_grid.{ext}"), dpi=PUB_DPI)
    plt.close(fig)


def write_all_checkpoints_summary(per_epoch, out_dir):
    with open(os.path.join(out_dir, "all_checkpoints_summary.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["epoch", "n_models", "AUC_mean", "AUC_sd", "AUC_sem", "AUC_min", "AUC_max",
                    "BestF1_mean", "BestF1_sd", "BestF1_min", "BestF1_max"])
        for r in per_epoch:
            a, bf = r['aucs'], r['bestf1s']
            n = len(a)
            sem = a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            w.writerow([r['epoch'], n, f"{a.mean():.4f}", f"{a.std(ddof=0):.4f}", f"{sem:.4f}",
                        f"{a.min():.4f}", f"{a.max():.4f}", f"{bf.mean():.4f}", f"{bf.std(ddof=0):.4f}",
                        f"{bf.min():.4f}", f"{bf.max():.4f}"])


# -----------------------------------------------------------------------------
# Violins (matches make_composite_les.py's, renamed to composite_native_*
# for structural parity with the ppiGPLM output tree)
# -----------------------------------------------------------------------------
def pooled_probs_for_epoch(run_dirs, epoch, kind, pos_col):
    out = []
    for d in run_dirs:
        fp = os.path.join(d, f"epoch_{epoch}", f"{kind}_epoch{epoch}_probabilities.csv")
        out.extend(read_pos_probs(fp, pos_col))
    return np.array(out)


def plot_native_violins(run_dirs, epochs, out_dir, pos_col, ensemble_label):
    prs = [pooled_probs_for_epoch(run_dirs, e, "PRS", pos_col) for e in epochs]
    rrs = [pooled_probs_for_epoch(run_dirs, e, "RRS", pos_col) for e in epochs]
    n_prs = len(prs[0]) if prs else 0
    n_rrs = len(rrs[0]) if rrs else 0

    gap = 2.6
    prs_pos = np.arange(1, len(epochs) + 1, dtype=float)
    rrs_pos = prs_pos + len(epochs) + gap
    divider = len(epochs) + gap / 2.0 + 0.5

    fig, ax = plt.subplots(figsize=(max(14.0, 1.2 * len(epochs) + 3), 6.6))

    def viol(data, pos, color):
        parts = ax.violinplot(data, positions=pos, widths=0.9, showmedians=True, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(color); b.set_alpha(0.5); b.set_edgecolor(color); b.set_linewidth(0.7)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.1)

    viol(prs, prs_pos, PRS_C)
    viol(rrs, rrs_pos, RRS_C)

    ax.axvline(divider, color="0.5", ls="--", lw=1.3)
    ax.set_xticks(list(prs_pos) + list(rrs_pos))
    ax.set_xticklabels([str(e) for e in epochs] * 2, rotation=90, fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.2, rrs_pos[-1] + 0.8)
    ax.set_ylabel("P(interaction) = Probability_Friends", fontsize=13)
    ax.set_xlabel("Training epoch", fontsize=13)
    ax.yaxis.grid(True, ls="--", alpha=0.4); ax.set_axisbelow(True)

    ax.text(prs_pos.mean(), 1.06, "PRS (positives)", ha="center", va="bottom",
            fontsize=13, color=PRS_C, fontweight="bold", transform=ax.transData)
    ax.text(rrs_pos.mean(), 1.06, "RRS (negatives)", ha="center", va="bottom",
            fontsize=13, color=RRS_C, fontweight="bold", transform=ax.transData)
    ax.set_title(f"{len(run_dirs)}-model composite - Native - PRS vs RRS\n"
                 f"pooled across the {len(run_dirs)} models; PRS n = {n_prs}, RRS n = {n_rrs} "
                 f"at each indicated epoch (training checkpoint)",
                 fontsize=13, pad=30)
    fig.tight_layout()
    base = os.path.join(out_dir, "composite_native_PRS-RRS_violins")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote composite_native_PRS-RRS_violins.png/.pdf (n_PRS={n_prs}, n_RRS={n_rrs}/epoch)")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Composite ROC-curve ensemble analysis for ppiDCE.")
    ap.add_argument("--parent", required=True, help="Directory containing per-replicate results dirs.")
    ap.add_argument("--replicate_glob", default="dce_V3-*_scratch12L_ml1024",
                    help="Glob for the per-replicate dirs under --parent.")
    ap.add_argument("--les_subdir", required=True, help="LES-wrapper output subdir to read, e.g. LES.")
    ap.add_argument("--out", required=True, help="Output directory for this condition's composite tree.")
    ap.add_argument("--ensemble_label", default="V3 ensemble",
                    help="Label used in plot titles (default: 'V3 ensemble').")
    ap.add_argument("--pos_col", default="Probability_Friends")
    ap.add_argument("--violins_only", action="store_true",
                    help="Only write composite_native_PRS-RRS_violins (skip ROC/trajectory outputs) "
                         "-- for conditions with no meaningful epoch trend, e.g. random controls.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_dirs = find_run_dirs(args.parent, args.replicate_glob, args.les_subdir)
    epoch_sets = [set(read_epochs(d)) for d in run_dirs]
    epochs = sorted(set.intersection(*epoch_sets))
    dropped = [sorted(s - set(epochs)) for s in epoch_sets]
    print(f"{len(run_dirs)} replicates, epochs = {epochs}")
    if any(dropped):
        print(f"  per-replicate dropped (incomplete rows): {dropped}")

    if not args.violins_only:
        roc_dir = os.path.join(args.out, "ROC")
        os.makedirs(roc_dir, exist_ok=True)
        print("Per-epoch composite ROC (raw):")
        per_epoch_raw = [per_epoch_roc(run_dirs, e, args.pos_col, roc_dir, args.ensemble_label, False)
                         for e in epochs]
        print("Per-epoch composite ROC (smoothed):")
        for e in epochs:
            per_epoch_roc(run_dirs, e, args.pos_col, roc_dir, args.ensemble_label, True)

        print("Aggregate ROC views:")
        write_all_checkpoints_summary(per_epoch_raw, roc_dir)
        plot_auc_vs_epoch(per_epoch_raw, args.ensemble_label, roc_dir)
        plot_all_checkpoints(per_epoch_raw, args.ensemble_label, roc_dir)
        plot_checkpoint_grid(per_epoch_raw, args.ensemble_label, roc_dir)
        print(f"  wrote all_checkpoints_summary.csv, AUC_vs_epoch, composite_ROC_all_checkpoints, "
              f"composite_ROC_checkpoint_grid -> {roc_dir}")

        print("Trajectories (AUC/F1, matching make_composite_les.py):")
        les_epochs = epochs
        auc_mat = np.array([r['aucs'] for r in per_epoch_raw]).T
        f1_mat = np.array([r['bestf1s'] for r in per_epoch_raw]).T
        _write_trajectory_figs(les_epochs, auc_mat, f1_mat, args.out)

    print("Native PRS/RRS violins:")
    plot_native_violins(run_dirs, epochs, args.out, args.pos_col, args.ensemble_label)
    print(f"DONE -> {args.out}")


def _write_trajectory_figs(epochs, auc_mat, f1_mat, out):
    """AUC/F1 trajectory figures + composite_trajectory_data.csv, matching
    make_composite_les.py's output (kept alongside the ROC/ subfolder to
    match the ppiGPLM tree, which has both at the same directory level)."""
    def compute_les(vals):
        e = np.asarray(epochs, float)
        en = (e - e.min()) / (e.max() - e.min())
        return float(np.trapezoid(vals, en))

    les_auc = np.array([compute_les(row) for row in auc_mat])
    les_f1 = np.array([compute_les(row) for row in f1_mat])

    def make_traj(mat, color, name, les_vals, base, floor=False):
        mean = mat.mean(axis=0); sd = mat.std(axis=0, ddof=0); n = mat.shape[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        if floor:
            ax.fill_between(epochs, 0, mean, color=color, alpha=0.18, label="Area under curve (LES)")
            ax.errorbar(epochs, mean, yerr=sd, fmt="o-", color=color, lw=2.5, markersize=7,
                        capsize=4, elinewidth=1.4, ecolor=color, label=f"Ensemble mean ±1 SD (n = {n})")
        else:
            lo, hi = np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1)
            ax.fill_between(epochs, lo, hi, color=color, alpha=0.20, label=f"±1 SD (n = {n} replicates)")
            ax.plot(epochs, mean, "o-", color=color, lw=2.5, markersize=7, label="Ensemble mean")
        for x, y in zip(epochs, mean):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9, color="0.15")
        ax.set_xlabel("Training epoch"); ax.set_ylabel(name)
        ax.set_ylim(0, 1); ax.set_xticks(list(epochs))
        ax.grid(True, linestyle="--", alpha=0.7); ax.set_axisbelow(True)
        ax.set_title(f"{name} trajectory — {n}-replicate V3 ensemble\n"
                     f"LES-{name} = {np.nanmean(les_vals):.4f} ± {np.nanstd(les_vals, ddof=0):.4f} "
                     f"(mean ± SD, n = {n})", fontsize=14, pad=12)
        ax.legend(loc="lower right", frameon=True)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{base}.{ext}", dpi=PUB_DPI)
        plt.close(fig)

    make_traj(auc_mat, MEAN_C, "AUC", les_auc, os.path.join(out, "composite_trajectory_AUC"))
    make_traj(f1_mat, "#2c8a3d", "Best F1", les_f1, os.path.join(out, "composite_trajectory_F1"))
    make_traj(auc_mat, MEAN_C, "AUC", les_auc, os.path.join(out, "composite_trajectory_AUC_area"), floor=True)
    make_traj(f1_mat, "#2c8a3d", "Best F1", les_f1, os.path.join(out, "composite_trajectory_F1_area"), floor=True)

    with open(os.path.join(out, "composite_trajectory_data.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "AUC_mean", "AUC_SD", "F1_mean", "F1_SD"])
        for i, e in enumerate(epochs):
            w.writerow([e, f"{auc_mat[:, i].mean():.6f}", f"{auc_mat[:, i].std(ddof=0):.6f}",
                        f"{f1_mat[:, i].mean():.6f}", f"{f1_mat[:, i].std(ddof=0):.6f}"])
        w.writerow([])
        w.writerow(["LES_AUC_mean", f"{les_auc.mean():.6f}", "LES_AUC_SD", f"{les_auc.std(ddof=0):.6f}"])
        w.writerow(["LES_F1_mean", f"{les_f1.mean():.6f}", "LES_F1_SD", f"{les_f1.std(ddof=0):.6f}"])
    print(f"  wrote composite_trajectory_AUC/F1(_area), composite_trajectory_data.csv -> {out}")


if __name__ == "__main__":
    main()
