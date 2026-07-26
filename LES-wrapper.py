#!/usr/bin/env python
"""
LES-wrapper.py — Learning Efficiency Score evaluation across ppiDCE checkpoints.

Runs ppiDCE inference on PRS (Positive Reference Set) and RRS (Random Reference
Set) sequence-pair files at every saved checkpoint in a directory, computes
ROC-AUC and Best-F1 at each checkpoint, then integrates these into a single
Learning Efficiency Score (LES) per metric — the area under the metric-vs-epoch
curve.

This is the ppiDCE port of the LES-wrapper family. Two things about it:

  1. Model-specific glue. ppiDCE saves one checkpoint per epoch as
     `ppiDCE_epoch{N}.pth` plus `ppiDCE_final.pth`, so the x-axis of every LES
     curve is the training *epoch*. Inference uses `inference_ppiDCE.py`, whose
     output columns are `seq1, seq2, Prediction, Probability_Friends,
     Probability_Enemies` (the ppiYYD / ppiBTEP convention) — so the
     positive-class ("friends" = interacting = label 1) probability is the
     column named `Probability_Friends`, the **second-to-last** column. This
     wrapper reads `Probability_Friends` by name (with a positional
     second-to-last-column fallback).

  2. Output shape follows ppiGPLM's LES-wrapper_v2.py (the deliberate port
     target). Relative to the earlier ppiDCE/ppiBTEP wrapper, the outputs here
     therefore:
       - drop the optimal-F1 *threshold* metric everywhere (no trajectory_Threshold
         plot, no LES-Threshold, no Best_F1_Threshold summary column, no manifest
         Threshold entry, and the combined trajectory figure is a 1x2 AUC+F1 grid);
       - always emit a vector **PDF** alongside every PNG (the old `--plot_format`
         switch is gone — both formats are always written);
       - add per-checkpoint probability-distribution plots (PRS vs RRS violins +
         jittered points) plus two summary distribution figures;
       - write a README.md legend for the analysis-level plots;
       - degrade gracefully to a single checkpoint (per-checkpoint analysis only,
         LES/trajectory/distribution summaries skipped).

Basic usage:
    python LES-wrapper.py \\
        --checkpoint_dir ROC_Checkpoints \\
        --prs_file MED4_PRS-RRS/PRS-V3-1.csv \\
        --rrs_file MED4_PRS-RRS/RRS-V3-1.csv \\
        --output_dir LES_results_MED4 \\
        --include_final
"""

import os
import sys
import re
import glob
import argparse
import subprocess
import csv
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score


# -----------------------------------------------------------------------------
# Publication-quality figure defaults
# -----------------------------------------------------------------------------
PUB_DPI = 600

def set_publication_style():
    """Apply consistent, publication-quality matplotlib defaults (600 dpi,
    tight bounding box, larger readable fonts, heavier axis lines)."""
    plt.rcParams.update({
        'savefig.dpi': PUB_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'font.family': 'DejaVu Sans',
        'font.size': 13,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'axes.linewidth': 1.2,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'lines.markersize': 7,
    })


# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='LES-wrapper: Learning Efficiency Score evaluation across ppiDCE checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python LES-wrapper.py --checkpoint_dir ROC_Checkpoints \\
      --prs_file MED4_PRS-RRS/PRS-V3-1.csv --rrs_file MED4_PRS-RRS/RRS-V3-1.csv \\
      --output_dir LES_results --include_final
        """
    )

    # Required arguments
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing ppiDCE checkpoints (ppiDCE_epoch*.pth)')
    parser.add_argument('--prs_file', type=str, required=True,
                        help='Path to Positive Reference Set CSV (seq1,seq2[,label])')
    parser.add_argument('--rrs_file', type=str, required=True,
                        help='Path to Random Reference Set CSV (seq1,seq2[,label])')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='LES_results',
                        help='Directory to save all outputs (default: LES_results)')

    # Checkpoint selection
    parser.add_argument('--checkpoint_pattern', type=str, default='ppiDCE_epoch*.pth',
                        help='Glob to match per-epoch checkpoints (default: ppiDCE_epoch*.pth)')
    parser.add_argument('--include_final', action='store_true',
                        help='Also evaluate ppiDCE_final.pth (plotted after the last epoch; '
                             'excluded from the LES integral)')

    # ppiDCE inference settings (forwarded to inference_ppiDCE.py)
    parser.add_argument('--inference_script', type=str, default=None,
                        help='Path to inference_ppiDCE.py (default: alongside this wrapper)')
    parser.add_argument('--model_config', type=str,
                        default='facebook/esm1b_t33_650M_UR50S',
                        help='ESM model name or local path for tokenizer/config')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Transformer layers the checkpoint was trained with. REQUIRED for '
                             'from-scratch models (e.g. 6); forwarded to inference_ppiDCE.py so '
                             'the config is rebuilt to match, otherwise the load is silently wrong.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Inference batch size (default: 4)')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Max total tokens seq1+seq2+special (default: 1024)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (default: cuda)')

    # Control flow
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference; reuse existing probability CSVs')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating trajectory / distribution plots')
    parser.add_argument('--color_threshold', action='store_true',
                        help='Color the ROC curve by decision threshold and add a '
                             'colorbar (default: plain single-color curve, no scale)')

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def extract_epoch_from_checkpoint(ckpt_name):
    """Extract the epoch number from a ppiDCE checkpoint filename."""
    # Match ppiDCE_epoch1.pth, ppiDCE_epoch12.pth, etc. (underscore optional).
    match = re.search(r'epoch[_-]?(\d+)\.pth$', ckpt_name)
    if match:
        return int(match.group(1))
    # Final checkpoint sorts after every numbered epoch.
    if ckpt_name == 'ppiDCE_final.pth':
        return float('inf')
    return None


def get_checkpoints(checkpoint_dir, pattern, include_final=False):
    """Return a list of (name, epoch, path) sorted by epoch."""
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, pattern))

    if include_final:
        final_ckpt = os.path.join(checkpoint_dir, 'ppiDCE_final.pth')
        if os.path.exists(final_ckpt) and final_ckpt not in ckpt_files:
            ckpt_files.append(final_ckpt)

    checkpoints = []
    for ckpt_path in ckpt_files:
        ckpt_name = os.path.basename(ckpt_path)
        epoch = extract_epoch_from_checkpoint(ckpt_name)
        if epoch is not None:
            checkpoints.append((ckpt_name, epoch, ckpt_path))

    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def run_inference(inference_script, ckpt_path, input_file, output_csv,
                  model_config, batch_size, max_length, device, num_layers=None):
    """Run ppiDCE inference for one checkpoint, writing directly to output_csv."""
    cmd = [
        sys.executable, inference_script,
        '--model_path', ckpt_path,
        '--model_config', model_config,
        '--input_file', input_file,
        '--output_file', output_csv,
        '--batch_size', str(batch_size),
        '--max_length', str(max_length),
        '--device', device,
    ]
    if num_layers is not None:
        cmd += ['--num_layers', str(num_layers)]

    print(f"  Running: inference_ppiDCE.py --model_path {os.path.basename(ckpt_path)} "
          f"--input_file {os.path.basename(input_file)} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: Inference failed for {os.path.basename(ckpt_path)}")
        print(f"  stderr: {result.stderr[-800:]}")
        return False

    return True


def extract_probabilities_from_csv(csv_path):
    """Extract the positive-class probability from a ppiDCE inference CSV.

    ppiDCE writes columns: seq1, seq2, Prediction, Probability_Friends,
    Probability_Enemies (the ppiYYD / ppiBTEP convention). The positive-class
    ("friends" = interacting = label 1) probability is the column named
    `Probability_Friends` — the second-to-last column. Sequences contain no
    commas, so a positional fallback to the second-to-last column is safe if
    the header is ever missing.
    """
    probabilities = []
    if not os.path.exists(csv_path):
        print(f"  WARNING: File not found: {csv_path}")
        return probabilities

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        prob_idx = None
        if header is not None:
            for i, col in enumerate(header):
                if col.strip().lower() == 'probability_friends':
                    prob_idx = i
                    break
        # Fallback: positive-class prob is the second-to-last column.
        if prob_idx is None:
            prob_idx = -2

        for row in reader:
            if not row:
                continue
            try:
                probabilities.append(float(row[prob_idx]))
            except (ValueError, IndexError):
                continue
    return probabilities


def combine_probabilities(prs_probs, rrs_probs, output_path):
    """Write PRS/RRS probabilities side by side for ROC analysis."""
    max_len = max(len(prs_probs), len(rrs_probs))
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # No header — the ROC reader expects two raw value columns (PRS, RRS).
        for i in range(max_len):
            prs_val = prs_probs[i] if i < len(prs_probs) else ''
            rrs_val = rrs_probs[i] if i < len(rrs_probs) else ''
            writer.writerow([prs_val, rrs_val])
    return output_path


def run_roc_analysis_internal(combined_csv_path, output_plot_path, color_threshold=False):
    """Compute AUC / Best-F1 and render the ROC plot. Returns (roc_auc, best_f1).

    By default the curve is a single color with no colorbar. Pass
    color_threshold=True to color the curve by decision threshold and add a
    threshold colorbar. Following LES-wrapper_v2.py, the optimal-F1 threshold is
    NOT reported (it was a degenerate diagnostic for non-discriminating controls).
    """
    prs_probs, rrs_probs = [], []

    with open(combined_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                prs_val = row[0].strip()
                rrs_val = row[1].strip()
                if prs_val:
                    prs_probs.append(float(prs_val))
                if rrs_val:
                    rrs_probs.append(float(rrs_val))

    if not prs_probs or not rrs_probs:
        return None, None

    # Assign labels (PRS = 1, RRS = 0)
    prs_labels = [1] * len(prs_probs)
    rrs_labels = [0] * len(rrs_probs)

    probs = np.array(prs_probs + rrs_probs)
    labels = np.array(prs_labels + rrs_labels)

    # ROC + AUC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Keep finite, in-range thresholds (only used for the optional colored ROC plot).
    finite_idxs = np.where(np.isfinite(thresholds))[0]
    fpr = fpr[finite_idxs]
    tpr = tpr[finite_idxs]
    thresholds = thresholds[finite_idxs]

    valid_thresholds_idxs = np.where((thresholds >= 0) & (thresholds <= 1))[0]
    fpr = fpr[valid_thresholds_idxs]
    tpr = tpr[valid_thresholds_idxs]
    thresholds = thresholds[valid_thresholds_idxs]

    # Best F1 over ALL candidate thresholds — every unique score, not just the
    # ROC vertices. roc_curve(drop_intermediate=True) prunes collinear vertices;
    # scanning the full unique-score set guarantees the true F1-optimal cutoff is
    # considered. Decision rule: prob >= threshold => positive.
    best_f1 = -1.0
    for thresh in np.unique(probs):
        current_f1 = f1_score(labels, (probs >= thresh).astype(int), zero_division=0)
        if current_f1 >= best_f1:
            best_f1 = current_f1

    # ROC plot. Default: clean single-color curve. With color_threshold: the
    # curve is colored by decision threshold and a colorbar is added.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if color_threshold:
        norm = plt.Normalize(vmin=thresholds.min(), vmax=thresholds.max())
        cmap = plt.cm.viridis
        for i in range(len(fpr) - 1):
            ax.plot(fpr[i:i + 2], tpr[i:i + 2], color=cmap(norm(thresholds[i])), lw=2.5)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Threshold', fontsize=15)
    else:
        ax.plot(fpr, tpr, color='#08519c', lw=2.5)

    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')

    legend_text = f'AUC = {roc_auc:.3f}, Best F1 = {best_f1:.3f}'
    ax.legend([legend_text], loc="lower right")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=PUB_DPI, format='png')
    # Also emit a vector PDF (scalable, no pixelation) for publication.
    plt.savefig(os.path.splitext(output_plot_path)[0] + '.pdf', format='pdf')
    plt.close(fig)

    return roc_auc, best_f1


def compute_les(epochs, values):
    """Learning Efficiency Score = area under the metric-vs-epoch curve.

    Epochs are normalized to [0, 1] before integration so LES is comparable
    across runs of different length. The final checkpoint (epoch == inf) is
    excluded from the integral so it does not distort the area.
    """
    if len(epochs) < 2 or len(values) < 2:
        return 0.0

    eps = np.array(epochs, dtype=float)
    vals = np.array(values, dtype=float)

    valid_mask = np.isfinite(eps)
    eps = eps[valid_mask]
    vals = vals[valid_mask]

    if len(eps) < 2:
        return 0.0

    eps_normalized = (eps - eps.min()) / (eps.max() - eps.min())
    return np.trapezoid(vals, eps_normalized)


def plot_metric_trajectory(epochs, values, metric_name, output_path, les_value,
                           les_label=None):
    """Plot a single metric trajectory across epochs (y-axis fixed to 0..1)."""
    plt.figure(figsize=(10, 6))

    valid_mask = [e < float('inf') for e in epochs]
    plot_eps = [e for e, v in zip(epochs, valid_mask) if v]
    plot_vals = [val for val, v in zip(values, valid_mask) if v]

    plt.plot(plot_eps, plot_vals, 'bo-', linewidth=2, markersize=8)
    plt.fill_between(plot_eps, plot_vals, alpha=0.3)

    plt.xlabel('Training Epoch', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    label = les_label if les_label is not None else f'LES-{metric_name}'
    plt.title(f'{metric_name} vs Training Epoch\n{label} = {les_value:.4f}', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, (e, val) in enumerate(zip(plot_eps, plot_vals)):
        if i % max(1, len(plot_eps) // 10) == 0:
            plt.annotate(f'{val:.3f}', (e, val), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PUB_DPI)
    plt.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
    plt.close()


def plot_combined_trajectories(epochs, auc_values, f1_values, output_path, les_values):
    """Plot AUC and Best-F1 trajectories on a single figure (1x2)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    valid_mask = [e < float('inf') for e in epochs]
    plot_eps = [e for e, v in zip(epochs, valid_mask) if v]
    plot_auc = [val for val, v in zip(auc_values, valid_mask) if v]
    plot_f1 = [val for val, v in zip(f1_values, valid_mask) if v]

    axes[0].plot(plot_eps, plot_auc, 'bo-', linewidth=2, markersize=6)
    axes[0].fill_between(plot_eps, plot_auc, alpha=0.3)
    axes[0].set_xlabel('Training Epoch')
    axes[0].set_ylabel('AUC')
    axes[0].set_title(f'AUC Trajectory\nLES-AUC = {les_values["AUC"]:.4f}')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_ylim([0, 1])

    axes[1].plot(plot_eps, plot_f1, 'go-', linewidth=2, markersize=6)
    axes[1].fill_between(plot_eps, plot_f1, alpha=0.3, color='green')
    axes[1].set_xlabel('Training Epoch')
    axes[1].set_ylabel('Best F1')
    axes[1].set_title(f'Best F1 Trajectory\nArea under the curve = {les_values["F1"]:.4f}')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=PUB_DPI)
    plt.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
    plt.close()


def plot_probability_distribution(prs_probs, rrs_probs, output_path, epoch_str, ax=None):
    """Probability-distribution plot for one checkpoint: P(interaction) for PRS
    (positives) vs RRS (negatives). y-axis = probability, fixed to [0, 1].
    Violin (distribution shape) + jittered points. Draws into `ax` if given
    (used by the summary grid); otherwise makes and saves its own figure.
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 6))

    data = [prs_probs, rrs_probs]
    colors = ['#2166ac', '#b2182b']            # PRS blue, RRS red
    parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.35); pc.set_edgecolor(colors[i])
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        if len(d):
            x = (i + 1) + (rng.random(len(d)) - 0.5) * 0.16
            ax.scatter(x, d, s=8, color=colors[i], alpha=0.5, edgecolors='none')
    ax.set_xticks([1, 2]); ax.set_xticklabels(['PRS', 'RRS'])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('P(interaction)')
    ax.set_title(f'epoch {epoch_str}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    if own:
        plt.tight_layout()
        plt.savefig(output_path, dpi=PUB_DPI)
        plt.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
        plt.close(fig)


def plot_summary_distributions(dist_data, output_path):
    """One summary figure: the per-checkpoint probability distributions (PRS vs RRS),
    one panel per checkpoint, every panel with y-axis fixed to [0, 1].
    dist_data: list of (epoch_str, prs_probs, rrs_probs) in checkpoint order.
    """
    n = len(dist_data)
    if n == 0:
        return
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)
    for idx, (epoch_str, prs, rrs) in enumerate(dist_data):
        r, c = divmod(idx, ncols)
        plot_probability_distribution(prs, rrs, None, epoch_str, ax=axes[r][c])
    for idx in range(n, nrows * ncols):          # hide unused panels
        r, c = divmod(idx, ncols)
        axes[r][c].axis('off')
    fig.suptitle('Probability distributions across checkpoints (PRS vs RRS)',
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=PUB_DPI)
    fig.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
    plt.close(fig)


def plot_summary_distributions_combined(dist_data, output_path):
    """All per-checkpoint probability distributions on ONE axes (publication quality):
    the x-axis is split into two sections — the PRS violins (left, one per checkpoint,
    labelled by its training epoch) then the RRS violins (right, same). y-axis is
    P(interaction), fixed to [0, 1]. Makes PRS-stays-high / RRS-stays-low visible at a
    glance across training.
    """
    n = len(dist_data)
    if n == 0:
        return
    epochs = [d[0] for d in dist_data]
    prs_list = [d[1] for d in dist_data]
    rrs_list = [d[2] for d in dist_data]

    prs_pos = list(range(1, n + 1))
    rrs_pos = list(range(n + 2, 2 * n + 2))     # gap of 1 at n+1 for the divider
    divider = n + 1

    fig, ax = plt.subplots(figsize=(max(11.0, 0.62 * (2 * n + 2)), 7.0))

    def _violins(data, positions, color):
        parts = ax.violinplot(data, positions=positions, showmedians=True,
                              showextrema=False, widths=0.82)
        for pc in parts['bodies']:
            pc.set_facecolor(color); pc.set_alpha(0.40); pc.set_edgecolor(color)
            pc.set_linewidth(0.8)
        if 'cmedians' in parts:
            parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.1)

    def _points(data, positions, color):
        rng = np.random.default_rng(0)
        for d, pos in zip(data, positions):
            if len(d):
                x = pos + (rng.random(len(d)) - 0.5) * 0.16
                ax.scatter(x, d, s=6, color=color, alpha=0.5, edgecolors='none',
                           zorder=3)

    _violins(prs_list, prs_pos, '#2166ac')       # PRS blue (positives)
    _violins(rrs_list, rrs_pos, '#b2182b')       # RRS red (negatives)
    _points(prs_list, prs_pos, '#2166ac')
    _points(rrs_list, rrs_pos, '#b2182b')

    ax.axvline(divider, color='gray', linestyle='--', linewidth=1.3)

    ax.set_xticks(prs_pos + rrs_pos)
    ax.set_xticklabels([str(e) for e in epochs] * 2, rotation=90, fontsize=10)
    ax.set_xlim(0.3, 2 * n + 2 - 0.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel('P(interaction)')
    ax.set_xlabel('Training epoch')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    tr = ax.get_xaxis_transform()
    ax.text((1 + n) / 2.0, 1.03, 'PRS (positives)', ha='center', va='bottom',
            fontsize=15, color='#2166ac', fontweight='bold', transform=tr)
    ax.text((n + 2 + 2 * n + 1) / 2.0, 1.03, 'RRS (negatives)', ha='center',
            va='bottom', fontsize=15, color='#b2182b', fontweight='bold', transform=tr)
    ax.set_title('Probability distributions across checkpoints — PRS vs RRS', pad=34)

    fig.tight_layout()
    fig.savefig(output_path, dpi=PUB_DPI)
    fig.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
    plt.close(fig)


def write_analysis_readme(output_dir):
    """Write a README.md legend for the analysis-level plots (not per-checkpoint)."""
    text = """# LES analysis — plot legend (ppiDCE)

This folder is the Learning Efficiency Score (LES) analysis for one ppiDCE model
evaluated on one PRS (Positive Reference Set) / RRS (Random Reference Set) pair,
across all saved training checkpoints (one per epoch). **PRS = blue (positives);
RRS = red (negatives).** The interaction score for each pair is
`Probability_Friends` in [0, 1] — see the next section for exactly what that is.

## What "probability" / `P(interaction)` means here (read this)
The value on every y-axis — labelled `P(interaction)` — is **not** an empirical
frequency or a calibrated statistical probability that two proteins interact. It is
the ppiDCE classifier's **softmax probability for class `1` (interacting)**.

ppiDCE encodes the protein pair and the classifier head emits 2 logits over the
classes `{0, 1}`. A softmax turns those into a probability for each class and we
read off the mass on class `1`:

```
P(interaction) = softmax(classifier logits)[ "1" ] = Probability_Friends
```

- It is a genuine 2-class softmax, so **`Probability_Friends + Probability_Enemies
  = 1`** exactly. Both are saved in each per-checkpoint `*_probabilities.csv`.
- **It is `Probability_Friends` for *both* PRS and RRS** — a single, shared score
  function, *not* "`Probability_Friends` for PRS and `Probability_Enemies` for
  RRS". That shared score is what makes the ROC/AUC valid: every pair gets the
  same score `Probability_Friends`, PRS pairs are labelled 1 and RRS pairs 0, and
  AUC measures how well `Probability_Friends` ranks true interactors above random
  ones. A good model pushes `Probability_Friends` high for PRS (blue, near 1) and
  low for RRS (red, near 0). `Probability_Enemies` is recorded but is **not** used
  for AUC, F1, LES, or any plot.

## Summary figures (this folder)
- **`trajectory_AUC.png`** — ROC-AUC vs training epoch (y-axis 0-1). Subtitle
  `LES-AUC` is the area under this AUC-vs-epoch curve (the learning-efficiency
  score for AUC).
- **`trajectory_F1.png`** — Best-F1 vs training epoch (y-axis 0-1). Subtitle
  `Area under the curve` is the integral of the F1-vs-epoch curve.
- **`trajectory_combined.png`** — the AUC and Best-F1 trajectories side by side.
- **`summary_prob_distributions.png`** — a grid, one panel per checkpoint; each panel
  shows the P(interaction) distribution for PRS vs RRS (violin + jittered points),
  y-axis 0-1.
- **`summary_prob_distributions_combined.png`** — the same distributions on a single
  axes with the x-axis split into two sections: all PRS violins (left, one per
  checkpoint, labelled by epoch) then all RRS violins (right). Lets you see PRS
  staying high and RRS staying low across training in one view.
- **`summary_table.csv`** — per-checkpoint AUC and Best-F1, plus a final LES row.
- **`manifest.json`** — run metadata and LES values.

Every PNG has a companion vector **`.pdf`** at the same path.

## How to read the violin plots
Each violin summarises the P(interaction) scores for one reference set (PRS or RRS)
at one checkpoint. The anatomy:

- **Width (the shaded shape):** a kernel-density estimate (KDE) of the score
  distribution — wider where more pairs fall, narrower where few do. It shows the
  *shape* of the distribution, not an absolute count.
- **Horizontal black line:** the **median** of the scores (not the mean). A PRS
  median near 1 and an RRS median near 0 is the signature of a well-separated,
  discriminating model.
- **Dots:** the individual pairs — all raw scores, jittered horizontally only (the
  jitter is cosmetic; vertical position is the true probability).
- **Colour:** blue = PRS (positives, should sit high), red = RRS (negatives, should
  sit low).

## Per-checkpoint folders (`epoch_<N>/`)
Each holds that checkpoint's ROC curve (`ROC_epoch<N>.png`), its probability
distribution (`prob_dist_epoch<N>.png`), and the raw PRS/RRS/combined probability
CSVs. (These folders intentionally have no README.)

## Reading it
A discriminating model shows PRS probabilities clustered near 1 and RRS near 0
(clear separation in the distribution plots), an AUC that rises and plateaus, and —
for a non-discriminating control — AUC ~ 0.5 with PRS/RRS distributions overlapping
(the expected null).
"""
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(text)


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    set_publication_style()

    # Validate inputs
    if not os.path.exists(args.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)
    if not os.path.exists(args.prs_file):
        print(f"ERROR: PRS file not found: {args.prs_file}")
        sys.exit(1)
    if not os.path.exists(args.rrs_file):
        print(f"ERROR: RRS file not found: {args.rrs_file}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Locate the ppiDCE inference script (defaults to alongside this wrapper).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_script = args.inference_script or os.path.join(script_dir, 'inference_ppiDCE.py')
    if not os.path.exists(inference_script):
        print(f"ERROR: Inference script not found: {inference_script}")
        sys.exit(1)

    checkpoints = get_checkpoints(args.checkpoint_dir, args.checkpoint_pattern, args.include_final)
    if not checkpoints:
        print(f"ERROR: No checkpoints found matching '{args.checkpoint_pattern}' in {args.checkpoint_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("LES-wrapper: Learning Efficiency Score Evaluation (ppiDCE)")
    print(f"{'='*60}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"PRS file: {args.prs_file}")
    print(f"RRS file: {args.rrs_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"{'='*60}\n")

    # Results storage
    results = []
    epochs = []
    auc_values = []
    f1_values = []
    dist_data = []   # (epoch_str, prs_probs, rrs_probs) per checkpoint, for the summary

    for idx, (ckpt_name, epoch, ckpt_path) in enumerate(checkpoints):
        epoch_str = str(epoch) if epoch < float('inf') else 'final'
        print(f"\n[{idx+1}/{len(checkpoints)}] Processing checkpoint: {ckpt_name} (epoch {epoch_str})")

        ckpt_subdir = os.path.join(args.output_dir, f"epoch_{epoch_str}")
        os.makedirs(ckpt_subdir, exist_ok=True)

        prs_csv = os.path.join(ckpt_subdir, f"PRS_epoch{epoch_str}_probabilities.csv")
        rrs_csv = os.path.join(ckpt_subdir, f"RRS_epoch{epoch_str}_probabilities.csv")

        if not args.skip_inference:
            print(f"  Running PRS inference...")
            if not run_inference(inference_script, ckpt_path, args.prs_file, prs_csv,
                                 args.model_config, args.batch_size, args.max_length, args.device,
                                 num_layers=args.num_layers):
                print(f"  SKIPPING checkpoint due to inference error")
                continue

            print(f"  Running RRS inference...")
            if not run_inference(inference_script, ckpt_path, args.rrs_file, rrs_csv,
                                 args.model_config, args.batch_size, args.max_length, args.device,
                                 num_layers=args.num_layers):
                print(f"  SKIPPING checkpoint due to inference error")
                continue

        print(f"  Extracting probabilities...")
        prs_probs = extract_probabilities_from_csv(prs_csv)
        rrs_probs = extract_probabilities_from_csv(rrs_csv)

        if not prs_probs or not rrs_probs:
            print(f"  WARNING: Could not extract probabilities, skipping")
            continue

        print(f"  PRS samples: {len(prs_probs)}, RRS samples: {len(rrs_probs)}")

        combined_csv = os.path.join(ckpt_subdir, f"combined_probabilities_epoch{epoch_str}.csv")
        combine_probabilities(prs_probs, rrs_probs, combined_csv)

        # Per-checkpoint probability-distribution plot (PRS vs RRS), y in [0, 1].
        plot_probability_distribution(
            prs_probs, rrs_probs,
            os.path.join(ckpt_subdir, f"prob_dist_epoch{epoch_str}.png"), epoch_str)
        dist_data.append((epoch_str, list(prs_probs), list(rrs_probs)))

        print(f"  Running ROC analysis...")
        roc_plot = os.path.join(ckpt_subdir, f"ROC_epoch{epoch_str}.png")
        roc_auc, best_f1 = run_roc_analysis_internal(
            combined_csv, roc_plot, color_threshold=args.color_threshold)

        if roc_auc is None:
            print(f"  WARNING: ROC analysis failed, skipping")
            continue

        print(f"  Results: AUC={roc_auc:.4f}, F1={best_f1:.4f}")

        results.append({
            'checkpoint': ckpt_name,
            'epoch': epoch if epoch < float('inf') else 'final',
            'AUC': roc_auc,
            'Best_F1': best_f1,
            'PRS_samples': len(prs_probs),
            'RRS_samples': len(rrs_probs)
        })

        epochs.append(epoch)
        auc_values.append(roc_auc)
        f1_values.append(best_f1)

    # LES only meaningful with >= 2 checkpoints. With a single checkpoint, do the
    # per-checkpoint analysis only and skip all summaries.
    multi = len(epochs) >= 2
    if multi:
        print(f"\n{'='*60}")
        print("Computing Learning Efficiency Scores (LES)")
        print(f"{'='*60}")
        les_auc = compute_les(epochs, auc_values)
        les_f1 = compute_les(epochs, f1_values)
        les_values = {'AUC': les_auc, 'F1': les_f1}
        print(f"  LES-AUC: {les_auc:.6f}")
        print(f"  LES-F1: {les_f1:.6f}")
    else:
        les_auc = les_f1 = None
        les_values = {}
        print(f"\nOnly {len(epochs)} checkpoint(s) analyzed — skipping LES, trajectory, "
              f"and distribution summaries (these need >= 2 checkpoints).")

    # Trajectory / distribution plots
    if not args.no_plots and multi:
        print(f"\nGenerating trajectory plots...")
        plot_metric_trajectory(epochs, auc_values, 'AUC',
                               os.path.join(args.output_dir, 'trajectory_AUC.png'), les_auc,
                               les_label='LES-AUC')
        plot_metric_trajectory(epochs, f1_values, 'Best F1',
                               os.path.join(args.output_dir, 'trajectory_F1.png'), les_f1,
                               les_label='Area under the curve')
        plot_combined_trajectories(epochs, auc_values, f1_values,
                                   os.path.join(args.output_dir, 'trajectory_combined.png'), les_values)
        plot_summary_distributions(dist_data,
                                   os.path.join(args.output_dir, 'summary_prob_distributions.png'))
        plot_summary_distributions_combined(
            dist_data,
            os.path.join(args.output_dir, 'summary_prob_distributions_combined.png'))
        write_analysis_readme(args.output_dir)
        print(f"  Saved trajectory / distribution plots to {args.output_dir}")

    # Summary table
    print(f"\nGenerating summary table...")
    summary_csv = os.path.join(args.output_dir, 'summary_table.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'epoch', 'AUC', 'Best_F1',
                                               'PRS_samples', 'RRS_samples'])
        writer.writeheader()
        writer.writerows(results)
    if multi:
        with open(summary_csv, 'a', newline='') as f:
            f.write(f"\nLES (Learning Efficiency Score),---,{les_auc:.6f},{les_f1:.6f},---,---\n")
    print(f"  Saved summary table to {summary_csv}")

    # JSON manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint_dir': args.checkpoint_dir,
        'prs_file': args.prs_file,
        'rrs_file': args.rrs_file,
        'output_dir': args.output_dir,
        'model_config': args.model_config,
        'num_checkpoints': len(checkpoints),
        'num_successful': len(results),
        'LES': ({'AUC': les_auc, 'F1': les_f1} if multi else None),
        'results': results
    }
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved manifest to {manifest_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Checkpoints processed: {len(results)}/{len(checkpoints)}")
    if multi:
        print(f"\nLearning Efficiency Scores (LES):")
        print(f"  LES-AUC:       {les_auc:.6f}")
        print(f"  LES-F1:        {les_f1:.6f}")
    else:
        print("\n(Single checkpoint: LES and trajectory/distribution summaries skipped.)")

    if results:
        final_result = results[-1]
        print(f"\nFinal Checkpoint Performance:")
        print(f"  AUC:       {final_result['AUC']:.4f}")
        print(f"  Best F1:   {final_result['Best_F1']:.4f}")

    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
