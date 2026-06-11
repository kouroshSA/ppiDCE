#!/usr/bin/env python
"""
LES-wrapper.py - Learning Efficiency Score evaluation across ppiDCE checkpoints.

Runs ppiDCE inference on PRS (Positive Reference Set) and RRS (Random Reference
Set) sequence-pair files at every saved checkpoint in a directory, computes
ROC-AUC, the optimal-F1 threshold, and Best-F1 at each checkpoint, then
integrates these into a single Learning Efficiency Score (LES) per metric -
the area under the metric-vs-epoch curve.

This is the ppiDCE port of the ppiGPLM LES-wrapper. The two differences that
matter here:

  1. Checkpoints. ppiDCE saves one checkpoint per epoch as
     `ppiDCE_epoch{N}.pth` (plus `ppiDCE_final.pth`), not nanoGPT-style
     `ckpt_{iter}.pt`. The x-axis of every LES curve is therefore the training
     *epoch*, not the iteration.

  2. Inference output format. ppiDCE's `inference_ppiDCE.py` writes a CSV with
     columns `seq1, seq2, [label,] pred_label, prob_0, prob_1` - so the
     positive-class probability is the column literally named `prob_1` (the
     last column), whereas ppiGPLM put `Probability_of_1` second-from-last.
     This wrapper reads `prob_1` by name (with a positional last-column
     fallback).

Basic usage:
    python LES-wrapper.py \\
        --checkpoint_dir ROC_Checkpoints \\
        --prs_file MED4_PRS_100.csv \\
        --rrs_file MED4_RRS_100.csv \\
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
PUB_DPI = 300

def set_publication_style():
    """Apply consistent, publication-quality matplotlib defaults (300 dpi,
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
      --prs_file MED4_PRS_100.csv --rrs_file MED4_RRS_100.csv \\
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
                        help='Also evaluate ppiDCE_final.pth (plotted after the last epoch)')

    # ppiDCE inference settings (forwarded to inference_ppiDCE.py)
    parser.add_argument('--inference_script', type=str, default=None,
                        help='Path to inference_ppiDCE.py (default: alongside this wrapper)')
    parser.add_argument('--model_config', type=str,
                        default='facebook/esm1b_t33_650M_UR50S',
                        help='ESM model name or local path for tokenizer/config')
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
                        help='Skip generating trajectory plots')
    parser.add_argument('--color_threshold', action='store_true',
                        help='Color the ROC curve by decision threshold and add a '
                             'colorbar (default: plain single-color curve, no scale)')

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def extract_epoch_from_checkpoint(ckpt_name):
    """Extract the epoch number from a ppiDCE checkpoint filename."""
    # Match ppiDCE_epoch1.pth, ppiDCE_epoch12.pth, etc.
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
                  model_config, batch_size, max_length, device):
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

    ppiDCE writes columns: seq1, seq2, [label,] pred_label, prob_0, prob_1.
    The positive-class probability is the column named `prob_1` (also the last
    column). Sequences contain no commas, so a positional fallback to the last
    column is safe if the header is ever missing.
    """
    probabilities = []
    if not os.path.exists(csv_path):
        print(f"  WARNING: File not found: {csv_path}")
        return probabilities

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        prob1_idx = None
        if header is not None:
            for i, col in enumerate(header):
                if col.strip().lower() == 'prob_1':
                    prob1_idx = i
                    break
        # Fallback: positive-class prob is the last column.
        if prob1_idx is None:
            prob1_idx = -1

        for row in reader:
            if not row:
                continue
            try:
                probabilities.append(float(row[prob1_idx]))
            except (ValueError, IndexError):
                continue
    return probabilities


def combine_probabilities(prs_probs, rrs_probs, output_path):
    """Write PRS/RRS probabilities side by side for ROC analysis."""
    max_len = max(len(prs_probs), len(rrs_probs))
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # No header - the ROC reader expects two raw value columns (PRS, RRS).
        for i in range(max_len):
            prs_val = prs_probs[i] if i < len(prs_probs) else ''
            rrs_val = rrs_probs[i] if i < len(rrs_probs) else ''
            writer.writerow([prs_val, rrs_val])
    return output_path


def run_roc_analysis_internal(combined_csv_path, output_plot_path, color_threshold=False):
    """Compute AUC / Best-F1 / optimal threshold and render the ROC plot.

    By default the curve is a single color with no colorbar. Pass
    color_threshold=True to color the curve by decision threshold and add a
    threshold colorbar. The optimal threshold is always computed and returned
    (it appears in the summary table) but is not annotated on the plot.
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
        return None, None, None

    # Assign labels (PRS = 1, RRS = 0)
    prs_labels = [1] * len(prs_probs)
    rrs_labels = [0] * len(rrs_probs)

    probs = np.array(prs_probs + rrs_probs)
    labels = np.array(prs_labels + rrs_labels)

    # ROC + AUC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Keep finite, in-range thresholds for plotting/F1 search.
    finite_idxs = np.where(np.isfinite(thresholds))[0]
    fpr = fpr[finite_idxs]
    tpr = tpr[finite_idxs]
    thresholds = thresholds[finite_idxs]

    valid_thresholds_idxs = np.where((thresholds >= 0) & (thresholds <= 1))[0]
    fpr = fpr[valid_thresholds_idxs]
    tpr = tpr[valid_thresholds_idxs]
    thresholds = thresholds[valid_thresholds_idxs]

    # Best F1 over candidate thresholds.
    best_f1 = -1.0
    best_thresh = None
    for thresh in thresholds:
        predicted_labels = (probs >= thresh).astype(int)
        current_f1 = f1_score(labels, predicted_labels)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

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

    # Per request: show only AUC and Best F1 on the curve (no threshold value).
    legend_text = f'AUC = {roc_auc:.3f}, Best F1 = {best_f1:.3f}'
    ax.legend([legend_text], loc="lower right")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=PUB_DPI, format='png')
    plt.close(fig)

    return roc_auc, best_f1, best_thresh


def compute_les(epochs, values):
    """Learning Efficiency Score = area under the metric-vs-epoch curve."""
    if len(epochs) < 2 or len(values) < 2:
        return 0.0

    eps = np.array(epochs, dtype=float)
    vals = np.array(values, dtype=float)

    # Drop the final checkpoint (epoch == inf) from the integral.
    valid_mask = np.isfinite(eps)
    eps = eps[valid_mask]
    vals = vals[valid_mask]

    if len(eps) < 2:
        return 0.0

    # Normalize epochs to [0, 1] so LES is comparable across training lengths.
    eps_normalized = (eps - eps.min()) / (eps.max() - eps.min())

    return np.trapezoid(vals, eps_normalized)


def plot_metric_trajectory(epochs, values, metric_name, output_path, les_value):
    """Plot a single metric trajectory across epochs."""
    plt.figure(figsize=(10, 6))

    valid_mask = [e < float('inf') for e in epochs]
    plot_eps = [e for e, v in zip(epochs, valid_mask) if v]
    plot_vals = [val for val, v in zip(values, valid_mask) if v]

    plt.plot(plot_eps, plot_vals, 'bo-', linewidth=2, markersize=8)
    plt.fill_between(plot_eps, plot_vals, alpha=0.3)

    plt.xlabel('Training Epoch', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(f'{metric_name} vs Training Epoch\nLES-{metric_name} = {les_value:.4f}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, (e, val) in enumerate(zip(plot_eps, plot_vals)):
        if i % max(1, len(plot_eps) // 10) == 0:
            plt.annotate(f'{val:.3f}', (e, val), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PUB_DPI)
    plt.close()


def plot_combined_trajectories(epochs, auc_vals, f1_vals, thresh_vals, output_path, les_values):
    """Plot AUC / Best-F1 / threshold trajectories on a single figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    valid_mask = [e < float('inf') for e in epochs]
    plot_eps = [e for e, v in zip(epochs, valid_mask) if v]
    plot_auc = [val for val, v in zip(auc_vals, valid_mask) if v]
    plot_f1 = [val for val, v in zip(f1_vals, valid_mask) if v]
    plot_thresh = [val for val, v in zip(thresh_vals, valid_mask) if v]

    axes[0].plot(plot_eps, plot_auc, 'bo-', linewidth=2, markersize=6)
    axes[0].fill_between(plot_eps, plot_auc, alpha=0.3)
    axes[0].set_xlabel('Training Epoch')
    axes[0].set_ylabel('AUC')
    axes[0].set_title(f'AUC Trajectory\nLES-AUC = {les_values["AUC"]:.4f}')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(plot_eps, plot_f1, 'go-', linewidth=2, markersize=6)
    axes[1].fill_between(plot_eps, plot_f1, alpha=0.3, color='green')
    axes[1].set_xlabel('Training Epoch')
    axes[1].set_ylabel('Best F1')
    axes[1].set_title(f'Best F1 Trajectory\nLES-F1 = {les_values["F1"]:.4f}')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim([0, 1.05])

    axes[2].plot(plot_eps, plot_thresh, 'ro-', linewidth=2, markersize=6)
    axes[2].fill_between(plot_eps, plot_thresh, alpha=0.3, color='red')
    axes[2].set_xlabel('Training Epoch')
    axes[2].set_ylabel('Best F1 Threshold')
    axes[2].set_title(f'Threshold Trajectory\nLES-Threshold = {les_values["Threshold"]:.4f}')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=PUB_DPI)
    plt.close()


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
    thresh_values = []

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
                                 args.model_config, args.batch_size, args.max_length, args.device):
                print(f"  SKIPPING checkpoint due to inference error")
                continue

            print(f"  Running RRS inference...")
            if not run_inference(inference_script, ckpt_path, args.rrs_file, rrs_csv,
                                 args.model_config, args.batch_size, args.max_length, args.device):
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

        print(f"  Running ROC analysis...")
        roc_plot = os.path.join(ckpt_subdir, f"ROC_epoch{epoch_str}.png")
        roc_auc, best_f1, best_thresh = run_roc_analysis_internal(
            combined_csv, roc_plot, color_threshold=args.color_threshold)

        if roc_auc is None:
            print(f"  WARNING: ROC analysis failed, skipping")
            continue

        print(f"  Results: AUC={roc_auc:.4f}, F1={best_f1:.4f}, Threshold={best_thresh:.4f}")

        results.append({
            'checkpoint': ckpt_name,
            'epoch': epoch if epoch < float('inf') else 'final',
            'AUC': roc_auc,
            'Best_F1': best_f1,
            'Best_F1_Threshold': best_thresh,
            'PRS_samples': len(prs_probs),
            'RRS_samples': len(rrs_probs)
        })

        epochs.append(epoch)
        auc_values.append(roc_auc)
        f1_values.append(best_f1)
        thresh_values.append(best_thresh)

    # Compute LES values
    print(f"\n{'='*60}")
    print("Computing Learning Efficiency Scores (LES)")
    print(f"{'='*60}")

    les_auc = compute_les(epochs, auc_values)
    les_f1 = compute_les(epochs, f1_values)
    les_thresh = compute_les(epochs, thresh_values)

    les_values = {'AUC': les_auc, 'F1': les_f1, 'Threshold': les_thresh}

    print(f"  LES-AUC: {les_auc:.6f}")
    print(f"  LES-F1: {les_f1:.6f}")
    print(f"  LES-Threshold: {les_thresh:.6f}")

    # Trajectory plots
    if not args.no_plots and len(epochs) >= 2:
        print(f"\nGenerating trajectory plots...")
        plot_metric_trajectory(epochs, auc_values, 'AUC',
                               os.path.join(args.output_dir, 'trajectory_AUC.png'), les_auc)
        plot_metric_trajectory(epochs, f1_values, 'Best F1',
                               os.path.join(args.output_dir, 'trajectory_F1.png'), les_f1)
        plot_metric_trajectory(epochs, thresh_values, 'Best F1 Threshold',
                               os.path.join(args.output_dir, 'trajectory_Threshold.png'), les_thresh)
        plot_combined_trajectories(epochs, auc_values, f1_values, thresh_values,
                                   os.path.join(args.output_dir, 'trajectory_combined.png'), les_values)
        print(f"  Saved trajectory plots to {args.output_dir}")

    # Summary table
    print(f"\nGenerating summary table...")
    summary_csv = os.path.join(args.output_dir, 'summary_table.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'epoch', 'AUC', 'Best_F1',
                                               'Best_F1_Threshold', 'PRS_samples', 'RRS_samples'])
        writer.writeheader()
        writer.writerows(results)
    with open(summary_csv, 'a', newline='') as f:
        f.write(f"\nLES (Learning Efficiency Score),---,{les_auc:.6f},{les_f1:.6f},{les_thresh:.6f},---,---\n")
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
        'LES': {'AUC': les_auc, 'F1': les_f1, 'Threshold': les_thresh},
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
    print(f"\nLearning Efficiency Scores (LES):")
    print(f"  LES-AUC:       {les_auc:.6f}")
    print(f"  LES-F1:        {les_f1:.6f}")
    print(f"  LES-Threshold: {les_thresh:.6f}")

    if results:
        final_result = results[-1]
        print(f"\nFinal Checkpoint Performance:")
        print(f"  AUC:       {final_result['AUC']:.4f}")
        print(f"  Best F1:   {final_result['Best_F1']:.4f}")
        print(f"  Threshold: {final_result['Best_F1_Threshold']:.4f}")

    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
