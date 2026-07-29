#!/usr/bin/env python3
"""
make_composite_les.py — Composite (N-replicate ensemble) LES figures for
ppiDCE, ported from ppiYYD's `make_composite_les.py`.

Aggregates the per-replicate LES-wrapper.py outputs of the 10 V3 replicates
into three publication-quality composite views, mirroring the ppiGPLM/ppiYYD
10-model ensembles but on ppiDCE's checkpoint naming (`ppiDCE_epoch{N}.pth`,
no underscore) and directory layout — one top-level results dir per
replicate (`results/dce_V3-k_scratch12L_ml1024/`), each containing **two**
LES-wrapper runs: `LES/` (full PRS/RRS) and `LES_no_homodimers/` (homodimer
pairs excluded). `--les_subdir` selects which one to composite; run this
script twice (once per subdir) to get both.

  1. composite_trajectory_AUC(.png/.pdf)   — across-replicate MEAN ROC-AUC per
     epoch, with a +/-1 SD band; title reports ensemble LES-AUC = mean +/- SD.
  2. composite_trajectory_F1(.png/.pdf)    — same for Best-F1.
     Each also gets an *_area version (shading to the floor + SD error bars).
  3. composite_PRS-RRS_violins(.png/.pdf)  — the PRS vs RRS P(interaction)
     distributions POOLED across all N replicates at each epoch.

Plus composite_trajectory_data.csv (per-epoch mean/SD + LES mean/SD) and two
legend .md files.

Example
-------
    python make_composite_les.py --parent results --les_subdir LES \\
        --out results/composite
    python make_composite_les.py --parent results --les_subdir LES_no_homodimers \\
        --out results/composite_no_homodimers
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

PUB_DPI = 600
AUC_C = "#1f4e9c"   # blue  (matches the per-replicate LES-wrapper AUC colour)
F1_C = "#2c8a3d"    # green (Best-F1)
PRS_C, RRS_C = "#2166ac", "#b2182b"   # PRS positives (blue), RRS negatives (red)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# -----------------------------------------------------------------------------
# Discover the per-replicate LES directories
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


# -----------------------------------------------------------------------------
# Trajectories (AUC / Best-F1): across-replicate mean +/- SD, ensemble LES
# -----------------------------------------------------------------------------
def read_summary(run_dir):
    """Return {epoch:int -> (AUC, Best_F1)} for the numbered-epoch rows only
    (the 'final' checkpoint and the trailing 'LES (...)' row are skipped)."""
    out = {}
    with open(os.path.join(run_dir, 'summary_table.csv')) as fh:
        for row in csv.DictReader(fh):
            ckpt = (row.get('checkpoint') or '').strip()
            ep = (row.get('epoch') or '').strip()
            if not ckpt.startswith('ppiDCE_epoch') or not ep.isdigit():
                continue
            out[int(ep)] = (float(row['AUC']), float(row['Best_F1']))
    return out


def compute_les(epochs, values):
    """Area under the metric-vs-epoch curve; epochs min-max normalized to [0,1]."""
    e = np.asarray(epochs, float)
    v = np.asarray(values, float)
    if len(e) < 2:
        return float('nan')
    en = (e - e.min()) / (e.max() - e.min())
    return float(np.trapezoid(v, en))


def _annotate_nodes(ax, xs, ys):
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color="0.15")


def make_trajectory(epochs, mat, color, metric_name, les_vals, base, floor=False):
    """mat: (n_replicates, n_epochs). floor=False -> +/-1 SD band; floor=True ->
    area shading to the floor with +/-1 SD error bars on each node."""
    mean = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=0)              # population SD across replicates
    n = mat.shape[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    if floor:
        ax.fill_between(epochs, 0, mean, color=color, alpha=0.18,
                        label="Area under curve (LES)")
        ax.errorbar(epochs, mean, yerr=sd, fmt="o-", color=color, lw=2.5,
                    markersize=7, capsize=4, elinewidth=1.4, ecolor=color,
                    label=f"Ensemble mean +/-1 SD (n = {n})")
    else:
        lo = np.clip(mean - sd, 0, 1)
        hi = np.clip(mean + sd, 0, 1)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.20,
                        label=f"+/-1 SD (n = {n} replicates)")
        ax.plot(epochs, mean, "o-", color=color, lw=2.5, markersize=7,
                label="Ensemble mean")
    _annotate_nodes(ax, epochs, mean)

    ax.set_xlabel("Training epoch", fontsize=15)
    ax.set_ylabel(metric_name, fontsize=15)
    ax.set_ylim(0, 1)
    ax.set_xlim(min(epochs) - 0.4, max(epochs) + 0.4)
    ax.set_xticks(list(epochs))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{metric_name} trajectory — {n}-replicate V3 ensemble\n"
        f"LES-{metric_name} = {np.nanmean(les_vals):.4f} +/- {np.nanstd(les_vals, ddof=0):.4f} "
        f"(mean +/- SD, n = {n})",
        fontsize=15, pad=12)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=PUB_DPI)
    plt.close(fig)
    print(f"  wrote {os.path.basename(base)}.png/.pdf   "
          f"LES = {np.nanmean(les_vals):.4f} +/- {np.nanstd(les_vals, ddof=0):.4f}")


def trajectories(run_dirs, out):
    summaries = [read_summary(d) for d in run_dirs]
    # common epoch grid across all replicates (intersection), sorted
    common = sorted(set.intersection(*[set(s) for s in summaries]))
    if len(common) < 2:
        raise SystemExit(f"Need >= 2 shared epochs across replicates; got {common}")
    dropped = [sorted(set(s) - set(common)) for s in summaries]
    if any(dropped):
        print(f"  epoch grid intersected to {common}; per-replicate dropped: {dropped}")

    auc_mat = np.array([[s[e][0] for e in common] for s in summaries])
    f1_mat = np.array([[s[e][1] for e in common] for s in summaries])
    les_auc = np.array([compute_les(common, row) for row in auc_mat])
    les_f1 = np.array([compute_les(common, row) for row in f1_mat])
    epochs = np.array(common, float)

    make_trajectory(epochs, auc_mat, AUC_C, "AUC", les_auc,
                    os.path.join(out, "composite_trajectory_AUC"))
    make_trajectory(epochs, f1_mat, F1_C, "Best F1", les_f1,
                    os.path.join(out, "composite_trajectory_F1"))
    make_trajectory(epochs, auc_mat, AUC_C, "AUC", les_auc,
                    os.path.join(out, "composite_trajectory_AUC_area"), floor=True)
    make_trajectory(epochs, f1_mat, F1_C, "Best F1", les_f1,
                    os.path.join(out, "composite_trajectory_F1_area"), floor=True)

    with open(os.path.join(out, "composite_trajectory_data.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "AUC_mean", "AUC_SD", "F1_mean", "F1_SD"])
        for i, e in enumerate(common):
            w.writerow([e,
                        f"{auc_mat[:, i].mean():.6f}", f"{auc_mat[:, i].std(ddof=0):.6f}",
                        f"{f1_mat[:, i].mean():.6f}", f"{f1_mat[:, i].std(ddof=0):.6f}"])
        w.writerow([])
        w.writerow(["LES_AUC_mean", f"{les_auc.mean():.6f}",
                    "LES_AUC_SD", f"{les_auc.std(ddof=0):.6f}"])
        w.writerow(["LES_F1_mean", f"{les_f1.mean():.6f}",
                    "LES_F1_SD", f"{les_f1.std(ddof=0):.6f}"])
    return common


# -----------------------------------------------------------------------------
# Violins: PRS vs RRS P(interaction) pooled across replicates per epoch
# -----------------------------------------------------------------------------
def read_pos_probs(path, pos_col):
    """Positive-class probability (Probability_Friends) per pair from a
    ppiDCE LES-wrapper per-checkpoint probabilities CSV. Reads the named
    column; falls back to the second-to-last column if the header lacks it."""
    vals = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="") as f:
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


def pooled(run_dirs, epoch, kind, pos_col):
    out = []
    for d in run_dirs:
        fp = os.path.join(d, f"epoch_{epoch}", f"{kind}_epoch{epoch}_probabilities.csv")
        out.extend(read_pos_probs(fp, pos_col))
    return np.array(out)


def violins(run_dirs, epochs, out, pos_col):
    prs = [pooled(run_dirs, e, "PRS", pos_col) for e in epochs]
    rrs = [pooled(run_dirs, e, "RRS", pos_col) for e in epochs]
    n = len(prs[0])
    if not (all(len(v) == n for v in prs) and all(len(v) == n for v in rrs)):
        print("  WARNING: uneven pooled counts across epochs "
              f"(PRS {[len(v) for v in prs]}, RRS {[len(v) for v in rrs]})")

    gap = 2.6
    prs_pos = np.arange(1, len(epochs) + 1, dtype=float)
    rrs_pos = prs_pos + len(epochs) + gap
    divider = len(epochs) + gap / 2.0 + 0.5

    fig, ax = plt.subplots(figsize=(max(14.0, 1.2 * len(epochs) + 3), 6.6))

    def viol(data, pos, color):
        parts = ax.violinplot(data, positions=pos, widths=0.9,
                              showmedians=True, showextrema=False)
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
    ax.set_ylabel("P(interaction) = Probability_Friends", fontsize=14)
    ax.set_xlabel("Training epoch", fontsize=13)
    ax.yaxis.grid(True, ls="--", alpha=0.4); ax.set_axisbelow(True)

    ax.text(prs_pos.mean(), 1.06, "PRS (positives)", ha="center", va="bottom",
            fontsize=14, color=PRS_C, fontweight="bold", transform=ax.transData)
    ax.text(rrs_pos.mean(), 1.06, "RRS (negatives)", ha="center", va="bottom",
            fontsize=14, color=RRS_C, fontweight="bold", transform=ax.transData)
    ax.set_title(f"{len(run_dirs)}-replicate composite — PRS vs RRS\n"
                 f"pooled across V3-1..V3-{len(run_dirs)}; n = {n} pairs per epoch",
                 fontsize=14, pad=30)
    fig.tight_layout()
    base = os.path.join(out, "composite_PRS-RRS_violins")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote composite_PRS-RRS_violins.png/.pdf  (n={n}/epoch)")


# -----------------------------------------------------------------------------
def _condition_note(les_subdir):
    """Human-readable description of the reference set behind a --les_subdir,
    for the generated legend files."""
    if les_subdir == "LES":
        return "the full PRS/RRS reference sets (homodimer pairs included)"
    if les_subdir == "LES_no_homodimers":
        return "homodimer-depleted PRS/RRS (seq1 == seq2 pairs excluded from both)"
    if les_subdir.startswith("LES_"):
        cond = les_subdir[len("LES_"):]
        return (f"the random-substituted PRS/RRS control set ('{cond}' — see "
                f"PRS-RRS_random_controls/README.md for what's substituted)")
    return f"the '{les_subdir}' reference set"


def write_legends(out, n_replicates, les_subdir):
    homo_note = _condition_note(les_subdir)
    traj = f"""# Composite LES trajectories — legend ({les_subdir})

`composite_trajectory_AUC.*` and `composite_trajectory_F1.*` show the
**across-replicate mean** of ROC-AUC / Best-F1 at each training epoch for the
{n_replicates}-replicate V3 ensemble, scored on {homo_note}, with a shaded
**+/-1 SD** band (population SD, ddof=0). The `_area` variants shade the area
under the mean curve and draw the SD as error bars on each node. Each
per-replicate value is read from that replicate's `summary_table.csv`
(produced by `LES-wrapper.py`); the per-replicate Learning Efficiency Score
(LES) is the area under its own metric-vs-epoch curve (epochs min-max
normalized to [0,1], trapezoidal rule). The title reports the ensemble LES as
**mean +/- SD** across the {n_replicates} replicates.
`composite_trajectory_data.csv` holds the per-epoch mean/SD and the LES
mean/SD.
"""
    viol = f"""# Composite PRS/RRS violins — legend ({les_subdir})

`composite_PRS-RRS_violins.*` pools the per-pair interaction score
`P(interaction) = Probability_Friends` across all {n_replicates} replicates at
each training epoch, scored on {homo_note} — each replicate contributes its
own matched reference pairs, so each violin summarises
{n_replicates} x (pairs/replicate) pooled values. Left section: PRS
(positives, blue) — should sit high. Right section: RRS (negatives, red) —
should sit low. Black line = median. y-axis is ppiDCE's 2-class softmax mass
on the "friends" (interacting) class, in [0, 1].
"""
    with open(os.path.join(out, "composite_trajectories_legend.md"), "w") as f:
        f.write(traj)
    with open(os.path.join(out, "composite_violins_legend.md"), "w") as f:
        f.write(viol)


def main():
    ap = argparse.ArgumentParser(
        description="Composite (N-replicate ensemble) LES figures for ppiDCE.")
    ap.add_argument("--parent", default="results",
                    help="Directory containing the per-replicate results dirs (default: results).")
    ap.add_argument("--replicate_glob", default="dce_V3-*_scratch12L_ml1024",
                    help="Glob for the per-replicate dirs under --parent.")
    ap.add_argument("--les_subdir", default="LES",
                    help="Which LES-wrapper run to composite, e.g. LES, LES_no_homodimers, "
                         "LES_ps1_random, LES_ps2_random, LES_ps1-ps2_random "
                         "(default: LES, the full PRS/RRS set).")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: <parent>/composite for --les_subdir LES, "
                         "else <parent>/composite_<suffix> stripping a leading 'LES_').")
    ap.add_argument("--pos_col", default="Probability_Friends",
                    help="Positive-class probability column name (default: Probability_Friends).")
    args = ap.parse_args()

    if args.les_subdir == "LES":
        default_out = "composite"
    elif args.les_subdir.startswith("LES_"):
        default_out = "composite_" + args.les_subdir[len("LES_"):]
    else:
        default_out = "composite_" + args.les_subdir
    out = args.out or os.path.join(args.parent, default_out)
    os.makedirs(out, exist_ok=True)

    run_dirs = find_run_dirs(args.parent, args.replicate_glob, args.les_subdir)
    print(f"Composite over {len(run_dirs)} replicates ({args.les_subdir}):")
    for d in run_dirs:
        print(f"  - {d}")

    print("Trajectories:")
    epochs = trajectories(run_dirs, out)
    print("Violins:")
    violins(run_dirs, epochs, out, args.pos_col)
    write_legends(out, len(run_dirs), args.les_subdir)
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
