#!/usr/bin/env bash
# run_all_replicates.sh — sequentially run run_replicate.sh across multiple
# MED4 V3 replicates. Each replicate is a full 10-epoch from-scratch training
# (with per-epoch ROC/Best-F1) plus two end-of-run LES-wrapper analyses (with
# and without homodimer pairs) — see run_replicate.sh / train-recipe.md.
#
# Default replicate list is 2..10 (V3-1 is done separately). Resumable: a
# replicate whose LES_no_homodimers/summary_table.csv already exists is
# skipped, so a killed/restarted run doesn't redo finished work.
#
#   ./run_all_replicates.sh                 # replicates 2..10
#   ./run_all_replicates.sh 2 3 4           # just these replicates
#   NUM_LAYERS=6 BATCH_SIZE=4 ./run_all_replicates.sh   # forwarded to run_replicate.sh
#
# All run_replicate.sh env overrides (NUM_LAYERS, BATCH_SIZE, MAX_LENGTH,
# DEPLETE, PY) are inherited by each replicate's run_replicate.sh invocation.
set -euo pipefail

cd "$(dirname "$0")"

NUM_LAYERS="${NUM_LAYERS:-12}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

if [[ $# -gt 0 ]]; then
  REPLICATES=("$@")
else
  REPLICATES=(2 3 4 5 6 7 8 9 10)
fi

for k in "${REPLICATES[@]}"; do
  REP="V3-${k}"
  OUT="results/dce_${REP}_scratch${NUM_LAYERS}L_ml${MAX_LENGTH}"
  DONE_MARKER="$OUT/LES_no_homodimers/summary_table.csv"

  if [[ -f "$DONE_MARKER" ]]; then
    echo "=== ${REP}: already done (found $DONE_MARKER), skipping ==="
    continue
  fi

  echo "############################################################"
  echo "### ${REP} START $(date)"
  echo "############################################################"
  REP="$REP" ./run_replicate.sh
  echo "############################################################"
  echo "### ${REP} DONE $(date)"
  echo "############################################################"
done

echo "All requested replicates complete."
