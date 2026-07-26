#!/usr/bin/env bash
# run_replicate.sh — ppiDCE V3 replicate driver: from-scratch training with
# the ppiYYD warmup+cosine LR recipe (see train-recipe.md), a 90/10 train/val
# split (val sampled but kept in train — recommended for a small training set
# like this one), per-epoch ROC/Best-F1 analysis, and two final LES-wrapper
# passes over every saved checkpoint — one on the full PRS/RRS set, one on the
# homodimer-depleted set in MED4_PRS-RRS_no_homodimers/ (see its README.md).
# Homodimers are ~22-32% of every V3 replicate's PRS set and score much closer
# to the RRS baseline than heterodimers do (checked across all 10 replicates,
# 2026-07-26), so the two LES runs read differently — see train-recipe.md for
# the numbers this produced on V3-1.
#
# Formerly run_V3-1.sh; renamed once REP became the standard way to drive any
# replicate (REP=V3-2, V3-3, ...). See run_all_replicates.sh to run several.
#
#   REP=V3-2 ./run_replicate.sh                # full run (max_length 1024, 12 layers, batch 2 by default)
#   MAX_LENGTH=512 ./run_replicate.sh          # override max_length (expect more truncation — see train-recipe.md)
#   NUM_LAYERS=6 BATCH_SIZE=4 ./run_replicate.sh   # override architecture / batch size
#   DEPLETE=1 ./run_replicate.sh               # carve val OUT of train instead (disjoint split)
#   LES_ONLY=1 ./run_replicate.sh              # rebuild both LES analyses from existing checkpoints
#
# Override the interpreter with PY=/path/to/python.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-/home/ksa/anaconda3/envs/esm2/bin/python}"
REP="${REP:-V3-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
NUM_LAYERS="${NUM_LAYERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUT="${OUT:-results/dce_${REP}_scratch${NUM_LAYERS}L_ml${MAX_LENGTH}}"

DEPLETE_FLAG=()
if [[ "${DEPLETE:-0}" == "1" ]]; then
  DEPLETE_FLAG=(--deplete)
fi

TRAIN_SRC="MED4_V3_Trains/depleted_training_set-${REP}.csv"
PRS="MED4_PRS-RRS/PRS-${REP}.csv"
RRS="MED4_PRS-RRS/RRS-${REP}.csv"
PRS_NOHOMO="MED4_PRS-RRS_no_homodimers/PRS-${REP}.csv"
RRS_NOHOMO="MED4_PRS-RRS_no_homodimers/RRS-${REP}.csv"

export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT/data" "$OUT/eval"

if [[ "${LES_ONLY:-0}" != "1" ]]; then
  echo "=== ppiDCE ${REP} SPLIT $(date) ==="
  # Val is sampled but kept in train by default — recommended for a small
  # training set like this one (see train-recipe.md). Set DEPLETE=1 for the
  # standard disjoint split instead.
  "$PY" make_split.py \
      --input "$TRAIN_SRC" \
      --output_dir "$OUT/data" \
      --val_frac 0.1 --seed 42 "${DEPLETE_FLAG[@]}"

  echo
  echo "=== ppiDCE ${REP} TRAIN START $(date) ==="
  "$PY" train_ppiDCE.py \
      --train_file "$OUT/data/train.csv" \
      --val_file   "$OUT/data/val.csv" \
      --model_config facebook/esm1b_t33_650M_UR50S \
      --from_scratch --num_layers "$NUM_LAYERS" \
      --max_length "$MAX_LENGTH" \
      --epochs 10 --batch_size "$BATCH_SIZE" \
      --lr_schedule warmup_cosine \
      --learning_rate 2e-5 --min_lr 2e-6 --warmup_ratio 0.1 \
      --eval_prs "$PRS" --eval_rrs "$RRS" --eval_dir "$OUT/eval" \
      --roc_script roc_analysis_color_threshold_F1e.py \
      --output_dir "$OUT" \
      --device cuda --suppress_warnings
  echo "=== ppiDCE ${REP} TRAIN DONE $(date) ==="
fi

echo
echo "=== ppiDCE ${REP} LES (with homodimers) START $(date) ==="
"$PY" LES-wrapper.py \
    --checkpoint_dir "$OUT" \
    --prs_file "$PRS" \
    --rrs_file "$RRS" \
    --output_dir "$OUT/LES" \
    --model_config facebook/esm1b_t33_650M_UR50S \
    --num_layers "$NUM_LAYERS" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --include_final
echo "=== ppiDCE ${REP} LES (with homodimers) DONE $(date) ==="

echo
echo "=== ppiDCE ${REP} LES (no homodimers) START $(date) ==="
"$PY" LES-wrapper.py \
    --checkpoint_dir "$OUT" \
    --prs_file "$PRS_NOHOMO" \
    --rrs_file "$RRS_NOHOMO" \
    --output_dir "$OUT/LES_no_homodimers" \
    --model_config facebook/esm1b_t33_650M_UR50S \
    --num_layers "$NUM_LAYERS" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --include_final
echo "=== ppiDCE ${REP} LES (no homodimers) DONE $(date) ==="

echo
echo "--- per-epoch metrics (with homodimers, from training-time eval) ---"
cat "$OUT/eval/metrics_by_epoch.csv"
