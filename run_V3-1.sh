#!/usr/bin/env bash
# run_V3-1.sh — ppiDCE V3-1 replicate: from-scratch 6-block training with the
# ppiYYD warmup+cosine LR recipe, per-epoch ROC/Best-F1 analysis, and a final
# LES-wrapper pass over every saved checkpoint.
#
#   ./run_V3-1.sh                    # full run
#   LES_ONLY=1 ./run_V3-1.sh         # rebuild the LES analysis from existing checkpoints
#
# Override the interpreter with PY=/path/to/python.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-/home/ksa/anaconda3/envs/esm2/bin/python}"
REP="${REP:-V3-1}"
OUT="${OUT:-results/dce_${REP}_scratch6L}"

TRAIN_SRC="MED4_V3_Trains/depleted_training_set-${REP}.csv"
PRS="MED4_PRS-RRS/PRS-${REP}.csv"
RRS="MED4_PRS-RRS/RRS-${REP}.csv"

export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT/data" "$OUT/eval"

if [[ "${LES_ONLY:-0}" != "1" ]]; then
  echo "=== ppiDCE ${REP} SPLIT $(date) ==="
  # Val is SAMPLED from train but NOT removed from it: the eval pairs stay in
  # train. The real holdout is PRS/RRS.
  "$PY" make_split.py \
      --input "$TRAIN_SRC" \
      --output_dir "$OUT/data" \
      --val_frac 0.1 --seed 42

  echo
  echo "=== ppiDCE ${REP} TRAIN START $(date) ==="
  "$PY" train_ppiDCE.py \
      --train_file "$OUT/data/train.csv" \
      --val_file   "$OUT/data/val.csv" \
      --model_config facebook/esm1b_t33_650M_UR50S \
      --from_scratch --num_layers 6 \
      --max_length 1024 \
      --epochs 10 --batch_size 4 \
      --lr_schedule warmup_cosine \
      --learning_rate 2e-5 --min_lr 2e-6 --warmup_ratio 0.1 \
      --eval_prs "$PRS" --eval_rrs "$RRS" --eval_dir "$OUT/eval" \
      --roc_script roc_analysis_color_threshold_F1e.py \
      --output_dir "$OUT" \
      --device cuda --suppress_warnings
  echo "=== ppiDCE ${REP} TRAIN DONE $(date) ==="
fi

echo
echo "=== ppiDCE ${REP} LES START $(date) ==="
"$PY" LES-wrapper.py \
    --checkpoint_dir "$OUT" \
    --prs_file "$PRS" \
    --rrs_file "$RRS" \
    --output_dir "$OUT/LES" \
    --model_config facebook/esm1b_t33_650M_UR50S \
    --num_layers 6 \
    --max_length 1024 \
    --batch_size 4 \
    --device cuda \
    --include_final
echo "=== ppiDCE ${REP} LES DONE $(date) ==="

echo
echo "--- per-epoch metrics ---"
cat "$OUT/eval/metrics_by_epoch.csv"
