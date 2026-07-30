#!/usr/bin/env bash
# run_random_controls_les.sh — LES-wrapper analysis of the already-trained V3
# replicate checkpoints against the random-substituted PRS/RRS control sets
# (V3_PRS-RRS/random_controls/), for all three randomization schemes:
#
#   ps1      — partner 1 sequence randomly substituted
#   ps2      — partner 2 sequence randomly substituted
#   ps1-ps2  — both partners randomly substituted
#
# Pure evaluation — no training. Results land under Results_random_controls/,
# mirroring results/'s per-replicate layout (dce_V3-k_scratch12L_ml1024/),
# with one LES_{condition}_random/ subdir per condition. Resumable: a
# replicate x condition combo whose summary_table.csv already exists is
# skipped.
#
#   ./run_random_controls_les.sh              # all 10 replicates x 3 conditions
#   ./run_random_controls_les.sh 1 2 3         # just these replicates
#
# Override the interpreter with PY=/path/to/python.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-/home/ksa/anaconda3/envs/esm2/bin/python}"
NUM_LAYERS="${NUM_LAYERS:-12}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SRC="${SRC:-V3_PRS-RRS/random_controls}"
OUT_ROOT="Results_random_controls"

export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

CONDITIONS=(ps1_random ps2_random ps1-ps2_random)

if [[ $# -gt 0 ]]; then
  REPLICATES=("$@")
else
  REPLICATES=(1 2 3 4 5 6 7 8 9 10)
fi

for k in "${REPLICATES[@]}"; do
  REP="V3-${k}"
  CKPT_DIR="results/dce_${REP}_scratch${NUM_LAYERS}L_ml${MAX_LENGTH}"
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "=== ${REP}: SKIP, no checkpoint dir $CKPT_DIR ==="
    continue
  fi

  for cond in "${CONDITIONS[@]}"; do
    PRS="$SRC/PRS-${REP}_${cond}.csv"
    RRS="$SRC/RRS-${REP}_${cond}.csv"
    OUT="$OUT_ROOT/dce_${REP}_scratch${NUM_LAYERS}L_ml${MAX_LENGTH}/LES_${cond}"

    if [[ -f "$OUT/summary_table.csv" ]]; then
      echo "=== ${REP} ${cond}: already done, skipping ==="
      continue
    fi

    mkdir -p "$OUT"
    echo "=== ppiDCE ${REP} LES (${cond}) START $(date) ==="
    "$PY" LES-wrapper.py \
        --checkpoint_dir "$CKPT_DIR" \
        --prs_file "$PRS" \
        --rrs_file "$RRS" \
        --output_dir "$OUT" \
        --model_config facebook/esm1b_t33_650M_UR50S \
        --num_layers "$NUM_LAYERS" \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --device cuda \
        --include_final
    echo "=== ppiDCE ${REP} LES (${cond}) DONE $(date) ==="
  done
done

echo "All requested replicate x condition LES runs complete."
