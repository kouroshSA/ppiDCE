#!/usr/bin/env bash
# run_homodimers_only_les.sh — LES-wrapper analysis of the already-trained V3
# replicate checkpoints against the homodimers-only PRS
# (V3_PRS-RRS/PRS-RRS_homodimers_only/), paired with the full RRS — there are no RRS
# homodimers to filter to, so the full 100-pair random set is used unchanged
# (see V3_PRS-RRS/PRS-RRS_homodimers_only/NOTE_RRS_homodimers.md). Override with SRC=.
#
# Pure evaluation — no training. Results land under Results_Homodimers_only/,
# mirroring results/'s per-replicate layout (dce_V3-k_scratch12L_ml1024/),
# each with a LES_homodimers_only/ subdir. Resumable: a replicate whose
# summary_table.csv already exists is skipped.
#
#   ./run_homodimers_only_les.sh          # all 10 replicates
#   ./run_homodimers_only_les.sh 1 2 3    # just these replicates
#
# Override the interpreter with PY=/path/to/python.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-/home/ksa/anaconda3/envs/esm2/bin/python}"
NUM_LAYERS="${NUM_LAYERS:-12}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SRC="${SRC:-V3_PRS-RRS/PRS-RRS_homodimers_only}"
OUT_ROOT="Results_Homodimers_only"

export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

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

  PRS="$SRC/PRS-${REP}.csv"
  RRS="$SRC/RRS-${REP}.csv"
  OUT="$OUT_ROOT/dce_${REP}_scratch${NUM_LAYERS}L_ml${MAX_LENGTH}/LES_homodimers_only"

  if [[ -f "$OUT/summary_table.csv" ]]; then
    echo "=== ${REP}: already done, skipping ==="
    continue
  fi

  mkdir -p "$OUT"
  echo "=== ppiDCE ${REP} LES (homodimers_only) START $(date) ==="
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
  echo "=== ppiDCE ${REP} LES (homodimers_only) DONE $(date) ==="
done

echo "All requested replicate LES (homodimers_only) runs complete."
