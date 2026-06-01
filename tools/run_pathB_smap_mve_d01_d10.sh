#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
MODEL_DATASET="${MODEL_DATASET:-SMAP}"
PATHB_OUTPUT_ROOT="${PATHB_OUTPUT_ROOT:-pathB_result}"
VALIDATION_THRESHOLD_ROOT="${VALIDATION_THRESHOLD_ROOT:-validation_threshold}"
LOG_DIR="${LOG_DIR:-pathB_run_logs}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"
VARIANTS="${VARIANTS:-}"
PATHB_MODE="${PATHB_MODE:-self}"
PATHB_PROTO_DATASET="${PATHB_PROTO_DATASET:-${MODEL_DATASET}}"
PATHB_PROTO_RECOMPUTE_FLAG="${PATHB_PROTO_RECOMPUTE_FLAG:-}"
PATHB_COMPARE_STEPS="${PATHB_COMPARE_STEPS:-49,45,40,35,30,25,20,15,10,5,0}"

mkdir -p "${LOG_DIR}"

if [[ -n "${VARIANTS}" ]]; then
  variant_list="${VARIANTS}"
else
  variant_list="$(seq -f 'SMAP_MVE_d%02g' 1 10)"
fi

for variant in ${variant_list}; do
  log_file="${LOG_DIR}/${variant}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] start ${variant}" | tee "${log_file}"
  python evaluate_machine_window_middle1.py \
    --device "${DEVICE}" \
    --dataset "${variant}" \
    --model_dataset "${MODEL_DATASET}" \
    --result_tag "${variant}" \
    --pathB_output_root "${PATHB_OUTPUT_ROOT}" \
    --validation_threshold_root "${VALIDATION_THRESHOLD_ROOT}" \
    --pathB_mode "${PATHB_MODE}" \
    --pathB_proto_dataset "${PATHB_PROTO_DATASET}" \
    --pathB_compare_steps "${PATHB_COMPARE_STEPS}" \
    ${PATHB_PROTO_RECOMPUTE_FLAG} \
    ${OVERWRITE_FLAG} 2>&1 | tee -a "${log_file}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] done ${variant}" | tee -a "${log_file}"
done

echo "Requested SMAP_MVE Path B inference runs finished."
