#!/usr/bin/env bash
set -euo pipefail

# One-click runner for PathB-HDSAC on SMAP-MVE.
# Prototype is always built from base_dataset=SMAP train 5% validation split.
# MVE clean/variant datasets are never used to estimate the prototype.

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:1}"
BASE_DATASET="${BASE_DATASET:-SMAP}"
MODEL_DATASET="${MODEL_DATASET:-SMAP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-pathB_result_hdsac}"
SELECTED_STEPS="${SELECTED_STEPS:-40,25,10,0}"
VAL_RATIO="${VAL_RATIO:-0.05}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-24}"
SPLIT="${SPLIT:-4}"
TAU="${TAU:-1.0}"
RIDGE="${RIDGE:-1e-4}"
TOP_R_DESCRIPTOR="${TOP_R_DESCRIPTOR:-3}"
CHANNEL_TOPK="${CHANNEL_TOPK:-3}"
FUSION_ALPHA="${FUSION_ALPHA:-0.5}"
ALPHA_VALUES="${ALPHA_VALUES:-0:1:0.05}"
ALPHA_METRIC="${ALPHA_METRIC:-auc_roc}"
OUT_CSV="${OUT_CSV:-pathB_hdsac_auc_summary.csv}"

SAVES_TEXT="${SAVES:-save0 save1 save2}"
SCORE_VARIANTS_TEXT="${SCORE_VARIANTS:-SMAP_MVE_clean SMAP_MVE_d01 SMAP_MVE_d02 SMAP_MVE_d03 SMAP_MVE_d05 SMAP_MVE_d08 SMAP_MVE_d10}"
EVAL_VARIANTS_TEXT="${EVAL_VARIANTS:-SMAP_MVE_d01 SMAP_MVE_d02 SMAP_MVE_d03 SMAP_MVE_d05 SMAP_MVE_d08 SMAP_MVE_d10}"

read -r -a SAVE_LIST <<< "${SAVES_TEXT}"
read -r -a SCORE_VARIANTS <<< "${SCORE_VARIANTS_TEXT}"
read -r -a EVAL_VARIANTS <<< "${EVAL_VARIANTS_TEXT}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-1}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

STEPS_KEY="$(printf '%s' "${SELECTED_STEPS}" | tr ',' ' ' | xargs | tr ' ' '_')"

echo "[HDSAC-BATCH] python=${PYTHON_BIN}"
echo "[HDSAC-BATCH] device=${DEVICE}"
echo "[HDSAC-BATCH] base_dataset=${BASE_DATASET}"
echo "[HDSAC-BATCH] model_dataset=${MODEL_DATASET}"
echo "[HDSAC-BATCH] output_root=${OUTPUT_ROOT}"
echo "[HDSAC-BATCH] selected_steps=${SELECTED_STEPS}"
echo "[HDSAC-BATCH] fusion_alpha=${FUSION_ALPHA}"
echo "[HDSAC-BATCH] alpha_values=${ALPHA_VALUES}"
echo "[HDSAC-BATCH] alpha_metric=${ALPHA_METRIC}"
echo "[HDSAC-BATCH] saves=${SAVE_LIST[*]}"
echo "[HDSAC-BATCH] score_variants=${SCORE_VARIANTS[*]}"
echo "[HDSAC-BATCH] eval_variants=${EVAL_VARIANTS[*]}"

echo "[HDSAC-BATCH] syntax check"
"${PYTHON_BIN}" -m py_compile \
  tools/generate_pathB_hdsac_scores.py \
  tools/evaluate_pathB_hdsac_auc.py \
  main_model.py \
  diff_models.py \
  utils.py

for save_id in "${SAVE_LIST[@]}"; do
  prototype_path="${OUTPUT_ROOT}/${BASE_DATASET}/prototype_steps_${STEPS_KEY}_seed${SEED}_${save_id}.npz"

  echo "[HDSAC-BATCH] build prototype save=${save_id}"
  "${PYTHON_BIN}" tools/generate_pathB_hdsac_scores.py \
    --mode build_prototype \
    --device "${DEVICE}" \
    --base_dataset "${BASE_DATASET}" \
    --model_dataset "${MODEL_DATASET}" \
    --saves "${save_id}" \
    --selected_steps "${SELECTED_STEPS}" \
    --val_ratio "${VAL_RATIO}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --split "${SPLIT}" \
    --tau "${TAU}" \
    --ridge "${RIDGE}" \
    --top_r_descriptor "${TOP_R_DESCRIPTOR}" \
    --channel_topk "${CHANNEL_TOPK}" \
    --output_root "${OUTPUT_ROOT}" \
    "${OVERWRITE_ARGS[@]}"

  if [[ ! -f "${prototype_path}" ]]; then
    echo "[HDSAC-BATCH][ERROR] missing prototype: ${prototype_path}" >&2
    exit 1
  fi

  for variant in "${SCORE_VARIANTS[@]}"; do
    echo "[HDSAC-BATCH] score save=${save_id} variant=${variant}"
    "${PYTHON_BIN}" tools/generate_pathB_hdsac_scores.py \
      --mode score \
      --device "${DEVICE}" \
      --dataset "${variant}" \
      --base_dataset "${BASE_DATASET}" \
      --model_dataset "${MODEL_DATASET}" \
      --saves "${save_id}" \
      --selected_steps "${SELECTED_STEPS}" \
      --prototype_path "${prototype_path}" \
      --batch_size "${BATCH_SIZE}" \
      --split "${SPLIT}" \
      --fusion_alpha "${FUSION_ALPHA}" \
      --output_root "${OUTPUT_ROOT}" \
      "${OVERWRITE_ARGS[@]}"
  done
done

echo "[HDSAC-BATCH] evaluate"
"${PYTHON_BIN}" tools/evaluate_pathB_hdsac_auc.py \
  --pathB_output_root "${OUTPUT_ROOT}" \
  --base_dataset "${BASE_DATASET}" \
  --variants "${EVAL_VARIANTS[@]}" \
  --alpha_values "${ALPHA_VALUES}" \
  --alpha_metric "${ALPHA_METRIC}" \
  --out "${OUT_CSV}"

echo "[HDSAC-BATCH] done: ${OUT_CSV}"
