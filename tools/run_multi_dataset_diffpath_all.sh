#!/usr/bin/env bash
set -Eeuo pipefail

# Multi-dataset DiffPath-6D-GMM pipeline.
# Defaults keep the agreed protocol:
# train batch size = 12, reconstruction scoring batch size = 24,
# DiffPath/6D scoring batch size = 128.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-base.yaml}"
DATASETS_TEXT="${DATASETS:-SMAP MSL PSM}"
DATASETS_TEXT="${DATASETS_TEXT//,/ }"
read -r -a DATASET_VALUES <<< "${DATASETS_TEXT}"
if [[ "${#DATASET_VALUES[@]}" -eq 0 ]]; then
  echo "[ERROR] DATASETS cannot be empty." >&2
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-12}"
RECON_BATCH_SIZE="${RECON_BATCH_SIZE:-24}"
DIFFPATH_BATCH_SIZE="${DIFFPATH_BATCH_SIZE:-128}"
DIFFPATH_STEPS_LIST="${DIFFPATH_STEPS_LIST:-5,10,25,50}"
ENABLE_6D_KDE="${ENABLE_6D_KDE:-0}"
NUM_RUNS="${NUM_RUNS:-3}"
BASE_SEED="${BASE_SEED:-1}"
KDE_BANDWIDTHS="${KDE_BANDWIDTHS:-0.05,0.1,0.2,0.5,1.0}"
KDE_BANDWIDTHS_6D="${KDE_BANDWIDTHS_6D:-0.2,0.5,1.0,2.0,5.0}"
GMM_COMPONENTS_6D="${GMM_COMPONENTS_6D:-2,4,8,16}"
GMM_COVARIANCE_TYPES_6D="${GMM_COVARIANCE_TYPES_6D:-diag,full}"
ALPHA_VALUES="${ALPHA_VALUES:-0:1:0.05}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-pathB_result_multi_diffpath_6d_gmm_summax}"
LOG_DIR_BASE="${LOG_DIR_BASE:-run_logs/multi_diffpath}"
REUSE_EXISTING_TRAIN="${REUSE_EXISTING_TRAIN:-1}"
OVERWRITE="${OVERWRITE:-0}"
TRAIN_SPLIT="${TRAIN_SPLIT:-10}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"

mkdir -p "${OUTPUT_ROOT_BASE}" "${LOG_DIR_BASE}"

echo "[MULTI-DIFFPATH] project=${ROOT_DIR}"
echo "[MULTI-DIFFPATH] datasets=${DATASET_VALUES[*]}"
echo "[MULTI-DIFFPATH] device=${DEVICE}"
echo "[MULTI-DIFFPATH] batch_size=${BATCH_SIZE}"
echo "[MULTI-DIFFPATH] recon_batch_size=${RECON_BATCH_SIZE}"
echo "[MULTI-DIFFPATH] diffpath_batch_size=${DIFFPATH_BATCH_SIZE}"
echo "[MULTI-DIFFPATH] diffpath_steps=${DIFFPATH_STEPS_LIST}"
echo "[MULTI-DIFFPATH] output_root_base=${OUTPUT_ROOT_BASE}"
echo "[MULTI-DIFFPATH] reuse_existing_train=${REUSE_EXISTING_TRAIN}"
echo "[MULTI-DIFFPATH] overwrite=${OVERWRITE}"

bash -n tools/run_dataset_diffpath_all.sh

"${PYTHON_BIN}" -m py_compile \
  tools/evaluate_pathB_diffpath_f1.py \
  evaluate_machine_window_middle1.py \
  diffpath_1d.py \
  main_model.py \
  exe_machine.py

for dataset in "${DATASET_VALUES[@]}"; do
  model_dir_name="${dataset}_unconditional:True_task:reconstruction_split:${TRAIN_SPLIT}_diffusion_step:${DIFFUSION_STEPS}"
  have_all_checkpoints=1
  for run_index in 0 1 2; do
    checkpoint="train_result/save${run_index}/${model_dir_name}/best-model.pth"
    if [[ ! -f "${checkpoint}" ]]; then
      have_all_checkpoints=0
      break
    fi
  done

  dataset_skip_train="${SKIP_TRAIN:-0}"
  if [[ "${REUSE_EXISTING_TRAIN}" == "1" && "${have_all_checkpoints}" == "1" ]]; then
    dataset_skip_train=1
  fi

  dataset_output_root="${OUTPUT_ROOT_BASE}/${dataset}"
  dataset_log_dir="${LOG_DIR_BASE}/${dataset}"
  dataset_summary="${dataset_output_root}/${dataset}_diffpath_f1_summary_all_nfe.csv"

  echo
  echo "[MULTI-DIFFPATH] ===== ${dataset} ====="
  echo "[MULTI-DIFFPATH] skip_train=${dataset_skip_train}"
  echo "[MULTI-DIFFPATH] output_root=${dataset_output_root}"
  echo "[MULTI-DIFFPATH] summary=${dataset_summary}"

  DATASET="${dataset}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  DEVICE="${DEVICE}" \
  CONFIG="${CONFIG}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  RECON_BATCH_SIZE="${RECON_BATCH_SIZE}" \
  DIFFPATH_BATCH_SIZE="${DIFFPATH_BATCH_SIZE}" \
  DIFFPATH_STEPS_LIST="${DIFFPATH_STEPS_LIST}" \
  ENABLE_6D_KDE="${ENABLE_6D_KDE}" \
  NUM_RUNS="${NUM_RUNS}" \
  BASE_SEED="${BASE_SEED}" \
  KDE_BANDWIDTHS="${KDE_BANDWIDTHS}" \
  KDE_BANDWIDTHS_6D="${KDE_BANDWIDTHS_6D}" \
  GMM_COMPONENTS_6D="${GMM_COMPONENTS_6D}" \
  GMM_COVARIANCE_TYPES_6D="${GMM_COVARIANCE_TYPES_6D}" \
  ALPHA_VALUES="${ALPHA_VALUES}" \
  OUTPUT_ROOT="${dataset_output_root}" \
  SUMMARY_CSV="${dataset_summary}" \
  LOG_DIR="${dataset_log_dir}" \
  OVERWRITE="${OVERWRITE}" \
  SKIP_TRAIN="${dataset_skip_train}" \
  TRAIN_SPLIT="${TRAIN_SPLIT}" \
  DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
  bash tools/run_dataset_diffpath_all.sh
done

COMBINED_SUMMARY="${COMBINED_SUMMARY:-${OUTPUT_ROOT_BASE}/all_datasets_diffpath_f1_summary.csv}"
"${PYTHON_BIN}" -c '
import csv
import os
import sys

out_path = sys.argv[1]
root = sys.argv[2]
datasets = sys.argv[3:]
rows = []
fieldnames = None
for dataset in datasets:
    summary_path = os.path.join(
        root,
        dataset,
        f"{dataset}_diffpath_f1_summary_all_nfe.csv",
    )
    if not os.path.exists(summary_path):
        raise SystemExit(f"[ERROR] Missing summary: {summary_path}")
    with open(summary_path, newline="") as stream:
        reader = csv.DictReader(stream)
        if fieldnames is None:
            fieldnames = ["source_dataset"] + list(reader.fieldnames)
        for row in reader:
            rows.append({"source_dataset": dataset, **row})

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[MULTI-DIFFPATH] Combined dataset summary: {out_path}")
' \
  "${COMBINED_SUMMARY}" \
  "${OUTPUT_ROOT_BASE}" \
  "${DATASET_VALUES[@]}"

echo
echo "[MULTI-DIFFPATH] All datasets completed."
echo "[MULTI-DIFFPATH] Combined summary: ${COMBINED_SUMMARY}"
