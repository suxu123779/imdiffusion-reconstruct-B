#!/usr/bin/env bash
set -Eeuo pipefail

# Complete DiffPath-6D-GMM pipeline for one single-file dataset:
# three independent training processes -> NFE sweep scoring -> per-save F1 -> mean/std.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${DATASET:-SMAP}"
MODEL_DATASET="${MODEL_DATASET:-${DATASET}}"
RESULT_TAG="${RESULT_TAG:-${DATASET}}"
PROTO_DATASET="${PROTO_DATASET:-${MODEL_DATASET}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-base.yaml}"
BATCH_SIZE="${BATCH_SIZE:-12}"
RECON_BATCH_SIZE="${RECON_BATCH_SIZE:-24}"
DIFFPATH_BATCH_SIZE="${DIFFPATH_BATCH_SIZE:-128}"
NUM_RUNS="${NUM_RUNS:-3}"
BASE_SEED="${BASE_SEED:-1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-10}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
if [[ -n "${DIFFPATH_STEPS_LIST:-}" ]]; then
  RAW_DIFFPATH_STEPS_LIST="${DIFFPATH_STEPS_LIST}"
elif [[ -n "${DIFFPATH_STEPS:-}" ]]; then
  RAW_DIFFPATH_STEPS_LIST="${DIFFPATH_STEPS}"
else
  RAW_DIFFPATH_STEPS_LIST="5,10,25,50"
fi
KDE_BANDWIDTHS="${KDE_BANDWIDTHS:-0.05,0.1,0.2,0.5,1.0}"
KDE_BANDWIDTHS_6D="${KDE_BANDWIDTHS_6D:-0.2,0.5,1.0,2.0,5.0}"
GMM_COMPONENTS_6D="${GMM_COMPONENTS_6D:-2,4,8,16}"
GMM_COVARIANCE_TYPES_6D="${GMM_COVARIANCE_TYPES_6D:-diag,full}"
ENABLE_6D_KDE="${ENABLE_6D_KDE:-0}"
ALPHA_VALUES="${ALPHA_VALUES:-0:1:0.05}"
OUTPUT_ROOT="${OUTPUT_ROOT:-pathB_result_${DATASET}_diffpath_nfe_sweep}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/${DATASET}_diffpath_f1_summary_all_nfe.csv}"
LOG_DIR="${LOG_DIR:-run_logs/${DATASET}_diffpath}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

case "${DATASET}" in
  SMD|machine-*) FEATURE_DIM=38 ;;
  PSM) FEATURE_DIM=25 ;;
  MSL) FEATURE_DIM=55 ;;
  SMAP|SMAP_MVE*) FEATURE_DIM=25 ;;
  GCP) FEATURE_DIM=19 ;;
  SWaT) FEATURE_DIM=45 ;;
  CODERED) FEATURE_DIM=48 ;;
  *)
    echo "[ERROR] Unknown DATASET=${DATASET}; add its feature dimension to this script." >&2
    exit 1
    ;;
esac

RAW_DIFFPATH_STEPS_LIST="${RAW_DIFFPATH_STEPS_LIST//,/ }"
read -r -a DIFFPATH_STEP_VALUES <<< "${RAW_DIFFPATH_STEPS_LIST}"
if [[ "${#DIFFPATH_STEP_VALUES[@]}" -eq 0 ]]; then
  echo "[ERROR] DIFFPATH_STEPS_LIST cannot be empty." >&2
  exit 1
fi
for diffpath_steps in "${DIFFPATH_STEP_VALUES[@]}"; do
  if ! [[ "${diffpath_steps}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid diffpath step count: ${diffpath_steps}" >&2
    exit 1
  fi
done

if [[ "${NUM_RUNS}" != "3" ]]; then
  echo "[ERROR] The requested repeated-experiment protocol requires NUM_RUNS=3." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

OVERWRITE_TRAIN_ARGS=()
OVERWRITE_INFERENCE_ARGS=()
KDE_6D_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_TRAIN_ARGS+=(--overwrite)
  OVERWRITE_INFERENCE_ARGS+=(
    --overwrite
    --diffpath_recompute_calibrator
  )
fi
if [[ "${ENABLE_6D_KDE}" != "1" ]]; then
  KDE_6D_ARGS+=(--diffpath_disable_6d_kde)
fi

run_step() {
  local step_name="$1"
  shift
  local log_path="${LOG_DIR}/${step_name}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${step_name}"
  "$@" 2>&1 | tee "${log_path}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${step_name}"
}

echo "[${DATASET}-DIFFPATH] project=${ROOT_DIR}"
echo "[${DATASET}-DIFFPATH] python=${PYTHON_BIN}"
echo "[${DATASET}-DIFFPATH] device=${DEVICE}"
echo "[${DATASET}-DIFFPATH] training_cap=500 (validation early stopping enabled)"
echo "[${DATASET}-DIFFPATH] batch_size=${BATCH_SIZE}"
echo "[${DATASET}-DIFFPATH] recon_batch_size=${RECON_BATCH_SIZE}"
echo "[${DATASET}-DIFFPATH] diffpath_batch_size=${DIFFPATH_BATCH_SIZE}"
echo "[${DATASET}-DIFFPATH] base_seed=${BASE_SEED}"
echo "[${DATASET}-DIFFPATH] diffpath_steps=${DIFFPATH_STEP_VALUES[*]}"
echo "[${DATASET}-DIFFPATH] output_root_base=${OUTPUT_ROOT}"
echo "[${DATASET}-DIFFPATH] overwrite=${OVERWRITE}"
echo "[${DATASET}-DIFFPATH] skip_train=${SKIP_TRAIN}"
echo "[${DATASET}-DIFFPATH] enable_6d_kde=${ENABLE_6D_KDE}"

if [[ ! -f "config/${CONFIG}" ]]; then
  echo "[ERROR] Missing config file: config/${CONFIG}" >&2
  exit 1
fi

for data_file in \
  "data/Machine/${DATASET}_train.pkl" \
  "data/Machine/${DATASET}_test.pkl" \
  "data/Machine/${DATASET}_test_label.pkl"; do
  if [[ ! -f "${data_file}" ]]; then
    echo "[ERROR] Missing ${DATASET} data file: ${data_file}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" -c '
import sys
import pickle
import numpy
import sklearn
import torch
import yaml

dataset, device, feature_dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
train_path, test_path, label_path = sys.argv[4:7]
print(f"[{dataset}-DIFFPATH] numpy {numpy.__version__}")
print(f"[{dataset}-DIFFPATH] sklearn {sklearn.__version__}")
print(f"[{dataset}-DIFFPATH] torch {torch.__version__}")

with open(train_path, "rb") as stream:
    train = numpy.asarray(pickle.load(stream))
with open(test_path, "rb") as stream:
    test = numpy.asarray(pickle.load(stream))
with open(label_path, "rb") as stream:
    labels = numpy.asarray(pickle.load(stream)).reshape(-1)
if train.ndim != 2 or test.ndim != 2:
    raise SystemExit(
        f"[ERROR] {dataset} train/test must be 2-D, got {train.shape} and {test.shape}"
    )
if train.shape[1] != feature_dim or test.shape[1] != feature_dim:
    raise SystemExit(
        f"[ERROR] {dataset} must have {feature_dim} features, got {train.shape[1]} and {test.shape[1]}"
    )
if len(test) != len(labels):
    raise SystemExit(
        f"[ERROR] {dataset} test/label lengths differ: {len(test)} vs {len(labels)}"
    )
if not numpy.isfinite(train).all() or not numpy.isfinite(test).all():
    raise SystemExit(f"[ERROR] {dataset} data contain NaN/Inf")
print(
    f"[{dataset}-DIFFPATH] data train={train.shape} test={test.shape} "
    f"anomalies={int(labels.sum())}"
)

if device.startswith("cuda"):
    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA device requested but torch.cuda.is_available() is false")
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    if index >= torch.cuda.device_count():
        raise SystemExit(
            f"[ERROR] Requested {device}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )
    print(f"[{dataset}-DIFFPATH] gpu {torch.cuda.get_device_name(index)}")
' \
  "${DATASET}" \
  "${DEVICE}" \
  "${FEATURE_DIM}" \
  "data/Machine/${DATASET}_train.pkl" \
  "data/Machine/${DATASET}_test.pkl" \
  "data/Machine/${DATASET}_test_label.pkl"

"${PYTHON_BIN}" -m py_compile \
  exe_machine.py \
  evaluate_machine_window_middle1.py \
  main_model.py \
  diffpath_1d.py \
  tools/evaluate_pathB_diffpath_f1.py

MODEL_DIR_NAME="${MODEL_DATASET}_unconditional:True_task:reconstruction_split:${TRAIN_SPLIT}_diffusion_step:${DIFFUSION_STEPS}"
if [[ "${OVERWRITE}" != "1" ]]; then
  for run_index in 0 1 2; do
    model_dir="train_result/save${run_index}/${MODEL_DIR_NAME}"
    if [[ "${SKIP_TRAIN}" != "1" ]] && [[ -d "${model_dir}" ]] && [[ -n "$(find "${model_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "[ERROR] Existing training output: ${model_dir}" >&2
      echo "Set OVERWRITE=1 only when you intend to replace this ${DATASET} run." >&2
      exit 1
    fi
    run_seed=$((BASE_SEED + run_index))
    for diffpath_steps in "${DIFFPATH_STEP_VALUES[@]}"; do
      step_output_root="${OUTPUT_ROOT}/nfe${diffpath_steps}"
      score_file="${step_output_root}/${RESULT_TAG}/ensemble/diffpath_1d_scores_save${run_index}.npz"
      calibrator_file="${step_output_root}/_prototypes/${PROTO_DATASET}/diffpath_1d_steps${diffpath_steps}_seed${run_seed}_save${run_index}.npz"
      calibrator_6d_file="${step_output_root}/_prototypes/${PROTO_DATASET}/diffpath_6d_steps${diffpath_steps}_seed${run_seed}_save${run_index}.npz"
      calibrator_6d_gmm_file="${step_output_root}/_prototypes/${PROTO_DATASET}/diffpath_6d_gmm_steps${diffpath_steps}_seed${run_seed}_save${run_index}.npz"
      for existing_output in "${score_file}" "${calibrator_file}" "${calibrator_6d_file}" "${calibrator_6d_gmm_file}"; do
        if [[ -e "${existing_output}" ]]; then
          echo "[ERROR] Existing DiffPath output: ${existing_output}" >&2
          echo "Set OVERWRITE=1 or choose another OUTPUT_ROOT." >&2
          exit 1
        fi
      done
    done
  done
  for diffpath_steps in "${DIFFPATH_STEP_VALUES[@]}"; do
    step_summary_csv="${OUTPUT_ROOT}/nfe${diffpath_steps}/${DATASET}_diffpath_f1_summary.csv"
    if [[ -e "${step_summary_csv}" ]]; then
      echo "[ERROR] Existing summary output: ${step_summary_csv}" >&2
      echo "Set OVERWRITE=1 or choose another OUTPUT_ROOT." >&2
      exit 1
    fi
  done
  if [[ -e "${SUMMARY_CSV}" ]]; then
    echo "[ERROR] Existing summary output: ${SUMMARY_CSV}" >&2
    echo "Set OVERWRITE=1 or choose another SUMMARY_CSV." >&2
    exit 1
  fi
fi

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "[${DATASET}-DIFFPATH] SKIP_TRAIN=1, reusing existing train_result/save0..save2 checkpoints."
else
  for run_index in 0 1 2; do
    run_step "train_${DATASET}_save${run_index}" \
      "${PYTHON_BIN}" exe_machine.py \
      --config "${CONFIG}" \
      --device "${DEVICE}" \
      --dataset "${MODEL_DATASET}" \
      --task_mode reconstruction \
      --batch_size "${BATCH_SIZE}" \
      --num_runs 1 \
      --run_start "${run_index}" \
      --seed "${BASE_SEED}" \
      --split "${TRAIN_SPLIT}" \
      --diffusion_step "${DIFFUSION_STEPS}" \
      "${OVERWRITE_TRAIN_ARGS[@]}"
  done
fi

for run_index in 0 1 2; do
  checkpoint="train_result/save${run_index}/${MODEL_DIR_NAME}/best-model.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

for diffpath_steps in "${DIFFPATH_STEP_VALUES[@]}"; do
  step_output_root="${OUTPUT_ROOT}/nfe${diffpath_steps}"
  step_summary_csv="${step_output_root}/${DATASET}_diffpath_f1_summary.csv"
  for run_index in 0 1 2; do
    run_step "diffpath_${DATASET}_nfe${diffpath_steps}_save${run_index}" \
      "${PYTHON_BIN}" evaluate_machine_window_middle1.py \
      --config "${CONFIG}" \
      --device "${DEVICE}" \
      --dataset "${DATASET}" \
      --model_dataset "${MODEL_DATASET}" \
      --result_tag "${RESULT_TAG}" \
      --pathB_mode diffpath \
      --pathB_proto_dataset "${PROTO_DATASET}" \
      --pathB_output_root "${step_output_root}" \
      --saves "save${run_index}" \
      --diffpath_num_steps "${diffpath_steps}" \
      --reconstruction_batch_size "${RECON_BATCH_SIZE}" \
      --diffpath_batch_size "${DIFFPATH_BATCH_SIZE}" \
      --diffpath_kde_bandwidths "${KDE_BANDWIDTHS}" \
      --diffpath_6d_kde_bandwidths "${KDE_BANDWIDTHS_6D}" \
      --diffpath_6d_gmm_components "${GMM_COMPONENTS_6D}" \
      --diffpath_6d_gmm_covariance_types "${GMM_COVARIANCE_TYPES_6D}" \
      --seed "${BASE_SEED}" \
      "${KDE_6D_ARGS[@]}" \
      "${OVERWRITE_INFERENCE_ARGS[@]}"
  done

  for run_index in 0 1 2; do
    score_file="${step_output_root}/${RESULT_TAG}/ensemble/diffpath_1d_scores_save${run_index}.npz"
    if [[ ! -f "${score_file}" ]]; then
      echo "[ERROR] Missing DiffPath score bundle: ${score_file}" >&2
      exit 1
    fi
  done

  mkdir -p "$(dirname "${step_summary_csv}")"
  run_step "evaluate_${DATASET}_f1_nfe${diffpath_steps}" \
    "${PYTHON_BIN}" tools/evaluate_pathB_diffpath_f1.py \
    --pathB_output_root "${step_output_root}" \
    --datasets "${RESULT_TAG}" \
    --saves save0 save1 save2 \
    --alpha_values "${ALPHA_VALUES}" \
    --out "${step_summary_csv}"
done

mkdir -p "$(dirname "${SUMMARY_CSV}")"
"${PYTHON_BIN}" -c '
import csv
import os
import sys

out_path, root, dataset = sys.argv[1:4]
steps = sys.argv[4:]
rows = []
fieldnames = None
for step in steps:
    summary_path = os.path.join(
        root,
        f"nfe{step}",
        f"{dataset}_diffpath_f1_summary.csv",
    )
    with open(summary_path, newline="") as stream:
        reader = csv.DictReader(stream)
        if fieldnames is None:
            fieldnames = ["nfe"] + list(reader.fieldnames)
        for row in reader:
            row = {"nfe": step, **row}
            rows.append(row)
with open(out_path, "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[{dataset}-DIFFPATH] Combined NFE summary: {out_path}")
' \
  "${SUMMARY_CSV}" \
  "${OUTPUT_ROOT}" \
  "${DATASET}" \
  "${DIFFPATH_STEP_VALUES[@]}"

echo
echo "[${DATASET}-DIFFPATH] All stages completed."
for diffpath_steps in "${DIFFPATH_STEP_VALUES[@]}"; do
  echo "[${DATASET}-DIFFPATH] NFE ${diffpath_steps} output: ${OUTPUT_ROOT}/nfe${diffpath_steps}"
done
echo "[${DATASET}-DIFFPATH] Combined summary: ${SUMMARY_CSV}"
echo "[${DATASET}-DIFFPATH] Logs: ${LOG_DIR}"
echo
cat "${SUMMARY_CSV}"
