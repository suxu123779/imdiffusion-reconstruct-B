#!/usr/bin/env bash
set -Eeuo pipefail

# Complete SMAP pipeline:
# three independent training processes -> DiffPath scoring -> per-save F1 -> mean/std.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-base.yaml}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_RUNS="${NUM_RUNS:-3}"
BASE_SEED="${BASE_SEED:-1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-10}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
DIFFPATH_STEPS="${DIFFPATH_STEPS:-10}"
KDE_BANDWIDTHS="${KDE_BANDWIDTHS:-0.05,0.1,0.2,0.5,1.0}"
KDE_BANDWIDTHS_6D="${KDE_BANDWIDTHS_6D:-0.2,0.5,1.0,2.0,5.0}"
ALPHA_VALUES="${ALPHA_VALUES:-0:1:0.05}"
OUTPUT_ROOT="${OUTPUT_ROOT:-pathB_result_diffpath}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/SMAP_diffpath_f1_summary.csv}"
LOG_DIR="${LOG_DIR:-run_logs/smap_diffpath}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${NUM_RUNS}" != "3" ]]; then
  echo "[ERROR] The requested repeated-experiment protocol requires NUM_RUNS=3." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

OVERWRITE_TRAIN_ARGS=()
OVERWRITE_INFERENCE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_TRAIN_ARGS+=(--overwrite)
  OVERWRITE_INFERENCE_ARGS+=(
    --overwrite
    --diffpath_recompute_calibrator
  )
fi

run_step() {
  local step_name="$1"
  shift
  local log_path="${LOG_DIR}/${step_name}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${step_name}"
  "$@" 2>&1 | tee "${log_path}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${step_name}"
}

echo "[SMAP-DIFFPATH] project=${ROOT_DIR}"
echo "[SMAP-DIFFPATH] python=${PYTHON_BIN}"
echo "[SMAP-DIFFPATH] device=${DEVICE}"
echo "[SMAP-DIFFPATH] training_cap=500 (validation early stopping enabled)"
echo "[SMAP-DIFFPATH] batch_size=${BATCH_SIZE}"
echo "[SMAP-DIFFPATH] base_seed=${BASE_SEED}"
echo "[SMAP-DIFFPATH] output_root=${OUTPUT_ROOT}"
echo "[SMAP-DIFFPATH] overwrite=${OVERWRITE}"

if [[ ! -f "config/${CONFIG}" ]]; then
  echo "[ERROR] Missing config file: config/${CONFIG}" >&2
  exit 1
fi

for data_file in \
  data/Machine/SMAP_train.pkl \
  data/Machine/SMAP_test.pkl \
  data/Machine/SMAP_test_label.pkl; do
  if [[ ! -f "${data_file}" ]]; then
    echo "[ERROR] Missing SMAP data file: ${data_file}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" -c '
import sys
import pickle
import numpy
import sklearn
import torch
import tqdm
import yaml

device = sys.argv[1]
train_path, test_path, label_path = sys.argv[2:5]
print("[SMAP-DIFFPATH] numpy", numpy.__version__)
print("[SMAP-DIFFPATH] sklearn", sklearn.__version__)
print("[SMAP-DIFFPATH] torch", torch.__version__)

with open(train_path, "rb") as stream:
    train = numpy.asarray(pickle.load(stream))
with open(test_path, "rb") as stream:
    test = numpy.asarray(pickle.load(stream))
with open(label_path, "rb") as stream:
    labels = numpy.asarray(pickle.load(stream)).reshape(-1)
if train.ndim != 2 or test.ndim != 2:
    raise SystemExit(
        f"[ERROR] SMAP train/test must be 2-D, got {train.shape} and {test.shape}"
    )
if train.shape[1] != 25 or test.shape[1] != 25:
    raise SystemExit(
        f"[ERROR] SMAP must have 25 features, got {train.shape[1]} and {test.shape[1]}"
    )
if len(test) != len(labels):
    raise SystemExit(
        f"[ERROR] SMAP test/label lengths differ: {len(test)} vs {len(labels)}"
    )
if not numpy.isfinite(train).all() or not numpy.isfinite(test).all():
    raise SystemExit("[ERROR] SMAP data contain NaN/Inf")
print(
    f"[SMAP-DIFFPATH] data train={train.shape} test={test.shape} "
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
    print("[SMAP-DIFFPATH] gpu", torch.cuda.get_device_name(index))
' \
  "${DEVICE}" \
  data/Machine/SMAP_train.pkl \
  data/Machine/SMAP_test.pkl \
  data/Machine/SMAP_test_label.pkl

"${PYTHON_BIN}" -m py_compile \
  exe_machine.py \
  evaluate_machine_window_middle1.py \
  main_model.py \
  diffpath_1d.py \
  tools/evaluate_pathB_diffpath_f1.py

MODEL_DIR_NAME="SMAP_unconditional:True_task:reconstruction_split:${TRAIN_SPLIT}_diffusion_step:${DIFFUSION_STEPS}"
if [[ "${OVERWRITE}" != "1" ]]; then
  for run_index in 0 1 2; do
    model_dir="train_result/save${run_index}/${MODEL_DIR_NAME}"
    if [[ -d "${model_dir}" ]] && [[ -n "$(find "${model_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "[ERROR] Existing training output: ${model_dir}" >&2
      echo "Set OVERWRITE=1 only when you intend to replace this SMAP run." >&2
      exit 1
    fi
    score_file="${OUTPUT_ROOT}/SMAP/ensemble/diffpath_1d_scores_save${run_index}.npz"
    run_seed=$((BASE_SEED + run_index))
    calibrator_file="${OUTPUT_ROOT}/_prototypes/SMAP/diffpath_1d_steps${DIFFPATH_STEPS}_seed${run_seed}_save${run_index}.npz"
    calibrator_6d_file="${OUTPUT_ROOT}/_prototypes/SMAP/diffpath_6d_steps${DIFFPATH_STEPS}_seed${run_seed}_save${run_index}.npz"
    for existing_output in "${score_file}" "${calibrator_file}" "${calibrator_6d_file}"; do
      if [[ -e "${existing_output}" ]]; then
        echo "[ERROR] Existing DiffPath output: ${existing_output}" >&2
        echo "Set OVERWRITE=1 or choose another OUTPUT_ROOT." >&2
        exit 1
      fi
    done
  done
  if [[ -e "${SUMMARY_CSV}" ]]; then
    echo "[ERROR] Existing summary output: ${SUMMARY_CSV}" >&2
    echo "Set OVERWRITE=1 or choose another SUMMARY_CSV." >&2
    exit 1
  fi
fi

for run_index in 0 1 2; do
  run_step "train_save${run_index}" \
    "${PYTHON_BIN}" exe_machine.py \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --dataset SMAP \
    --task_mode reconstruction \
    --batch_size "${BATCH_SIZE}" \
    --num_runs 1 \
    --run_start "${run_index}" \
    --seed "${BASE_SEED}" \
    --split "${TRAIN_SPLIT}" \
    --diffusion_step "${DIFFUSION_STEPS}" \
    "${OVERWRITE_TRAIN_ARGS[@]}"
done

for run_index in 0 1 2; do
  checkpoint="train_result/save${run_index}/${MODEL_DIR_NAME}/best-model.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] Training finished without checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

for run_index in 0 1 2; do
  run_step "diffpath_save${run_index}" \
    "${PYTHON_BIN}" evaluate_machine_window_middle1.py \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --dataset SMAP \
    --model_dataset SMAP \
    --result_tag SMAP \
    --pathB_mode diffpath \
    --pathB_proto_dataset SMAP \
    --pathB_output_root "${OUTPUT_ROOT}" \
    --saves "save${run_index}" \
    --diffpath_num_steps "${DIFFPATH_STEPS}" \
    --diffpath_kde_bandwidths "${KDE_BANDWIDTHS}" \
    --diffpath_6d_kde_bandwidths "${KDE_BANDWIDTHS_6D}" \
    --seed "${BASE_SEED}" \
    "${OVERWRITE_INFERENCE_ARGS[@]}"
done

for run_index in 0 1 2; do
  score_file="${OUTPUT_ROOT}/SMAP/ensemble/diffpath_1d_scores_save${run_index}.npz"
  if [[ ! -f "${score_file}" ]]; then
    echo "[ERROR] Missing DiffPath score bundle: ${score_file}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${SUMMARY_CSV}")"
run_step evaluate_f1 \
  "${PYTHON_BIN}" tools/evaluate_pathB_diffpath_f1.py \
  --pathB_output_root "${OUTPUT_ROOT}" \
  --datasets SMAP \
  --saves save0 save1 save2 \
  --alpha_values "${ALPHA_VALUES}" \
  --out "${SUMMARY_CSV}"

echo
echo "[SMAP-DIFFPATH] All SMAP stages completed."
echo "[SMAP-DIFFPATH] Summary: ${SUMMARY_CSV}"
echo "[SMAP-DIFFPATH] Logs: ${LOG_DIR}"
echo
cat "${SUMMARY_CSV}"
