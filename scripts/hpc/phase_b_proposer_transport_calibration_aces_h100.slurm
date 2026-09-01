#!/bin/bash
#SBATCH --job-name=phase-b-proposer-cal
#SBATCH --partition=gpu
#SBATCH --array=0-5%2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:2
#SBATCH --requeue
#SBATCH --output=logs/phase-b-proposer-cal-%A_%a.out
#SBATCH --error=logs/phase-b-proposer-cal-%A_%a.err

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_PYTHON_MODULE:=Python/3.11.5-GCCcore-13.2.0}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_TARGET_CONTRACT_ROOT:?AF_TARGET_CONTRACT_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_CALIBRATION_CONFIG:?AF_CALIBRATION_CONFIG is required}"
: "${AF_VLLM_IMAGE:?AF_VLLM_IMAGE is required}"
: "${AF_HF_HOME:=${SCRATCH:?SCRATCH is required}/huggingface-cache}"
: "${AF_TENSOR_PARALLEL_SIZE:=2}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly frozen_root="${AF_OUTPUT_ROOT}/frozen"
readonly plan="${frozen_root}/plan.json"
readonly task_plan="${frozen_root}/task_plan.jsonl"
readonly freeze_manifest="${frozen_root}/freeze_manifest.json"
readonly endpoint_port="${AF_VLLM_PORT:-$((8100 + AF_TASK_INDEX))}"
readonly endpoint="http://127.0.0.1:${endpoint_port}"
readonly model="$(jq -r '.model_contract.model' "${plan}")"
readonly served_context="$(jq -r '.model_contract.served_context_tokens' "${plan}")"
readonly server_log="${AF_OUTPUT_ROOT}/vllm-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${AF_TASK_INDEX}.log"
readonly gpu_inventory="${AF_OUTPUT_ROOT}/runtime/task_${AF_TASK_INDEX}_gpu_inventory.txt"

module load "${AF_PYTHON_MODULE}"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data: ${AF_PUBLIC_DATA_ROOT}" >&2; exit 2; }
[[ -d "${AF_TARGET_CONTRACT_ROOT}" ]] || { echo "missing target contracts: ${AF_TARGET_CONTRACT_ROOT}" >&2; exit 2; }
[[ -f "${AF_CALIBRATION_CONFIG}" ]] || { echo "missing calibration config" >&2; exit 2; }
[[ -s "${AF_VLLM_IMAGE}" ]] || { echo "missing vLLM image: ${AF_VLLM_IMAGE}" >&2; exit 2; }
[[ -f "${plan}" && -f "${task_plan}" && -f "${freeze_manifest}" ]] || {
  echo "missing frozen calibration inputs" >&2
  exit 2
}
cmp --silent "${AF_CALIBRATION_CONFIG}" "${plan}" || {
  echo "repository calibration plan differs from frozen plan" >&2
  exit 2
}
(
  cd "${frozen_root}"
  sha256sum -c plan.json.sha256
  sha256sum -c task_plan.jsonl.sha256
  sha256sum -c freeze_manifest.json.sha256
)

container_runtime=""
if command -v apptainer >/dev/null 2>&1; then
  container_runtime="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  container_runtime="singularity"
else
  echo "neither Apptainer nor Singularity is available" >&2
  exit 2
fi

module load WebProxy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}/runtime" "${AF_HF_HOME}"
cd "${AF_REPO_ROOT}"

server_pid=""
cleanup() {
  local status=$?
  trap - EXIT
  set +e
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  | tee "${gpu_inventory}"
"${container_runtime}" exec --nv \
  --bind "${AF_REPO_ROOT}:${AF_REPO_ROOT},${AF_PUBLIC_DATA_ROOT}:${AF_PUBLIC_DATA_ROOT},${AF_OUTPUT_ROOT}:${AF_OUTPUT_ROOT},${AF_HF_HOME}:${AF_HF_HOME}" \
  --env "HF_HOME=${AF_HF_HOME}" \
  "${AF_VLLM_IMAGE}" \
  vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${endpoint_port}" \
    --max-model-len "${served_context}" \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size "${AF_TENSOR_PARALLEL_SIZE}" \
    >"${server_log}" 2>&1 &
server_pid=$!

ready=false
for _ in $(seq 1 1200); do
  if curl --silent --fail "${endpoint}/v1/models" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "vLLM exited before becoming ready; see ${server_log}" >&2
    tail -n 100 "${server_log}" >&2
    exit 1
  fi
  sleep 2
done
if [[ "${ready}" != true ]]; then
  echo "vLLM did not become ready; see ${server_log}" >&2
  exit 1
fi

"${AF_PYTHON}" scripts/run_phase_b_proposer_transport_calibration.py \
  --plan "${plan}" \
  --task-index "${AF_TASK_INDEX}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}" \
  --vllm-base-url "${endpoint}"
