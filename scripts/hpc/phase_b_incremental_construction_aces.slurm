#!/bin/bash
# Run one incremental-construction task on two ACES H100 GPUs.
#SBATCH --job-name=incremental-construction
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:2
#SBATCH --requeue

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_VLLM_IMAGE:?AF_VLLM_IMAGE is required}"
: "${AF_HF_HOME:?AF_HF_HOME is required}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_TENSOR_PARALLEL_SIZE:=2}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}" WebProxy
readonly plan="${AF_OUTPUT_ROOT}/frozen/plan.json"
readonly model="$(jq -r '.model_contract.model' "${plan}")"
readonly context_tokens="$(jq -r '.model_contract.served_context_tokens' "${plan}")"
readonly endpoint_port=8100
readonly endpoint="http://127.0.0.1:${endpoint_port}"
readonly server_log="${AF_OUTPUT_ROOT}/runtime/vllm-${SLURM_JOB_ID}.log"

mkdir -p "${AF_OUTPUT_ROOT}/runtime" "${AF_HF_HOME}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

runtime=""
if command -v apptainer >/dev/null 2>&1; then
  runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
  runtime=singularity
else
  echo "neither Apptainer nor Singularity is available" >&2
  exit 2
fi
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
  >"${AF_OUTPUT_ROOT}/runtime/gpu-${SLURM_JOB_ID}.txt"
"${runtime}" exec --nv \
  --bind "${AF_HF_HOME}:${AF_HF_HOME}" \
  --env "HF_HOME=${AF_HF_HOME}" \
  "${AF_VLLM_IMAGE}" \
  vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${endpoint_port}" \
    --max-model-len "${context_tokens}" \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size "${AF_TENSOR_PARALLEL_SIZE}" \
    >"${server_log}" 2>&1 &
server_pid=$!

ready=false
for _ in $(seq 1 1200); do
  if curl --noproxy "*" --silent --fail "${endpoint}/v1/models" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -n 100 "${server_log}" >&2
    exit 1
  fi
  sleep 2
done
[[ "${ready}" == true ]] || { echo "vLLM readiness timed out" >&2; exit 1; }

export AF_VLLM_ENDPOINT="${endpoint}"
bash "${AF_REPO_ROOT}/scripts/hpc/run_phase_b_incremental_construction_task.sh"
