#!/bin/bash
# Run one staged-search task on four Delta A40 GPUs.
#SBATCH --job-name=staged-search
#SBATCH --account=bibo-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine Slurm user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_MECHANISM_SPEC_ROOT:=${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/staged-search-pilot-v1-delta-a40x4}"
: "${AF_STAGED_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_staged_search_pilot_v1.json}"
: "${AF_VLLM_IMAGE:=${AF_PROJECT}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_HF_HOME:=${AF_PROJECT}/huggingface-cache}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly plan="${AF_OUTPUT_ROOT}/frozen/plan.json"
readonly model="$(jq -r '.model_contract.model' "${plan}")"
readonly context_tokens="$(jq -r '.model_contract.served_context_tokens' "${plan}")"
readonly endpoint_port=8100
readonly endpoint="http://127.0.0.1:${endpoint_port}"
readonly server_log="${AF_OUTPUT_ROOT}/runtime/vllm-${SLURM_JOB_ID}.log"

mkdir -p "${AF_OUTPUT_ROOT}/runtime" "${AF_HF_HOME}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
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
apptainer exec --nv \
  --bind "${AF_PROJECT}:${AF_PROJECT},${AF_WORK}:${AF_WORK}" \
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
  if curl --silent --fail "${endpoint}/v1/models" >/dev/null; then
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
bash "${AF_REPO_ROOT}/scripts/hpc/run_phase_b_staged_search_task.sh"
