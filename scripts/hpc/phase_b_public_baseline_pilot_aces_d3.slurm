#!/bin/bash
#SBATCH --job-name=phase-b-baseline-d3
#SBATCH --partition=gpu
#SBATCH --time=02:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:2
#SBATCH --requeue

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_PYTHON_MODULE:=Python/3.11.5-GCCcore-13.2.0}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_VLLM_IMAGE:?AF_VLLM_IMAGE is required}"
: "${AF_HF_HOME:=${SCRATCH:?SCRATCH is required}/huggingface-cache}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${AF_TENSOR_PARALLEL_SIZE:=2}"

readonly task_plan="${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl"
readonly task="$(sed -n "$((AF_TASK_INDEX + 1))p" "${task_plan}")"
readonly operating_point="${AF_OUTPUT_ROOT}/frozen/d3_llm_operating_point.json"
[[ -n "${task}" ]] || { echo "missing baseline task ${AF_TASK_INDEX}" >&2; exit 2; }
[[ "$(jq -r '.method' <<<"${task}")" == "d3_native_no_tools" ]] || {
  echo "task ${AF_TASK_INDEX} is not a D3 task" >&2
  exit 2
}
[[ -f "${operating_point}" ]] || { echo "missing D3 operating point" >&2; exit 2; }
readonly benchmark_id="$(jq -r '.benchmark_id' <<<"${task}")"
readonly tier="$(jq -r '.tier' <<<"${task}")"
readonly repetition="$(jq -r '.repetition' <<<"${task}")"
readonly timeout="$(jq -r '.wall_timeout_seconds' <<<"${task}")"
readonly generations="$(jq -r '.d3_generations' <<<"${task}")"
readonly patience="$(jq -r '.d3_patience' <<<"${task}")"
readonly model="$(jq -r '.model' "${operating_point}")"
readonly reasoning="$(jq -r '.reasoning_effort' "${operating_point}")"
readonly output_tokens="$(jq -r '.max_output_tokens' "${operating_point}")"
readonly endpoint_port="${AF_VLLM_PORT:-$((8300 + AF_TASK_INDEX))}"
readonly endpoint="http://127.0.0.1:${endpoint_port}"
readonly server_log="${AF_OUTPUT_ROOT}/vllm-d3-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${AF_TASK_INDEX}.log"

module load "${AF_PYTHON_MODULE}"
module load WebProxy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}/runs" "${AF_HF_HOME}"
cd "${AF_REPO_ROOT}"

container_runtime=""
if command -v apptainer >/dev/null 2>&1; then
  container_runtime="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  container_runtime="singularity"
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

"${container_runtime}" exec --nv \
  --bind "${AF_REPO_ROOT}:${AF_REPO_ROOT},${AF_PUBLIC_DATA_ROOT}:${AF_PUBLIC_DATA_ROOT},${AF_OUTPUT_ROOT}:${AF_OUTPUT_ROOT},${AF_HF_HOME}:${AF_HF_HOME}" \
  --env "HF_HOME=${AF_HF_HOME}" \
  "${AF_VLLM_IMAGE}" \
  vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${endpoint_port}" \
    --max-model-len 32768 \
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
[[ "${ready}" == true ]] || { echo "vLLM did not become ready" >&2; exit 1; }

"${AF_PYTHON}" scripts/run_baseline.py \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --benchmark-id "${benchmark_id}" \
  --tier "${tier}" \
  --method d3_native_no_tools \
  --seed "${repetition}" \
  --model "vllm:${model}" \
  --output-root "${AF_OUTPUT_ROOT}/runs" \
  --wall-timeout-seconds "${timeout}" \
  --d3-generations "${generations}" \
  --d3-patience "${patience}" \
  --llm-max-output-tokens "${output_tokens}" \
  --llm-timeout-seconds 1200 \
  --llm-max-attempts 3 \
  --vllm-base-url "${endpoint}" \
  --vllm-reasoning-effort "${reasoning}" \
  --vllm-temperature 0.2 \
  --development-only
