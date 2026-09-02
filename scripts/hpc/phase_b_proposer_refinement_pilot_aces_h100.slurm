#!/bin/bash
# Run one feedback-rich proposer refinement task on two ACES H100 GPUs.
#SBATCH --job-name=proposer-refinement
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:2
#SBATCH --requeue

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_TARGET_CONTRACT_ROOT:?AF_TARGET_CONTRACT_ROOT is required}"
: "${AF_MECHANISM_SPEC_ROOT:?AF_MECHANISM_SPEC_ROOT is required}"
: "${AF_JUDGE_PROTOCOL:?AF_JUDGE_PROTOCOL is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_REFINEMENT_CONFIG:?AF_REFINEMENT_CONFIG is required}"
: "${AF_VLLM_IMAGE:?AF_VLLM_IMAGE is required}"
: "${AF_HF_HOME:=${SCRATCH:?SCRATCH is required}/huggingface-cache}"
: "${AF_TENSOR_PARALLEL_SIZE:=2}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly frozen_root="${AF_OUTPUT_ROOT}/frozen"
readonly plan="${frozen_root}/plan.json"
readonly task_plan="${frozen_root}/task_plan.jsonl"
readonly freeze_manifest="${frozen_root}/freeze_manifest.json"
readonly task_line=$((AF_TASK_INDEX + 1))

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in \
  "${AF_PYTHON}" \
  "${AF_REFINEMENT_CONFIG}" \
  "${AF_JUDGE_PROTOCOL}" \
  "${AF_VLLM_IMAGE}" \
  "${plan}" \
  "${task_plan}" \
  "${freeze_manifest}"; do
  [[ -e "${path}" ]] || { echo "missing refinement input: ${path}" >&2; exit 2; }
done
for directory in \
  "${AF_PUBLIC_DATA_ROOT}" \
  "${AF_TARGET_CONTRACT_ROOT}" \
  "${AF_MECHANISM_SPEC_ROOT}"; do
  [[ -d "${directory}" ]] || {
    echo "missing refinement directory: ${directory}" >&2
    exit 2
  }
done
cmp --silent "${AF_REFINEMENT_CONFIG}" "${plan}" || {
  echo "repository refinement plan differs from frozen plan" >&2
  exit 2
}
(
  cd "${frozen_root}"
  sha256sum -c plan.json.sha256
  sha256sum -c task_plan.jsonl.sha256
  sha256sum -c freeze_manifest.json.sha256
)

readonly task_json="$(sed -n "${task_line}p" "${task_plan}")"
[[ -n "${task_json}" ]] || { echo "missing task ${AF_TASK_INDEX}" >&2; exit 2; }
[[ "$(jq -r '.task_index' <<<"${task_json}")" == "${AF_TASK_INDEX}" ]] || {
  echo "task ledger index mismatch" >&2
  exit 2
}
readonly arm_id="$(jq -r '.arm_id' <<<"${task_json}")"
readonly benchmark_id="$(jq -r '.benchmark_id' <<<"${task_json}")"
readonly tier="$(jq -r '.tier' <<<"${task_json}")"
readonly repetition="$(jq -r '.repetition' <<<"${task_json}")"
readonly feedback_mode="$(jq -r '.proposer_feedback_mode' <<<"${task_json}")"
readonly proposal_policy="$(jq -r '.proposal_policy' <<<"${task_json}")"
readonly require_cache="$(jq -r '.require_shared_round_zero_cache_hit' <<<"${task_json}")"
readonly expected_prompt_sha="$(jq -r '.public_prompt_sha256' <<<"${task_json}")"
readonly expected_target_sha="$(jq -r '.public_target_contract_sha256' <<<"${task_json}")"
readonly expected_mechanism_sha="$(jq -r '.public_mechanism_spec_sha256' <<<"${task_json}")"

readonly prompt_path="${AF_PUBLIC_DATA_ROOT}/phase_b_v1/${benchmark_id}/proposer_prompt.txt"
readonly target_contract="${AF_TARGET_CONTRACT_ROOT}/specs/${benchmark_id}.json"
readonly mechanism_spec="${AF_MECHANISM_SPEC_ROOT}/specs/${benchmark_id}.json"
for path in "${prompt_path}" "${target_contract}" "${mechanism_spec}"; do
  [[ -f "${path}" ]] || { echo "missing public task input: ${path}" >&2; exit 2; }
done
[[ "$(sha256sum "${prompt_path}" | awk '{print $1}')" == "${expected_prompt_sha}" ]] || {
  echo "public prompt differs from frozen task" >&2
  exit 2
}
[[ "$(sha256sum "${target_contract}" | awk '{print $1}')" == "${expected_target_sha}" ]] || {
  echo "target contract differs from frozen task" >&2
  exit 2
}
[[ "$(sha256sum "${mechanism_spec}" | awk '{print $1}')" == "${expected_mechanism_sha}" ]] || {
  echo "mechanism specification differs from frozen task" >&2
  exit 2
}
readonly expected_judge_sha="$(jq -r '.model_contract.judge_protocol_config_sha256' "${plan}")"
[[ "$(sha256sum "${AF_JUDGE_PROTOCOL}" | awk '{print $1}')" == "${expected_judge_sha}" ]] || {
  echo "judge protocol differs from frozen plan" >&2
  exit 2
}

readonly model="$(jq -r '.model_contract.model' "${plan}")"
readonly proposer_reasoning="$(jq -r '.model_contract.proposer_reasoning_effort' "${plan}")"
readonly judge_reasoning="$(jq -r '.model_contract.judge_reasoning_effort' "${plan}")"
readonly temperature="$(jq -r '.model_contract.temperature' "${plan}")"
readonly output_tokens="$(jq -r '.model_contract.proposer_max_output_tokens' "${plan}")"
readonly served_context="$(jq -r '.model_contract.served_context_tokens' "${plan}")"
readonly request_timeout="$(jq -r '.model_contract.request_timeout_seconds' "${plan}")"
readonly iterations="$(jq -r '.search_budget.iteration_budget' "${plan}")"
readonly beam_size="$(jq -r '.search_budget.beam_size' "${plan}")"
readonly fit_starts="$(jq -r '.search_budget.fit_starts' "${plan}")"
readonly fit_nfev="$(jq -r '.search_budget.fit_max_nfev' "${plan}")"
readonly fit_timeout="$(jq -r '.search_budget.fit_timeout_seconds' "${plan}")"
readonly retry_starts="$(jq -r '.search_budget.fit_retry_starts' "${plan}")"
readonly retry_nfev="$(jq -r '.search_budget.fit_retry_max_nfev' "${plan}")"
readonly retry_timeout="$(jq -r '.search_budget.fit_retry_timeout_seconds' "${plan}")"
readonly final_nfev="$(jq -r '.search_budget.final_fit_max_nfev' "${plan}")"
readonly final_timeout="$(jq -r '.search_budget.final_fit_timeout_seconds' "${plan}")"
readonly fit_strategy="$(jq -r '.search_budget.parameter_fit_strategy' "${plan}")"

readonly run_name="${benchmark_id}_${tier}_seed${repetition}"
readonly arm_output_root="${AF_OUTPUT_ROOT}/searches/${arm_id}"
readonly run_root="${arm_output_root}/runs/${run_name}"
readonly shared_cache="${AF_OUTPUT_ROOT}/shared-round-zero-cache/${run_name}"
readonly process_time="${run_root}/search_process_time.txt"
readonly gpu_inventory="${run_root}/gpu_inventory.txt"
readonly started_epoch_seconds="$(date +%s.%N)"
readonly endpoint_port="$((8100 + AF_TASK_INDEX))"
readonly endpoint="http://127.0.0.1:${endpoint_port}"
readonly server_log="${AF_OUTPUT_ROOT}/vllm-${SLURM_JOB_ID}.log"

module load WebProxy
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
mkdir -p \
  "${AF_REPO_ROOT}/logs" \
  "${AF_OUTPUT_ROOT}/runtime" \
  "${AF_HF_HOME}" \
  "${shared_cache}" \
  "${run_root}"
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
  "${AF_PYTHON}" scripts/write_phase_b_search_task_runtime.py \
    --output "${run_root}/task_runtime.json" \
    --task-index "${AF_TASK_INDEX}" \
    --arm-id "${arm_id}" \
    --benchmark-id "${benchmark_id}" \
    --tier "${tier}" \
    --repetition "${repetition}" \
    --started-epoch-seconds "${started_epoch_seconds}" \
    --exit-code "${status}" \
    --allocated-cpus "${SLURM_CPUS_PER_TASK:-16}" \
    --allocated-gpus "${AF_TENSOR_PARALLEL_SIZE}" \
    --gpu-inventory "${gpu_inventory}" \
    --process-time "${process_time}" >/dev/null
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  | tee "${gpu_inventory}"
"${container_runtime}" exec --nv \
  --bind "${AF_HF_HOME}:${AF_HF_HOME}" \
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
  if curl --noproxy "*" --silent --fail "${endpoint}/v1/models" >/dev/null; then
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
[[ "${ready}" == true ]] || { echo "vLLM readiness timed out" >&2; exit 1; }

resume_arguments=()
if [[ -f "${run_root}/checkpoints/run.json" ]]; then
  resume_arguments=(--resume)
fi
cache_arguments=()
if [[ "${require_cache}" == true ]]; then
  cache_arguments=(--require-initial-proposer-cache-hit)
fi

/usr/bin/time -o "${process_time}" \
  -f $'elapsed_seconds=%e\nuser_cpu_seconds=%U\nsystem_cpu_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' \
  "${AF_PYTHON}" scripts/run_experiment.py \
    --data-root "${AF_PUBLIC_DATA_ROOT}" \
    --benchmark-id "${benchmark_id}" \
    --tier "${tier}" \
    --seed "${repetition}" \
    --proposer-model "vllm:${model}" \
    --judge-model "vllm:${model}" \
    --vllm-base-url "${endpoint}" \
    --vllm-proposer-reasoning-effort "${proposer_reasoning}" \
    --vllm-judge-reasoning-effort "${judge_reasoning}" \
    --vllm-temperature "${temperature}" \
    --vllm-seed "${repetition}" \
    --development-only \
    --selection-policy incumbent_relative_hybrid \
    --hybrid-science-weight 0.5 \
    --hybrid-judge-seed-base "$((12000 + 2 * repetition))" \
    --iteration-budget "${iterations}" \
    --stagnation-iterations "${iterations}" \
    --beam-size "${beam_size}" \
    --llm-timeout-seconds "${request_timeout}" \
    --llm-max-output-tokens "${output_tokens}" \
    --llm-cache-root "${shared_cache}" \
    --fit-starts "${fit_starts}" \
    --fit-max-nfev "${fit_nfev}" \
    --fit-timeout-seconds "${fit_timeout}" \
    --fit-retry-starts "${retry_starts}" \
    --fit-retry-max-nfev "${retry_nfev}" \
    --fit-retry-timeout-seconds "${retry_timeout}" \
    --final-fit-max-nfev "${final_nfev}" \
    --final-fit-timeout-seconds "${final_timeout}" \
    --parameter-fit-strategy "${fit_strategy}" \
    --public-target-contract "${target_contract}" \
    --public-mechanism-spec "${mechanism_spec}" \
    --proposer-feedback-mode "${feedback_mode}" \
    --proposal-policy "${proposal_policy}" \
    --output-root "${arm_output_root}/runs" \
    "${cache_arguments[@]}" \
    "${resume_arguments[@]}"
