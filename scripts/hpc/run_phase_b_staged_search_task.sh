#!/bin/bash
# Run one frozen staged-search task against an already-ready vLLM endpoint.

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_TARGET_CONTRACT_ROOT:?AF_TARGET_CONTRACT_ROOT is required}"
: "${AF_MECHANISM_SPEC_ROOT:?AF_MECHANISM_SPEC_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_STAGED_CONFIG:?AF_STAGED_CONFIG is required}"
: "${AF_VLLM_ENDPOINT:?AF_VLLM_ENDPOINT is required}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly frozen_root="${AF_OUTPUT_ROOT}/frozen"
readonly plan="${frozen_root}/plan.json"
readonly task_plan="${frozen_root}/task_plan.jsonl"
readonly freeze_manifest="${frozen_root}/freeze_manifest.json"
readonly task_line=$((AF_TASK_INDEX + 1))

for path in \
  "${AF_PYTHON}" \
  "${AF_STAGED_CONFIG}" \
  "${plan}" \
  "${task_plan}" \
  "${freeze_manifest}"; do
  [[ -e "${path}" ]] || { echo "missing staged-search input: ${path}" >&2; exit 2; }
done
cmp --silent "${AF_STAGED_CONFIG}" "${plan}" || {
  echo "repository staged-search plan differs from frozen plan" >&2
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
readonly benchmark_id="$(jq -r '.benchmark_id' <<<"${task_json}")"
readonly tier="$(jq -r '.tier' <<<"${task_json}")"
readonly repetition="$(jq -r '.repetition' <<<"${task_json}")"
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

readonly model="$(jq -r '.model_contract.model' "${plan}")"
readonly proposer_reasoning="$(jq -r '.model_contract.proposer_reasoning_effort' "${plan}")"
readonly judge_reasoning="$(jq -r '.model_contract.judge_reasoning_effort' "${plan}")"
readonly temperature="$(jq -r '.model_contract.temperature' "${plan}")"
readonly output_tokens="$(jq -r '.model_contract.max_output_tokens' "${plan}")"
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
readonly run_output_root="${AF_OUTPUT_ROOT}/search/runs"
readonly run_root="${run_output_root}/${run_name}"
readonly process_time="${run_root}/search_process_time.json"

mkdir -p "${run_root}"
resume_arguments=()
if [[ -f "${run_root}/checkpoints/run.json" ]]; then
  resume_arguments=(--resume)
fi

"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/run_with_resource_timing.py" \
  --output "${process_time}" \
  -- \
  "${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/run_experiment.py" \
    --data-root "${AF_PUBLIC_DATA_ROOT}" \
    --benchmark-id "${benchmark_id}" \
    --tier "${tier}" \
    --seed "${repetition}" \
    --proposer-model "vllm:${model}" \
    --judge-model "vllm:${model}" \
    --vllm-base-url "${AF_VLLM_ENDPOINT}" \
    --vllm-proposer-reasoning-effort "${proposer_reasoning}" \
    --vllm-judge-reasoning-effort "${judge_reasoning}" \
    --vllm-temperature "${temperature}" \
    --vllm-seed "${repetition}" \
    --llm-timeout-seconds "${request_timeout}" \
    --llm-max-output-tokens "${output_tokens}" \
    --development-only \
    --selection-policy validation_only \
    --iteration-budget "${iterations}" \
    --stagnation-iterations "${iterations}" \
    --beam-size "${beam_size}" \
    --proposer-construction-mode staged_v2 \
    --proposer-feedback-mode rich_v1 \
    --proposal-policy incumbent_refinement_v1 \
    --disable-postfit-pruning \
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
    --output-root "${run_output_root}" \
    "${resume_arguments[@]}"
