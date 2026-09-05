#!/bin/bash
# Run one frozen incremental-construction task against a ready vLLM endpoint.

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_TARGET_CONTRACT_ROOT:?AF_TARGET_CONTRACT_ROOT is required}"
: "${AF_MECHANISM_SPEC_ROOT:?AF_MECHANISM_SPEC_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_VLLM_ENDPOINT:?AF_VLLM_ENDPOINT is required}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly frozen="${AF_OUTPUT_ROOT}/frozen"
readonly plan="${frozen}/plan.json"
readonly tasks="${frozen}/task_plan.jsonl"
readonly task_line=$((AF_TASK_INDEX + 1))
for path in "${AF_PYTHON}" "${plan}" "${tasks}" "${frozen}/freeze_manifest.json"; do
  [[ -e "${path}" ]] || { echo "missing incremental-construction input: ${path}" >&2; exit 2; }
done
(
  cd "${frozen}"
  sha256sum -c plan.json.sha256
  sha256sum -c task_plan.jsonl.sha256
  sha256sum -c freeze_manifest.json.sha256
)
readonly task_json="$(sed -n "${task_line}p" "${tasks}")"
[[ -n "${task_json}" ]] || { echo "missing task ${AF_TASK_INDEX}" >&2; exit 2; }
[[ "$(jq -r '.task_index' <<<"${task_json}")" == "${AF_TASK_INDEX}" ]] || {
  echo "task ledger index mismatch" >&2
  exit 2
}

"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/run_phase_b_incremental_construction_task.py" \
  --plan "${plan}" \
  --task-index "${AF_TASK_INDEX}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --mechanism-spec-root "${AF_MECHANISM_SPEC_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}/results" \
  --vllm-base-url "${AF_VLLM_ENDPOINT}"
