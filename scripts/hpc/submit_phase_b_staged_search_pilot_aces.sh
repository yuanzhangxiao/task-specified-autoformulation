#!/bin/bash
# Freeze and submit the public-only staged-search pilot on ACES H100 GPUs.

set -euo pipefail

: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_ACES_GRES:=gpu:h100:2}"
: "${AF_TENSOR_PARALLEL_SIZE:=2}"
: "${AF_ARRAY_SPEC:=0-5%2}"
: "${AF_PUBLIC_DATA_ROOT:=${SCRATCH}/phase_b/inputs/public-prompt-v3}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/staged-search-pilot-v1-aces-h100x2}"
: "${AF_STAGED_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_staged_search_pilot_v1.json}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_MECHANISM_SPEC_ROOT:=${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1}"
: "${AF_VLLM_IMAGE:=${PROJECT:-${SCRATCH}}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_HF_HOME:=${SCRATCH}/huggingface-cache}"
: "${AF_WORKER_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_staged_search_pilot_aces.slurm}"
: "${AF_SUMMARY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_staged_search_summary.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
[[ "${AF_TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid tensor-parallel size: ${AF_TENSOR_PARALLEL_SIZE}" >&2
  exit 2
}
[[ "${AF_ACES_GRES}" == "gpu:h100:${AF_TENSOR_PARALLEL_SIZE}" ]] || {
  echo "GPU request and tensor-parallel size differ" >&2
  exit 2
}
module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in \
  "${AF_PYTHON}" \
  "${AF_STAGED_CONFIG}" \
  "${AF_VLLM_IMAGE}" \
  "${AF_WORKER_JOB}" \
  "${AF_SUMMARY_JOB}"; do
  [[ -e "${path}" ]] || { echo "missing staged-search input: ${path}" >&2; exit 2; }
done
for directory in \
  "${AF_PUBLIC_DATA_ROOT}" \
  "${AF_TARGET_CONTRACT_ROOT}" \
  "${AF_MECHANISM_SPEC_ROOT}"; do
  [[ -d "${directory}" ]] || {
    echo "missing staged-search directory: ${directory}" >&2
    exit 2
  }
done
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_HF_HOME}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_staged_search_pilot.py \
  --config "${AF_STAGED_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --mechanism-spec-root "${AF_MECHANISM_SPEC_ROOT}"
readonly task_count_raw="$(wc -l <"${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl")"
readonly task_count="${task_count_raw//[[:space:]]/}"
[[ "${task_count}" == "6" ]] || {
  echo "staged-search pilot must contain exactly six tasks: ${task_count}" >&2
  exit 2
}

readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_MECHANISM_SPEC_ROOT=${AF_MECHANISM_SPEC_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_STAGED_CONFIG=${AF_STAGED_CONFIG},AF_VLLM_IMAGE=${AF_VLLM_IMAGE},AF_HF_HOME=${AF_HF_HOME},AF_TENSOR_PARALLEL_SIZE=${AF_TENSOR_PARALLEL_SIZE}"
search_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --gres="${AF_ACES_GRES}" \
    --array="${AF_ARRAY_SPEC}" \
    --output="${AF_REPO_ROOT}/logs/staged-search-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/staged-search-%A_%a.err" \
    --export="${common_export}" \
    "${AF_WORKER_JOB}"
)"
readonly search_job="${search_submission%%;*}"
summary_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --dependency="afterany:${search_job}" \
    --output="${AF_REPO_ROOT}/logs/staged-search-summary-%j.out" \
    --error="${AF_REPO_ROOT}/logs/staged-search-summary-%j.err" \
    --export="${common_export}" \
    "${AF_SUMMARY_JOB}"
)"
readonly summary_job="${summary_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg search_job "${search_job}" \
  --arg summary_job "${summary_job}" \
  --arg array_spec "${AF_ARRAY_SPEC}" \
  --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/plan.json" | awk '{print $1}')" \
  --arg requested_gres "${AF_ACES_GRES}" \
  --arg tensor_parallel_size "${AF_TENSOR_PARALLEL_SIZE}" \
  '{
    schema_version: "phase-b-staged-search-pilot-submission-1",
    submitted_at_utc: $submitted_at_utc,
    platform: ("aces-h100x" + $tensor_parallel_size),
    requested_gres: $requested_gres,
    tensor_parallel_size: ($tensor_parallel_size | tonumber),
    array_spec: $array_spec,
    search_job: $search_job,
    summary_job: $summary_job,
    plan_sha256: $plan_sha256,
    test_data_opened: false,
    private_reference_opened: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "ACES_STAGED_SEARCH_JOB=${search_job}"
echo "ACES_STAGED_SEARCH_SUMMARY_JOB=${summary_job}"
echo "ACES_STAGED_SEARCH_ROOT=${AF_OUTPUT_ROOT}"
