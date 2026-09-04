#!/bin/bash
# Freeze and submit the public-only staged-search pilot on Delta A40 GPUs.

set -euo pipefail

readonly af_user="${USER:?USER is unset}"
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
: "${AF_ARRAY_SPEC:=0-5%2}"
: "${AF_GPU_ACCOUNT:=bibo-delta-gpu}"
: "${AF_CPU_ACCOUNT:=bibo-delta-cpu}"
: "${AF_WORKER_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_staged_search_pilot_delta.slurm}"
: "${AF_SUMMARY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_staged_search_summary.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
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

readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_MECHANISM_SPEC_ROOT=${AF_MECHANISM_SPEC_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_STAGED_CONFIG=${AF_STAGED_CONFIG},AF_VLLM_IMAGE=${AF_VLLM_IMAGE},AF_HF_HOME=${AF_HF_HOME},AF_TENSOR_PARALLEL_SIZE=${AF_TENSOR_PARALLEL_SIZE}"
search_submission="$(
  sbatch --parsable \
    --account="${AF_GPU_ACCOUNT}" \
    --array="${AF_ARRAY_SPEC}" \
    --output="${AF_REPO_ROOT}/logs/staged-search-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/staged-search-%A_%a.err" \
    --export="${common_export}" \
    "${AF_WORKER_JOB}"
)"
readonly search_job="${search_submission%%;*}"
summary_submission="$(
  sbatch --parsable \
    --account="${AF_CPU_ACCOUNT}" \
    --partition=cpu \
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
  '{
    schema_version: "phase-b-staged-search-pilot-submission-1",
    submitted_at_utc: $submitted_at_utc,
    platform: "delta-a40x4",
    array_spec: $array_spec,
    search_job: $search_job,
    summary_job: $summary_job,
    plan_sha256: $plan_sha256,
    test_data_opened: false,
    private_reference_opened: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "DELTA_STAGED_SEARCH_JOB=${search_job}"
echo "DELTA_STAGED_SEARCH_SUMMARY_JOB=${summary_job}"
echo "DELTA_STAGED_SEARCH_ROOT=${AF_OUTPUT_ROOT}"
