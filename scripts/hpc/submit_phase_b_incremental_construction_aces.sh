#!/bin/bash
# Freeze and submit the one-seed incremental-construction gate on ACES.

set -euo pipefail

: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_ACES_GRES:=gpu:h100:2}"
: "${AF_TENSOR_PARALLEL_SIZE:=2}"
: "${AF_ARRAY_SPEC:=0}"
: "${AF_PUBLIC_DATA_ROOT:=${SCRATCH}/phase_b/inputs/public-prompt-v3}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/incremental-construction-pilot-v1-aces-h100x2-smoke}"
: "${AF_CONSTRUCTION_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_incremental_construction_pilot_v1.json}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v2}"
: "${AF_MECHANISM_SPEC_ROOT:=${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1}"
: "${AF_VLLM_IMAGE:=${PROJECT:-${SCRATCH}}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_HF_HOME:=${SCRATCH}/huggingface-cache}"
: "${AF_WORKER_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_incremental_construction_aces.slurm}"
: "${AF_SUMMARY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_incremental_construction_summary.slurm}"

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
for path in "${AF_PYTHON}" "${AF_CONSTRUCTION_CONFIG}" "${AF_VLLM_IMAGE}" "${AF_WORKER_JOB}" "${AF_SUMMARY_JOB}"; do
  [[ -e "${path}" ]] || { echo "missing incremental-construction input: ${path}" >&2; exit 2; }
done
for directory in "${AF_PUBLIC_DATA_ROOT}" "${AF_TARGET_CONTRACT_ROOT}" "${AF_MECHANISM_SPEC_ROOT}"; do
  [[ -d "${directory}" ]] || { echo "missing incremental-construction directory: ${directory}" >&2; exit 2; }
done
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_HF_HOME}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_incremental_construction_pilot.py \
  --config "${AF_CONSTRUCTION_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --mechanism-spec-root "${AF_MECHANISM_SPEC_ROOT}"
readonly task_count_raw="$(wc -l <"${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl")"
readonly task_count="${task_count_raw//[[:space:]]/}"
[[ "${task_count}" == "6" ]] || {
  echo "incremental-construction plan must contain six tasks: ${task_count}" >&2
  exit 2
}

readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_MECHANISM_SPEC_ROOT=${AF_MECHANISM_SPEC_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_VLLM_IMAGE=${AF_VLLM_IMAGE},AF_HF_HOME=${AF_HF_HOME},AF_TENSOR_PARALLEL_SIZE=${AF_TENSOR_PARALLEL_SIZE}"
worker_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --gres="${AF_ACES_GRES}" --array="${AF_ARRAY_SPEC}" --output="${AF_REPO_ROOT}/logs/incremental-construction-%A_%a.out" --error="${AF_REPO_ROOT}/logs/incremental-construction-%A_%a.err" --export="${common_export}" "${AF_WORKER_JOB}")"
readonly worker_job="${worker_submission%%;*}"
summary_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --dependency="afterany:${worker_job}" --output="${AF_REPO_ROOT}/logs/incremental-construction-summary-%j.out" --error="${AF_REPO_ROOT}/logs/incremental-construction-summary-%j.err" --export="${common_export}" "${AF_SUMMARY_JOB}")"
readonly summary_job="${summary_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg worker_job "${worker_job}" \
  --arg summary_job "${summary_job}" \
  --arg array_spec "${AF_ARRAY_SPEC}" \
  --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/plan.json" | awk '{print $1}')" \
  --arg requested_gres "${AF_ACES_GRES}" \
  '{
    schema_version: "phase-b-incremental-construction-submission-1",
    submitted_at_utc: $submitted_at_utc,
    platform: "aces-h100x2",
    requested_gres: $requested_gres,
    array_spec: $array_spec,
    worker_job: $worker_job,
    summary_job: $summary_job,
    plan_sha256: $plan_sha256,
    parameter_fitting_performed: false,
    scientific_judge_called: false,
    test_data_opened: false,
    private_reference_opened: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "ACES_INCREMENTAL_CONSTRUCTION_JOB=${worker_job}"
echo "ACES_INCREMENTAL_CONSTRUCTION_SUMMARY_JOB=${summary_job}"
echo "ACES_INCREMENTAL_CONSTRUCTION_ROOT=${AF_OUTPUT_ROOT}"
