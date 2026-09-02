#!/bin/bash
# Freeze and submit the matched feedback-rich refinement pilot on ACES.

set -euo pipefail

: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_PUBLIC_DATA_ROOT:=${SCRATCH}/phase_b/inputs/public-prompt-v3}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/proposer-refinement-pilot-v1-aces-h100x2}"
: "${AF_REFINEMENT_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_proposer_refinement_pilot_v1.json}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_MECHANISM_SPEC_ROOT:=${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1}"
: "${AF_JUDGE_PROTOCOL:=${AF_REPO_ROOT}/configs/hybrid_judge_consensus_operating_point_v1.json}"
: "${AF_VLLM_IMAGE:=${PROJECT:-${SCRATCH}}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_HF_HOME:=${SCRATCH}/huggingface-cache}"
: "${AF_WORKER_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_refinement_pilot_aces_h100.slurm}"
: "${AF_SUMMARY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_refinement_pilot_summary_aces.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in \
  "${AF_PYTHON}" \
  "${AF_REFINEMENT_CONFIG}" \
  "${AF_JUDGE_PROTOCOL}" \
  "${AF_VLLM_IMAGE}" \
  "${AF_WORKER_JOB}" \
  "${AF_SUMMARY_JOB}"; do
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
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_HF_HOME}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_proposer_refinement_pilot.py \
  --config "${AF_REFINEMENT_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --mechanism-spec-root "${AF_MECHANISM_SPEC_ROOT}" \
  --judge-protocol "${AF_JUDGE_PROTOCOL}"
readonly task_count_raw="$(wc -l <"${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl")"
readonly task_count="${task_count_raw//[[:space:]]/}"
if [[ ! "${task_count}" =~ ^[1-9][0-9]*$ ]] \
  || ((task_count < 2 || task_count % 2 != 0)); then
  echo "invalid matched task count: ${task_count}" >&2
  exit 2
fi
readonly matched_count=$((task_count / 2))
readonly exploratory_end=$((matched_count - 1))
readonly refinement_start="${matched_count}"
readonly refinement_end=$((task_count - 1))

readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_MECHANISM_SPEC_ROOT=${AF_MECHANISM_SPEC_ROOT},AF_JUDGE_PROTOCOL=${AF_JUDGE_PROTOCOL},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_REFINEMENT_CONFIG=${AF_REFINEMENT_CONFIG},AF_VLLM_IMAGE=${AF_VLLM_IMAGE},AF_HF_HOME=${AF_HF_HOME}"
exploratory_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --array="0-${exploratory_end}%2" \
    --output="${AF_REPO_ROOT}/logs/proposer-refinement-explore-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/proposer-refinement-explore-%A_%a.err" \
    --export="${common_export}" \
    "${AF_WORKER_JOB}"
)"
readonly exploratory_job="${exploratory_submission%%;*}"
refinement_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --array="${refinement_start}-${refinement_end}%2" \
    --dependency="afterany:${exploratory_job}" \
    --output="${AF_REPO_ROOT}/logs/proposer-refinement-incumbent-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/proposer-refinement-incumbent-%A_%a.err" \
    --export="${common_export}" \
    "${AF_WORKER_JOB}"
)"
readonly refinement_job="${refinement_submission%%;*}"
summary_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --dependency="afterany:${refinement_job}" \
    --output="${AF_REPO_ROOT}/logs/proposer-refinement-summary-%j.out" \
    --error="${AF_REPO_ROOT}/logs/proposer-refinement-summary-%j.err" \
    --export="${common_export}" \
    "${AF_SUMMARY_JOB}"
)"
readonly summary_job="${summary_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg exploratory_job "${exploratory_job}" \
  --arg refinement_job "${refinement_job}" \
  --arg summary_job "${summary_job}" \
  --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/plan.json" | awk '{print $1}')" \
  '{
    schema_version: "phase-b-proposer-refinement-pilot-submission-1",
    submitted_at_utc: $submitted_at_utc,
    platform: "aces-h100x2",
    exploratory_job: $exploratory_job,
    refinement_job: $refinement_job,
    summary_job: $summary_job,
    plan_sha256: $plan_sha256,
    test_data_opened: false,
    private_reference_opened: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "ACES_REFINEMENT_EXPLORATORY_JOB=${exploratory_job}"
echo "ACES_REFINEMENT_INCUMBENT_JOB=${refinement_job}"
echo "ACES_REFINEMENT_SUMMARY_JOB=${summary_job}"
echo "ACES_REFINEMENT_ROOT=${AF_OUTPUT_ROOT}"
