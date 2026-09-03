#!/bin/bash
# Submit the 2-cell x 3-seed reciprocal fitting pilot on ACES CPUs.

set -euo pipefail
: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_PRIVATE_DATA_ROOT:=${AF_REPO_ROOT}/data_raw}"
: "${AF_SOURCE_REPLAY_ROOT:=${SCRATCH}/phase_b/proposer-repair-replay-v4-aces-cpu}"
: "${AF_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_reciprocal_fitting_pilot_v1.json}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/reciprocal-fitting-pilot-v1-aces-cpu}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in \
  "${AF_PYTHON}" \
  "${AF_CONFIG}" \
  "${AF_SOURCE_REPLAY_ROOT}/proposer_repair_replay.json" \
  "${AF_PRIVATE_DATA_ROOT}/benchmark6_alien_device/private/selected_system_spec.json"; do
  [[ -e "${path}" ]] || { echo "missing reciprocal pilot input: ${path}" >&2; exit 2; }
done
if jq -e '.schema_version == "phase-b-parameter-range-ownership-pilot-1"' \
  "${AF_CONFIG}" >/dev/null; then
  readonly result_prefix="ACES_PARAMETER_RANGE"
else
  readonly result_prefix="ACES_RECIPROCAL"
fi
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || {
  echo "missing public data: ${AF_PUBLIC_DATA_ROOT}" >&2
  exit 2
}
[[ ! -e "${AF_OUTPUT_ROOT}/submission_manifest.json" ]] || {
  echo "submission manifest already exists: ${AF_OUTPUT_ROOT}" >&2
  exit 2
}
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
readonly exports="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_CONFIG=${AF_CONFIG},AF_SOURCE_REPLAY_ROOT=${AF_SOURCE_REPLAY_ROOT},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_PRIVATE_DATA_ROOT=${AF_PRIVATE_DATA_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}"
prepare_submission="$(
  sbatch --parsable --account="${AF_ACES_ACCOUNT}" \
    --output="${AF_REPO_ROOT}/logs/reciprocal-fit-prepare-%j.out" \
    --error="${AF_REPO_ROOT}/logs/reciprocal-fit-prepare-%j.err" \
    --export="${exports}" \
    "${AF_REPO_ROOT}/scripts/hpc/phase_b_reciprocal_fitting_prepare_cpu.slurm"
)"
readonly prepare_job="${prepare_submission%%;*}"
fit_submission="$(
  sbatch --parsable --account="${AF_ACES_ACCOUNT}" \
    --array="0-17%12" \
    --dependency="afterok:${prepare_job}" \
    --output="${AF_REPO_ROOT}/logs/reciprocal-fit-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/reciprocal-fit-%A_%a.err" \
    --export="${exports}" \
    "${AF_REPO_ROOT}/scripts/hpc/phase_b_reciprocal_fitting_pilot_cpu.slurm"
)"
readonly fit_job="${fit_submission%%;*}"
summary_submission="$(
  sbatch --parsable --account="${AF_ACES_ACCOUNT}" \
    --dependency="afterany:${fit_job}" \
    --output="${AF_REPO_ROOT}/logs/reciprocal-fit-summary-%j.out" \
    --error="${AF_REPO_ROOT}/logs/reciprocal-fit-summary-%j.err" \
    --export="${exports}" \
    "${AF_REPO_ROOT}/scripts/hpc/phase_b_reciprocal_fitting_summary_cpu.slurm"
)"
readonly summary_job="${summary_submission%%;*}"
jq -n \
  --arg prepare_job "${prepare_job}" \
  --arg fit_job "${fit_job}" \
  --arg summary_job "${summary_job}" \
  '{schema_version:"phase-b-reciprocal-fitting-submission-1",prepare_job:$prepare_job,fit_job:$fit_job,summary_job:$summary_job,test_data_opened:false}' \
  >"${AF_OUTPUT_ROOT}/submission_manifest.json"
echo "${result_prefix}_PREPARE_JOB=${prepare_job}"
echo "${result_prefix}_FIT_JOB=${fit_job}"
echo "${result_prefix}_SUMMARY_JOB=${summary_job}"
echo "${result_prefix}_ROOT=${AF_OUTPUT_ROOT}"
