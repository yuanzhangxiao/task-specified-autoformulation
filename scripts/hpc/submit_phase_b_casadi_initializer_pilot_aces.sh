#!/bin/bash
# Submit the 2-cell x 3-seed matched CasADi initializer pilot on ACES CPUs.

set -euo pipefail
: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_PUBLIC_DATA_ROOT:=${SCRATCH}/phase_b/inputs/public-prompt-v3}"
: "${AF_SOURCE_REPLAY_ROOT:=${SCRATCH}/phase_b/proposer-repair-replay-v4-aces-cpu}"
: "${AF_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_casadi_initializer_pilot_v1.json}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/casadi-initializer-pilot-v1-aces-cpu}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in \
  "${AF_PYTHON}" \
  "${AF_CONFIG}" \
  "${AF_SOURCE_REPLAY_ROOT}/proposer_repair_replay.json" \
  "${AF_PUBLIC_DATA_ROOT}"; do
  [[ -e "${path}" ]] || { echo "missing CasADi pilot input: ${path}" >&2; exit 2; }
done
"${AF_PYTHON}" -c 'import casadi'
[[ ! -e "${AF_OUTPUT_ROOT}/submission_manifest.json" ]] || {
  echo "submission manifest already exists: ${AF_OUTPUT_ROOT}" >&2
  exit 2
}
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
readonly exports="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_CONFIG=${AF_CONFIG},AF_SOURCE_REPLAY_ROOT=${AF_SOURCE_REPLAY_ROOT},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}"
prepare_submission="$(
  sbatch --parsable --account="${AF_ACES_ACCOUNT}" \
    --output="${AF_REPO_ROOT}/logs/casadi-init-prepare-%j.out" \
    --error="${AF_REPO_ROOT}/logs/casadi-init-prepare-%j.err" \
    --export="${exports}" \
    "${AF_REPO_ROOT}/scripts/hpc/phase_b_casadi_initializer_prepare_cpu.slurm"
)"
readonly prepare_job="${prepare_submission%%;*}"
fit_submission="$(
  sbatch --parsable --account="${AF_ACES_ACCOUNT}" \
    --array="0-11%12" \
    --dependency="afterok:${prepare_job}" \
    --output="${AF_REPO_ROOT}/logs/casadi-init-pilot-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/casadi-init-pilot-%A_%a.err" \
    --export="${exports}" \
    "${AF_REPO_ROOT}/scripts/hpc/phase_b_casadi_initializer_pilot_cpu.slurm"
)"
readonly fit_job="${fit_submission%%;*}"
summary_submission="$(
  sbatch --parsable --account="${AF_ACES_ACCOUNT}" \
    --dependency="afterany:${fit_job}" \
    --output="${AF_REPO_ROOT}/logs/casadi-init-summary-%j.out" \
    --error="${AF_REPO_ROOT}/logs/casadi-init-summary-%j.err" \
    --export="${exports}" \
    "${AF_REPO_ROOT}/scripts/hpc/phase_b_casadi_initializer_summary_cpu.slurm"
)"
readonly summary_job="${summary_submission%%;*}"
jq -n \
  --arg prepare_job "${prepare_job}" \
  --arg fit_job "${fit_job}" \
  --arg summary_job "${summary_job}" \
  '{schema_version:"phase-b-casadi-initializer-submission-1",prepare_job:$prepare_job,fit_job:$fit_job,summary_job:$summary_job,test_data_opened:false}' \
  >"${AF_OUTPUT_ROOT}/submission_manifest.json"
echo "ACES_CASADI_PREPARE_JOB=${prepare_job}"
echo "ACES_CASADI_FIT_JOB=${fit_job}"
echo "ACES_CASADI_SUMMARY_JOB=${summary_job}"
echo "ACES_CASADI_ROOT=${AF_OUTPUT_ROOT}"
