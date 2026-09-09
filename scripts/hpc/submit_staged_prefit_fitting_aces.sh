#!/bin/bash
# Submit the exact six-candidate public fitting handoff on ACES CPUs.

set -euo pipefail
: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_SOURCE_ROOT:?set AF_SOURCE_ROOT to the passed staged function pre-fit root}"
: "${AF_PUBLIC_ROOT:?set AF_PUBLIC_ROOT to the public-prompt-v3 root}"
: "${AF_CONFIG:=${AF_REPO_ROOT}/configs/staged_prefit_fitting_handoff_v1.json}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/staged-prefit-fitting-handoff-v1}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in "${AF_PYTHON}" "${AF_CONFIG}" "${AF_SOURCE_ROOT}/plan.json" "${AF_SOURCE_ROOT}/results/summary.json" "${AF_PUBLIC_ROOT}"; do [[ -e "${path}" ]] || { echo "missing staged pre-fit fitting input: ${path}" >&2; exit 2; }; done
[[ -z "$(git -C "${AF_REPO_ROOT}" status --porcelain)" ]] || { echo "fitting checkout must be clean before freezing" >&2; exit 2; }
[[ ! -e "${AF_OUTPUT_ROOT}/submission_manifest.json" ]] || { echo "submission manifest already exists: ${AF_OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${AF_OUTPUT_ROOT}/logs"
readonly exports="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_CONFIG=${AF_CONFIG},AF_SOURCE_ROOT=${AF_SOURCE_ROOT},AF_PUBLIC_ROOT=${AF_PUBLIC_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}"
prepare_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --output="${AF_OUTPUT_ROOT}/logs/prepare-%j.out" --error="${AF_OUTPUT_ROOT}/logs/prepare-%j.err" --export="${exports}" "${AF_REPO_ROOT}/scripts/hpc/staged_prefit_fitting_prepare_cpu.slurm")"
readonly prepare_job="${prepare_submission%%;*}"
fit_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --array="0-5%6" --dependency="afterok:${prepare_job}" --output="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.err" --export="${exports}" "${AF_REPO_ROOT}/scripts/hpc/staged_prefit_fitting_worker_cpu.slurm")"
readonly fit_job="${fit_submission%%;*}"
summary_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --dependency="afterany:${fit_job}" --output="${AF_OUTPUT_ROOT}/logs/summary-%j.out" --error="${AF_OUTPUT_ROOT}/logs/summary-%j.err" --export="${exports}" "${AF_REPO_ROOT}/scripts/hpc/staged_prefit_fitting_summary_cpu.slurm")"
readonly summary_job="${summary_submission%%;*}"
jq -n --arg prepare_job "${prepare_job}" --arg fit_job "${fit_job}" --arg summary_job "${summary_job}" --arg commit "$(git -C "${AF_REPO_ROOT}" rev-parse HEAD)" '{schema_version:"scientific-staged-prefit-fitting-submission-1",prepare_job:$prepare_job,fit_job:$fit_job,summary_job:$summary_job,commit:$commit,candidate_regeneration_performed:false,scientific_judge_called:false,test_data_opened:false,private_reference_opened:false}' >"${AF_OUTPUT_ROOT}/submission_manifest.json"
echo "ACES_PREFIT_FITTING_PREPARE_JOB=${prepare_job}"
echo "ACES_PREFIT_FITTING_JOB=${fit_job}"
echo "ACES_PREFIT_FITTING_SUMMARY_JOB=${summary_job}"
echo "ACES_PREFIT_FITTING_ROOT=${AF_OUTPUT_ROOT}"
