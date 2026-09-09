#!/bin/bash
# Submit the matched six-candidate fitter-rescue chain on ACES CPUs.

set -euo pipefail
: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_SOURCE_FITTING_ROOT:?set AF_SOURCE_FITTING_ROOT to the completed fitting handoff root}"
: "${AF_CONFIG:=${AF_REPO_ROOT}/configs/staged_fitter_rescue_v1.json}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/staged-fitter-rescue-v1}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in "${AF_PYTHON}" "${AF_CONFIG}" "${AF_SOURCE_FITTING_ROOT}/plan.json" "${AF_SOURCE_FITTING_ROOT}/summary/summary.json"; do [[ -e "${path}" ]] || { echo "missing fitter-rescue input: ${path}" >&2; exit 2; }; done
[[ -z "$(git -C "${AF_REPO_ROOT}" status --porcelain)" ]] || { echo "fitter-rescue checkout must be clean before freezing" >&2; exit 2; }
[[ ! -e "${AF_OUTPUT_ROOT}/submission_manifest.json" ]] || { echo "submission manifest already exists: ${AF_OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${AF_OUTPUT_ROOT}/logs"
readonly exports="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_CONFIG=${AF_CONFIG},AF_SOURCE_FITTING_ROOT=${AF_SOURCE_FITTING_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}"
prepare_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --output="${AF_OUTPUT_ROOT}/logs/prepare-%j.out" --error="${AF_OUTPUT_ROOT}/logs/prepare-%j.err" --export="${exports}" "${AF_REPO_ROOT}/scripts/hpc/staged_fitter_rescue_prepare_cpu.slurm")"
readonly prepare_job="${prepare_submission%%;*}"
rescue_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --array="0-5%6" --dependency="afterok:${prepare_job}" --output="${AF_OUTPUT_ROOT}/logs/rescue-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/rescue-%A_%a.err" --export="${exports}" "${AF_REPO_ROOT}/scripts/hpc/staged_fitter_rescue_worker_cpu.slurm")"
readonly rescue_job="${rescue_submission%%;*}"
summary_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --dependency="afterany:${rescue_job}" --output="${AF_OUTPUT_ROOT}/logs/summary-%j.out" --error="${AF_OUTPUT_ROOT}/logs/summary-%j.err" --export="${exports}" "${AF_REPO_ROOT}/scripts/hpc/staged_fitter_rescue_summary_cpu.slurm")"
readonly summary_job="${summary_submission%%;*}"
jq -n --arg prepare_job "${prepare_job}" --arg rescue_job "${rescue_job}" --arg summary_job "${summary_job}" --arg commit "$(git -C "${AF_REPO_ROOT}" rev-parse HEAD)" '{schema_version:"scientific-staged-fitter-rescue-submission-1",prepare_job:$prepare_job,rescue_job:$rescue_job,summary_job:$summary_job,commit:$commit,candidate_regeneration_performed:false,topology_or_function_revision_performed:false,scientific_judge_called:false,test_data_opened:false,private_reference_opened:false,multiple_round_search_performed:false}' >"${AF_OUTPUT_ROOT}/submission_manifest.json"
echo "ACES_FITTER_RESCUE_PREPARE_JOB=${prepare_job}"
echo "ACES_FITTER_RESCUE_JOB=${rescue_job}"
echo "ACES_FITTER_RESCUE_SUMMARY_JOB=${summary_job}"
echo "ACES_FITTER_RESCUE_ROOT=${AF_OUTPUT_ROOT}"
