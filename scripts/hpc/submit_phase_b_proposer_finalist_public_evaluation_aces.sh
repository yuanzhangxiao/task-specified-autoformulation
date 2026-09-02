#!/bin/bash
# Freeze and submit the paired repaired-finalist public evaluation on ACES.

set -euo pipefail

: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_SOURCE_REPLAY_ROOT:=${SCRATCH}/phase_b/proposer-repair-replay-v4-aces-cpu}"
: "${AF_FINALIST_OUTPUT_ROOT:=${SCRATCH}/phase_b/proposer-finalist-public-evaluation-v1-aces-cpu}"
: "${AF_FINALIST_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_proposer_finalist_public_evaluation_v1.json}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_MECHANISM_SPEC_ROOT:=${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1}"
: "${AF_TASK_IDS:=0-11%12}"
: "${AF_FINALIST_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_finalist_public_evaluation_aces.slurm}"
: "${AF_SUMMARY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_finalist_public_evaluation_summary_aces.slurm}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in \
  "${AF_PYTHON}" \
  "${AF_FINALIST_CONFIG}" \
  "${AF_FINALIST_JOB}" \
  "${AF_SUMMARY_JOB}" \
  "${AF_SOURCE_REPLAY_ROOT}/proposer_repair_replay.json" \
  "${AF_SOURCE_REPLAY_ROOT}/artifact_ledger.jsonl"; do
  [[ -e "${path}" ]] || { echo "missing finalist input: ${path}" >&2; exit 2; }
done
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || {
  echo "missing public data: ${AF_PUBLIC_DATA_ROOT}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_FINALIST_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_proposer_finalist_public_evaluation.py \
  --config "${AF_FINALIST_CONFIG}" \
  --source-replay-root "${AF_SOURCE_REPLAY_ROOT}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --mechanism-spec-root "${AF_MECHANISM_SPEC_ROOT}" \
  --output-root "${AF_FINALIST_OUTPUT_ROOT}"

readonly frozen_plan="${AF_FINALIST_OUTPUT_ROOT}/frozen/plan.json"
readonly shared_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_FINALIST_PLAN=${frozen_plan},AF_SOURCE_REPLAY_ROOT=${AF_SOURCE_REPLAY_ROOT},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_MECHANISM_SPEC_ROOT=${AF_MECHANISM_SPEC_ROOT},AF_FINALIST_OUTPUT_ROOT=${AF_FINALIST_OUTPUT_ROOT}"
fit_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --array="${AF_TASK_IDS}" \
    --output="${AF_REPO_ROOT}/logs/proposer-finalist-fit-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/proposer-finalist-fit-%A_%a.err" \
    --export="${shared_export}" \
    "${AF_FINALIST_JOB}"
)"
fit_job="${fit_submission%%;*}"
summary_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --dependency="afterany:${fit_job}" \
    --output="${AF_REPO_ROOT}/logs/proposer-finalist-summary-%j.out" \
    --error="${AF_REPO_ROOT}/logs/proposer-finalist-summary-%j.err" \
    --export="${shared_export}" \
    "${AF_SUMMARY_JOB}"
)"
summary_job="${summary_submission%%;*}"

echo "ACES_PROPOSER_FINALIST_FIT_JOB=${fit_job}"
echo "ACES_PROPOSER_FINALIST_SUMMARY_JOB=${summary_job}"
echo "ACES_PROPOSER_FINALIST_ROOT=${AF_FINALIST_OUTPUT_ROOT}"
