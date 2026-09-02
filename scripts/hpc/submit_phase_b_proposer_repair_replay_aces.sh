#!/bin/bash
# Submit the no-call proposer first-attempt repair replay on ACES CPUs.

set -euo pipefail

: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_SOURCE_OUTPUT_ROOT:=${SCRATCH}/phase_b/proposer-reasoning-calibration-v3-aces-h100x2}"
: "${AF_REPLAY_OUTPUT_ROOT:=${SCRATCH}/phase_b/proposer-repair-replay-v4-aces-cpu}"
: "${AF_REPLAY_PLAN:=${AF_REPO_ROOT}/configs/phase_b_proposer_repair_replay_v4.json}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_REPLAY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_repair_replay_aces.slurm}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in \
  "${AF_PYTHON}" \
  "${AF_REPLAY_PLAN}" \
  "${AF_REPLAY_JOB}" \
  "${AF_SOURCE_OUTPUT_ROOT}/frozen/plan.json" \
  "${AF_SOURCE_OUTPUT_ROOT}/analysis/proposer_transport_calibration.json"; do
  [[ -e "${path}" ]] || { echo "missing replay input: ${path}" >&2; exit 2; }
done
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || {
  echo "missing public data: ${AF_PUBLIC_DATA_ROOT}" >&2
  exit 2
}
[[ -d "${AF_TARGET_CONTRACT_ROOT}" ]] || {
  echo "missing target contracts: ${AF_TARGET_CONTRACT_ROOT}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_REPLAY_OUTPUT_ROOT}"
submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --output="${AF_REPO_ROOT}/logs/proposer-repair-replay-%j.out" \
    --error="${AF_REPO_ROOT}/logs/proposer-repair-replay-%j.err" \
    --export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_REPLAY_PLAN=${AF_REPLAY_PLAN},AF_SOURCE_OUTPUT_ROOT=${AF_SOURCE_OUTPUT_ROOT},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_REPLAY_OUTPUT_ROOT=${AF_REPLAY_OUTPUT_ROOT}" \
    "${AF_REPLAY_JOB}"
)"
job_id="${submission%%;*}"
echo "ACES_PROPOSER_REPAIR_REPLAY_JOB=${job_id}"
echo "ACES_PROPOSER_REPAIR_REPLAY_ROOT=${AF_REPLAY_OUTPUT_ROOT}"
