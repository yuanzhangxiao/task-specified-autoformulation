#!/bin/bash
# Run one repaired proposer finalist on public development data using ACES CPUs.
#SBATCH --job-name=proposer-finalist-fit
#SBATCH --partition=cpu
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_FINALIST_PLAN:?AF_FINALIST_PLAN is required}"
: "${AF_SOURCE_REPLAY_ROOT:?AF_SOURCE_REPLAY_ROOT is required}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_TARGET_CONTRACT_ROOT:?AF_TARGET_CONTRACT_ROOT is required}"
: "${AF_MECHANISM_SPEC_ROOT:?AF_MECHANISM_SPEC_ROOT is required}"
: "${AF_FINALIST_OUTPUT_ROOT:?AF_FINALIST_OUTPUT_ROOT is required}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/run_phase_b_proposer_finalist_public_evaluation.py \
  --config "${AF_FINALIST_PLAN}" \
  --source-replay-root "${AF_SOURCE_REPLAY_ROOT}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --mechanism-spec-root "${AF_MECHANISM_SPEC_ROOT}" \
  --output-root "${AF_FINALIST_OUTPUT_ROOT}" \
  --task-index "${AF_TASK_INDEX}"
