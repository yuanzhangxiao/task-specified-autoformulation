#!/bin/bash
# Fit one frozen candidate with one initializer policy.
#SBATCH --job-name=casadi-init-pilot
#SBATCH --partition=cpu
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue

set -euo pipefail
: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

if [[ -n "${AF_GCCCORE_MODULE:-}" && -n "${AF_PYTHON_MODULE:-}" ]]; then
  module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
fi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONHASHSEED=0
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/run_phase_b_casadi_initializer_pilot.py \
  --experiment-root "${AF_OUTPUT_ROOT}" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --task-index "${AF_TASK_INDEX}"
