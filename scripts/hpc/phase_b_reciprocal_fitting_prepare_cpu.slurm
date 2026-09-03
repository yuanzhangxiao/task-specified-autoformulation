#!/bin/bash
# Build the oracle-observed-derivative overlay and freeze all pilot inputs.
#SBATCH --job-name=reciprocal-fit-prepare
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_CONFIG:?AF_CONFIG is required}"
: "${AF_SOURCE_REPLAY_ROOT:?AF_SOURCE_REPLAY_ROOT is required}"
: "${AF_PUBLIC_DATA_ROOT:?AF_PUBLIC_DATA_ROOT is required}"
: "${AF_PRIVATE_DATA_ROOT:?AF_PRIVATE_DATA_ROOT is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"

if [[ -n "${AF_GCCCORE_MODULE:-}" && -n "${AF_PYTHON_MODULE:-}" ]]; then
  module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
fi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONHASHSEED=0
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/build_phase_b_exact_derivative_overlay.py \
  --config "${AF_CONFIG}" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --private-data-root "${AF_PRIVATE_DATA_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}/derivative-overlay"
"${AF_PYTHON}" scripts/prepare_phase_b_reciprocal_fitting_pilot.py \
  --config "${AF_CONFIG}" \
  --source-replay-root "${AF_SOURCE_REPLAY_ROOT}" \
  --derivative-overlay-root "${AF_OUTPUT_ROOT}/derivative-overlay" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}"
