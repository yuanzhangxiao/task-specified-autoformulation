#!/bin/bash
#SBATCH --job-name=fixed-latent-fit
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/fixed-latent-basis-fitting-smoke-v1}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONHASHSEED=0
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/run_fixed_latent_basis_fitting_smoke.py \
  --output "${AF_OUTPUT_ROOT}/smoke_result.json"
