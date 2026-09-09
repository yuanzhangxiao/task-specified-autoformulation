#!/bin/bash
# Fit one exact frozen candidate under its precommitted runtime route.
#SBATCH --job-name=prefit-fit
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
export PYTHONPATH="${AF_REPO_ROOT:?}/src${PYTHONPATH:+:${PYTHONPATH}}" PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON:?}" scripts/staged_prefit_fitting_campaign.py run --output-root "${AF_OUTPUT_ROOT:?}" --task-index "${SLURM_ARRAY_TASK_ID:?}"
