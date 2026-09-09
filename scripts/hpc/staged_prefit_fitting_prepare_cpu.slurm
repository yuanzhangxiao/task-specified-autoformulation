#!/bin/bash
# Freeze the exact six passed candidates and public train/validation assets.
#SBATCH --job-name=prefit-fit-prepare
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
export PYTHONPATH="${AF_REPO_ROOT:?}/src${PYTHONPATH:+:${PYTHONPATH}}" PYTHONHASHSEED=0
cd "${AF_REPO_ROOT}"
"${AF_PYTHON:?}" scripts/staged_prefit_fitting_campaign.py freeze --config "${AF_CONFIG:?}" --source-root "${AF_SOURCE_ROOT:?}" --public-root "${AF_PUBLIC_ROOT:?}" --output-root "${AF_OUTPUT_ROOT:?}"
