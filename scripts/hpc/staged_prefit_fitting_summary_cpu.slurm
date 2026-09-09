#!/bin/bash
# Summarize separate structural, fitting, validation, and resource endpoints.
#SBATCH --job-name=prefit-fit-summary
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
export PYTHONPATH="${AF_REPO_ROOT:?}/src${PYTHONPATH:+:${PYTHONPATH}}" PYTHONHASHSEED=0
cd "${AF_REPO_ROOT}"
"${AF_PYTHON:?}" scripts/staged_prefit_fitting_campaign.py summarize --output-root "${AF_OUTPUT_ROOT:?}"
cat "${AF_OUTPUT_ROOT}/summary/summary.md"
