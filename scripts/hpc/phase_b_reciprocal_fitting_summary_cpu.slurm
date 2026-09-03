#!/bin/bash
# Summarize all reciprocal fitting pilot endpoints.
#SBATCH --job-name=reciprocal-fit-summary
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"

if [[ -n "${AF_GCCCORE_MODULE:-}" && -n "${AF_PYTHON_MODULE:-}" ]]; then
  module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
fi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONHASHSEED=0
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/summarize_phase_b_reciprocal_fitting_pilot.py \
  --experiment-root "${AF_OUTPUT_ROOT}"
cd "${AF_OUTPUT_ROOT}"
jq -r '"\(.sha256)  \(.path)"' summary/task_artifact_ledger.jsonl \
  | sha256sum -c -
cd summary
sha256sum -c manifest.json.sha256
cat "${AF_OUTPUT_ROOT}/summary/reciprocal_fitting_pilot_summary.md"
