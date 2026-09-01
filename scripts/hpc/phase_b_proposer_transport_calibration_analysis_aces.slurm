#!/bin/bash
#SBATCH --job-name=proposer-cal-analysis
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/proposer-cal-analysis-%j.out
#SBATCH --error=logs/proposer-cal-analysis-%j.err

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/analyze_phase_b_proposer_transport_calibration.py \
  --plan "${AF_OUTPUT_ROOT}/frozen/plan.json" \
  --results-root "${AF_OUTPUT_ROOT}/results" \
  --output-root "${AF_OUTPUT_ROOT}/analysis"
