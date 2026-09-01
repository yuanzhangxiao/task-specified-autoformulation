#!/bin/bash
#SBATCH --job-name=phase-b-baseline-summary
#SBATCH --account=bibo-delta-cpu
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

cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/summarize_phase_b_public_baseline_pilot.py \
  --task-plan "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" \
  --runs-root "${AF_OUTPUT_ROOT}/runs" \
  --output-root "${AF_OUTPUT_ROOT}/summary"
cat "${AF_OUTPUT_ROOT}/summary/baseline_development_summary.json"
