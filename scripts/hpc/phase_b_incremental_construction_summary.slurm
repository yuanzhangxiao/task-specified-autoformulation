#!/bin/bash
# Summarize incremental-construction tasks without fitting or private data.
#SBATCH --job-name=incremental-construction-summary
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"

readonly frozen="${AF_OUTPUT_ROOT}/frozen"
readonly output="${AF_OUTPUT_ROOT}/summary/incremental_construction_summary.json"
"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/summarize_phase_b_incremental_construction_pilot.py" \
  --plan "${frozen}/plan.json" \
  --task-plan "${frozen}/task_plan.jsonl" \
  --result-root "${AF_OUTPUT_ROOT}/results" \
  --output "${output}"
sha256sum "${output}" >"${output}.sha256"
