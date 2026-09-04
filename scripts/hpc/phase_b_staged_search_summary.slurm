#!/bin/bash
# Summarize public-only staged-search tasks after an array completes.
#SBATCH --job-name=staged-search-summary
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_STAGED_CONFIG:?AF_STAGED_CONFIG is required}"
: "${AF_PYTHON_MODULE:=}"
: "${AF_GCCCORE_MODULE:=}"

if [[ -n "${AF_GCCCORE_MODULE}" ]]; then
  module load "${AF_GCCCORE_MODULE}"
fi
if [[ -n "${AF_PYTHON_MODULE}" ]]; then
  module load "${AF_PYTHON_MODULE}"
fi
readonly frozen="${AF_OUTPUT_ROOT}/frozen"
readonly output="${AF_OUTPUT_ROOT}/summary/staged_search_summary.json"
"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/summarize_phase_b_staged_search_pilot.py" \
  --plan "${frozen}/plan.json" \
  --task-plan "${frozen}/task_plan.jsonl" \
  --search-root "${AF_OUTPUT_ROOT}/search" \
  --output "${output}"
sha256sum "${output}" >"${output}.sha256"
