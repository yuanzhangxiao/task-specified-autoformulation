#!/bin/bash
# Summarize the feedback-rich refinement pilot on an ACES CPU node.
#SBATCH --job-name=proposer-refine-summary
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_TARGET_CONTRACT_ROOT:?AF_TARGET_CONTRACT_ROOT is required}"
: "${AF_MECHANISM_SPEC_ROOT:?AF_MECHANISM_SPEC_ROOT is required}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/summarize_phase_b_proposer_refinement_pilot.py \
  --plan "${AF_OUTPUT_ROOT}/frozen/plan.json" \
  --search-root "${AF_OUTPUT_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --mechanism-spec-root "${AF_MECHANISM_SPEC_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}/analysis"

cd "${AF_OUTPUT_ROOT}/analysis"
sha256sum proposer_refinement_report.json proposer_refinement_tasks.csv \
  proposer_refinement_report.md >artifact_sha256.txt
