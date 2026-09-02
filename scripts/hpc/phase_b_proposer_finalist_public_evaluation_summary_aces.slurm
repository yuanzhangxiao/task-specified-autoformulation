#!/bin/bash
# Summarize the complete repaired-finalist public evaluation on ACES CPUs.
#SBATCH --job-name=proposer-finalist-summary
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
: "${AF_PYTHON:?AF_PYTHON is required}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_FINALIST_OUTPUT_ROOT:?AF_FINALIST_OUTPUT_ROOT is required}"

module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/summarize_phase_b_proposer_finalist_public_evaluation.py \
  --experiment-root "${AF_FINALIST_OUTPUT_ROOT}"

cd "${AF_FINALIST_OUTPUT_ROOT}/summary"
sha256sum -c finalist_public_evaluation_manifest.json.sha256
jq -r '"\(.sha256)  \(.path)"' artifact_ledger.jsonl \
  | while read -r digest relative; do
      printf '%s  %s\n' "${digest}" "${AF_FINALIST_OUTPUT_ROOT}/${relative}"
    done \
  | sha256sum -c -
