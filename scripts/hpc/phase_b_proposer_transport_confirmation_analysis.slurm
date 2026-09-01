#!/bin/bash
#SBATCH --job-name=phase-b-proposer-confirm
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
: "${AF_PRIMARY_ANALYSIS:?AF_PRIMARY_ANALYSIS is required}"

readonly analysis_root="${AF_OUTPUT_ROOT}/analysis"
readonly confirmation_analysis="${analysis_root}/proposer_transport_calibration.json"
readonly handoff="${AF_OUTPUT_ROOT}/cross_cluster_handoff.json"

cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/analyze_phase_b_proposer_transport_calibration.py \
  --plan "${AF_OUTPUT_ROOT}/frozen/plan.json" \
  --results-root "${AF_OUTPUT_ROOT}/results" \
  --output-root "${analysis_root}"
"${AF_PYTHON}" scripts/verify_phase_b_proposer_transport_confirmation.py \
  --source-analysis "${AF_PRIMARY_ANALYSIS}" \
  --confirmation-analysis "${confirmation_analysis}" \
  --handoff "${handoff}" \
  --output-root "${analysis_root}"
