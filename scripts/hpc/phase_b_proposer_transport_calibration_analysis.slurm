#!/bin/bash
#SBATCH --job-name=phase-b-proposer-cal-analysis
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/phase-b-proposer-cal-analysis-%j.out
#SBATCH --error=logs/phase-b-proposer-cal-analysis-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/proposer-transport-calibration-v1}"

cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/analyze_phase_b_proposer_transport_calibration.py \
  --plan "${AF_OUTPUT_ROOT}/frozen/plan.json" \
  --results-root "${AF_OUTPUT_ROOT}/results" \
  --output-root "${AF_OUTPUT_ROOT}/analysis"
