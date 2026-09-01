#!/bin/bash
#SBATCH --job-name=phase-b-raw-eval-merge
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/phase-b-raw-eval-merge-%j.out
#SBATCH --error=logs/phase-b-raw-eval-merge-%j.err

set -euo pipefail
readonly af_user="${SLURM_JOB_USER:-${USER:?}}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/raw-agent-deterministic-evaluation-v1}"
mkdir -p "${AF_REPO_ROOT}/logs"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/merge_phase_b_postfreeze.py \
  --subjects "${AF_OUTPUT_ROOT}/adapted/frozen_evaluation_subjects.jsonl" \
  --input-root "${AF_OUTPUT_ROOT}/postfreeze" \
  --output-root "${AF_OUTPUT_ROOT}/postfreeze"
