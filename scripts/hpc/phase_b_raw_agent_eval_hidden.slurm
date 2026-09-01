#!/bin/bash
#SBATCH --job-name=phase-b-raw-eval-hidden
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-23%12
#SBATCH --output=logs/phase-b-raw-eval-hidden-%A_%a.out
#SBATCH --error=logs/phase-b-raw-eval-hidden-%A_%a.err

set -euo pipefail
readonly af_user="${SLURM_JOB_USER:-${USER:?}}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_PRIVATE_DATA_ROOT:=${AF_REPO_ROOT}/data_raw}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/raw-agent-deterministic-evaluation-v1}"
: "${AF_MAX_TRAJECTORY_SECONDS:=300}"
readonly shard_index="${SLURM_ARRAY_TASK_ID:?submit as an array}"
readonly shard_count=24
readonly mechanism_root="${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1/specs"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
mkdir -p "${AF_REPO_ROOT}/logs"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/evaluate_phase_b_hidden_subspace.py \
  --subjects "${AF_OUTPUT_ROOT}/postfreeze/postfreeze_subjects.jsonl" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --private-data-root "${AF_PRIVATE_DATA_ROOT}" \
  --mechanism-config-root "${mechanism_root}" \
  --output-root "${AF_OUTPUT_ROOT}/hidden" \
  --shard-index "${shard_index}" \
  --shard-count "${shard_count}" \
  --maximum-trajectory-wall-time-seconds "${AF_MAX_TRAJECTORY_SECONDS}"
