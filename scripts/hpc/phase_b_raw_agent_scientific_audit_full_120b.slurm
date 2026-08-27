#!/bin/bash
#SBATCH --job-name=phase-b-raw-audit-full
#SBATCH --account=bibo-delta-gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=logs/phase-b-raw-audit-full-%A_%a.out
#SBATCH --error=logs/phase-b-raw-audit-full-%A_%a.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine Slurm user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_SHARD_COUNT:=4}"
: "${AF_RAW_FITTED_EVAL_ROOT:=${AF_WORK}/phase_b/raw-data-agent-fitted-full-evaluation-v1}"
: "${AF_CALIBRATION_ROOT:=${AF_RAW_FITTED_EVAL_ROOT}/gpt-oss-120b}"
: "${AF_CALIBRATION_PAIRS:=${AF_RAW_FITTED_EVAL_ROOT}/pairs.jsonl}"
: "${AF_PAIR_IDS:=}"
: "${AF_REPETITIONS:=1}"
: "${AF_REQUIRED_REPETITIONS:=1}"
: "${AF_MAX_ATTEMPTS:=10}"
: "${AF_JUDGE_SEED_BASE:=18000}"
: "${AF_REQUIRED_JUDGE_SEED_BASE:=18000}"
: "${AF_MODEL_SEMANTIC_CONTRACT:=true}"
: "${AF_REPAIR_MISSING_ATOMIC_UNITS:=true}"

export AF_PROJECT AF_WORK AF_REPO_ROOT AF_LOCAL_MODEL AF_TENSOR_PARALLEL_SIZE
export AF_SHARD_COUNT AF_CALIBRATION_ROOT AF_CALIBRATION_PAIRS AF_PAIR_IDS
export AF_REPETITIONS AF_REQUIRED_REPETITIONS AF_MAX_ATTEMPTS
export AF_JUDGE_SEED_BASE AF_REQUIRED_JUDGE_SEED_BASE AF_MODEL_SEMANTIC_CONTRACT
export AF_REPAIR_MISSING_ATOMIC_UNITS

exec bash "${AF_REPO_ROOT}/scripts/hpc/run_vllm_atomic_judge.sh"
