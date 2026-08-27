#!/bin/bash
#SBATCH --job-name=phase-b-raw-method-judge
#SBATCH --account=bibo-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-raw-method-judge-%j.out
#SBATCH --error=logs/phase-b-raw-method-judge-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_SHARD_COUNT:=1}"
: "${AF_CALIBRATION_ROOT:=${AF_WORK}/phase_b/raw-data-agent-method-judge-v1/gpt-oss-120b}"
: "${AF_CALIBRATION_PAIRS:=${AF_WORK}/phase_b/raw-data-agent-method-judge-v1/pairs.jsonl}"
: "${AF_PAIR_IDS:=}"
: "${AF_REPETITIONS:=1}"
: "${AF_REQUIRED_REPETITIONS:=1}"
: "${AF_MAX_ATTEMPTS:=10}"
: "${AF_JUDGE_SEED_BASE:=16000}"
: "${AF_REQUIRED_JUDGE_SEED_BASE:=16000}"
: "${AF_COMPARATIVE_INDETERMINATE_POLICY:=neutral_fixed_denominator}"

export AF_PROJECT AF_WORK AF_REPO_ROOT AF_LOCAL_MODEL
export AF_TENSOR_PARALLEL_SIZE AF_SHARD_COUNT AF_CALIBRATION_ROOT
export AF_CALIBRATION_PAIRS AF_PAIR_IDS AF_REPETITIONS AF_REQUIRED_REPETITIONS
export AF_MAX_ATTEMPTS AF_JUDGE_SEED_BASE AF_REQUIRED_JUDGE_SEED_BASE
export AF_COMPARATIVE_INDETERMINATE_POLICY

exec bash "${AF_REPO_ROOT}/scripts/hpc/run_vllm_atomic_judge.sh"
