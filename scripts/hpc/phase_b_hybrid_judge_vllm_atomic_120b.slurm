#!/bin/bash
#SBATCH --job-name=phase-b-judge-atomic-120b
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-judge-atomic-120b-%j.out
#SBATCH --error=logs/phase-b-judge-atomic-120b-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_LOCAL_MODEL:=openai/gpt-oss-120b}"
: "${AF_TENSOR_PARALLEL_SIZE:=4}"
: "${AF_SHARD_COUNT:=1}"
: "${AF_CALIBRATION_ROOT:=${AF_WORK}/phase_b/judge-hybrid-atomic-v1/gpt-oss-120b}"

export AF_PROJECT AF_WORK AF_REPO_ROOT AF_LOCAL_MODEL
export AF_TENSOR_PARALLEL_SIZE AF_SHARD_COUNT AF_CALIBRATION_ROOT

exec bash "${AF_REPO_ROOT}/scripts/hpc/run_vllm_atomic_judge.sh"
