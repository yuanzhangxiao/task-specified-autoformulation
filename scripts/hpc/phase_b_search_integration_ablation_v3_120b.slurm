#!/bin/bash
#SBATCH --job-name=phase-b-search-v4
#SBATCH --account=bibo-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-11%2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --requeue
#SBATCH --output=logs/phase-b-search-v4-%A_%a.out
#SBATCH --error=logs/phase-b-search-v4-%A_%a.err

# Reuse the version-2 serving harness; only the frozen plan and output root vary.
set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/search-integration-ablation-v4}"
: "${AF_SEARCH_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_search_integration_ablation_v3.json}"

export AF_PROJECT AF_WORK AF_REPO_ROOT AF_OUTPUT_ROOT AF_SEARCH_CONFIG
exec "${AF_REPO_ROOT}/scripts/hpc/phase_b_search_integration_ablation_v2_120b.slurm"
