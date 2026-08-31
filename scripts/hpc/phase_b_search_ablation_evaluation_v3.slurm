#!/bin/bash
#SBATCH --job-name=phase-b-search-v4-eval
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/phase-b-search-v4-eval-%j.out
#SBATCH --error=logs/phase-b-search-v4-eval-%j.err

# Reuse the sealed postfreeze pipeline and request its arm-level endpoint bundle.
set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_SEARCH_ROOT:=${AF_WORK}/phase_b/search-integration-ablation-v4}"

export AF_PROJECT AF_WORK AF_REPO_ROOT AF_SEARCH_ROOT
export AF_WRITE_ARM_ENDPOINT_REPORT=true
exec "${AF_REPO_ROOT}/scripts/hpc/phase_b_search_ablation_evaluation_v2.slurm"
