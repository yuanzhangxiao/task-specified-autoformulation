#!/bin/bash
#SBATCH --job-name=phase-b-fit-recovery-summary
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/phase-b-fit-recovery-summary-%j.out
#SBATCH --error=logs/phase-b-fit-recovery-summary-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/search-fit-recovery-v1}"

readonly config="${AF_REPO_ROOT}/configs/phase_b_search_fit_recovery_v1.json"
readonly report="${AF_OUTPUT_ROOT}/fit_recovery_summary.json"

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"

"${AF_PYTHON}" scripts/summarize_phase_b_search_fit_recovery.py \
  --config "${config}" \
  --input-root "${AF_OUTPUT_ROOT}" \
  --output "${report}"

sha256sum "${report}" | tee "${report}.sha256"
