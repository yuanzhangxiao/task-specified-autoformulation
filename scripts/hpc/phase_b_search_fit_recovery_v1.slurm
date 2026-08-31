#!/bin/bash
#SBATCH --job-name=phase-b-fit-recovery
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/phase-b-fit-recovery-%A_%a.out
#SBATCH --error=logs/phase-b-fit-recovery-%A_%a.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_SEARCH_ROOT:=${AF_WORK}/phase_b/search-integration-ablation-v3}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/search-fit-recovery-v1}"
: "${AF_TASK_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

readonly config="${AF_REPO_ROOT}/configs/phase_b_search_fit_recovery_v1.json"

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data: ${AF_PUBLIC_DATA_ROOT}" >&2; exit 2; }
[[ -d "${AF_SEARCH_ROOT}" ]] || { echo "missing search root: ${AF_SEARCH_ROOT}" >&2; exit 2; }
[[ -f "${config}" ]] || { echo "missing recovery config: ${config}" >&2; exit 2; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"

"${AF_PYTHON}" scripts/run_phase_b_search_fit_recovery.py \
  --config "${config}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --search-root "${AF_SEARCH_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}" \
  --task-index "${AF_TASK_INDEX}"
