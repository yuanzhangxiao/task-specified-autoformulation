#!/bin/bash
#SBATCH --job-name=fitter-initials
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_CODE_COMMIT:?}"
cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_CODE_COMMIT}" && -z "$(git status --porcelain)" ]] || {
  echo 'Checkout changed after submission.' >&2; exit 2;
}
export PYTHONPATH="${AF_REPO_ROOT}/src:${AF_CASADI_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:-run}" in
  summarize|smoke)
    timeout --signal=TERM --kill-after=10s 540s "${AF_PYTHON}" scripts/run_initialization_campaign.py "$1" --output "${AF_OUTPUT_ROOT}"
    ;;
  run)
    timeout --signal=TERM --kill-after=15s 1650s "${AF_PYTHON}" scripts/run_initialization_campaign.py run --output "${AF_OUTPUT_ROOT}" --task-index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  *) exit 2 ;;
esac
