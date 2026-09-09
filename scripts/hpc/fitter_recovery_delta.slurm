#!/bin/bash
#SBATCH --job-name=fitter-recovery
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_CODE_COMMIT:?}"
cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_CODE_COMMIT}" && -z "$(git status --porcelain)" ]] || {
  echo 'Recovery checkout changed after submission.' >&2; exit 2;
}
export PYTHONPATH="${AF_REPO_ROOT}/src"
if [[ "${1:-run}" == summarize ]]; then
  "${AF_PYTHON}" scripts/run_fitter_recovery.py summarize --output "${AF_OUTPUT_ROOT}"
else
  "${AF_PYTHON}" scripts/run_fitter_recovery.py run --output "${AF_OUTPUT_ROOT}" --task-index "${SLURM_ARRAY_TASK_ID:?}"
fi
