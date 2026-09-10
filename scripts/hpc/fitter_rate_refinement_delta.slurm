#!/bin/bash
#SBATCH --job-name=fitter-rates
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_CODE_COMMIT:?}"
cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_CODE_COMMIT}" && -z "$(git status --porcelain)" ]] || {
  echo 'Methods checkout changed after submission.' >&2; exit 2;
}
export AF_CASADI_ROOT="${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}"
export PYTHONPATH="${AF_REPO_ROOT}/src:${AF_CASADI_ROOT}"
if [[ "${1:-run}" == summarize ]]; then
  "${AF_PYTHON}" scripts/run_fitter_rate_refinement.py summarize --output "${AF_OUTPUT_ROOT}"
else
  "${AF_PYTHON}" scripts/run_fitter_rate_refinement.py run --output "${AF_OUTPUT_ROOT}" --task-index "${SLURM_ARRAY_TASK_ID:?}"
fi
