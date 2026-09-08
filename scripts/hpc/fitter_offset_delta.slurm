#!/bin/bash
#SBATCH --job-name=fitter-offset
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_CODE_COMMIT:?}"
cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_CODE_COMMIT}" ]] || {
  echo 'Checkout changed after submission.' >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo 'Use an unchanged isolated checkout.' >&2; exit 2;
}
export PYTHONPATH="${AF_REPO_ROOT}/src"
case "${1:-run}" in
  run)
    "${AF_PYTHON}" scripts/run_fitter_offset.py run --output "${AF_OUTPUT_ROOT}" \
      --task-index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  summarize)
    "${AF_PYTHON}" scripts/run_fitter_offset.py summarize --output "${AF_OUTPUT_ROOT}"
    ;;
  *) echo 'Unknown stage.' >&2; exit 2 ;;
esac
