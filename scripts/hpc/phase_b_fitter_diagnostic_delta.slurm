#!/bin/bash
#SBATCH --job-name=fitter-diagnostic
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_CODE_COMMIT:?}"
cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_CODE_COMMIT}" ]] || {
  echo 'Checkout changed after submission; aborting.' >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo 'Checkout was edited after submission; aborting.' >&2; exit 2;
}
export PYTHONPATH="${AF_REPO_ROOT}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
readonly runner="${AF_REPO_ROOT}/scripts/run_phase_b_fitter_diagnostic.py"
case "${1:?stage required}" in
  prepare)
    "${AF_PYTHON}" "${runner}" prepare --config "${AF_CONFIG}" \
      --output-root "${AF_OUTPUT_ROOT}" --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
      --historical-root "${AF_HISTORICAL_RAW_ROOT}" --refresh-root "${AF_REFRESH_RAW_ROOT}"
    ;;
  run)
    "${AF_PYTHON}" "${runner}" run --output-root "${AF_OUTPUT_ROOT}" \
      --task-index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  summarize)
    "${AF_PYTHON}" "${runner}" summarize --output-root "${AF_OUTPUT_ROOT}"
    ;;
  *) echo 'Unknown diagnostic stage' >&2; exit 2 ;;
esac
