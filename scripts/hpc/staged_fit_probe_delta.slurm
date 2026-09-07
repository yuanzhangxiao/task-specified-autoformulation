#!/bin/bash
#SBATCH --job-name=staged-fit-probe
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
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
  echo 'Checkout is dirty.' >&2; exit 2;
}
export PYTHONPATH="${AF_REPO_ROOT}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"${AF_PYTHON}" scripts/run_staged_fit_probe.py run --output "${AF_OUTPUT_ROOT}"
