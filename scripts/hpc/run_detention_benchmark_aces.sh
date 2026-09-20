#!/bin/bash
# A distinct development audit; neither old campaigns nor frozen fitting change.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_COMMIT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]] || exit 2
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]] || exit 2
fi
case "${1:?stage}" in
  prepare)
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_detention_benchmark.py tests/test_detention_submission.py
    exec "$AF_PYTHON" scripts/run_detention_benchmark.py prepare --output "$AF_OUTPUT_ROOT"
    ;;
  fit)
    exec "$AF_PYTHON" scripts/run_detention_benchmark.py fit --output "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  report)
    exec "$AF_PYTHON" scripts/run_detention_benchmark.py report --output "$AF_OUTPUT_ROOT"
    ;;
  *) exit 2 ;;
esac
