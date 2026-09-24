#!/bin/bash
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}" "${AF_INPUTS:?}"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]
fi
case "${1:?stage}" in
 prepare)
   "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_dalla_rescue.py tests/test_sibling_fit.py tests/test_process_pruning.py
   "$AF_PYTHON" scripts/smoke_dalla_rescue.py
   exec "$AF_PYTHON" scripts/dalla_rescue.py prepare --inputs "$AF_INPUTS" --root "$AF_OUTPUT_ROOT"
   ;;
 fit) exec "$AF_PYTHON" scripts/dalla_rescue.py fit --root "$AF_OUTPUT_ROOT" ;;
 report) exec "$AF_PYTHON" scripts/dalla_rescue.py report --root "$AF_OUTPUT_ROOT" ;;
 *) exit 2 ;;
esac
