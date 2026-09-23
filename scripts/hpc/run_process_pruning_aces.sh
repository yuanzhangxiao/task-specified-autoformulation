#!/bin/bash
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
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
   "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_process_pruning.py tests/test_process_pruning_campaign.py
   "$AF_PYTHON" scripts/smoke_process_pruning.py
   exec "$AF_PYTHON" scripts/process_pruning.py freeze --source "${AF_SOURCE_ROOT:?}" --root "$AF_OUTPUT_ROOT"
   ;;
 fit) exec "$AF_PYTHON" scripts/process_pruning.py fit --root "$AF_OUTPUT_ROOT" ;;
 report) exec "$AF_PYTHON" scripts/process_pruning.py report --root "$AF_OUTPUT_ROOT" ;;
 *) exit 2 ;;
esac
