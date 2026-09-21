#!/bin/bash
# CPU-only read-only source audit. Output checkpoints live in a separate directory.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_SOURCE_ROOT:?}"
: "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$AF_REPO_ROOT/src"
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]] || exit 2
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]] || exit 2
fi
"$AF_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_process_handoff.py tests/test_process_equation_diagnostics.py \
  tests/test_signed_processes.py tests/test_process_gain_comparison.py
exec "$AF_PYTHON" scripts/audit_process_handoff.py \
  --source "$AF_SOURCE_ROOT" --output "$AF_OUTPUT_ROOT"
