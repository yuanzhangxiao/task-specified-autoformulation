#!/bin/bash
# CPU-only, immutable saved models. No provider, refitting or test access.
set -euo pipefail
repo="${AF_REPO_ROOT:?set AF_REPO_ROOT}"
python="${AF_PYTHON:?set AF_PYTHON}"
root="${AF_OUTPUT_ROOT:?set AF_OUTPUT_ROOT}"
[[ "$(git -C "$repo" rev-parse HEAD)" == "${AF_COMMIT:?set AF_COMMIT}" ]]
export PYTHONPATH="$repo/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${1:-run}" == report ]]; then
  exec "$python" "$repo/scripts/assess_mechanisms.py" report --root "$root"
fi
exec "$python" "$repo/scripts/assess_mechanisms.py" run --root "$root" \
  --index "${SLURM_ARRAY_TASK_ID:?array task required}"
