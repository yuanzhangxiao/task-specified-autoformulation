#!/bin/bash
# One selected, previously unstarted CPU fit; no provider calls or promotion.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPATH="$AF_REPO_ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
[[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
# The component protocol fixes its outer fit worker allowance at 1800 seconds.
exec timeout --signal=TERM --kill-after=15s 1800s \
  "$AF_PYTHON" scripts/recover_initialization_namespace.py run \
  --root "$AF_OUTPUT_ROOT" --task-index "${SLURM_ARRAY_TASK_ID:?}"
