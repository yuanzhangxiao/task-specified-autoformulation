#!/bin/bash
# CPU-only evaluation of two empty-vector failures; source campaign stays intact.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_COMMIT:?}" "${AF_SOURCE_ROOT:?}" "${AF_OUTPUT_ROOT:?}"
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
[[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]] || exit 2
"$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_parameter_free_fitting.py tests/test_shared_process_recovery.py
exec "$AF_PYTHON" scripts/recover_shared_process_fixed_models.py \
  --source "$AF_SOURCE_ROOT" --output "$AF_OUTPUT_ROOT"
