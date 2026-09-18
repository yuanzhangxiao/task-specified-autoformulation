#!/bin/bash
# Standalone recovery code, original frozen campaign Python and package source.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_RECOVERY_ROOT:?}" "${AF_RECOVERY_CODE:?}"
module load GCCcore/13.2.0 Python/3.11.5
[[ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" = dfc6f81a613e186ddffdd0b2406feca583a58944 ]]
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "${1:?stage}" in
  worker)
    exec "$AF_PYTHON" "$AF_RECOVERY_CODE/recover_review_quota.py" worker \
      --root "$AF_RECOVERY_ROOT" --slot "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  finalize)
    exec "$AF_PYTHON" "$AF_RECOVERY_CODE/recover_review_quota.py" finalize \
      --root "$AF_RECOVERY_ROOT"
    ;;
  *) echo 'Expected worker or finalize' >&2; exit 2 ;;
esac
