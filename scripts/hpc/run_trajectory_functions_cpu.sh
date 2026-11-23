#!/bin/bash
# Slurm supplies the checkout and result root, independent of the spooled script.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${SLURM_JOB_ID:?}"
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
AF_CLI="$AF_REPO_ROOT/scripts/trajectory_functions.py"
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src"
fi
case "${1:-}" in
  fit)
    exec "$AF_PYTHON" "$AF_CLI" fit --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}" ;;
  validate)
    "$AF_PYTHON" "$AF_CLI" validate --root "$AF_OUTPUT_ROOT" --validation "$AF_REPO_ROOT/inputs/validation.json"
    exec "$AF_PYTHON" "$AF_CLI" inspect --root "$AF_OUTPUT_ROOT" ;;
  *) echo 'Expected fit or validate.' >&2; exit 2 ;;
esac
