#!/bin/bash
# Slurm uses a spool copy, so preserve the submitted repository explicitly.
set -euo pipefail
AF_STAGE=${1:?Use run or report}
shift
: "${AF_REPO_ROOT:?Slurm workers require exported AF_REPO_ROOT}"
: "${AF_INSULIN_POSITIVE_CONTINUATION_ROOT:?Slurm workers require the frozen campaign root}"
: "${AF_PYTHON:?Slurm workers require the submitted AF_PYTHON}"
AF_CLI="$AF_REPO_ROOT/scripts/run_insulin_positive_continuation.py"
[[ -x "$AF_PYTHON" && -f "$AF_CLI" ]] || { echo 'Submitted Python or repository is unavailable.' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_REPO_ROOT/vendor${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf 'Insulin positive continuation worker: stage=%s host=%s started=%s\n' "$AF_STAGE" "$(hostname)" "$(date -u +%FT%TZ)" >&2
cd "$AF_REPO_ROOT"
case "$AF_STAGE" in
  run)
    AF_INDEX=${SLURM_ARRAY_TASK_ID:-${1:-}}
    [[ "$AF_INDEX" =~ ^[0-9]+$ ]] || { echo 'Expected a nonnegative task index.' >&2; exit 2; }
    exec "$AF_PYTHON" "$AF_CLI" run --root "$AF_INSULIN_POSITIVE_CONTINUATION_ROOT" --index "$AF_INDEX"
    ;;
  report)
    exec "$AF_PYTHON" "$AF_CLI" report --root "$AF_INSULIN_POSITIVE_CONTINUATION_ROOT"
    ;;
  *) echo 'Use run or report.' >&2; exit 2 ;;
esac
