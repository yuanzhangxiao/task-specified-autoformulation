#!/bin/bash
# Native CPU runtime. Prepare validates schemas; numerical qualification is a task stage.
set -euo pipefail
: "${AF_REPO_ROOT:?required}" "${AF_ENDPOINT_REQUALIFICATION_ROOT:?required}"
AF_STAGE=${1:?Use prepare, submit, run, status, report, or inspect}
printf 'endpoint-requalification: worker-enter stage=%s host=%s time=%s\n' "$AF_STAGE" "${HOSTNAME:-unknown}" "$(date -u +%FT%TZ)" >&2
shift
export AF_PLATFORM=${AF_PLATFORM:-delta}
case "$AF_PLATFORM" in
  delta)
    export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
    export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
    ;;
  aces)
    export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
    export AF_CASADI_ROOT=${AF_CASADI_ROOT-}
    ;;
  *) echo 'AF_PLATFORM must be delta or aces.' >&2; exit 2 ;;
esac
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to an existing scientific Python.' >&2; exit 2; }
export AF_FIT_RUNTIME=native
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_REPO_ROOT/vendor${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf 'endpoint-requalification: dependency-import time=%s\n' "$(date -u +%FT%TZ)" >&2
AF_PROBE='import ctypes, numpy, scipy, pandas, pydantic, casadi, mpmath'
if ! "$AF_PYTHON" -c "$AF_PROBE" >/dev/null 2>&1; then
  if [[ "$AF_PLATFORM" == aces ]] && type module >/dev/null 2>&1; then
    module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  else
    echo 'Scientific Python import failed. Select AF_PYTHON and AF_CASADI_ROOT; no work was started.' >&2
    exit 2
  fi
fi
# Keep the submission and compute workers identical after EasyBuild modifies paths.
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_REPO_ROOT/vendor${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
"$AF_PYTHON" -c "$AF_PROBE"
cd "$AF_REPO_ROOT"
printf 'endpoint-requalification: cli-enter time=%s\n' "$(date -u +%FT%TZ)" >&2
AF_CLI="$AF_REPO_ROOT/scripts/run_construction_endpoint_requalification.py"
case "$AF_STAGE" in
  prepare|submit|verify-execution)
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_ENDPOINT_REQUALIFICATION_ROOT" "$@"
    ;;
  status|report|inspect)
    # Scheduler helpers append a task-index placeholder to the report job.
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_ENDPOINT_REQUALIFICATION_ROOT"
    ;;
  run)
    AF_INDEX=${SLURM_ARRAY_TASK_ID:-${1:-}}
    [[ "$AF_INDEX" =~ ^[0-9]+$ ]] || { echo 'Expected a nonnegative task index.' >&2; exit 2; }
    # The run CLI verifies the sealed execution identity before any numerical call.
    exec "$AF_PYTHON" "$AF_CLI" run --root "$AF_ENDPOINT_REQUALIFICATION_ROOT" --index "$AF_INDEX"
    ;;
  *) echo 'Use prepare, submit, run, status, report, or inspect.' >&2; exit 2 ;;
esac
