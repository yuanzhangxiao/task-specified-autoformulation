#!/bin/bash
# One CPU and no provider calls; scheduler options are sealed by the submitter.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_LLMSR_CONTROL_ROOT:?}"
AF_STAGE=${1:?Use prepare, submit, run, status, report, or inspect}
shift
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to an existing project Python.' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_REPO_ROOT/vendor${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
AF_CLI="$AF_REPO_ROOT/scripts/run_llm_sr_numerical_controls.py"
case "$AF_STAGE" in
  prepare|submit|verify-execution)
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_LLMSR_CONTROL_ROOT" "$@"
    ;;
  status|report|inspect)
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_LLMSR_CONTROL_ROOT"
    ;;
  run)
    AF_INDEX=${SLURM_ARRAY_TASK_ID:-${1:-}}
    [[ "$AF_INDEX" =~ ^[01]$ ]] || { echo 'Expected task index 0 or 1.' >&2; exit 2; }
    exec "$AF_PYTHON" "$AF_CLI" run --root "$AF_LLMSR_CONTROL_ROOT" --index "$AF_INDEX"
    ;;
  *) echo 'Unknown control stage.' >&2; exit 2 ;;
esac
