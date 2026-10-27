#!/bin/bash
# One CPU and no provider calls; scheduler options are sealed by the submitter.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_LLMSR_DETENTION_LAZY_CONTROLS_ROOT:?}"
AF_STAGE=${1:?Use prepare, submit, qualify, run, status, report, or inspect}
printf 'Detention lazy controls: stage=%s host=%s started=%s\n' "$AF_STAGE" "$(hostname)" "$(date -u +%FT%TZ)" >&2
shift
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to an existing project Python.' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_REPO_ROOT/vendor${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
AF_CLI="$AF_REPO_ROOT/scripts/run_llm_sr_detention_lazy_controls.py"
case "$AF_STAGE" in
  prepare|submit|verify-execution)
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_LLMSR_DETENTION_LAZY_CONTROLS_ROOT" "$@"
    ;;
  status|report|inspect)
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_LLMSR_DETENTION_LAZY_CONTROLS_ROOT"
    ;;
  qualify|run)
    AF_INDEX=${SLURM_ARRAY_TASK_ID:-${1:-}}
    [[ "$AF_INDEX" =~ ^([0-9]|1[0-9]|2[0-6])$ ]] || { echo 'Expected task index 0 through 26.' >&2; exit 2; }
    if [[ "$AF_STAGE" == qualify && "$AF_INDEX" -gt 5 ]] || [[ "$AF_STAGE" == run && "$AF_INDEX" -lt 6 ]]; then
      echo 'Task index does not belong to this scheduler stage.' >&2; exit 2
    fi
    exec "$AF_PYTHON" "$AF_CLI" run --root "$AF_LLMSR_DETENTION_LAZY_CONTROLS_ROOT" --index "$AF_INDEX"
    ;;
  *) echo 'Unknown control stage.' >&2; exit 2 ;;
esac
