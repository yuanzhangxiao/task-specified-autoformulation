#!/bin/bash
# Separate saved-evidence recovery campaign; also serves as its Slurm worker.
set -euo pipefail
AF_STAGE=${1:?Use run, submit, verify, status, report, or inspect}
shift
case "$AF_STAGE" in
  worker-run|worker-report)
    # Slurm executes a spool copy; its location is not the submitted repository.
    : "${AF_REPO_ROOT:?Slurm workers require exported AF_REPO_ROOT}"
    export AF_REPO_ROOT
    ;;
  *) export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) ;;
esac
export AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT=${AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT:-/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-recovery-v1}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
printf 'Construction portfolio recovery: stage=%s host=%s started=%s\n' "$AF_STAGE" "$(hostname)" "$(date -u +%FT%TZ)" >&2
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to an existing project Python.' >&2; exit 2; }
AF_CLI="$AF_REPO_ROOT/scripts/run_construction_portfolio_recovery.py"
[[ -f "$AF_CLI" ]] || { echo 'Use the extracted source bundle, not an empty directory.' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_REPO_ROOT/vendor${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
case "$AF_STAGE" in
  run|prepare)
    AF_SOURCE=${AF_CONSTRUCTION_PORTFOLIO_SOURCE:-/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-fitting-v2}
    if [[ -z "${AF_CONSTRUCTION_PORTFOLIO_SOURCE:-}" && ! -d "$AF_SOURCE" && -d "$AF_REPO_ROOT/inputs/portfolio-source" ]]; then
      AF_SOURCE="$AF_REPO_ROOT/inputs/portfolio-source"
    fi
    AF_CONFIG=${AF_CONSTRUCTION_PORTFOLIO_RECOVERY_CONFIG:-$AF_REPO_ROOT/configs/construction_portfolio_recovery_v1.json}
    [[ -f "$AF_SOURCE/plan.json" && -f "$AF_SOURCE/inputs.json" ]] || { echo 'Set AF_CONSTRUCTION_PORTFOLIO_SOURCE to the complete saved portfolio campaign.' >&2; exit 2; }
    "$AF_PYTHON" "$AF_CLI" prepare --root "$AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT" --source "$AF_SOURCE" --config "$AF_CONFIG"
    if [[ "$AF_STAGE" == run ]]; then
      exec "$AF_PYTHON" "$AF_CLI" submit --root "$AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT"
    fi
    ;;
  worker-run)
    AF_INDEX=${SLURM_ARRAY_TASK_ID:-${1:-}}
    [[ "$AF_INDEX" =~ ^[0-9]+$ ]] || { echo 'Expected a nonnegative task index.' >&2; exit 2; }
    exec "$AF_PYTHON" "$AF_CLI" run --root "$AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT" --index "$AF_INDEX"
    ;;
  worker-report)
    exec "$AF_PYTHON" "$AF_CLI" report --root "$AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT"
    ;;
  submit|verify|verify-execution|status|report|inspect)
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT"
    ;;
  *) echo 'Use run, prepare, submit, verify, status, report, or inspect.' >&2; exit 2 ;;
esac
