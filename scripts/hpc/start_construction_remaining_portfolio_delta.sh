#!/bin/bash
# Login-node preparation, submission and read-only result inspection.
set -euo pipefail
AF_STAGE=${1:?Use run, prepare, submit, verify, status, report, or inspect}
shift
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_REMAINING_PORTFOLIO_ROOT=${AF_REMAINING_PORTFOLIO_ROOT:-/work/hdd/bibo/yxiao2/phase_c/construction-remaining-portfolio-v1}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
printf 'Construction remaining portfolio: stage=%s host=%s started=%s\n' "$AF_STAGE" "$(hostname)" "$(date -u +%FT%TZ)" >&2
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to an existing project Python.' >&2; exit 2; }
AF_CLI="$AF_REPO_ROOT/scripts/run_construction_remaining_portfolio.py"
[[ -f "$AF_CLI" ]] || { echo 'Use the extracted source bundle, not an empty directory.' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_REPO_ROOT/vendor${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
case "$AF_STAGE" in
  run|prepare)
    AF_SOURCE=${AF_REMAINING_PORTFOLIO_SOURCE:-$AF_REPO_ROOT/inputs/remaining-portfolio-source}
    AF_CONFIG=${AF_REMAINING_PORTFOLIO_CONFIG:-$AF_REPO_ROOT/configs/construction_remaining_portfolio_v1.json}
    [[ -f "$AF_SOURCE" || -f "$AF_SOURCE/inputs.json" ]] || { echo 'Set AF_REMAINING_PORTFOLIO_SOURCE to the authenticated compact intake or its inputs.json.' >&2; exit 2; }
    "$AF_PYTHON" "$AF_CLI" prepare --root "$AF_REMAINING_PORTFOLIO_ROOT" --source "$AF_SOURCE" --config "$AF_CONFIG"
    if [[ "$AF_STAGE" == run ]]; then
      exec "$AF_PYTHON" "$AF_CLI" submit --root "$AF_REMAINING_PORTFOLIO_ROOT"
    fi
    ;;
  submit|verify|verify-execution|status|report|inspect)
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_REMAINING_PORTFOLIO_ROOT"
    ;;
  *) echo 'Use run, prepare, submit, verify, status, report, or inspect.' >&2; exit 2 ;;
esac
