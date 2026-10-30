#!/bin/bash
# Separate training-only CPU campaign; this script also serves as its Slurm worker.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export AF_CONSTRUCTION_PORTFOLIO_ROOT=${AF_CONSTRUCTION_PORTFOLIO_ROOT:-/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-fitting-v1}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
AF_STAGE=${1:?Use run, submit, verify, status, report, or inspect}
shift
printf 'Construction portfolio: stage=%s host=%s started=%s\n' "$AF_STAGE" "$(hostname)" "$(date -u +%FT%TZ)" >&2
[[ -x "$AF_PYTHON" ]] || { echo 'Set AF_PYTHON to an existing project Python.' >&2; exit 2; }
AF_CLI="$AF_REPO_ROOT/scripts/run_construction_portfolio_fitting.py"
[[ -f "$AF_CLI" ]] || { echo 'Use the extracted source bundle, not an empty directory.' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_REPO_ROOT/vendor${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
case "$AF_STAGE" in
  run|prepare)
    AF_INTAKE=${AF_CONSTRUCTION_PORTFOLIO_INTAKE:-$AF_REPO_ROOT/inputs/scaled-readout-intake}
    AF_CONFIG=${AF_CONSTRUCTION_PORTFOLIO_CONFIG:-$AF_REPO_ROOT/configs/construction_portfolio_fitting_v1.json}
    [[ -f "$AF_INTAKE/manifest.json" ]] || { echo 'Set AF_CONSTRUCTION_PORTFOLIO_INTAKE to the extracted, complete 23-model delivery.' >&2; exit 2; }
    "$AF_PYTHON" "$AF_CLI" prepare --root "$AF_CONSTRUCTION_PORTFOLIO_ROOT" --source "$AF_INTAKE" --config "$AF_CONFIG"
    if [[ "$AF_STAGE" == run ]]; then
      exec "$AF_PYTHON" "$AF_CLI" submit --root "$AF_CONSTRUCTION_PORTFOLIO_ROOT"
    fi
    ;;
  worker-run)
    AF_INDEX=${SLURM_ARRAY_TASK_ID:-${1:-}}
    [[ "$AF_INDEX" =~ ^[0-9]+$ ]] || { echo 'Expected a nonnegative task index.' >&2; exit 2; }
    exec "$AF_PYTHON" "$AF_CLI" run --root "$AF_CONSTRUCTION_PORTFOLIO_ROOT" --index "$AF_INDEX"
    ;;
  worker-report)
    exec "$AF_PYTHON" "$AF_CLI" report --root "$AF_CONSTRUCTION_PORTFOLIO_ROOT"
    ;;
  submit|verify|verify-execution|status|report|inspect)
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_CONSTRUCTION_PORTFOLIO_ROOT"
    ;;
  *) echo 'Use run, prepare, submit, verify, status, report, or inspect.' >&2; exit 2 ;;
esac
