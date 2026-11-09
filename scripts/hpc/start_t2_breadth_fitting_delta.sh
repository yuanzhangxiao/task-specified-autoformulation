#!/bin/bash
set -euo pipefail
AF_STAGE=${1:?Use run, prepare, submit, status, report, export, or inspect}
if [[ "$AF_STAGE" == slurm-* ]]; then
  : "${AF_REPO_ROOT:?Submitted worker requires AF_REPO_ROOT}"
else
  AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
export AF_REPO_ROOT
export AF_T2_BREADTH_FIT_ROOT=${AF_T2_BREADTH_FIT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/t2-breadth-fitting-v1}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export AF_FIT_RUNTIME=native
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT:$AF_REPO_ROOT/vendor:$AF_CASADI_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
AF_CLI="$AF_REPO_ROOT/scripts/run_t2_breadth_fitting.py"
printf 'T2 breadth fitting %s on %s, root=%s\n' "$AF_STAGE" "$(hostname)" "$AF_T2_BREADTH_FIT_ROOT" >&2
case "$AF_STAGE" in
  run|prepare)
    AF_ARCHIVE=${AF_T2_BREADTH_HANDOFF:-$AF_REPO_ROOT/inputs/orion-fitting-handoff.tar.gz}
    "$AF_PYTHON" "$AF_CLI" prepare --root "$AF_T2_BREADTH_FIT_ROOT" --archive "$AF_ARCHIVE"
    if [[ "$AF_STAGE" == run ]]; then
      exec "$AF_PYTHON" "$AF_CLI" submit --root "$AF_T2_BREADTH_FIT_ROOT"
    fi
    ;;
  slurm-task)
    exec "$AF_PYTHON" "$AF_CLI" slurm-task --root "$AF_T2_BREADTH_FIT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?Missing array task index}"
    ;;
  submit|status|report|slurm-report|inspect|export)
    exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_T2_BREADTH_FIT_ROOT"
    ;;
  *) echo "Unknown fitting stage: $AF_STAGE" >&2; exit 2 ;;
esac
