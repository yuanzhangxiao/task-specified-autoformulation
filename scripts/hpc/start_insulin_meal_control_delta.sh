#!/bin/bash
set -euo pipefail
AF_STAGE=${1:?Expected run, prepare, submit, status, report or inspect}
case "$AF_STAGE" in
  worker-*) : "${AF_INSULIN_MEAL_CODE:?Worker requires the submitted checkout}" ;;
  *) AF_INSULIN_MEAL_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) ;;
esac
export AF_INSULIN_MEAL_CODE
export AF_INSULIN_MEAL_ROOT=${AF_INSULIN_MEAL_ROOT:-/work/hdd/bibo/yxiao2/phase_c/insulin-meal-control-v1}
export AF_PYTHON=${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}
export AF_CASADI_ROOT=${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}
export PYTHONPATH="$AF_INSULIN_MEAL_CODE/src:$AF_INSULIN_MEAL_CODE:$AF_INSULIN_MEAL_CODE/vendor${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
AF_CLI="$AF_INSULIN_MEAL_CODE/scripts/run_insulin_meal_control.py"
cd "$AF_INSULIN_MEAL_CODE"
case "$AF_STAGE" in
  run) "$AF_PYTHON" "$AF_CLI" prepare --root "$AF_INSULIN_MEAL_ROOT"
       exec "$AF_PYTHON" "$AF_CLI" submit --root "$AF_INSULIN_MEAL_ROOT" ;;
  worker-run) exec "$AF_PYTHON" "$AF_CLI" worker --root "$AF_INSULIN_MEAL_ROOT" --index "${SLURM_ARRAY_TASK_ID:?missing array index}" ;;
  worker-report) exec "$AF_PYTHON" "$AF_CLI" report --root "$AF_INSULIN_MEAL_ROOT" ;;
  prepare|submit|status|report|inspect)
       exec "$AF_PYTHON" "$AF_CLI" "$AF_STAGE" --root "$AF_INSULIN_MEAL_ROOT" ;;
  *) echo "Unknown stage: $AF_STAGE" >&2; exit 2 ;;
esac
