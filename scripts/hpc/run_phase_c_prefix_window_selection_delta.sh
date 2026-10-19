#!/bin/bash
# M44: opt-in prefix-window selection with frozen per-task input and policy seals.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}"
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]
fi
[[ ! -f SHA256SUMS ]] || sha256sum -c SHA256SUMS --quiet
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:?prepare, run or report}" in
  prepare)
    "$AF_PYTHON" -u scripts/phase_c_prefix_window_selection.py verify --root "$AF_OUTPUT_ROOT"
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider \
      tests/test_prefix_window_selection_guards.py \
      tests/test_prefix_window_diagnostics.py \
      tests/test_submit_prefix_window_selection.py
    ;;
  run)
    "$AF_PYTHON" -u scripts/phase_c_prefix_window_selection.py run \
      --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  report)
    "$AF_PYTHON" -u scripts/phase_c_prefix_window_selection.py report --root "$AF_OUTPUT_ROOT"
    ;;
  *) exit 2 ;;
esac
