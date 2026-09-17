#!/bin/bash
# One task per array element; completed native generations and calls are cached.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
[[ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" == "$AF_COMMIT" ]] || exit 2
git -C "$AF_REPO_ROOT" diff --quiet
git -C "$AF_REPO_ROOT" diff --cached --quiet
export PYTHONPATH="$AF_REPO_ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$AF_REPO_ROOT"
cli="$AF_REPO_ROOT/scripts/phase_b_d3.py"
case "${1:-run}" in
  prepare)
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_d3_rollout.py tests/test_phase_b_d3.py
    "$AF_PYTHON" scripts/smoke_phase_b_d3.py --require-torch
    "$AF_PYTHON" "$cli" prepare --config "${AF_D3_CONFIG:?}" \
      --public-root "${AF_PUBLIC_ROOT:?}" --model "${AF_D3_MODEL:?}" --root "$AF_OUTPUT_ROOT"
    ;;
  run) "$AF_PYTHON" "$cli" run --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}" ;;
  report) "$AF_PYTHON" "$cli" report --root "$AF_OUTPUT_ROOT" ;;
  *) exit 2 ;;
esac
