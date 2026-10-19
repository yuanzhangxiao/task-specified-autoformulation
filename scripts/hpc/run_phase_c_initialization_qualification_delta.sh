#!/bin/bash
# M43: known equations, training-only shared fit, strict-prefix latent inference.
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
    "$AF_PYTHON" -u scripts/phase_c_initialization_qualification.py verify --root "$AF_OUTPUT_ROOT"
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_initialization_qualification.py
    ;;
  run)
    "$AF_PYTHON" -u scripts/phase_c_initialization_qualification.py run \
      --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  report)
    "$AF_PYTHON" -u scripts/phase_c_initialization_qualification.py report --root "$AF_OUTPUT_ROOT"
    ;;
  *) exit 2 ;;
esac
