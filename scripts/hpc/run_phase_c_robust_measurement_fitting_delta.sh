#!/bin/bash
# Existing Delta CPU environment; fixed equations, no provider/GPU/test access.
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
    "$AF_PYTHON" scripts/phase_c_robust_measurement_fitting.py verify --root "$AF_OUTPUT_ROOT"
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider \
      tests/test_robust_measurement.py tests/test_robust_measurement_fit.py \
      tests/test_robust_measurement_fixtures.py tests/test_robust_measurement_campaign.py
    ;;
  run)
    "$AF_PYTHON" scripts/phase_c_robust_measurement_fitting.py run \
      --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  report)
    "$AF_PYTHON" scripts/phase_c_robust_measurement_fitting.py report --root "$AF_OUTPUT_ROOT"
    ;;
  *) exit 2 ;;
esac
