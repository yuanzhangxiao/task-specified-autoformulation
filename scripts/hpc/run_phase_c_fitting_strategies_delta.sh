#!/bin/bash
# One CPU per fit, no model server, GPU or API credential.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}"
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]
fi
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:?prepare, fit or report}" in
  prepare)
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_fitting_strategies.py tests/test_fitting_budget_reuse.py
    if [[ -f tests/test_fitting_reuse_diagnostic.py ]]; then
      "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_fitting_reuse_diagnostic.py
    fi
    if [[ -f tests/test_challenging_fitting.py ]]; then
      "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_challenging_fitting.py
    fi
    if [[ -f tests/test_fitting_numerical_diagnostic.py ]]; then
      "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_fitting_numerical_diagnostic.py
    fi
    if [[ -f tests/test_fitting_checkpoint_diagnostic.py ]]; then
      "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_fitting_checkpoint_diagnostic.py
    fi
    if [[ -f tests/test_fitting_screening_diagnostic.py ]]; then
      "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_fitting_screening_diagnostic.py
    fi
    "$AF_PYTHON" scripts/phase_c_fitting_strategies.py qualify --root "$AF_OUTPUT_ROOT"
    ;;
  fit)
    "$AF_PYTHON" scripts/phase_c_fitting_strategies.py run --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  report)
    "$AF_PYTHON" scripts/phase_c_fitting_strategies.py report --root "$AF_OUTPUT_ROOT"
    ;;
  *) exit 2 ;;
esac
