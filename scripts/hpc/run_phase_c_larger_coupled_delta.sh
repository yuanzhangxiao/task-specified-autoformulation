#!/bin/bash
# M18 uses existing Delta CPU dependencies; no provider or GPU.
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
case "${1:?prepare, run or report}" in
  prepare)
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_larger_coupled.py tests/test_larger_coupled_inputs.py tests/test_coupled_polishing.py tests/test_operation_timing.py tests/test_polishing_fit.py tests/test_polishing_campaign.py tests/test_profiled_coupled.py tests/test_coupled_campaign.py tests/test_profiled_campaign.py tests/test_profiled_output.py tests/test_checkpoint_publication.py tests/test_generic_recovery.py tests/test_mesh_refinement.py tests/test_screening_replay.py tests/test_fitting_screening_diagnostic.py
    "$AF_PYTHON" scripts/phase_c_larger_coupled.py qualify --root "$AF_OUTPUT_ROOT"
    ;;
  run)
    "$AF_PYTHON" scripts/phase_c_larger_coupled.py "$1" --root "$AF_OUTPUT_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  report)
    "$AF_PYTHON" scripts/phase_c_larger_coupled.py report --root "$AF_OUTPUT_ROOT"
    ;;
  *) exit 2 ;;
esac
