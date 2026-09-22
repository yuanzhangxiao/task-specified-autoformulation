#!/bin/bash
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$AF_REPO_ROOT/src"
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]
fi
"$AF_PYTHON" scripts/basin_repair_pilot.py verify --root "$AF_OUTPUT_ROOT"
case "${1:?stage}" in
 prepare)
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_basin_model_repair.py tests/test_basin_repair_pilot.py
    actual="$(sha256sum "$AF_VLLM_IMAGE")"
    [[ "${actual%% *}" == "$(jq -r '.config.serving_image_sha256' "$AF_OUTPUT_ROOT/plan.json")" ]]
    ;;
 propose) module load WebProxy; exec bash scripts/hpc/run_staged_topology_server.sh ;;
 fit) exec "$AF_PYTHON" scripts/basin_repair_pilot.py fit --root "$AF_OUTPUT_ROOT" ;;
 report) exec "$AF_PYTHON" scripts/basin_repair_pilot.py report --root "$AF_OUTPUT_ROOT" ;;
 *) exit 2 ;;
esac
