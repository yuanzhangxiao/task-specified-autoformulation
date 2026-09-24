#!/bin/bash
# Persistent services; no submission recursion and no test-data access.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPATH="$AF_REPO_ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "${AF_SITE:-aces}" == aces ]]; then
  module load GCCcore/13.2.0 Python/3.11.5
fi
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]
fi
"$AF_PYTHON" scripts/component_campaign.py verify --root "$AF_OUTPUT_ROOT"
case "${1:?stage}" in
  propose)
    if [[ "${AF_SITE:-aces}" == aces ]]; then module load WebProxy; fi
    exec bash scripts/hpc/run_staged_topology_server.sh
    ;;
  critic)
    : "${AF_JETSTREAM_API_KEY:?Supply via environment; never include in command arguments}"
    if [[ "${AF_SITE:-aces}" == aces ]]; then module load WebProxy; fi
    exec "$AF_PYTHON" scripts/component_campaign.py work --stage critic --root "$AF_OUTPUT_ROOT" --wall-seconds "${AF_WORKER_SECONDS:-21600}"
    ;;
  fit|prune)
    exec "$AF_PYTHON" scripts/component_campaign.py work --stage "$1" --root "$AF_OUTPUT_ROOT" --wall-seconds "${AF_WORKER_SECONDS:-21600}"
    ;;
  *) echo 'expected propose, critic, fit or prune' >&2; exit 2 ;;
esac
