#!/bin/bash
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}"
site="$(jq -er '.site' "$AF_OUTPUT_ROOT/plan.json")"
case "$site" in
 aces) module load GCCcore/13.2.0 Python/3.11.5 ;;
 delta) ;;
 *) echo 'Unsupported diagnostic site' >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
 [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
else
 [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]
fi
case "${1:?stage}" in
 fit) exec "$AF_PYTHON" scripts/dalla_sign_diagnostic.py fit --root "$AF_OUTPUT_ROOT" ;;
 report) exec "$AF_PYTHON" scripts/dalla_sign_diagnostic.py report --root "$AF_OUTPUT_ROOT" ;;
 *) exit 2 ;;
esac
