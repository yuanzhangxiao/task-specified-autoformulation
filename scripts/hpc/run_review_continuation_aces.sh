#!/bin/bash
# Real script arguments survive the ACES sbatch wrapper; no --wrap command.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]] || { echo 'Pinned commit changed' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mode="${1:?stage}"; index="${2:-0}"
if [[ "$mode" == dispatch ]]; then
  exec "$AF_PYTHON" scripts/submit_review_continuation.py --root "$AF_OUTPUT_ROOT" --round "$index"
fi
if [[ "$mode" == prepare ]]; then
  "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_review_continuation.py
  if [[ "$(jq -r '.protocol' "$AF_OUTPUT_ROOT/plan.json")" == review-deadline-4 ]]; then
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_review_parameters.py
    "$AF_PYTHON" scripts/smoke_review_parameters.py
    "$AF_PYTHON" scripts/audit_review_parameters.py --root "$AF_OUTPUT_ROOT"
  else
    "$AF_PYTHON" scripts/smoke_review_continuation.py
    "$AF_PYTHON" scripts/audit_review_continuation.py --root "$AF_OUTPUT_ROOT"
  fi
fi
exec bash scripts/hpc/run_review_deadline_aces.sh "$mode" "$index"
