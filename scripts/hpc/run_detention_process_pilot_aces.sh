#!/bin/bash
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$AF_REPO_ROOT/src"
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]] || exit 2
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]] || exit 2
fi
"$AF_PYTHON" scripts/detention_process_pilot.py verify --root "$AF_OUTPUT_ROOT"
case "${1:?stage}" in
 prepare)
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_process_review.py tests/test_shared_process_contract.py tests/test_detention_process_pilot.py tests/test_detention_process_submission.py tests/test_detention_bound_pilot.py tests/test_signed_processes.py tests/test_process_gain_comparison.py
    if jq -e '.handoff_confirmation' "$AF_OUTPUT_ROOT/plan.json" >/dev/null; then
      "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_process_handoff.py tests/test_process_handoff_confirmation.py
    fi
    actual="$(sha256sum "$AF_VLLM_IMAGE")"
    [[ "${actual%% *}" == "$(jq -r '.config.serving_image_sha256' "$AF_OUTPUT_ROOT/plan.json")" ]] || exit 2
    ;;
 propose) module load WebProxy; exec bash scripts/hpc/run_staged_topology_server.sh ;;
 fit) exec "$AF_PYTHON" scripts/detention_process_pilot.py fit --root "$AF_OUTPUT_ROOT" ;;
 report)
    if jq -e '.handoff_confirmation' "$AF_OUTPUT_ROOT/plan.json" >/dev/null; then
      exec "$AF_PYTHON" scripts/process_handoff_confirmation.py report --root "$AF_OUTPUT_ROOT"
    fi
    exec "$AF_PYTHON" scripts/detention_process_pilot.py report --root "$AF_OUTPUT_ROOT"
    ;;
 *) exit 2 ;;
esac
