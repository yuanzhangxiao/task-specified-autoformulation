#!/bin/bash
# Public-only matched construction/revision; no new fitter or automatic expansion.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
export AF_ROUND="${2:?round}"
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
"$AF_PYTHON" scripts/review_deadline.py verify --root "$AF_OUTPUT_ROOT"
case "${1:?stage}" in
 prepare)
    "$AF_PYTHON" -c 'import casadi, scipy; print(casadi.__version__, scipy.__version__)'
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_shared_process_pilot.py tests/test_review_revision_v6.py tests/test_shared_process_pilot_submission.py
    if [[ "$(jq -r '.protocol' "$AF_OUTPUT_ROOT/plan.json")" == shared-process-integration-1 ]]; then
      "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_shared_process_integration.py
      "$AF_PYTHON" scripts/smoke_shared_process_integration.py
    else
      "$AF_PYTHON" scripts/smoke_shared_process_pilot.py
    fi
    actual="$(sha256sum "$AF_VLLM_IMAGE")"
    [[ "${actual%% *}" == "$(jq -r '.config.serving_image_sha256' "$AF_OUTPUT_ROOT/plan.json")" ]] || exit 2
    ;;
 propose) module load WebProxy; exec bash scripts/hpc/run_staged_topology_server.sh ;;
 fit) exec "$AF_PYTHON" scripts/review_deadline.py fit-task --root "$AF_OUTPUT_ROOT" --round "$AF_ROUND" ;;
 finish)
    "$AF_PYTHON" scripts/review_deadline.py finish-round --root "$AF_OUTPUT_ROOT" --round "$AF_ROUND"
    "$AF_PYTHON" - "$AF_OUTPUT_ROOT" "$AF_ROUND" <<'PY'
import sys
from pathlib import Path
from autoformalism.rebuttal import review_deadline_io as io
root, index = Path(sys.argv[1]), int(sys.argv[2])
plan = io.verify(root)
missing = [t['task_id'] for t in plan['tasks'] if io.read_round(root, t, index) is None]
if missing:
    raise SystemExit('Incomplete round; no automatic rerun or new allowance: ' + str(missing))
PY
    ;;
 *) echo 'Unknown stage' >&2; exit 2 ;;
esac
