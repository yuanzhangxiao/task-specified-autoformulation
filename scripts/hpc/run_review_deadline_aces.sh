#!/bin/bash
# One server per proposer visit; independent CPU fits, then a dependency barrier.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_OUTPUT_ROOT:?}" "${AF_COMMIT:?}"
mode="${1:?prepare, propose, fit, finish, demo, or submit-next}"
export AF_ROUND="${2:-0}"
module load GCCcore/13.2.0 Python/3.11.5
cd "$AF_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]] || { echo 'Pinned commit changed' >&2; exit 2; }
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$AF_PYTHON" scripts/review_deadline.py verify --root "$AF_OUTPUT_ROOT"
case "$mode" in
 prepare)
    module load WebProxy
    "$AF_PYTHON" -c 'import casadi, scipy, numpy; print(casadi.__version__, scipy.__version__, numpy.__version__)'
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_review_deadline.py tests/test_review_deadline_submission.py tests/test_single_target_profile.py
    "$AF_PYTHON" scripts/smoke_review_deadline.py
    if [[ "$(jq -r '.config.protocol' "$AF_OUTPUT_ROOT/plan.json")" == review-deadline-2 ]]; then
      "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_review_deadline_v2.py tests/test_review_deadline_v2_submission.py
      "$AF_PYTHON" scripts/smoke_review_deadline_v2.py
    fi
    runtime="$(command -v apptainer || command -v singularity)"
    actual="$(sha256sum "$AF_VLLM_IMAGE")"
    [[ "${actual%% *}" == "$(jq -r '.config.serving_image_sha256' "$AF_OUTPUT_ROOT/plan.json")" ]] || exit 2
    model="$(jq -r '.config.model_settings.model' "$AF_OUTPUT_ROOT/plan.json")"
    revision="$(jq -r '.config.model_settings.model_revision' "$AF_OUTPUT_ROOT/plan.json")"
    "$runtime" exec --bind "$AF_HF_HOME:$AF_HF_HOME" --env "HF_HOME=$AF_HF_HOME" "$AF_VLLM_IMAGE" python3 -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2])' "$model" "$revision"
    ;;
 propose) module load WebProxy; exec bash scripts/hpc/run_staged_topology_server.sh ;;
 fit) exec "$AF_PYTHON" scripts/review_deadline.py fit-task --root "$AF_OUTPUT_ROOT" --round "$AF_ROUND" ;;
 finish) exec "$AF_PYTHON" scripts/review_deadline.py finish-round --root "$AF_OUTPUT_ROOT" --round "$AF_ROUND" ;;
 submit-next) exec bash scripts/hpc/submit_review_deadline_v2_aces.sh --round "$AF_ROUND" ;;
 demo) exec "$AF_PYTHON" scripts/review_deadline.py demo --root "$AF_OUTPUT_ROOT/controlled-demo" ;;
 *) echo 'unknown stage' >&2; exit 2 ;;
esac
