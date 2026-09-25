#!/bin/bash
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}"
module load GCCcore/13.2.0 Python/3.11.5
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
 prepare)
   "$AF_PYTHON" scripts/dalla_sign_repair.py verify --root "$AF_OUTPUT_ROOT"
   "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_sign_review.py tests/test_directional_sign_review.py tests/test_dalla_sign_repair.py tests/test_dalla_rescue.py
   "$AF_PYTHON" scripts/smoke_dalla_sign_repair.py --protocol "$(jq -er '.protocol' "$AF_OUTPUT_ROOT/plan.json")"
   actual="$(sha256sum "$AF_VLLM_IMAGE")"
   [[ "${actual%% *}" == "$(jq -er '.config.serving_image_sha256' "$AF_OUTPUT_ROOT/plan.json")" ]]
   runtime="$(command -v apptainer || command -v singularity)"
   model="$(jq -er '.config.model_settings.model' "$AF_OUTPUT_ROOT/plan.json")"
   revision="$(jq -er '.config.model_settings.model_revision' "$AF_OUTPUT_ROOT/plan.json")"
   "$runtime" exec --bind "$AF_HF_HOME:$AF_HF_HOME" --env "HF_HOME=$AF_HF_HOME,HF_HUB_OFFLINE=1" \
     "$AF_VLLM_IMAGE" python3 -c 'from huggingface_hub import snapshot_download; import sys; print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))' "$model" "$revision"
   ;;
 review) module load WebProxy; exec bash scripts/hpc/run_staged_topology_server.sh ;;
 fit) exec "$AF_PYTHON" scripts/dalla_sign_repair.py fit --root "$AF_OUTPUT_ROOT" ;;
 report) exec "$AF_PYTHON" scripts/dalla_sign_repair.py report --root "$AF_OUTPUT_ROOT" ;;
 *) exit 2 ;;
esac
