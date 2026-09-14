#!/bin/bash
# CPU replay/freeze and preparation; GPU local repair. There is no fitting stage.
set -euo pipefail
: "${AF_REPO_ROOT:?required}" "${AF_PYTHON:?required}" "${AF_OUTPUT_ROOT:?required}"
mode="${1:?expected prepare or repair}"
[[ "$mode" == prepare || "$mode" == repair ]] || { echo 'Unknown worker mode' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$AF_OUTPUT_ROOT/runtime" "$AF_OUTPUT_ROOT/tmp"
export TMPDIR="$AF_OUTPUT_ROOT/tmp"
cd "$AF_REPO_ROOT"
if [[ "$mode" == prepare ]]; then
  : "${AF_SOURCE_ROOT:?required}" "${AF_CONFIG:?required}" "${AF_VLLM_IMAGE:?required}" "${AF_HF_HOME:?required}"
  preflight_log="$AF_OUTPUT_ROOT/runtime/preflight-${SLURM_JOB_ID}.log"
  printf 'Running preflight tests; full output: %s\n' "$preflight_log"
  if "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_prefit_replay.py tests/test_prefit_feedback.py tests/test_prefit_feedback_submission.py > "$preflight_log" 2>&1; then
    tail -n 2 "$preflight_log"
  else
    preflight_status=$?
    printf 'Preflight failed; full output: %s\n' "$preflight_log" >&2
    tail -n 80 "$preflight_log" >&2
    exit "$preflight_status"
  fi
  "$AF_PYTHON" scripts/smoke_prefit_feedback.py > "$AF_OUTPUT_ROOT/runtime/smoke-${SLURM_JOB_ID}.json"
  "$AF_PYTHON" scripts/prefit_response_replay.py --source "$AF_SOURCE_ROOT" --output "$AF_OUTPUT_ROOT/replay.json" > "$AF_OUTPUT_ROOT/runtime/replay-${SLURM_JOB_ID}.json"
  "$AF_PYTHON" scripts/prefit_feedback_campaign.py freeze --corpus "$AF_OUTPUT_ROOT/replay.json" --config "$AF_CONFIG" --output "$AF_OUTPUT_ROOT" > "$AF_OUTPUT_ROOT/runtime/freeze-${SLURM_JOB_ID}.json"
  "$AF_PYTHON" scripts/prefit_feedback_campaign.py cases --root "$AF_OUTPUT_ROOT" > "$AF_OUTPUT_ROOT/runtime/selected-cases.json"
  module load WebProxy
  runtime="$(command -v apptainer || command -v singularity)"
  expected="$(jq -er '.config.serving_image_sha256' "$AF_OUTPUT_ROOT/plan.json")"
  actual="$(sha256sum "$AF_VLLM_IMAGE")"
  [[ "${actual%% *}" == "$expected" ]] || { echo 'Frozen image SHA differs' >&2; exit 2; }
  container_python="$("$runtime" exec "$AF_VLLM_IMAGE" sh -c 'command -v python3 || command -v python')"
  [[ "$container_python" == /* && "$container_python" != *$'\n'* ]] || { echo 'Invalid container Python' >&2; exit 2; }
  model="$(jq -er '.config.model_settings.model' "$AF_OUTPUT_ROOT/plan.json")"
  revision="$(jq -er '.config.model_settings.model_revision' "$AF_OUTPUT_ROOT/plan.json")"
  "$runtime" exec --bind "$AF_HF_HOME:$AF_HF_HOME" --env "HF_HOME=$AF_HF_HOME" "$AF_VLLM_IMAGE" "$container_python" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2])' "$model" "$revision"
else
  "$AF_PYTHON" scripts/prefit_feedback_campaign.py verify --root "$AF_OUTPUT_ROOT"
  module load WebProxy
  exec bash scripts/hpc/run_staged_topology_server.sh
fi
