#!/bin/bash
# Separate replay, optional one-RHS proposal, and frozen-profile child fitting.
set -euo pipefail
: "${AF_REPO_ROOT:?required}" "${AF_PYTHON:?required}" "${AF_OUTPUT_ROOT:?required}"
mode="${1:?expected prepare, propose, or fit}"
[[ "$mode" == prepare || "$mode" == propose || "$mode" == fit ]] || exit 2
module load GCCcore/13.2.0 Python/3.11.5
verify_submission() {
  "$AF_PYTHON" - "$AF_OUTPUT_ROOT/submission-intent/identity.json" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
identity = json.loads(Path(sys.argv[1]).read_text())
repo = os.environ['AF_REPO_ROOT']
head = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
dirty = subprocess.check_output(['git', '-C', repo, 'status', '--porcelain'], text=True)
if head != identity['commit'] or dirty:
    raise SystemExit('Queued checkout changed; refusing numerical/provider work')
for name, expected in identity['paths'].items():
    if os.environ.get(name) != expected:
        raise SystemExit('Queued path changed: ' + name)
for name, expected in identity['sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Queued historical/config content changed: ' + name)
PY
}
verify_submission
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$AF_OUTPUT_ROOT/runtime" "$AF_OUTPUT_ROOT/tmp"
export TMPDIR="$AF_OUTPUT_ROOT/tmp"
cd "$AF_REPO_ROOT"
cli="$AF_REPO_ROOT/scripts/prefit_numerical_sibling.py"
if [[ "$mode" == prepare ]]; then
  log="$AF_OUTPUT_ROOT/runtime/preflight-${SLURM_JOB_ID}.log"
  if "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_residual_evidence.py tests/test_fit_residual_feedback.py tests/test_numerical_sibling.py tests/test_sibling_fit.py tests/test_prefit_numerical_sibling_submission.py tests/test_prefit_numerical_sibling_integration.py > "$log" 2>&1; then
    tail -n 2 "$log"
  else
    tail -n 100 "$log" >&2
    exit 1
  fi
  "$AF_PYTHON" scripts/smoke_prefit_numerical_sibling.py --feedback-policy "$(jq -r '.feedback_policy // "optional-review-1"' "$AF_CONFIG")" > "$AF_OUTPUT_ROOT/runtime/smoke-${SLURM_JOB_ID}.json"
  verify_submission
  "$AF_PYTHON" "$cli" prepare --parent-fit "$AF_PARENT_FIT" --continuation-fit "$AF_CONTINUATION_FIT" --source "$AF_SOURCE_ROOT" --construction-root "$AF_CONSTRUCTION_ROOT" --config "$AF_CONFIG" --root "$AF_OUTPUT_ROOT"
  "$AF_PYTHON" "$cli" replay --root "$AF_OUTPUT_ROOT" > "$AF_OUTPUT_ROOT/runtime/replay.json"
  if [[ "$(jq -r '.status' "$AF_OUTPUT_ROOT/runtime/replay.json")" != ready ]]; then
    "$AF_PYTHON" "$cli" run --root "$AF_OUTPUT_ROOT" --base-url http://unused
    exit 0
  fi
  module load WebProxy
  runtime="$(command -v apptainer || command -v singularity)"
  expected="$(jq -er '.config.serving_image_sha256' "$AF_OUTPUT_ROOT/plan.json")"
  actual="$(sha256sum "$AF_VLLM_IMAGE")"
  [[ "${actual%% *}" == "$expected" ]] || { echo 'Frozen image SHA differs' >&2; exit 2; }
  container_python="$("$runtime" exec "$AF_VLLM_IMAGE" sh -c 'command -v python3 || command -v python')"
  [[ "$container_python" == /* && "$container_python" != *$'\n'* ]] || exit 2
  model="$(jq -er '.config.model_settings.model' "$AF_OUTPUT_ROOT/plan.json")"
  revision="$(jq -er '.config.model_settings.model_revision' "$AF_OUTPUT_ROOT/plan.json")"
  "$runtime" exec --bind "$AF_HF_HOME:$AF_HF_HOME" --env "HF_HOME=$AF_HF_HOME" "$AF_VLLM_IMAGE" "$container_python" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2])' "$model" "$revision"
  verify_submission
elif [[ "$mode" == propose ]]; then
  "$AF_PYTHON" "$cli" verify --root "$AF_OUTPUT_ROOT" > "$AF_OUTPUT_ROOT/runtime/verify-${SLURM_JOB_ID}.json"
  if [[ "$(jq -r '.replay_status' "$AF_OUTPUT_ROOT/runtime/verify-${SLURM_JOB_ID}.json")" != ready ]]; then
    "$AF_PYTHON" "$cli" run --root "$AF_OUTPUT_ROOT" --base-url http://unused
    exit 0
  fi
  verify_submission
  module load WebProxy
  exec bash scripts/hpc/run_staged_topology_server.sh
else
  if [[ ! -f "$AF_OUTPUT_ROOT/plan.json" ]]; then
    jq -n '{protocol:"prefit-numerical-sibling-1",status:"preparation_missing",message:"No frozen plan; inspect preparation logs. No child fitting was attempted."}' > "$AF_OUTPUT_ROOT/summary.json"
    cat "$AF_OUTPUT_ROOT/summary.json"
    exit 1
  fi
  verify_submission
  status=0
  "$AF_PYTHON" "$cli" fit --root "$AF_OUTPUT_ROOT" || status=$?
  "$AF_PYTHON" "$cli" report --root "$AF_OUTPUT_ROOT"
  exit "$status"
fi
