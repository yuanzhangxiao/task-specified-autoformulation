#!/bin/bash
# One CPU: verified saved repair, unchanged numerical profile, no live LLM.
set -euo pipefail
: "${AF_REPO_ROOT:?required}" "${AF_OUTPUT_ROOT:?required}" "${AF_PYTHON:?required}"
: "${AF_SOURCE_ROOT:?required}" "${AF_CONSTRUCTION_ROOT:?required}" "${AF_PUBLIC_ROOT:?required}" "${AF_SELECTION:?required}"
module load GCCcore/13.2.0 Python/3.11.5
verify_submission() {
  local manifest="$AF_OUTPUT_ROOT/submission-intent/identity.json" selected
  [[ -f "$manifest" ]] || { echo 'Missing submission identity; use the submit script' >&2; return 2; }
  [[ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" == "$(jq -er '.commit' "$manifest")" && -z "$(git -C "$AF_REPO_ROOT" status --porcelain)" ]] || { echo 'Pinned checkout changed after submission' >&2; return 2; }
  selected="$(sha256sum "$AF_SELECTION" | cut -d' ' -f1)"
  [[ "$selected" == "$(jq -er '.selection_sha256' "$manifest")" ]] || { echo 'Selection file changed after submission' >&2; return 2; }
  if [[ -f "$AF_OUTPUT_ROOT/submission_manifest.json" ]]; then
    [[ "$(jq -c '.identity' "$AF_OUTPUT_ROOT/submission_manifest.json")" == "$(jq -c '.' "$manifest")" ]] || { echo 'Submission manifest identity differs' >&2; return 2; }
  fi
}
verify_submission
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$AF_OUTPUT_ROOT/runtime" "$AF_OUTPUT_ROOT/tmp"
export TMPDIR="$AF_OUTPUT_ROOT/tmp"
cd "$AF_REPO_ROOT"
log="$AF_OUTPUT_ROOT/runtime/preflight-${SLURM_JOB_ID:-local}.log"
if "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_prefit_fit_handoff.py tests/test_public_fitting.py tests/test_requirement_feedback.py tests/test_requirement_sign_policy.py > "$log" 2>&1; then
  tail -n 2 "$log"
else
  status=$?
  tail -n 80 "$log" >&2
  exit "$status"
fi
"$AF_PYTHON" -c 'import casadi, json, platform, numpy, scipy; print(json.dumps({"platform":"aces-cpu","python":platform.python_version(),"casadi":casadi.__version__,"numpy":numpy.__version__,"scipy":scipy.__version__}))' > "$AF_OUTPUT_ROOT/runtime/environment.json"
"$AF_PYTHON" scripts/smoke_prefit_fit_handoff.py > "$AF_OUTPUT_ROOT/runtime/smoke.json"
verify_submission
"$AF_PYTHON" scripts/prefit_fit_handoff.py prepare --source "$AF_SOURCE_ROOT" --construction-root "$AF_CONSTRUCTION_ROOT" --public-root "$AF_PUBLIC_ROOT" --selection "$AF_SELECTION" --output "$AF_OUTPUT_ROOT" > "$AF_OUTPUT_ROOT/runtime/prepared.json"
"$AF_PYTHON" scripts/prefit_fit_handoff.py run --output "$AF_OUTPUT_ROOT"
