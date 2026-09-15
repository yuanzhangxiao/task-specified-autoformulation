#!/bin/bash
# One CPU, one declared warm-start extension, with the historical parent read-only.
set -euo pipefail
: "${AF_REPO_ROOT:?required}" "${AF_OUTPUT_ROOT:?required}" "${AF_PYTHON:?required}"
: "${AF_PARENT_FIT:?required}" "${AF_SELECTION:?required}"
module load GCCcore/13.2.0 Python/3.11.5
file_sha() { sha256sum "$1" | cut -d' ' -f1; }
verify_submission() {
  local manifest="$AF_OUTPUT_ROOT/submission-intent/identity.json" artifact field
  [[ -f "$manifest" ]] || { echo 'Missing submission identity; use the submit script' >&2; return 2; }
  [[ "$AF_REPO_ROOT" == "$(jq -er '.repo_root' "$manifest")" && "$AF_OUTPUT_ROOT" == "$(jq -er '.output_root' "$manifest")" && "$AF_PARENT_FIT" == "$(jq -er '.parent_fit' "$manifest")" ]] || { echo 'Submission paths changed after submission' >&2; return 2; }
  # Recheck the whole experiment boundary even for a manually supplied old intent.
  "$AF_PYTHON" - "$AF_REPO_ROOT" "$AF_OUTPUT_ROOT" "$AF_PARENT_FIT" "$AF_SELECTION" <<'PY' || return 2
import sys
from pathlib import Path
repo, destination, parent, selection = [Path(v).resolve() for v in sys.argv[1:]]
for source in (parent.parent, repo, selection):
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise SystemExit("Output must be separate from the parent experiment, checkout and selection")
PY
  [[ "$(git -C "$AF_REPO_ROOT" rev-parse HEAD)" == "$(jq -er '.commit' "$manifest")" && -z "$(git -C "$AF_REPO_ROOT" status --porcelain)" ]] || { echo 'Pinned checkout changed after submission' >&2; return 2; }
  [[ "$(file_sha "$AF_SELECTION")" == "$(jq -er '.selection_sha256' "$manifest")" ]] || { echo 'Selection file changed after submission' >&2; return 2; }
  for artifact in freeze result backend_result; do
    case "$artifact" in
      freeze) field=parent_freeze_sha256 ;;
      result) field=parent_result_sha256 ;;
      backend_result) field=parent_backend_sha256 ;;
    esac
    [[ -f "$AF_PARENT_FIT/$artifact.json" && "$(file_sha "$AF_PARENT_FIT/$artifact.json")" == "$(jq -er ".$field" "$manifest")" ]] || { echo "Parent $artifact changed after submission" >&2; return 2; }
  done
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
if "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_public_fit_continuation.py tests/test_public_fit_continuation_submission.py > "$log" 2>&1; then
  tail -n 2 "$log"
else
  status=$?
  tail -n 80 "$log" >&2
  exit "$status"
fi
"$AF_PYTHON" -c 'import casadi, json, platform, numpy, scipy; print(json.dumps({"platform":"aces-cpu","python":platform.python_version(),"casadi":casadi.__version__,"numpy":numpy.__version__,"scipy":scipy.__version__}))' > "$AF_OUTPUT_ROOT/runtime/environment.json"
"$AF_PYTHON" scripts/smoke_public_fit_continuation.py > "$AF_OUTPUT_ROOT/runtime/smoke.json"
verify_submission
directory="$AF_OUTPUT_ROOT/continuation"
"$AF_PYTHON" scripts/run_public_fit_continuation.py prepare --parent "$AF_PARENT_FIT" --selection "$AF_SELECTION" --output "$directory" > "$AF_OUTPUT_ROOT/runtime/prepared.json"
"$AF_PYTHON" scripts/run_public_fit_continuation.py inspect --output "$directory" > "$AF_OUTPUT_ROOT/runtime/inspection.json"
"$AF_PYTHON" scripts/run_public_fit_continuation.py run --output "$directory" > "$AF_OUTPUT_ROOT/runtime/run.json"
"$AF_PYTHON" scripts/run_public_fit_continuation.py report --output "$directory" > "$AF_OUTPUT_ROOT/summary.json.tmp"
mv "$AF_OUTPUT_ROOT/summary.json.tmp" "$AF_OUTPUT_ROOT/summary.json"
cat "$AF_OUTPUT_ROOT/summary.json"
