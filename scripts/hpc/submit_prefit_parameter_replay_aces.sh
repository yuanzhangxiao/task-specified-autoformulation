#!/bin/bash
# One CPU allocation per explicit stage; replay never calls a proposer or fitter.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:?set the clean pinned checkout}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v2}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-parameter-replay-v1}"
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
export AF_REPLAY_ACTION="${AF_REPLAY_ACTION:-replay}"
[[ "$AF_REPLAY_ACTION" == replay || "$AF_REPLAY_ACTION" == fit ]] || exit 2
[[ -x "$AF_PYTHON" && -f "$AF_SOURCE_ROOT/results/state.json" ]] || { echo 'Missing Python or saved proposer state' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
# Resolve paths and refuse overlap before creating logs or submission records.
"$AF_PYTHON" - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['AF_OUTPUT_ROOT']).resolve()
for name in ('AF_REPO_ROOT', 'AF_SOURCE_ROOT'):
    source = Path(os.environ[name]).resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise SystemExit('Output must be separate from source and checkout')
PY
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$AF_REPO_ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo 'Use a clean pinned checkout' >&2; exit 2; }
verify_worker() {
  [[ "$(git rev-parse HEAD)" == "${AF_EXPECTED_COMMIT:?missing commit}" ]] || { echo 'Queued checkout changed' >&2; exit 2; }
  [[ -z "$(git status --porcelain)" ]] || { echo 'Queued checkout changed' >&2; exit 2; }
  [[ "$(sha256sum "$AF_SOURCE_ROOT/results/state.json" | cut -d ' ' -f 1)" == "${AF_EXPECTED_STATE:?missing state digest}" ]] || { echo 'Queued source state changed' >&2; exit 2; }
}
if [[ "${1:-submit}" == worker ]]; then
  verify_worker
  mkdir -p "$AF_OUTPUT_ROOT/tmp"
  export TMPDIR="$AF_OUTPUT_ROOT/tmp"
  if [[ "$AF_REPLAY_ACTION" == replay ]]; then
    "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_prefit_parameter_replay.py tests/test_numerical_sibling.py tests/test_sibling_fit.py
    "$AF_PYTHON" scripts/smoke_prefit_parameter_replay.py
    verify_worker
    "$AF_PYTHON" scripts/prefit_parameter_replay.py replay --source "$AF_SOURCE_ROOT" --root "$AF_OUTPUT_ROOT" --attempt 0
  else
    "$AF_PYTHON" scripts/prefit_parameter_replay.py fit --root "$AF_OUTPUT_ROOT"
  fi
  exit 0
fi
[[ "$AF_REPLAY_ACTION" != fit || -f "$AF_OUTPUT_ROOT/plan.json" ]] || { echo 'Run replay first' >&2; exit 2; }
export AF_EXPECTED_COMMIT="$(git rev-parse HEAD)"
export AF_EXPECTED_STATE="$(sha256sum "$AF_SOURCE_ROOT/results/state.json" | cut -d ' ' -f 1)"
identity="$(jq -ncS --arg commit "$AF_EXPECTED_COMMIT" --arg repo "$AF_REPO_ROOT" --arg source "$AF_SOURCE_ROOT" --arg output "$AF_OUTPUT_ROOT" --arg action "$AF_REPLAY_ACTION" --arg state "$AF_EXPECTED_STATE" '{commit:$commit,repo:$repo,source:$source,output:$output,action:$action,state_sha256:$state}')"
intent="$AF_OUTPUT_ROOT/submission-$AF_REPLAY_ACTION"
if [[ -f "$intent/manifest.json" ]]; then
  [[ "$(jq -cS '.identity' "$intent/manifest.json")" == "$identity" ]] || { echo 'Submission identity changed' >&2; exit 2; }
  cat "$intent/manifest.json"
  exit 0
fi
mkdir -p "$AF_OUTPUT_ROOT/logs"
mkdir "$intent" || { echo 'Submission intent exists; inspect the queue before retrying' >&2; exit 2; }
printf '%s\n' "$identity" > "$intent/identity.json"
job="$(sbatch --parsable --account="${AF_ACCOUNT:-156264627414}" --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=01:00:00 --export=ALL --job-name="parameter-$AF_REPLAY_ACTION" --output="$AF_OUTPUT_ROOT/logs/$AF_REPLAY_ACTION-%j.out" --error="$AF_OUTPUT_ROOT/logs/$AF_REPLAY_ACTION-%j.err" "$AF_REPO_ROOT/scripts/hpc/submit_prefit_parameter_replay_aces.sh" worker)"
job="${job%%;*}"
[[ "$job" =~ ^[0-9]+$ ]] || { echo 'Uncertain job ID; inspect submission intent' >&2; exit 2; }
jq -n --arg job_id "$job" --argjson identity "$identity" '{job_id:$job_id,identity:$identity}' > "$intent/manifest.json.tmp"
mv "$intent/manifest.json.tmp" "$intent/manifest.json"
cat "$intent/manifest.json"
