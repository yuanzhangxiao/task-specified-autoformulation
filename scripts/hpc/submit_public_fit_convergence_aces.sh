#!/bin/bash
# One bounded diagnostic; uncertain submission cannot replenish numerical budgets.
set -euo pipefail
repo="${AF_REPO_ROOT:?set the clean pinned checkout}"
root="${AF_OUTPUT_ROOT:-/scratch/user/u.yx126462/phase_b/public-fit-convergence-v1}"
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
export AF_PARENT_FIT="${AF_PARENT_FIT:-/scratch/user/u.yx126462/phase_b/prefit-public-fit-v1/fit}"
export AF_CONTINUATION_FIT="${AF_CONTINUATION_FIT:-/scratch/user/u.yx126462/phase_b/public-fit-continuation-v1/continuation}"
export AF_SELECTION="${AF_SELECTION:-$repo/configs/public_fit_convergence_v1.json}"
[[ -x "$AF_PYTHON" && -f "$AF_SELECTION" ]] || { echo 'Missing Python or selection' >&2; exit 2; }
for artifact in freeze.json result.json backend_result.json; do
  [[ -f "$AF_PARENT_FIT/$artifact" ]] || { echo "Missing parent artifact: $artifact" >&2; exit 2; }
  [[ -f "$AF_CONTINUATION_FIT/$artifact" ]] || { echo "Missing continuation artifact: $artifact" >&2; exit 2; }
done
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo 'Use a clean pinned checkout' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
# Canonical paths also prevent aliases/symlinks from placing outputs in the parent.
paths="$("$AF_PYTHON" - "$repo" "$root" "$AF_PARENT_FIT" "$AF_CONTINUATION_FIT" "$AF_SELECTION" <<'PY'
import json
import sys
from pathlib import Path
repo, destination, parent, continuation, selection = [Path(v).resolve() for v in sys.argv[1:]]
for source in (parent.parent, continuation.parent, repo, selection):
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise SystemExit("Output must be separate from both historical experiments, checkout and selection")
print(json.dumps([str(p) for p in (repo, destination, parent, continuation, selection)]))
PY
)"
repo="$(jq -r '.[0]' <<< "$paths")"
root="$(jq -r '.[1]' <<< "$paths")"
export AF_PARENT_FIT="$(jq -r '.[2]' <<< "$paths")"
export AF_CONTINUATION_FIT="$(jq -r '.[3]' <<< "$paths")"
export AF_SELECTION="$(jq -r '.[4]' <<< "$paths")"
export AF_REPO_ROOT="$repo" AF_OUTPUT_ROOT="$root"
file_sha() { sha256sum "$1" | cut -d' ' -f1; }
identity="$(jq -cn --arg commit "$(git -C "$repo" rev-parse HEAD)" \
  --arg repo_root "$repo" --arg output_root "$root" \
  --arg selection_sha256 "$(file_sha "$AF_SELECTION")" \
  --arg parent_fit "$AF_PARENT_FIT" \
  --arg parent_freeze_sha256 "$(file_sha "$AF_PARENT_FIT/freeze.json")" \
  --arg parent_result_sha256 "$(file_sha "$AF_PARENT_FIT/result.json")" \
  --arg parent_backend_sha256 "$(file_sha "$AF_PARENT_FIT/backend_result.json")" \
  --arg continuation_fit "$AF_CONTINUATION_FIT" \
  --arg continuation_freeze_sha256 "$(file_sha "$AF_CONTINUATION_FIT/freeze.json")" \
  --arg continuation_result_sha256 "$(file_sha "$AF_CONTINUATION_FIT/result.json")" \
  --arg continuation_backend_sha256 "$(file_sha "$AF_CONTINUATION_FIT/backend_result.json")" \
  '{commit:$commit,repo_root:$repo_root,output_root:$output_root,selection_sha256:$selection_sha256,parent_fit:$parent_fit,parent_freeze_sha256:$parent_freeze_sha256,parent_result_sha256:$parent_result_sha256,parent_backend_sha256:$parent_backend_sha256,continuation_fit:$continuation_fit,continuation_freeze_sha256:$continuation_freeze_sha256,continuation_result_sha256:$continuation_result_sha256,continuation_backend_sha256:$continuation_backend_sha256}')"
manifest="$root/submission_manifest.json"
if [[ -f "$manifest" ]]; then
  [[ "$(jq -c '.identity' "$manifest")" == "$identity" ]] || { echo 'Submission identity changed; use the original checkout and inputs' >&2; exit 2; }
  cat "$manifest"
  exit 0
fi
[[ ! -e "$root/diagnostic" ]] || { echo 'A diagnostic directory already exists; inspect it using the CLI' >&2; exit 2; }
mkdir -p "$root/logs"
mkdir "$root/submission-intent" || { echo 'Submission intent exists; inspect queue/logs before retrying' >&2; exit 2; }
printf '%s\n' "$identity" > "$root/submission-intent/identity.json"
if ! job="$(sbatch --parsable --account="${AF_ACCOUNT:-156264627414}" --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=03:00:00 --job-name=public-fit-converge --export=ALL --output="$root/logs/converge-%j.out" --error="$root/logs/converge-%j.err" "$repo/scripts/hpc/run_public_fit_convergence_aces.sh")"; then
  echo 'Submission failed or is uncertain; saved intent prevents automatic duplication' >&2
  exit 2
fi
job="${job%%;*}"
[[ "$job" =~ ^[0-9]+$ ]] || { echo 'Invalid scheduler job ID; inspect saved intent' >&2; exit 2; }
jq -n --arg job_id "$job" --argjson identity "$identity" '{job_id:$job_id,identity:$identity,platform:"aces-cpu",cpus:1}' > "$manifest.tmp"
mv "$manifest.tmp" "$manifest"
cat "$manifest"
