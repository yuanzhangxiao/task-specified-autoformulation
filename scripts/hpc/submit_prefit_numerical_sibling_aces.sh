#!/bin/bash
# Idempotent three-job pipeline; a persisted submission intent blocks uncertain retries.
set -euo pipefail
repo="${AF_REPO_ROOT:?set the clean pinned checkout}"
root="${AF_OUTPUT_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-numerical-sibling-v1}"
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
export AF_PARENT_FIT="${AF_PARENT_FIT:-/scratch/user/u.yx126462/phase_b/prefit-public-fit-v1/fit}"
export AF_CONTINUATION_FIT="${AF_CONTINUATION_FIT:-/scratch/user/u.yx126462/phase_b/public-fit-continuation-v1/continuation}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-requirements-v2}"
export AF_CONSTRUCTION_ROOT="${AF_CONSTRUCTION_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1}"
export AF_CONFIG="${AF_CONFIG:-$repo/configs/prefit_numerical_sibling_v1.json}"
export AF_VLLM_IMAGE="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
export AF_HF_HOME="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
export AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-/scratch/user/u.yx126462/autoformalism-runtime-cache/prefit-numerical-sibling}"
export AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-/scratch/user/u.yx126462/af-ipc}"
[[ -x "$AF_PYTHON" && -f "$AF_CONFIG" && -f "$AF_VLLM_IMAGE" ]] || { echo 'Missing Python, config or image' >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo 'Use a clean pinned checkout' >&2; exit 2; }
for artifact in freeze.json result.json backend_result.json; do
  [[ -f "$AF_PARENT_FIT/$artifact" && -f "$AF_CONTINUATION_FIT/$artifact" ]] || { echo "Missing historical $artifact" >&2; exit 2; }
done
[[ -f "$AF_SOURCE_ROOT/plan.json" && -f "$AF_CONSTRUCTION_ROOT/plan.json" && -f "$AF_PARENT_FIT/../handoff.json" ]] || { echo 'Missing historical construction/repair handoff' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
paths="$("$AF_PYTHON" - "$repo" "$root" "$AF_PARENT_FIT" "$AF_CONTINUATION_FIT" "$AF_SOURCE_ROOT" "$AF_CONSTRUCTION_ROOT" "$AF_CONFIG" <<'PY'
import json, sys
from pathlib import Path
repo, root, parent, continuation, source, construction, config = [Path(v).resolve() for v in sys.argv[1:]]
for protected in (repo, parent.parent, continuation.parent, source, construction, config):
    if root.is_relative_to(protected) or protected.is_relative_to(root):
        raise SystemExit('Output must be separate from checkout and all historical experiments')
print(json.dumps([str(p) for p in (repo, root, parent, continuation, source, construction, config)]))
PY
)"
export AF_REPO_ROOT="$(jq -r '.[0]' <<< "$paths")" AF_OUTPUT_ROOT="$(jq -r '.[1]' <<< "$paths")"
export AF_PARENT_FIT="$(jq -r '.[2]' <<< "$paths")" AF_CONTINUATION_FIT="$(jq -r '.[3]' <<< "$paths")"
export AF_SOURCE_ROOT="$(jq -r '.[4]' <<< "$paths")" AF_CONSTRUCTION_ROOT="$(jq -r '.[5]' <<< "$paths")" AF_CONFIG="$(jq -r '.[6]' <<< "$paths")"
repo="$AF_REPO_ROOT"; root="$AF_OUTPUT_ROOT"
export PYTHONPATH="$repo/src"
identity="$("$AF_PYTHON" - "$(git -C "$repo" rev-parse HEAD)" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
names = ('AF_REPO_ROOT','AF_OUTPUT_ROOT','AF_PARENT_FIT','AF_CONTINUATION_FIT','AF_SOURCE_ROOT','AF_CONSTRUCTION_ROOT','AF_CONFIG','AF_VLLM_IMAGE')
paths = {k: os.environ[k] for k in names}
files = [Path(paths['AF_CONFIG']), Path(paths['AF_SOURCE_ROOT'])/'plan.json', Path(paths['AF_CONSTRUCTION_ROOT'])/'plan.json', Path(paths['AF_PARENT_FIT']).parent/'handoff.json']
files += [Path(paths[k])/n for k in ('AF_PARENT_FIT','AF_CONTINUATION_FIT') for n in ('freeze.json','result.json','backend_result.json')]
print(json.dumps({'commit':sys.argv[1], 'paths':paths, 'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}, sort_keys=True, separators=(',',':')))
PY
)"
manifest="$root/submission_manifest.json"
if [[ -f "$manifest" ]]; then
  [[ "$(jq -cS '.identity' "$manifest")" == "$identity" ]] || { echo 'Submission identity differs; use original inputs' >&2; exit 2; }
  cat "$manifest"
  exit 0
fi
[[ ! -e "$root/plan.json" && ! -e "$root/evidence" ]] || { echo 'Prepared experiment already exists; use its CLI checkpoints' >&2; exit 2; }
mkdir -p "$root/logs" "$AF_HF_HOME"
mkdir "$root/submission-intent" || { echo 'Submission intent exists; inspect queue before retrying' >&2; exit 2; }
printf '%s\n' "$identity" > "$root/submission-intent/identity.json"
worker="$repo/scripts/hpc/run_prefit_numerical_sibling_aces.sh"
submit() {
  local stage="$1" job
  shift
  if ! job="$(sbatch --parsable --account="${AF_ACCOUNT:-156264627414}" --nodes=1 --ntasks=1 --export=ALL --output="$root/logs/$stage-%j.out" --error="$root/logs/$stage-%j.err" "$@" "$worker" "$stage")"; then
    echo 'Submission failed or uncertain; saved intent prevents automatic duplication' >&2
    return 2
  fi
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || { echo 'Invalid scheduler ID; inspect submission intent' >&2; return 2; }
  printf '%s\n' "$job" > "$root/submission-intent/$stage-job.txt"
  printf '%s\n' "$job"
}
prepare="$(submit prepare --job-name=sibling-prepare --partition=cpu --cpus-per-task=1 --mem=16G --time=01:00:00)"
propose="$(submit propose --dependency="afterok:$prepare" --kill-on-invalid-dep=yes --job-name=sibling-propose --partition=gpu --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --signal=B:TERM@300)"
fit="$(submit fit --dependency="afterany:$propose" --job-name=sibling-fit --partition=cpu --cpus-per-task=1 --mem=16G --time=01:00:00)"
jq -n --arg prepare_job "$prepare" --arg propose_job "$propose" --arg fit_job "$fit" --argjson identity "$identity" '{prepare_job:$prepare_job,propose_job:$propose_job,fit_job:$fit_job,identity:$identity}' > "$manifest.tmp"
mv "$manifest.tmp" "$manifest"
cat "$manifest"
