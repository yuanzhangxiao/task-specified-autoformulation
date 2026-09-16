#!/bin/bash
# Submit the bounded matrix once. Never silently duplicate uncertain submissions.
set -euo pipefail
repo="${AF_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
root="${AF_OUTPUT_ROOT:?set a NEW campaign output directory}"
public="${AF_PUBLIC_ROOT:?set the complete six-cell public release directory}"
config="${AF_CONFIG:-$repo/configs/review_deadline_v1.json}"
image="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
hf="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
account="${AF_ACCOUNT:-156264627414}"
[[ -x "$python" && -f "$image" && -f "$config" ]] || { echo 'Missing Python, image or config' >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo 'Use a clean pinned experiment checkout' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$repo/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" "$repo/scripts/review_deadline.py" prepare --config "$config" --public-root "$public" --root "$root"
if [[ -f "$root/submission_manifest.json" ]]; then cat "$root/submission_manifest.json"; exit 0; fi
mkdir -p "$root/logs" "$hf"
mkdir "$root/submission-intent" || { echo 'Submission intent exists. Inspect recorded job IDs and squeue before retrying.' >&2; exit 2; }
export AF_REPO_ROOT="$repo" AF_PYTHON="$python" AF_OUTPUT_ROOT="$root" AF_VLLM_IMAGE="$image" AF_HF_HOME="$hf"
export AF_COMMIT="$(git -C "$repo" rev-parse HEAD)"
export AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-/scratch/user/u.yx126462/autoformalism-runtime-cache/review}"
export AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-/scratch/user/u.yx126462/af-ipc}"
worker="$repo/scripts/hpc/run_review_deadline_aces.sh"
submit() {
  local stage="$1" round="$2" id key
  shift 2
  key="$stage-$round"
  id="$(sbatch --parsable --kill-on-invalid-dep=yes --account="$account" --nodes=1 --ntasks=1 --export=ALL --job-name="review-$key" --output="$root/logs/$key-%A_%a.out" --error="$root/logs/$key-%A_%a.err" "$@" "$worker" "$stage" "$round")" || return 2
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo 'Uncertain scheduler reply: inspect squeue' >&2; return 2; }
  printf '%s\n' "$id" > "$root/submission-intent/$key.id"
  printf '%s\n' "$id"
}
prepare="$(submit prepare 0 --partition=cpu --cpus-per-task=1 --mem=16G --time=01:00:00)"
submit demo 0 --dependency="afterok:$prepare" --partition=cpu --cpus-per-task=1 --mem=16G --time=00:45:00 > /dev/null
prior="$prepare"
rounds="$(jq -r '.config.rounds' "$root/plan.json")"
for ((round=0; round<rounds; round++)); do
  indices="$(jq -r --argjson r "$round" '[.tasks[] | select($r > 0 or .arm != "refit_only") | .index] | join(",")' "$root/plan.json")"
  # First visit depends on successful preparation; later visits still report partial failures.
  dependency="afterok:$prepare,afterany:$prior"
  if [[ "$round" == 0 ]]; then dependency="afterok:$prior"; fi
  gpu="$(submit propose "$round" --dependency="$dependency" --partition=gpu --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=06:30:00 --signal=B:TERM@300)"
  cpu="$(submit fit "$round" --dependency="afterany:$gpu" --partition=cpu --array="${indices}%16" --cpus-per-task=1 --mem=16G --time=00:40:00 --signal=B:TERM@120)"
  prior="$(submit finish "$round" --dependency="afterany:$cpu" --partition=cpu --cpus-per-task=1 --mem=4G --time=00:15:00)"
done
"$python" - "$root" "$AF_COMMIT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
value={"commit":sys.argv[2], "jobs":{p.stem:p.read_text().strip() for p in sorted((root/"submission-intent").glob("*.id"))}, "one_h100_per_proposer_job":True, "one_cpu_per_fit":True,"automatic_test_access":False,"submission_complete":True}
temporary=root/"submission_manifest.tmp"
temporary.write_text(json.dumps(value,indent=2)+"\n")
temporary.replace(root/"submission_manifest.json")
print(json.dumps(value,indent=2))
PY
