#!/bin/bash
# Queue one visit at a time to stay below ACES's per-user submission limit.
set -euo pipefail
round=0
if [[ "$#" != 0 ]]; then
  [[ "$#" == 2 && "$1" == --round && "$2" =~ ^[0-9]+$ ]] || {
    echo 'Usage: submit_review_deadline_v2_aces.sh [--round N]' >&2; exit 2;
  }
  round="$2"
fi
repo="${AF_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
export AF_REPO_ROOT="$repo"
export AF_CONFIG="${AF_CONFIG:-$repo/configs/review_deadline_v2.json}"
: "${AF_OUTPUT_ROOT:?set a separate v2 campaign output directory}"
[[ -f "$AF_CONFIG" ]] || { echo 'Missing v2 configuration' >&2; exit 2; }
[[ "$(jq -er '.protocol' "$AF_CONFIG")" == review-deadline-2 ]] || {
  echo 'This launcher requires protocol review-deadline-2' >&2
  exit 2
}
if [[ -f "$AF_OUTPUT_ROOT/plan.json" ]]; then
  [[ "$(jq -er '.config.protocol' "$AF_OUTPUT_ROOT/plan.json")" == review-deadline-2 ]] || {
    echo 'Historical output is preserved. Choose a new v2 output directory.' >&2
    exit 2
  }
elif [[ -d "$AF_OUTPUT_ROOT" && -n "$(ls -A "$AF_OUTPUT_ROOT")" ]]; then
  echo 'Output is nonempty without a plan. Inspect it before choosing a new directory.' >&2
  exit 2
fi
: "${AF_PUBLIC_ROOT:?set the staged public development directory}"
# Check the entire matrix before freeze copies any files into the new output.
missing=0
cells="$(jq -er '.public_cells | select(type == "array" and length > 0) | .[]' "$AF_CONFIG")"
while IFS= read -r cell; do
  for name in manifest.json proposer_prompt.txt train.csv validation.csv; do
    file="$AF_PUBLIC_ROOT/phase_b_v1/$cell/$name"
    if [[ ! -r "$file" ]]; then
      printf 'Missing public development file: %s\n' "$file" >&2
      missing=1
    fi
  done
done <<< "$cells"
[[ "$missing" == 0 ]] || exit 2
bash "$repo/scripts/hpc/run_staged_topology_server.sh" --check-config "$AF_CONFIG" >/dev/null
python="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
root="$AF_OUTPUT_ROOT"
image="${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}"
hf="${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}"
account="${AF_ACCOUNT:-156264627414}"
[[ -x "$python" && -f "$image" ]] || { echo 'Missing Python or image' >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo 'Use a clean pinned experiment checkout' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$repo/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" "$repo/scripts/review_deadline.py" prepare --config "$AF_CONFIG" --public-root "$AF_PUBLIC_ROOT" --root "$root"
rounds="$(jq -r '.config.rounds' "$root/plan.json")"
(( round < rounds )) || { echo 'Round outside frozen campaign' >&2; exit 2; }
manifest="$root/submission-round-$round.json"
if [[ -f "$manifest" ]]; then cat "$root/submission_manifest.json"; exit 0; fi
mkdir -p "$root/logs" "$hf" "$root/submission-intent"
if [[ "$round" != 0 && ! -f "$root/submission-round-$((round-1)).json" ]]; then
  echo 'Previous round submission is incomplete; inspect its recorded job IDs.' >&2
  exit 2
fi
mkdir "$root/submission-intent/round-$round" || {
  echo "Round $round submission intent exists. Inspect recorded IDs and squeue; do not duplicate uncertain jobs." >&2
  exit 2
}
export AF_PYTHON="$python" AF_VLLM_IMAGE="$image" AF_HF_HOME="$hf"
export AF_COMMIT="$(git -C "$repo" rev-parse HEAD)"
export AF_COMPUTE_CACHE_ROOT="${AF_COMPUTE_CACHE_ROOT:-/scratch/user/u.yx126462/autoformalism-runtime-cache/review}"
export AF_IPC_TMP_ROOT="${AF_IPC_TMP_ROOT:-/scratch/user/u.yx126462/af-ipc}"
worker="$repo/scripts/hpc/run_review_deadline_aces.sh"
submit() {
  local stage="$1" index="$2" id key
  shift 2
  key="$stage-$index"
  id="$(sbatch --parsable --kill-on-invalid-dep=yes --account="$account" --nodes=1 --ntasks=1 --export=ALL --job-name="review-v2-$key" --output="$root/logs/$key-%A_%a.out" --error="$root/logs/$key-%A_%a.err" "$@" "$worker" "$stage" "$index")" || return 2
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo 'Uncertain scheduler reply: inspect squeue' >&2; return 2; }
  printf '%s\n' "$id" > "$root/submission-intent/$key.id"
  printf '%s\n' "$id"
}
if [[ "$round" == 0 ]]; then
  prepare="$(submit prepare 0 --partition=cpu --cpus-per-task=1 --mem=16G --time=01:00:00)"
  submit demo 0 --dependency="afterok:$prepare" --partition=cpu --cpus-per-task=1 --mem=16G --time=00:45:00 > /dev/null
  dependency="afterok:$prepare"
else
  prepare="$(cat "$root/submission-intent/prepare-0.id")"
  prior="$(cat "$root/submission-intent/finish-$((round-1)).id")"
  dependency="afterok:$prepare,afterany:$prior"
fi
indices="$(jq -r --argjson r "$round" '[.tasks[] | select($r > 0 or .arm != "refit_only") | .index] | join(",")' "$root/plan.json")"
gpu="$(submit propose "$round" --dependency="$dependency" --partition=gpu --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=06:30:00 --signal=B:TERM@300)"
cpu="$(submit fit "$round" --dependency="afterany:$gpu" --partition=cpu --array="${indices}%16" --cpus-per-task=1 --mem=16G --time=00:40:00 --signal=B:TERM@120)"
finish="$(submit finish "$round" --dependency="afterany:$cpu" --partition=cpu --cpus-per-task=1 --mem=4G --time=00:15:00)"
if (( round + 1 < rounds )); then
  submit submit-next "$((round+1))" --dependency="afterany:$finish" --partition=cpu --cpus-per-task=1 --mem=4G --time=00:15:00 > /dev/null
fi
"$python" - "$root" "$AF_COMMIT" "$round" "$rounds" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
index, rounds=map(int,sys.argv[3:])
value = {
    "commit": sys.argv[2],
    "protocol": "review-deadline-2",
    "jobs": {
        p.stem: p.read_text().strip()
        for p in sorted((root / "submission-intent").glob("*.id"))
    },
    "submitted_through_round": index,
    "planned_rounds": rounds,
    "all_rounds_submitted": index + 1 == rounds,
    "submission_complete": index + 1 == rounds,
    "round_submission_complete": True,
    "submission_scope": "through recorded round; dependent dispatcher submits next round",
    "one_h100_per_proposer_job": True,
    "one_cpu_per_fit": True,
    "automatic_test_access": False,
}
def write(path):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)
write(root/'submission_manifest.json')
write(root/f'submission-round-{index}.json')
print(json.dumps(value,indent=2))
PY
