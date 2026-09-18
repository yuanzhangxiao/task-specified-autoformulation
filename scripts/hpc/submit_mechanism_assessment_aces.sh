#!/bin/bash
# Prepare explicitly first. Submit once; preserve uncertain scheduler intents.
set -euo pipefail
repo="${AF_REPO_ROOT:?set AF_REPO_ROOT}"
root="${AF_OUTPUT_ROOT:?set AF_OUTPUT_ROOT}"
python="${AF_PYTHON:?set AF_PYTHON}"
export AF_COMMIT="$(git -C "$repo" rev-parse HEAD)"
count="$("$python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["rows"]))' "$root/plan.json")"
[[ "$count" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$root/logs"
mkdir "$root/submission-intent" || {
  echo 'Existing submission intent. Inspect saved job IDs and squeue before retrying.' >&2
  exit 2
}
submit() {
  local name="$1" id
  shift
  id="$(sbatch --parsable --account="${AF_ACCOUNT:-156264627414}" --partition=cpu \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --export=ALL \
    --job-name="mechanism-$name" --output="$root/logs/$name-%A_%a.out" \
    --error="$root/logs/$name-%A_%a.err" "$@")"
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || {
    echo 'Uncertain scheduler reply; inspect squeue before any retry.' >&2
    exit 2
  }
  printf '%s\n' "$id" > "$root/submission-intent/$name.id"
  printf '%s\n' "$id"
}
array="$(submit assess --array="0-$((count-1))%8" --mem=8G --time=03:00:00 \
  "$repo/scripts/hpc/run_mechanism_assessment.sh")"
summary="$(submit summary --dependency="afterany:$array" --mem=4G --time=00:15:00 \
  "$repo/scripts/hpc/run_mechanism_assessment.sh" report)"
"$python" - "$root" "$AF_COMMIT" "$array" "$summary" "$count" <<'PY'
import json, sys
from pathlib import Path
value = dict(commit=sys.argv[2], assessment_job_id=sys.argv[3],
             summary_job_id=sys.argv[4], models=int(sys.argv[5]),
             cpus_per_task=1, gpus=0, llm_calls=0, parameter_refits=0)
path = Path(sys.argv[1]) / 'submission.json'
path.write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps(value, indent=2))
PY
