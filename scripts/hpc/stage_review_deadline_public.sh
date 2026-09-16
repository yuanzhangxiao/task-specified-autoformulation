#!/bin/bash
# Run on the Mac: transfer only six development cells through existing SSH aliases.
set -euo pipefail
repo="${AF_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python="${AF_LOCAL_PYTHON:-$repo/.venv/bin/python}"
source_root="${AF_DELTA_PUBLIC_ROOT:-/work/hdd/bibo/yxiao2/phase_b/inputs/public-prompt-v3}"
destination="${AF_PUBLIC_ROOT:-/scratch/user/u.yx126462/phase_b/review-deadline-inputs-v1}"
[[ "$source_root" =~ ^/[a-zA-Z0-9_./-]+$ && "$destination" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Use simple absolute remote paths' >&2; exit 2; }
local_stage="$(mktemp -d "${TMPDIR:-/tmp}/review-development.XXXXXX")"
trap 'rm -rf "$local_stage"' EXIT
"$python" - "$repo/configs/review_deadline_v1.json" > "$local_stage/files.txt" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    config=json.load(stream)
for cell in config['public_cells']:
    for name in ('manifest.json','proposer_prompt.txt','train.csv','validation.csv'):
        print('phase_b_v1/'+cell+'/'+name)
PY
mkdir "$local_stage/public"
rsync -a --files-from="$local_stage/files.txt" "delta:$source_root/" "$local_stage/public/"
ssh aces "mkdir -p '$destination'"
# Existing differing files are not silently replaced. A new release directory is required.
rsync -ain --checksum "$local_stage/public/" "aces:$destination/" > "$local_stage/diff.txt"
if awk 'substr($0,1,1)=="<" && substr($0,3,9)!="+++++++++" {bad=1} END {exit !bad}' "$local_stage/diff.txt"; then
  echo 'Existing ACES development files differ; choose a new AF_PUBLIC_ROOT.' >&2
  cat "$local_stage/diff.txt" >&2
  exit 2
fi
rsync -a --ignore-existing "$local_stage/public/" "aces:$destination/"
printf 'ACES public development root: %s\n' "$destination"
