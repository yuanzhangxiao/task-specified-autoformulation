#!/bin/bash
# Explicit smoke/full phases share immutable plans and durable call allowances.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_ACTION=${1:?Use run, status, report, inspect, or preview followed by 20b or 120b}
AF_MODEL_SIZE=${2:?Specify 20b or 120b}
AF_PHASE=${3:-smoke}
AF_WAVE=${4:-wave1}
AF_PROFILE=${5:-breadth-five}
case "$AF_PROFILE" in three-policy|breadth-five|focused-retest) ;; *) echo 'Profile must be three-policy, breadth-five, or focused-retest.' >&2; exit 2 ;; esac
case "$AF_PHASE" in smoke|full) ;; *) echo 'Phase must be smoke or full.' >&2; exit 2 ;; esac
[[ "$AF_WAVE" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || { echo 'Invalid wave name.' >&2; exit 2; }
case "$AF_MODEL_SIZE" in
  20b) AF_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee ;;
  120b) AF_REVISION=b5c939de8f754692c1647ca79fbf85e8c1e70f8a ;;
  *) echo 'Model size must be 20b or 120b.' >&2; exit 2 ;;
esac
case "$AF_ACTION" in run|status|report|inspect|preview) ;; *) echo 'Unknown command.' >&2; exit 2 ;; esac
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
[[ -d "$AF_GROUP" ]] || { echo 'Run on ACES with AF_GROUP set to writable scratch.' >&2; exit 2; }
# Do not inherit a generic root from an earlier fitting or construction campaign.
AF_PROFILE_VERSION=v3
[[ "$AF_PROFILE" != focused-retest ]] || AF_PROFILE_VERSION=v4
export AF_OUTPUT_ROOT=${AF_CONSTRUCTION_SEARCH_ROOT:-$AF_GROUP/phase-c-search-${AF_PROFILE}-${AF_MODEL_SIZE}-${AF_PROFILE_VERSION}}
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
AF_CLI="$AF_REPO_ROOT/scripts/construction_search_campaign.py"
printf 'Construction search %s (%s, %s) output: %s\n' "$AF_MODEL_SIZE" "$AF_PROFILE" "$AF_PHASE" "$AF_OUTPUT_ROOT"
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  echo 'Loading scientific Python and its compiler/library hierarchy.' >&2
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  "$AF_PYTHON" "$AF_CLI" --help >/dev/null
fi
case "$AF_ACTION" in
  run)
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    AF_SNAPSHOT="hub/models--openai--gpt-oss-$AF_MODEL_SIZE/snapshots/$AF_REVISION/config.json"
    if [[ -z "${AF_HF_HOME:-}" ]]; then
      if [[ -f "$AF_GROUP/huggingface-cache/$AF_SNAPSHOT" ]]; then
        export AF_HF_HOME="$AF_GROUP/huggingface-cache"
      else
        export AF_HF_HOME=/scratch/user/u.yx126462/huggingface-cache
      fi
    fi
    [[ -f "$AF_HF_HOME/$AF_SNAPSHOT" && -f "$AF_VLLM_IMAGE" ]] || {
      echo 'Pinned offline model snapshot or image missing. Set AF_HF_HOME/AF_VLLM_IMAGE to existing assets; nothing is downloaded.' >&2; exit 2;
    }
    export AF_HARMONY_ASSETS=${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}
    "$AF_PYTHON" "$AF_REPO_ROOT/scripts/harmony_vocab.py" verify --directory "$AF_HARMONY_ASSETS"
    "$AF_PYTHON" "$AF_CLI" prepare --source "${AF_CONSTRUCTION_SEARCH_SOURCE:-$AF_REPO_ROOT/inputs/saved-source.tar.gz}" --root "$AF_OUTPUT_ROOT" --model-size "$AF_MODEL_SIZE" --profile "$AF_PROFILE"
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_construction_search_aces.py" --root "$AF_OUTPUT_ROOT" --phase "$AF_PHASE" --wave "$AF_WAVE"
    ;;
  status)
    squeue -u "$USER" -o '%.18i %.32j %.10T %.10M %R'
    "$AF_PYTHON" - "$AF_OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = sorted((root / 'submission').glob('*/*/manifest.json'))
paths += sorted((root / 'submission').glob('*/*/propose.reply.json'))
paths.append(root / 'summary.json')
for path in paths:
    relative = str(path.relative_to(root))
    if path.exists():
        value = json.loads(path.read_text())
        print(relative)
        if relative == "summary.json":
            value = {key: value.get(key) for key in ("status", "smoke_complete", "smoke_eligible_episodes", "status_counts", "cost")}
        print(json.dumps(value, indent=2))
if not (root / "summary.json").exists():
    print("No summary yet; consult the submitted job and logs.")
print("Logs:", root / "logs")
PY
    ;;
  report|preview) exec "$AF_PYTHON" "$AF_CLI" "$AF_ACTION" --root "$AF_OUTPUT_ROOT" ;;
  inspect)
    "$AF_PYTHON" "$AF_CLI" inspect --root "$AF_OUTPUT_ROOT"
    printf 'Download: %s/inspection.tar.gz\n' "$AF_OUTPUT_ROOT"
    ;;
esac
