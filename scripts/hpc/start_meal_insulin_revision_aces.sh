#!/bin/bash
# One explicit, training-only 120B proposal after the reviewed candidate-15 fits.
set -euo pipefail
export AF_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_ACTION=${1:?Use run, status, inspect, or preview}
case "$AF_ACTION" in run|status|inspect|preview) ;; *) echo 'Unknown command.' >&2; exit 2 ;; esac
[[ $# -eq 1 ]] || { echo 'This launcher accepts one action and uses the sealed 120B plan.' >&2; exit 2; }
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
[[ -d "$AF_GROUP" ]] || { echo 'Run on ACES with AF_GROUP set to writable scratch.' >&2; exit 2; }
# Never inherit another campaign's generic output path.
export AF_OUTPUT_ROOT=${AF_MEAL_INSULIN_REVISION_ROOT:-$AF_GROUP/phase-c-insulin-revision-120b-v1}
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
AF_CLI="$AF_REPO_ROOT/scripts/propose_meal_insulin_revision.py"
printf 'Candidate-15 insulin revision output: %s\n' "$AF_OUTPUT_ROOT"
if ! "$AF_PYTHON" "$AF_CLI" --help >/dev/null 2>&1; then
  echo 'Loading scientific Python and its compiler/library hierarchy.' >&2
  module load "${AF_GCCCORE_MODULE:-GCCcore/13.2.0}" "${AF_PYTHON_MODULE:-Python/3.11.5}"
  # Module dependencies may extend PYTHONPATH; this checkout must remain first.
  export PYTHONPATH="$AF_REPO_ROOT:$AF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  "$AF_PYTHON" "$AF_CLI" --help >/dev/null
fi
case "$AF_ACTION" in
  run)
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    AF_SNAPSHOT=hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json
    if [[ -z "${AF_HF_HOME:-}" ]]; then
      if [[ -f "$AF_GROUP/huggingface-cache/$AF_SNAPSHOT" ]]; then
        export AF_HF_HOME="$AF_GROUP/huggingface-cache"
      else
        export AF_HF_HOME=/scratch/user/u.yx126462/huggingface-cache
      fi
    fi
    export AF_HARMONY_ASSETS=${AF_HARMONY_ASSETS:-$AF_REPO_ROOT/assets/harmony}
    "$AF_PYTHON" "$AF_CLI" prepare --source "${AF_MEAL_INSULIN_REVISION_SOURCE:-$AF_REPO_ROOT/inputs/revision-intake.json}" --root "$AF_OUTPUT_ROOT" --model-size 120b
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_meal_insulin_revision_aces.py" --root "$AF_OUTPUT_ROOT"
    ;;
  status)
    if command -v squeue >/dev/null 2>&1; then
      squeue -u "$USER" -o '%.18i %.36j %.10T %.10M %R' || echo 'Scheduler status unavailable; showing local receipts.' >&2
    fi
    "$AF_PYTHON" - "$AF_OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
from scripts.submit_structural_revision_aces import _read
root = Path(sys.argv[1])
manifest = root / "submission/manifest.json"
intent = root / "submission/propose.intent.json"
receipt = root / "submission/propose.receipt.json"
record = {"logs": str(root / "logs")}
if manifest.exists():
    record.update(job=_read(manifest)["job"], manifest=str(manifest), receipt=str(receipt))
elif intent.exists():
    record.update(submission="unconfirmed; inspect scheduler before recovery", intent=str(intent))
else:
    record["submission"] = "not submitted"
print(json.dumps(record, indent=2))
PY
    exec "$AF_PYTHON" "$AF_CLI" status --root "$AF_OUTPUT_ROOT"
    ;;
  preview)
    if [[ ! -f "$AF_OUTPUT_ROOT/plan.json" ]]; then
      "$AF_PYTHON" "$AF_CLI" prepare --source "${AF_MEAL_INSULIN_REVISION_SOURCE:-$AF_REPO_ROOT/inputs/revision-intake.json}" --root "$AF_OUTPUT_ROOT" --model-size 120b
    fi
    exec "$AF_PYTHON" "$AF_CLI" preview --root "$AF_OUTPUT_ROOT" --html "$AF_OUTPUT_ROOT/PROMPT.html"
    ;;
  inspect)
    "$AF_PYTHON" "$AF_CLI" inspect --root "$AF_OUTPUT_ROOT"
    printf 'Download: %s/inspection.tar.gz\n' "$AF_OUTPUT_ROOT"
    ;;
esac
