#!/bin/bash
# Idempotent submission; a saved intent prevents uncertain duplicate jobs.
set -euo pipefail
repo="${AF_REPO_ROOT:?set the clean pinned checkout}"
root="${AF_OUTPUT_ROOT:?set a new output root}"
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-requirements-v2}"
export AF_CONSTRUCTION_ROOT="${AF_CONSTRUCTION_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-construction-audit-v1-fix1}"
export AF_PUBLIC_ROOT="${AF_PUBLIC_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1-e7ffd12/public}"
export AF_SELECTION="${AF_SELECTION:-$repo/configs/prefit_public_fit_handoff_v1.json}"
[[ -x "$AF_PYTHON" && -f "$AF_SELECTION" && -f "$AF_SOURCE_ROOT/plan.json" && -f "$AF_CONSTRUCTION_ROOT/plan.json" ]] || { echo 'Missing Python, selection, or source plan' >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo 'Use a clean pinned checkout' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
export AF_REPO_ROOT="$repo" AF_OUTPUT_ROOT="$root"
commit="$(git -C "$repo" rev-parse HEAD)"
selection_sha="$(sha256sum "$AF_SELECTION" | cut -d' ' -f1)"
"$AF_PYTHON" - "$root" "$AF_SOURCE_ROOT" "$AF_CONSTRUCTION_ROOT" "$AF_PUBLIC_ROOT" <<'PY'
import sys
from pathlib import Path
destination = Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    source = Path(value).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise SystemExit("Output must be separate from every source")
PY
identity="$(jq -cn --arg commit "$commit" --arg selection_sha256 "$selection_sha" --arg source_root "$AF_SOURCE_ROOT" --arg construction_root "$AF_CONSTRUCTION_ROOT" --arg public_root "$AF_PUBLIC_ROOT" '{commit:$commit,selection_sha256:$selection_sha256,source_root:$source_root,construction_root:$construction_root,public_root:$public_root}')"
manifest="$root/submission_manifest.json"
if [[ -f "$manifest" ]]; then
  [[ "$(jq -c '.identity' "$manifest")" == "$identity" ]] || { echo 'Submission identity changed; use the original checkout and inputs' >&2; exit 2; }
  cat "$manifest"
  exit 0
fi
[[ ! -e "$root/handoff.json" ]] || { echo 'A handoff already exists; inspect or run it using the CLI' >&2; exit 2; }
mkdir -p "$root/logs"
mkdir "$root/submission-intent" || { echo 'Submission intent exists; inspect queue/logs before retrying' >&2; exit 2; }
printf '%s\n' "$identity" > "$root/submission-intent/identity.json"
if ! job="$(sbatch --parsable --account="${AF_ACCOUNT:-156264627414}" --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=00:30:00 --job-name=prefit-public-fit --export=ALL --output="$root/logs/fit-%j.out" --error="$root/logs/fit-%j.err" "$repo/scripts/hpc/run_prefit_fit_handoff_aces.sh")"; then
  echo 'Submission failed or is uncertain; saved intent prevents automatic duplication' >&2
  exit 2
fi
job="${job%%;*}"
[[ "$job" =~ ^[0-9]+$ ]] || { echo 'Invalid scheduler job ID; inspect saved intent' >&2; exit 2; }
jq -n --arg job_id "$job" --argjson identity "$identity" '{job_id:$job_id,identity:$identity,platform:"aces-cpu",cpus:1}' > "$manifest.tmp"
mv "$manifest.tmp" "$manifest"
cat "$manifest"
