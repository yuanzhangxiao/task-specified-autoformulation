#!/bin/bash
# New orchestration, original frozen fitter; one CPU-only matched allocation.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:?set the new clean driver checkout}"
export AF_FIT_REPO="${AF_FIT_REPO:-/scratch/user/u.yx126462/repos/autoformalism-prefit-parameter-replay-v1}"
export AF_FIT_COMMIT="${AF_FIT_COMMIT:-9196877e484f7dea4f560966bd1040544c219000}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-parameter-replay-v1}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/scratch/user/u.yx126462/phase_b/prefit-matched-refit-v1}"
export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
[[ -x "$AF_PYTHON" ]] || { echo 'Missing Python' >&2; exit 2; }
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$AF_REPO_ROOT"
[[ -z "$(git status --porcelain)" && -z "$(git -C "$AF_FIT_REPO" status --porcelain)" ]] || { echo 'Use clean pinned checkouts' >&2; exit 2; }
[[ "$(git -C "$AF_FIT_REPO" rev-parse HEAD)" == "$AF_FIT_COMMIT" ]] || { echo 'Original fitter commit differs' >&2; exit 2; }
# Check paths and pin saved child bytes before creating logs or submission intent.
identity="$("$AF_PYTHON" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
paths = {k: str(Path(os.environ[k]).resolve()) for k in
         ('AF_REPO_ROOT', 'AF_FIT_REPO', 'AF_SOURCE_ROOT', 'AF_OUTPUT_ROOT')}
source, root = Path(paths['AF_SOURCE_ROOT']), Path(paths['AF_OUTPUT_ROOT'])
protected = [Path(paths[k]) for k in ('AF_REPO_ROOT', 'AF_FIT_REPO', 'AF_SOURCE_ROOT')]
source_plan = json.loads((source / 'plan.json').read_text())
history = Path(source_plan['source_root'])
old = json.loads((history / 'plan.json').read_text())
protected += [history, Path(old['paths']['source']), Path(old['paths']['construction']),
              Path(old['paths']['parent']).parent, Path(old['paths']['continuation']).parent]
for path in protected:
    path = path.resolve()
    if root.is_relative_to(path) or path.is_relative_to(root):
        raise SystemExit('Output must be separate from every historical experiment and checkout')
hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in
          ('plan.json', 'child_fit/freeze.json', 'child_fit/result.json', 'child_fit/backend_result.json')}
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print(json.dumps({'commit': commit, 'fit_commit': os.environ['AF_FIT_COMMIT'],
                  'paths': paths, 'source_sha256': hashes}, sort_keys=True, separators=(',', ':')))
PY
)"
intent="$AF_OUTPUT_ROOT/submission"
if [[ "${1:-submit}" == worker ]]; then
  [[ "$(jq -cS . "$intent/identity.json")" == "$identity" ]] || { echo 'Queued identity changed' >&2; exit 2; }
  mkdir -p "$AF_OUTPUT_ROOT/tmp"
  export TMPDIR="$AF_OUTPUT_ROOT/tmp"
  # Test the new driver with current synthetic fixtures; no benchmark work here.
  PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" -m pytest -q -p no:cacheprovider tests/test_prefit_matched_refit.py
  PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" scripts/smoke_prefit_matched_refit.py
  [[ "$(git rev-parse HEAD)" == "$(jq -r .commit "$intent/identity.json")" && -z "$(git status --porcelain)" ]] || { echo 'Driver changed during preflight' >&2; exit 2; }
  [[ "$(git -C "$AF_FIT_REPO" rev-parse HEAD)" == "$AF_FIT_COMMIT" && -z "$(git -C "$AF_FIT_REPO" status --porcelain)" ]] || { echo 'Executor changed during preflight' >&2; exit 2; }
  # Historical verification and the actual control use exactly the original code.
  export PYTHONPATH="$AF_FIT_REPO/src"
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/prefit_matched_refit.py" run --source "$AF_SOURCE_ROOT" --root "$AF_OUTPUT_ROOT"
  exit 0
fi
if [[ -f "$intent/manifest.json" ]]; then
  [[ "$(jq -cS '.identity' "$intent/manifest.json")" == "$identity" ]] || { echo 'Submission identity changed' >&2; exit 2; }
  cat "$intent/manifest.json"
  exit 0
fi
mkdir -p "$AF_OUTPUT_ROOT/logs"
mkdir "$intent" || { echo 'Submission intent exists; inspect the queue before retrying' >&2; exit 2; }
printf '%s\n' "$identity" > "$intent/identity.json"
job="$(sbatch --parsable --account="${AF_ACCOUNT:-156264627414}" --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=01:00:00 --export=ALL --job-name=matched-refit --output="$AF_OUTPUT_ROOT/logs/refit-%j.out" --error="$AF_OUTPUT_ROOT/logs/refit-%j.err" "$AF_REPO_ROOT/scripts/hpc/submit_prefit_matched_refit_aces.sh" worker)"
job="${job%%;*}"
[[ "$job" =~ ^[0-9]+$ ]] || { echo 'Uncertain job ID; inspect submission intent' >&2; exit 2; }
jq -n --arg job_id "$job" --argjson identity "$identity" '{job_id:$job_id,identity:$identity}' > "$intent/manifest.json.tmp"
mv "$intent/manifest.json.tmp" "$intent/manifest.json"
cat "$intent/manifest.json"
