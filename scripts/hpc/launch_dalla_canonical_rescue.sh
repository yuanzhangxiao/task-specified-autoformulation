#!/bin/bash
# CPU-only, paired R9/R2 diagnostic; run on one chosen site.
set -euo pipefail
export AF_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AF_COMMIT="$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")"
site="${1:-aces}"
case "$site" in
 aces)
   module load GCCcore/13.2.0 Python/3.11.5
   export AF_PYTHON="${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}"
   export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/scratch/group/p.nairr260351.000/u.yx126462/dalla-canonical-rescue-v1}"
   ;;
 delta)
   export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
   export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/dalla-canonical-rescue-v1}"
   ;;
 *) echo 'Usage: launch_dalla_canonical_rescue.sh aces|delta' >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[[ -x "$AF_PYTHON" ]] || { echo "Missing Python: $AF_PYTHON" >&2; exit 2; }
[[ -s "$AF_REPO_ROOT/inputs/canonical-sources.json" ]] || { echo 'Missing canonical source packet' >&2; exit 2; }
command -v sbatch >/dev/null
command -v jq >/dev/null
exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_dalla_sign_diagnostic.py" \
  --source "$AF_REPO_ROOT/inputs/canonical-sources.json" \
  --decisions "$AF_REPO_ROOT/configs/dalla_canonical_rescue_v1.json" \
  --root "$AF_OUTPUT_ROOT" --site "$site" --campaign canonical "$@"
