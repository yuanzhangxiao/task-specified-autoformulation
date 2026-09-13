#!/bin/bash
# Profiles -> fixed-node controls -> gate -> fits. Never renew an old run's budget.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_CASADI_ROOT="${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-attainability-v2}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-resolution-v3}"
readonly concurrency="${AF_ARRAY_CONCURRENCY:-2}"
[[ "${concurrency}" =~ ^[12]$ ]] || exit 2
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" && -z "$(git status --porcelain)" ]] || { echo 'Use a clean pinned checkout.' >&2; exit 2; }
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git rev-parse HEAD)"
export PYTHONPATH="${AF_REPO_ROOT}/src:${AF_CASADI_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "${AF_OUTPUT_ROOT}/logs"
"${AF_PYTHON}" scripts/run_resolution_campaign.py prepare --source "${AF_SOURCE_ROOT}" --output "${AF_OUTPUT_ROOT}" --config configs/fitter_resolution_v3.json
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  cat "${AF_OUTPUT_ROOT}/submission.json"
  echo 'Existing submission retained. Do not resubmit completed or interrupted tasks with a fresh budget.'
  exit 0
fi
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || { echo 'Reconcile partial submission with the queue.' >&2; exit 3; }
record_job() {
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}" "$1" "$2" "${AF_CODE_COMMIT}" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1]) / 'submission.json'
r=json.loads(path.read_text()) if path.exists() else {}
r.update({sys.argv[2]:sys.argv[3], 'commit':sys.argv[4], 'cpus_per_task':1, 'gpus':0,
          'submission_complete':sys.argv[2]=='summary_job_id'})
tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(path)
PY
}
submit() {
  local name="$1"; shift
  local job
  job="$(sbatch --parsable --output="${AF_OUTPUT_ROOT}/logs/${name}-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/${name}-%A_%a.err" --export=ALL "$@")"
  job="${job%%;*}"
  record_job "${name}_job_id" "${job}"
  echo "${job}"
}
smoke="$(submit smoke --time=00:10:00 scripts/hpc/resolution_delta.slurm smoke)"
profile="$(submit profile --time=00:10:00 --dependency="afterok:${smoke}" --kill-on-invalid-dep=yes --array="0-5%${concurrency}" scripts/hpc/resolution_delta.slurm profile)"
nodes="$(submit node --dependency="afterok:${smoke},afterany:${profile}" --kill-on-invalid-dep=yes --array="0-1%${concurrency}" scripts/hpc/resolution_delta.slurm run)"
gate="$(submit gate --time=00:10:00 --dependency="afterany:${nodes}" --kill-on-invalid-dep=yes scripts/hpc/resolution_delta.slurm gate)"
fit="$(submit fit --dependency="afterok:${gate}" --kill-on-invalid-dep=yes --array="2-7%${concurrency}" scripts/hpc/resolution_delta.slurm run)"
submit summary --time=00:10:00 --dependency="afterany:${fit}:${gate}" --kill-on-invalid-dep=yes scripts/hpc/resolution_delta.slurm summarize >/dev/null
cat "${AF_OUTPUT_ROOT}/submission.json"
