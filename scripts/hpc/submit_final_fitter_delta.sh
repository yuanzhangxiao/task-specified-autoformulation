#!/bin/bash
# One final finite matrix. Never create additional attempts after its results.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_CASADI_ROOT="${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-parameter-freedom-v1-mem64}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-final-alternatives-v1}"
readonly concurrency="${AF_ARRAY_CONCURRENCY:-2}"
[[ "${concurrency}" =~ ^[12]$ ]] || exit 2
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" && -z "$(git status --porcelain)" ]] || { echo 'Use a clean pinned checkout.' >&2; exit 2; }
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git rev-parse HEAD)"
export PYTHONPATH="${AF_REPO_ROOT}/src:${AF_CASADI_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "${AF_OUTPUT_ROOT}/logs"
"${AF_PYTHON}" scripts/run_final_fitter.py prepare --source "${AF_SOURCE_ROOT}" --output "${AF_OUTPUT_ROOT}" --config "${AF_CONFIG:-configs/fitter_final_alternatives_v1.json}"
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  cat "${AF_OUTPUT_ROOT}/submission.json"
  echo 'Existing submission retained; reconcile incomplete submission with the queue.'
  exit 0
fi
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || { echo 'Reconcile partial submission with the queue.' >&2; exit 3; }
record_job() {
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}" "$1" "$2" "${AF_CODE_COMMIT}" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])/'submission.json'
r=json.loads(p.read_text()) if p.exists() else {}
r.update({sys.argv[2]:sys.argv[3], 'commit':sys.argv[4], 'cpus_per_task':1,
          'fit_memory_gb':64, 'fit_tasks':9, 'gpus':0, 'no_automatic_followup':True,
          'submission_complete':sys.argv[2]=='summary_job_id'})
t=p.with_suffix('.tmp'); t.write_text(json.dumps(r,indent=2)+'\n'); t.replace(p)
PY
}
submit() {
  local name="$1"; shift
  local job
  job="$(sbatch --parsable --output="${AF_OUTPUT_ROOT}/logs/${name}-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/${name}-%A_%a.err" --export=ALL "$@")"
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || return 1
  record_job "${name}_job_id" "$job"
  echo "$job"
}
gate="$(submit gate --mem=8G --time=00:10:00 scripts/hpc/final_fitter_delta.slurm gate)"
fit="$(submit fit --dependency="afterok:${gate}" --kill-on-invalid-dep=yes --array="0-8%${concurrency}" scripts/hpc/final_fitter_delta.slurm run)"
submit summary --mem=2G --time=00:10:00 --dependency="afterany:${fit}:${gate}" --kill-on-invalid-dep=yes scripts/hpc/final_fitter_delta.slurm summarize >/dev/null
cat "${AF_OUTPUT_ROOT}/submission.json"
