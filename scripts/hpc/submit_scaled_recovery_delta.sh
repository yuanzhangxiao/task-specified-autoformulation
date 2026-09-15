#!/bin/bash
# Explicit recovery of the existing two checkpoints, not another initializer run.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_CASADI_ROOT="${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-scaled-alternating-v1}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-scaled-recovery-v2}"
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" && -z "$(git status --porcelain)" ]] || { echo 'Use a clean pinned checkout.' >&2; exit 2; }
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git rev-parse HEAD)"
mkdir -p "${AF_OUTPUT_ROOT}/logs"
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  cat "${AF_OUTPUT_ROOT}/submission.json"
  echo 'Existing submission retained; reconcile a partial submission rather than duplicating jobs.'
  exit 0
fi
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || { echo 'Reconcile partial submission with the queue.' >&2; exit 3; }
submit() {
  local name="$1"; shift
  local job
  job="$(sbatch --parsable --output="${AF_OUTPUT_ROOT}/logs/${name}-%A_%a.out" \
    --error="${AF_OUTPUT_ROOT}/logs/${name}-%A_%a.err" --export=ALL "$@")"
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || return 1
  "${AF_PYTHON}" -S - "${AF_OUTPUT_ROOT}" "${name}" "${job}" "${AF_CODE_COMMIT}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / 'submission.json'
r = json.loads(p.read_text()) if p.exists() else {}
r.update({sys.argv[2] + '_job_id': sys.argv[3], 'commit': sys.argv[4],
          'cpus_per_task': 1, 'fit_memory_gb': 16, 'fit_tasks': 2, 'gpus': 0,
          'collocation_reruns': 0, 'submission_complete': sys.argv[2] == 'summary'})
t = p.with_suffix('.tmp'); t.write_text(json.dumps(r, indent=2) + '\n'); t.replace(p)
PY
  echo "${job}"
}
prep="$(submit prepare --mem=8G --time=00:45:00 scripts/hpc/scaled_recovery_delta.slurm prepare)"
fit="$(submit fit --dependency="afterok:${prep}" --kill-on-invalid-dep=yes --array=0-1%2 scripts/hpc/scaled_recovery_delta.slurm run)"
submit summary --mem=2G --time=00:10:00 --dependency="afterany:${prep}:${fit}" --kill-on-invalid-dep=yes \
  scripts/hpc/scaled_recovery_delta.slurm summarize >/dev/null
cat "${AF_OUTPUT_ROOT}/submission.json"
