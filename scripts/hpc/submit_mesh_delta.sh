#!/bin/bash
# Paired mesh recovery and model-feature diagnostics: one arm per CPU task.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_CASADI_ROOT="${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-feasibility-v1}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-mesh-v1}"
readonly concurrency="${AF_ARRAY_CONCURRENCY:-2}"
[[ "${concurrency}" =~ ^[12]$ ]] || { echo 'Concurrency must be 1 or 2.' >&2; exit 2; }
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" && -z "$(git status --porcelain)" ]] || {
  echo 'Use the existing Python and a clean isolated checkout.' >&2; exit 2;
}
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git rev-parse HEAD)"
export PYTHONPATH="${AF_REPO_ROOT}/src:${AF_CASADI_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "${AF_OUTPUT_ROOT}/logs"
"${AF_PYTHON}" scripts/run_mesh_campaign.py prepare --source "${AF_SOURCE_ROOT}" --output "${AF_OUTPUT_ROOT}" --config configs/fitter_mesh_v1.json
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  cat "${AF_OUTPUT_ROOT}/submission.json"
  echo 'Existing submission retained. Use the documented resume command for incomplete tasks.'
  exit 0
fi
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || {
  echo 'Reconcile the queue before retrying a partial submission.' >&2; exit 3;
}
record_job() {
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}" "$1" "$2" "${AF_CODE_COMMIT}" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); path=root / "submission.json"
record=json.loads(path.read_text()) if path.exists() else {}
record.update({sys.argv[2]:sys.argv[3], "commit":sys.argv[4],
    "cpus_per_task":1, "gpus":0, "submission_complete":sys.argv[2]=="summary_job_id"})
temporary=path.with_suffix(".tmp"); temporary.write_text(json.dumps(record, indent=2)+"\n"); temporary.replace(path)
PY
}
last="$("${AF_PYTHON}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"])-1)' "${AF_OUTPUT_ROOT}/freeze.json")"
smoke="$(sbatch --parsable --time=00:10:00 --output="${AF_OUTPUT_ROOT}/logs/smoke-%j.out" --error="${AF_OUTPUT_ROOT}/logs/smoke-%j.err" --export=ALL scripts/hpc/mesh_delta.slurm smoke)"
smoke="${smoke%%;*}"
record_job smoke_job_id "${smoke}"
fit="$(sbatch --parsable --dependency="afterok:${smoke}" --kill-on-invalid-dep=yes --array="0-${last}%${concurrency}" --output="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.err" --export=ALL scripts/hpc/mesh_delta.slurm run)"
fit="${fit%%;*}"
record_job fit_job_id "${fit}"
summary="$(sbatch --parsable --dependency="afterany:${fit}" --time=00:10:00 --output="${AF_OUTPUT_ROOT}/logs/summary-%j.out" --error="${AF_OUTPUT_ROOT}/logs/summary-%j.err" --export=ALL scripts/hpc/mesh_delta.slurm summarize)"
summary="${summary%%;*}"
record_job summary_job_id "${summary}"
cat "${AF_OUTPUT_ROOT}/submission.json"
