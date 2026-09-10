#!/bin/bash
# User invokes this launcher; preparation performs no integration or fitting.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-methods-v1}"
export AF_CONFIG="${AF_CONFIG:-${AF_REPO_ROOT}/configs/fitter_methods_v1.json}"
readonly concurrency="${AF_ARRAY_CONCURRENCY:-2}"
[[ "${concurrency}" =~ ^[12]$ ]] || { echo 'Concurrency must be 1 or 2.' >&2; exit 2; }
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" && -z "$(git status --porcelain)" ]] || {
  echo 'Use the existing Python environment and a clean isolated checkout.' >&2; exit 2;
}
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git rev-parse HEAD)"
export AF_CASADI_ROOT="${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}"
export PYTHONPATH="${AF_REPO_ROOT}/src:${AF_CASADI_ROOT}"
mkdir -p "${AF_OUTPUT_ROOT}/logs"
"${AF_PYTHON}" scripts/run_fitter_methods.py prepare --config "${AF_CONFIG}" --output "${AF_OUTPUT_ROOT}"
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}/submission.json" <<'PY_CHECK'
import json, sys
record=json.load(open(sys.argv[1]))
print(json.dumps(record, indent=2))
if record.get("submission_complete") is not True:
    raise SystemExit("Partial submission: reconcile recorded jobs before continuing.")
PY_CHECK
  exit 0
fi
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || {
  echo 'Submission may already exist; reconcile the queue before retrying.' >&2; exit 3;
}
task_indices() {
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}/freeze.json" "$1" <<'PY'
import json, sys
from pathlib import Path
tasks=json.loads(Path(sys.argv[1]).read_text())["tasks"]
print(",".join(str(i) for i,t in enumerate(tasks) if t["kind"]==sys.argv[2]))
PY
}
record_job() {
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}/submission.json" "$1" "$2" "${AF_CODE_COMMIT}" <<'PY'
import sys
from pathlib import Path
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json
path=Path(sys.argv[1])
previous=read_json(path) if path.exists() else {}
tasks=read_json(path.parent / "freeze.json")["tasks"]
write_json(path, {**previous, sys.argv[2]:sys.argv[3], "commit":sys.argv[4],
    "cpus_per_task":1, "gpus":0,
    "guard_tasks":sum(t["kind"]=="guard" for t in tasks),
    "initializer_tasks":sum(t["kind"]=="initializer" for t in tasks),
    "fit_tasks":sum(t["kind"]=="fit" for t in tasks),
    "submission_complete":sys.argv[2]=="summary_job_id"})
PY
}
guard="$(sbatch --parsable --array="$(task_indices guard)%${concurrency}" --time=00:15:00 --output="${AF_OUTPUT_ROOT}/logs/guard-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/guard-%A_%a.err" --export=ALL scripts/hpc/fitter_methods_delta.slurm run)"
guard="${guard%%;*}"
record_job guard_job_id "${guard}"
dependency="${guard}"
initializers="$(task_indices initializer)"
if [[ -n "${initializers}" ]]; then
  initializer="$(sbatch --parsable --array="${initializers}%${concurrency}" --dependency="afterany:${guard}" --time=00:05:00 --output="${AF_OUTPUT_ROOT}/logs/init-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/init-%A_%a.err" --export=ALL scripts/hpc/fitter_methods_delta.slurm run)"
  initializer="${initializer%%;*}"
  record_job initializer_job_id "${initializer}"
  dependency="${initializer}"
fi
fit="$(sbatch --parsable --array="$(task_indices fit)%${concurrency}" --dependency="afterany:${dependency}" --output="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.err" --export=ALL scripts/hpc/fitter_methods_delta.slurm run)"
fit="${fit%%;*}"
record_job fit_job_id "${fit}"
summary="$(sbatch --parsable --dependency="afterany:${fit}" --time=00:05:00 --output="${AF_OUTPUT_ROOT}/logs/summary-%j.out" --error="${AF_OUTPUT_ROOT}/logs/summary-%j.err" --export=ALL scripts/hpc/fitter_methods_delta.slurm summarize)"
summary="${summary%%;*}"
record_job summary_job_id "${summary}"
cat "${AF_OUTPUT_ROOT}/submission.json"
