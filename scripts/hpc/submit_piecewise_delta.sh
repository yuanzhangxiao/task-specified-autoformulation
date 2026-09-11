#!/bin/bash
# Import the portable public-only bundle and submit four paired arms for every case.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/work/hdd/bibo/yxiao2/phase_b/collocation-node-cases-v1}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/piecewise-fitter-v1}"
export AF_CONFIG="${AF_CONFIG:-${AF_REPO_ROOT}/configs/piecewise_fitter_v1.json}"
export AF_CASADI_ROOT="${AF_CASADI_ROOT:-/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps}"
readonly concurrency="${AF_ARRAY_CONCURRENCY:-2}"
[[ "${concurrency}" =~ ^[12]$ ]] || { echo 'Concurrency must be 1 or 2.' >&2; exit 2; }
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" && -z "$(git status --porcelain)" ]] || {
  echo 'Use the existing Python and a clean isolated checkout.' >&2; exit 2;
}
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git rev-parse HEAD)"
export PYTHONPATH="${AF_REPO_ROOT}/src:${AF_CASADI_ROOT}"
mkdir -p "${AF_OUTPUT_ROOT}/logs"
"${AF_PYTHON}" scripts/run_piecewise_campaign.py prepare --source "${AF_SOURCE_ROOT}" --config "${AF_CONFIG}" --output "${AF_OUTPUT_ROOT}"
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}/submission.json" <<'PY'
import json, sys
record=json.load(open(sys.argv[1]))
print(json.dumps(record, indent=2))
if not record.get("submission_complete"):
    raise SystemExit("Partial submission: reconcile recorded jobs before continuing.")
PY
  exit 0
fi
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || {
  echo 'A submission may already exist; reconcile the queue before retrying.' >&2; exit 3;
}
record_job() {
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}" "$1" "$2" "${AF_CODE_COMMIT}" <<'PY'
import sys
from pathlib import Path
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json
root=Path(sys.argv[1]); path=root / "submission.json"
previous=read_json(path) if path.exists() else {}
frozen=read_json(root / "freeze.json")
write_json(path, {**previous, sys.argv[2]:sys.argv[3], "commit":sys.argv[4],
    "fit_tasks":len(frozen["cases"]), "fits":4*len(frozen["cases"]),
    "cpus_per_task":1, "gpus":0, "submission_complete":sys.argv[2]=="summary_job_id"})
PY
}
last="$("${AF_PYTHON}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["cases"])-1)' "${AF_OUTPUT_ROOT}/freeze.json")"
fit="$(sbatch --parsable --array="0-${last}%${concurrency}" --output="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.err" --export=ALL scripts/hpc/piecewise_delta.slurm run)"
fit="${fit%%;*}"
record_job fit_job_id "${fit}"
summary="$(sbatch --parsable --dependency="afterany:${fit}" --time=00:10:00 --output="${AF_OUTPUT_ROOT}/logs/summary-%j.out" --error="${AF_OUTPUT_ROOT}/logs/summary-%j.err" --export=ALL scripts/hpc/piecewise_delta.slurm summarize)"
summary="${summary%%;*}"
record_job summary_job_id "${summary}"
cat "${AF_OUTPUT_ROOT}/submission.json"
