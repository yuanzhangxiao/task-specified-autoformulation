#!/bin/bash
# User-invoked diagnostic submission. Codex does not execute this automatically.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/work/hdd/bibo/yxiao2/phase_b/staged-fit-v1-2f4d10c}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-stagnation-v1}"
export AF_CONFIG="${AF_CONFIG:-${AF_REPO_ROOT}/configs/fitter_stagnation_v1.json}"
readonly concurrency="${AF_ARRAY_CONCURRENCY:-2}"
[[ "${concurrency}" =~ ^[1-5]$ ]] || { echo 'Concurrency must be 1 through 5.' >&2; exit 2; }
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" && -f "${AF_SOURCE_ROOT}/freeze.json" ]] || {
  echo 'Python or the original fitting snapshot is unavailable.' >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo 'Use a clean isolated diagnostic checkout.' >&2; exit 2;
}
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git rev-parse HEAD)"
export PYTHONPATH="${AF_REPO_ROOT}/src"
mkdir -p "${AF_OUTPUT_ROOT}/logs"
"${AF_PYTHON}" scripts/run_fitter_stagnation.py prepare --config "${AF_CONFIG}" \
  --source "${AF_SOURCE_ROOT}" --output "${AF_OUTPUT_ROOT}"
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  cat "${AF_OUTPUT_ROOT}/submission.json"
  exit 0
fi
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || {
  echo 'Submission may already exist; reconcile the queue before retrying.' >&2; exit 3;
}
job="$(sbatch --parsable --array="0-4%${concurrency}" \
  --output="${AF_OUTPUT_ROOT}/logs/slurm-%A_%a.out" \
  --error="${AF_OUTPUT_ROOT}/logs/slurm-%A_%a.err" \
  --export=ALL scripts/hpc/fitter_stagnation_delta.slurm run)"
job="${job%%;*}"
# Record the array before attempting the separate dependent summary job.
"${AF_PYTHON}" - "${AF_OUTPUT_ROOT}" "${job}" "${AF_CODE_COMMIT}" <<'PY'
import sys
from pathlib import Path
from autoformalism.rebuttal.fitter_diagnostic import write_json
write_json(Path(sys.argv[1]) / "submission.json", {
    "array_job_id": sys.argv[2], "commit": sys.argv[3], "task_count": 5,
    "cpus_per_task": 1, "gpus": 0, "wall_minutes": 15,
})
PY
summary="$(sbatch --parsable --dependency="afterany:${job}" --time=00:05:00 \
  --output="${AF_OUTPUT_ROOT}/logs/summary-%j.out" \
  --error="${AF_OUTPUT_ROOT}/logs/summary-%j.err" \
  --export=ALL scripts/hpc/fitter_stagnation_delta.slurm summarize)"
summary="${summary%%;*}"
"${AF_PYTHON}" - "${AF_OUTPUT_ROOT}" "${summary}" <<'PY'
import sys
from pathlib import Path
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json
path = Path(sys.argv[1]) / "submission.json"
write_json(path, {**read_json(path), "summary_job_id": sys.argv[2]})
PY
cat "${AF_OUTPUT_ROOT}/submission.json"
