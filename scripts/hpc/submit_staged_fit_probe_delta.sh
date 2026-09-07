#!/bin/bash
# Freeze the reviewed candidate before submitting exactly one CPU job.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_FUNCTION_PLAN:?}" "${AF_FUNCTION_RESULTS:?}" "${AF_OUTPUT_ROOT:?}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_PUBLIC_DATA_ROOT="${AF_PUBLIC_DATA_ROOT:-/work/hdd/bibo/yxiao2/phase_b/inputs/public-prompt-v3}"
export AF_CONFIG="${AF_CONFIG:-${AF_REPO_ROOT}/configs/staged_fit_probe_v1.json}"
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" && -f "${AF_FUNCTION_PLAN}" && -d "${AF_FUNCTION_RESULTS}" ]] || {
  echo 'Python or source artifact path is unavailable.' >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo 'Use a clean isolated checkout.' >&2; exit 2;
}
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git rev-parse HEAD)"
export PYTHONPATH="${AF_REPO_ROOT}/src"
mkdir -p "${AF_OUTPUT_ROOT}/logs"
"${AF_PYTHON}" scripts/run_staged_fit_probe.py prepare \
  --config "${AF_CONFIG}" --function-plan "${AF_FUNCTION_PLAN}" \
  --function-results "${AF_FUNCTION_RESULTS}" --public-root "${AF_PUBLIC_DATA_ROOT}" \
  --output "${AF_OUTPUT_ROOT}"
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  cat "${AF_OUTPUT_ROOT}/submission.json"
  exit 0
fi
# Atomic directory creation guards concurrent/uncertain submissions too.
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || {
  echo 'Submission may already exist; reconcile the queue before retrying.' >&2; exit 3;
}
job="$(sbatch --parsable \
  --output="${AF_OUTPUT_ROOT}/logs/slurm-%j.out" \
  --error="${AF_OUTPUT_ROOT}/logs/slurm-%j.err" \
  --export=ALL scripts/hpc/staged_fit_probe_delta.slurm)"
job="${job%%;*}"
"${AF_PYTHON}" - "${AF_OUTPUT_ROOT}" "${job}" "${AF_CODE_COMMIT}" <<'PY'
import sys
from pathlib import Path
from autoformalism.rebuttal.fitter_diagnostic import write_json
write_json(Path(sys.argv[1]) / "submission.json", {
    "job_id": sys.argv[2], "commit": sys.argv[3], "cpus": 1, "gpus": 0,
    "wall_minutes": 15,
})
PY
cat "${AF_OUTPUT_ROOT}/submission.json"
