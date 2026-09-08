#!/bin/bash
# Explicitly user-invoked. No automatic submission from a Codex task or monitor.
set -euo pipefail
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-runtime-v2}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_b/fitter-signed-offset-v1}"
export AF_CONFIG="${AF_CONFIG:-${AF_REPO_ROOT}/configs/fitter_signed_offset_v1.json}"
readonly concurrency="${AF_ARRAY_CONCURRENCY:-2}"
[[ "${concurrency}" =~ ^[1-6]$ ]] || { echo 'Concurrency must be 1 through 6.' >&2; exit 2; }
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" && -f "${AF_SOURCE_ROOT}/freeze.json" ]] || {
  echo 'Python or the previous diagnostic snapshot is unavailable.' >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo 'Use a clean isolated runtime checkout.' >&2; exit 2;
}
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git rev-parse HEAD)"
export PYTHONPATH="${AF_REPO_ROOT}/src"
mkdir -p "${AF_OUTPUT_ROOT}/logs"
"${AF_PYTHON}" scripts/run_fitter_offset.py prepare --config "${AF_CONFIG}" \
  --source "${AF_SOURCE_ROOT}" --output "${AF_OUTPUT_ROOT}"
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  cat "${AF_OUTPUT_ROOT}/submission.json"
  exit 0
fi
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || {
  echo 'Submission may already exist; reconcile the queue before retrying.' >&2; exit 3;
}
record_job() {
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}" "$1" "$2" "${AF_CODE_COMMIT}" <<'PY'
import sys
from pathlib import Path
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json
path = Path(sys.argv[1]) / "submission.json"
previous = read_json(path) if path.exists() else {}
write_json(path, {**previous, sys.argv[2]:sys.argv[3], "commit":sys.argv[4],
                 "profile_tasks":1, "fit_tasks":4, "previous_replay_tasks":2,
                 "cpus_per_task":1, "gpus":0})
PY
}
profile="$(sbatch --parsable --array="0%${concurrency}" --time=00:15:00 \
  --output="${AF_OUTPUT_ROOT}/logs/profile-%A_%a.out" \
  --error="${AF_OUTPUT_ROOT}/logs/profile-%A_%a.err" \
  --export=ALL scripts/hpc/fitter_offset_delta.slurm run)"
profile="${profile%%;*}"
record_job profile_job_id "${profile}"
fit="$(sbatch --parsable --array="1-6%${concurrency}" --dependency="afterany:${profile}" \
  --output="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.out" \
  --error="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.err" \
  --export=ALL scripts/hpc/fitter_offset_delta.slurm run)"
fit="${fit%%;*}"
record_job fit_job_id "${fit}"
summary="$(sbatch --parsable --dependency="afterany:${fit}" --time=00:05:00 \
  --output="${AF_OUTPUT_ROOT}/logs/summary-%j.out" \
  --error="${AF_OUTPUT_ROOT}/logs/summary-%j.err" \
  --export=ALL scripts/hpc/fitter_offset_delta.slurm summarize)"
summary="${summary%%;*}"
record_job summary_job_id "${summary}"
cat "${AF_OUTPUT_ROOT}/submission.json"
