#!/bin/bash
# Export the public v4 candidates and submit the paired CPU comparison on ACES.

set -euo pipefail
: "${SCRATCH:?SCRATCH is unset}"
export AF_REPO_ROOT="${AF_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
export AF_PYTHON="${AF_PYTHON:-${AF_REPO_ROOT}/.venv/bin/python}"
export AF_GCCCORE_MODULE="${AF_GCCCORE_MODULE:-GCCcore/13.2.0}"
export AF_PYTHON_MODULE="${AF_PYTHON_MODULE:-Python/3.11.5}"
export AF_ACES_ACCOUNT="${AF_ACES_ACCOUNT:-156264627414}"
: "${AF_SOURCE_MULTIROUND_ROOT:?set AF_SOURCE_MULTIROUND_ROOT to the completed v4 multiround root}"
export AF_SOURCE_ROOT="${AF_SOURCE_ROOT:-${SCRATCH}/phase_b/collocation-node-cases-v4}"
export AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-${SCRATCH}/phase_b/collocation-node-v4-aces-cpu}"
export AF_CONFIG="${AF_CONFIG:-${AF_REPO_ROOT}/configs/collocation_node_v1.json}"
readonly concurrency="${AF_ARRAY_CONCURRENCY:-6}"
[[ "${concurrency}" =~ ^[1-6]$ ]] || {
  echo 'Concurrency must be between 1 and 6.' >&2
  exit 2
}
module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
for path in "${AF_PYTHON}" "${AF_CONFIG}" "${AF_SOURCE_MULTIROUND_ROOT}/plan.json"; do
  [[ -e "${path}" ]] || { echo "missing collocation-node input: ${path}" >&2; exit 2; }
done
[[ -z "$(git -C "${AF_REPO_ROOT}" status --porcelain)" ]] || {
  echo 'Use a clean isolated checkout.' >&2
  exit 2
}
export PYTHONPATH="${AF_REPO_ROOT}/src"
"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/export_collocation_node_cases.py" --source-root "${AF_SOURCE_MULTIROUND_ROOT}" --output "${AF_SOURCE_ROOT}"
"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/run_collocation_node_campaign.py" prepare --source "${AF_SOURCE_ROOT}" --config "${AF_CONFIG}" --output "${AF_OUTPUT_ROOT}"
if [[ -f "${AF_OUTPUT_ROOT}/submission.json" ]]; then
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}/submission.json" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1]))
print(json.dumps(record, indent=2))
if not record.get("submission_complete"):
    raise SystemExit("Partial submission: reconcile recorded jobs before continuing.")
PY
  exit 0
fi
mkdir "${AF_OUTPUT_ROOT}/submission.intent" 2>/dev/null || {
  echo 'A submission may already exist; reconcile the queue before retrying.' >&2
  exit 3
}
export AF_CODE_COMMIT
AF_CODE_COMMIT="$(git -C "${AF_REPO_ROOT}" rev-parse HEAD)"
record_job() {
  "${AF_PYTHON}" - "${AF_OUTPUT_ROOT}" "$1" "$2" "${AF_CODE_COMMIT}" <<'PY'
import sys
from pathlib import Path

from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json

root = Path(sys.argv[1])
path = root / "submission.json"
previous = read_json(path) if path.exists() else {}
frozen = read_json(root / "freeze.json")
write_json(
    path,
    {
        **previous,
        sys.argv[2]: sys.argv[3],
        "commit": sys.argv[4],
        "fit_tasks": len(frozen["tasks"]),
        "cases": len(frozen["cases"]),
        "cpus_per_task": 1,
        "gpus": 0,
        "submission_complete": sys.argv[2] == "summary_job_id",
    },
)
PY
}
mkdir -p "${AF_OUTPUT_ROOT}/logs"
last="$("${AF_PYTHON}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"])-1)' "${AF_OUTPUT_ROOT}/freeze.json")"
fit_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --array="0-${last}%${concurrency}" --output="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.out" --error="${AF_OUTPUT_ROOT}/logs/fit-%A_%a.err" --export=ALL "${AF_REPO_ROOT}/scripts/hpc/collocation_node_aces.slurm" run)"
readonly fit_job="${fit_submission%%;*}"
record_job fit_job_id "${fit_job}"
summary_submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --dependency="afterany:${fit_job}" --output="${AF_OUTPUT_ROOT}/logs/summary-%j.out" --error="${AF_OUTPUT_ROOT}/logs/summary-%j.err" --export=ALL "${AF_REPO_ROOT}/scripts/hpc/collocation_node_aces.slurm" summarize)"
readonly summary_job="${summary_submission%%;*}"
record_job summary_job_id "${summary_job}"
cat "${AF_OUTPUT_ROOT}/submission.json"
