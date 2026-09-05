#!/bin/bash
# Submit from an isolated, clean checkout; no GPU allocation or API calls.
set -euo pipefail
readonly af_user="${USER:?}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_HISTORICAL_RAW_ROOT:=${AF_WORK}/phase_b/raw-data-agent-fitted-v1}"
: "${AF_REFRESH_RAW_ROOT:=${AF_WORK}/phase_b/raw-data-agent-fitted-prompt-v3-refresh-v1}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/fitter-diagnostic-v1}"
: "${AF_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_fitter_diagnostic_v1.json}"
: "${AF_ARRAY_CONCURRENCY:=6}"
cd "${AF_REPO_ROOT}"
[[ -x "${AF_PYTHON}" ]] || { echo "Missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo 'Use a clean, dedicated checkout before submitting.' >&2; exit 2;
}
for directory in "${AF_PUBLIC_DATA_ROOT}" "${AF_HISTORICAL_RAW_ROOT}" "${AF_REFRESH_RAW_ROOT}"; do
  [[ -d "${directory}" ]] || { echo "Missing input directory: ${directory}" >&2; exit 2; }
done
[[ "${AF_ARRAY_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid concurrency' >&2; exit 2; }
export AF_REPO_ROOT AF_PYTHON AF_OUTPUT_ROOT AF_CONFIG AF_PUBLIC_DATA_ROOT
export AF_HISTORICAL_RAW_ROOT AF_REFRESH_RAW_ROOT
export AF_CODE_COMMIT="$(git rev-parse HEAD)"
export PYTHONPATH="${AF_REPO_ROOT}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_count="$("${AF_PYTHON}" - "${AF_CONFIG}" <<'PY'
import sys
from pathlib import Path
from autoformalism.rebuttal.fitter_diagnostic import ARMS, DiagnosticPlan, read_json
plan = DiagnosticPlan.model_validate(read_json(Path(sys.argv[1])))
if plan.fit_seconds + plan.replay_seconds + plan.supervisor_grace_seconds > 1380:
    raise SystemExit('Plan exceeds the 25-minute Slurm allocation; update Slurm time first.')
print(len(plan.cells) * len(plan.repetitions) * len(ARMS))
PY
)"
mkdir -p "${AF_OUTPUT_ROOT}/logs"
readonly job_script="${AF_REPO_ROOT}/scripts/hpc/phase_b_fitter_diagnostic_delta.slurm"
readonly log_template="${AF_OUTPUT_ROOT}/logs/%x-%A_%a.out"
submission_log="${AF_OUTPUT_ROOT}/logs/submission-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
printf 'commit=%s\n' "${AF_CODE_COMMIT}" > "${submission_log}"
prep="$(sbatch --parsable --output="${log_template}" "${job_script}" prepare)"
prep="${prep%%;*}"
printf 'prepare=%s\n' "${prep}" | tee -a "${submission_log}"
fits="$(sbatch --parsable --dependency="afterok:${prep}" \
  --array="0-$((task_count - 1))%${AF_ARRAY_CONCURRENCY}" \
  --output="${log_template}" "${job_script}" run)"
fits="${fits%%;*}"
printf 'fits=%s\n' "${fits}" | tee -a "${submission_log}"
summary="$(sbatch --parsable --dependency="afterany:${fits}" \
  --output="${log_template}" "${job_script}" summarize)"
summary="${summary%%;*}"
printf 'summary=%s\noutput=%s\n' "${summary}" "${AF_OUTPUT_ROOT}" | tee -a "${submission_log}"
