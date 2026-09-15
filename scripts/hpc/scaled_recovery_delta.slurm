#!/bin/bash
#SBATCH --job-name=fitter-saved-recovery
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_PYTHON:?}" "${AF_CASADI_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_SOURCE_ROOT:?}" "${AF_CODE_COMMIT:?}"
cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_CODE_COMMIT}" && -z "$(git status --porcelain)" ]] || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
readonly work="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/scaled-recovery-${SLURM_JOB_ID}.XXXXXX")"
echo "Node-local work: ${work}"
date -u '+Bootstrap begins: %Y-%m-%dT%H:%M:%SZ'
finish_preparation() {
  local result=$?
  trap - EXIT
  set +e
  # Even outer timeout/termination leaves an explicit preparation outcome.
  # Publication has its own bound; preserve the original failure exit code.
  timeout --kill-after=5s 120s "${AF_PYTHON}" -S "${work}/stage_scaled_recovery_runtime.py" publish-preparation \
    --output "${AF_OUTPUT_ROOT}" --work "${work}" --exit-code "${result}"
  local publication=$?
  if [[ "${result}" -ne 0 ]]; then exit "${result}"; fi
  exit "${publication}"
}
if [[ "${1:?}" == prepare ]]; then trap finish_preparation EXIT; fi
timeout --kill-after=5s 120s cp scripts/stage_scaled_recovery_runtime.py scripts/scaled_recovery_io.py \
  scripts/scaled_runtime_dependencies.py "${work}/"
case "${1:?}" in
  prepare)
    timeout --kill-after=10s 2400s "${AF_PYTHON}" -S "${work}/stage_scaled_recovery_runtime.py" build \
      --repo "${AF_REPO_ROOT}" --python "${AF_PYTHON}" --extras "${AF_CASADI_ROOT}" \
      --source "${AF_SOURCE_ROOT}" --output "${AF_OUTPUT_ROOT}" --work "${work}"
    ;;
  run)
    # Keep the supervisor stdout local too; periodic publication retains diagnostics.
    timeout --kill-after=10s 8400s "${AF_PYTHON}" -S "${work}/stage_scaled_recovery_runtime.py" restore \
      --output "${AF_OUTPUT_ROOT}" --work "${work}" --task-index "${SLURM_ARRAY_TASK_ID:?}" \
      >"${work}/run.log" 2>&1 || { tail -80 "${work}/run.log"; exit 1; }
    tail -30 "${work}/run.log"
    ;;
  summarize)
    timeout --kill-after=5s 500s "${AF_PYTHON}" -S scripts/run_scaled_recovery.py summarize --output "${AF_OUTPUT_ROOT}"
    ;;
  *) exit 2 ;;
esac
