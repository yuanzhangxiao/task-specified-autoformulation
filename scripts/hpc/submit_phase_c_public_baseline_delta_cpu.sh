#!/bin/bash
# Freeze and submit SINDy and PySR on the Phase C roster cells on Delta.
#
# Development data only: the release has no test split. Each method keeps its
# Phase B settings; only the data source changes, to the verified release.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-phase-c-baselines}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PHASE_C_RELEASE:=${AF_WORK}/phase_c/release-public-v2}"
: "${AF_BASELINE_CONFIG:=${AF_REPO_ROOT}/configs/phase_c_public_baseline_delta_cpu_v1.json}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_c/public-baselines-classical-v1}"
: "${AF_JULIA_DEPOT:=${AF_WORK}/julia-depot-pysr-1.5.9}"
: "${AF_SINDY_CONCURRENCY:=16}"
: "${AF_PYSR_CONCURRENCY:=16}"
: "${AF_SINDY_TIME:=00:40:00}"
: "${AF_PYSR_TIME:=01:15:00}"
: "${AF_PREPARE_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_delta_pysr_prepare.slurm}"
: "${AF_CPU_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_c_public_baseline_delta_cpu.slurm}"
: "${AF_SUMMARY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_delta_summary.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -f "${AF_PHASE_C_RELEASE}/summary.json" ]] || {
  echo "missing Phase C release receipt: ${AF_PHASE_C_RELEASE}/summary.json" >&2
  exit 2
}
for script in "${AF_PREPARE_JOB}" "${AF_CPU_JOB}" "${AF_SUMMARY_JOB}"; do
  [[ -f "${script}" ]] || { echo "missing job script: ${script}" >&2; exit 2; }
done
for value in "${AF_SINDY_CONCURRENCY}" "${AF_PYSR_CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid array concurrency: ${value}" >&2; exit 2; }
done
"${AF_PYTHON}" -c \
  'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pysr") else 2)' \
  || {
    echo "PySR is missing; install with: ${AF_PYTHON} -m pip install 'pysr==1.5.9'" >&2
    exit 2
  }
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  echo "use a new AF_OUTPUT_ROOT; completed runs are never silently overwritten" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_JULIA_DEPOT}"
cd "${AF_REPO_ROOT}"
readonly source_code_commit="$(git rev-parse HEAD)"
[[ "${source_code_commit}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "cannot resolve the source commit" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "submission requires a clean tracked worktree" >&2
  exit 2
}
# The shared environment may have another checkout installed; this one must win.
export PYTHONPATH="${AF_REPO_ROOT}/src"
"${AF_PYTHON}" scripts/prepare_phase_c_public_baseline.py \
  --config "${AF_BASELINE_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --release "${AF_PHASE_C_RELEASE}"

readonly sindy_indices="$(
  jq -r 'select(.method == "sindy") | .task_index' \
    "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | paste -sd, -
)"
readonly pysr_indices="$(
  jq -r 'select(.method == "pysr") | .task_index' \
    "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | paste -sd, -
)"
[[ -n "${sindy_indices}" && -n "${pysr_indices}" ]] || {
  echo "plan lacks SINDy or PySR tasks" >&2
  exit 2
}

# No value here may contain a comma: sbatch --export splits on commas.
readonly common_export="ALL,PYTHONPATH=${AF_REPO_ROOT}/src,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_PHASE_C_RELEASE=${AF_PHASE_C_RELEASE},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_JULIA_DEPOT=${AF_JULIA_DEPOT},AF_SOURCE_CODE_COMMIT=${source_code_commit}"
prepare_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --output="${AF_REPO_ROOT}/logs/phase-c-pysr-prepare-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-c-pysr-prepare-%j.err" \
    --export="${common_export}" \
    "${AF_PREPARE_JOB}"
)"
readonly prepare_job_id="${prepare_submission%%;*}"
sindy_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --time="${AF_SINDY_TIME}" \
    --array="${sindy_indices}%${AF_SINDY_CONCURRENCY}" \
    --output="${AF_REPO_ROOT}/logs/phase-c-sindy-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-c-sindy-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CPU_JOB}"
)"
readonly sindy_job_id="${sindy_submission%%;*}"
pysr_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --time="${AF_PYSR_TIME}" \
    --array="${pysr_indices}%${AF_PYSR_CONCURRENCY}" \
    --dependency="afterok:${prepare_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-c-pysr-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-c-pysr-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CPU_JOB}"
)"
readonly pysr_job_id="${pysr_submission%%;*}"
summary_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterany:${sindy_job_id}:${pysr_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-c-baseline-summary-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-c-baseline-summary-%j.err" \
    --export="ALL,PYTHONPATH=${AF_REPO_ROOT}/src,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}" \
    "${AF_SUMMARY_JOB}"
)"
readonly summary_job_id="${summary_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_code_commit "${source_code_commit}" \
  --arg release "${AF_PHASE_C_RELEASE}" \
  --arg prepare_job_id "${prepare_job_id}" \
  --arg sindy_job_id "${sindy_job_id}" \
  --arg pysr_job_id "${pysr_job_id}" \
  --arg summary_job_id "${summary_job_id}" \
  --arg sindy_task_indices "${sindy_indices}" \
  --arg pysr_task_indices "${pysr_indices}" \
  --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/plan.json" | awk '{print $1}')" \
  '{
    schema_version: "phase-c-public-baseline-submission-1",
    submitted_at_utc: $submitted_at_utc,
    source_code_commit: $source_code_commit,
    platform: "delta_cpu",
    phase_c_release: $release,
    pysr_prepare_job_id: $prepare_job_id,
    sindy_job_id: $sindy_job_id,
    pysr_job_id: $pysr_job_id,
    summary_job_id: $summary_job_id,
    sindy_task_indices: $sindy_task_indices,
    pysr_task_indices: $pysr_task_indices,
    plan_sha256: $plan_sha256,
    test_data_opened: false,
    private_reference_opened: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "PHASE_C_PYSR_PREPARE_JOB=${prepare_job_id}"
echo "PHASE_C_SINDY_JOB=${sindy_job_id}"
echo "PHASE_C_PYSR_JOB=${pysr_job_id}"
echo "PHASE_C_BASELINE_SUMMARY_JOB=${summary_job_id}"
echo "PHASE_C_BASELINE_MANIFEST=${submission_manifest}"
