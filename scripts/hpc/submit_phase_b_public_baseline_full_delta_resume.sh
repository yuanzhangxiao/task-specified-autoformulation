#!/bin/bash
# Retry only incomplete tasks from the complete Delta baseline matrix.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/public-baselines-full-v1}"
: "${AF_JULIA_DEPOT:=${AF_WORK}/julia-depot-pysr-1.5.9}"
: "${AF_PERSISTENCE_CONCURRENCY:=12}"
: "${AF_SINDY_CONCURRENCY:=12}"
: "${AF_PYSR_CONCURRENCY:=12}"
: "${AF_PREPARE_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_delta_pysr_prepare.slurm}"
: "${AF_CPU_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_delta_cpu.slurm}"
: "${AF_SUMMARY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_delta_summary.slurm}"
: "${AF_READINESS_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_full_delta_readiness.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
[[ -x "${AF_PYTHON}" && -f "${submission_manifest}" ]] || {
  echo "the original full-matrix submission is unavailable" >&2
  exit 2
}
cd "${AF_REPO_ROOT}"
readonly source_code_commit="$(jq -r '.source_code_commit' "${submission_manifest}")"
[[ "$(git rev-parse HEAD)" == "${source_code_commit}" ]] || {
  echo "resume requires the original source commit ${source_code_commit}" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "resume requires a clean tracked worktree" >&2
  exit 2
}
(
  cd "${AF_OUTPUT_ROOT}/frozen"
  sha256sum -c plan.json.sha256
  sha256sum -c task_plan.jsonl.sha256
  sha256sum -c planned_resource_ledger.jsonl.sha256
  sha256sum -c freeze_manifest.json.sha256
)

readonly resume_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
readonly retry_audit="${AF_OUTPUT_ROOT}/resume_audits/resume_audit_${resume_stamp}.json"
mkdir -p "$(dirname "${retry_audit}")"
"${AF_PYTHON}" scripts/find_incomplete_phase_b_public_baseline_tasks.py \
  --task-plan "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" \
  --runs-root "${AF_OUTPUT_ROOT}/runs" >"${retry_audit}.tmp"
mv "${retry_audit}.tmp" "${retry_audit}"
readonly persistence_indices="$(jq -r '.incomplete_indices_by_method.persistence | join(",")' "${retry_audit}")"
readonly sindy_indices="$(jq -r '.incomplete_indices_by_method.sindy | join(",")' "${retry_audit}")"
readonly pysr_indices="$(jq -r '.incomplete_indices_by_method.pysr | join(",")' "${retry_audit}")"
readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_JULIA_DEPOT=${AF_JULIA_DEPOT},AF_SOURCE_CODE_COMMIT=${source_code_commit}"

dependency_ids=()
prepare_job_id=""
persistence_job_id=""
sindy_job_id=""
pysr_job_id=""
if [[ -n "${persistence_indices}" ]]; then
  submission="$(sbatch --parsable --account=bibo-delta-cpu \
    --array="${persistence_indices}%${AF_PERSISTENCE_CONCURRENCY}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-persistence-retry-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-persistence-retry-%A_%a.err" \
    --export="${common_export}" "${AF_CPU_JOB}")"
  persistence_job_id="${submission%%;*}"
  dependency_ids+=("${persistence_job_id}")
fi
if [[ -n "${sindy_indices}" ]]; then
  submission="$(sbatch --parsable --account=bibo-delta-cpu \
    --array="${sindy_indices}%${AF_SINDY_CONCURRENCY}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-sindy-retry-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-sindy-retry-%A_%a.err" \
    --export="${common_export}" "${AF_CPU_JOB}")"
  sindy_job_id="${submission%%;*}"
  dependency_ids+=("${sindy_job_id}")
fi
if [[ -n "${pysr_indices}" ]]; then
  submission="$(sbatch --parsable --account=bibo-delta-cpu \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-pysr-prepare-retry-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-pysr-prepare-retry-%j.err" \
    --export="${common_export}" "${AF_PREPARE_JOB}")"
  prepare_job_id="${submission%%;*}"
  submission="$(sbatch --parsable --account=bibo-delta-cpu \
    --array="${pysr_indices}%${AF_PYSR_CONCURRENCY}" \
    --dependency="afterok:${prepare_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-pysr-retry-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-pysr-retry-%A_%a.err" \
    --export="${common_export}" "${AF_CPU_JOB}")"
  pysr_job_id="${submission%%;*}"
  dependency_ids+=("${pysr_job_id}")
fi

summary_args=(
  --parsable
  --account=bibo-delta-cpu
  --output="${AF_REPO_ROOT}/logs/phase-b-full-summary-retry-%j.out"
  --error="${AF_REPO_ROOT}/logs/phase-b-full-summary-retry-%j.err"
  --export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}"
)
if [[ "${#dependency_ids[@]}" -gt 0 ]]; then
  readonly dependency="$(IFS=:; echo "${dependency_ids[*]}")"
  summary_args+=(--dependency="afterany:${dependency}")
fi
submission="$(sbatch "${summary_args[@]}" "${AF_SUMMARY_JOB}")"
readonly summary_job_id="${submission%%;*}"
submission="$(sbatch --parsable --account=bibo-delta-cpu \
  --dependency="afterok:${summary_job_id}" \
  --output="${AF_REPO_ROOT}/logs/phase-b-full-readiness-retry-%j.out" \
  --error="${AF_REPO_ROOT}/logs/phase-b-full-readiness-retry-%j.err" \
  --export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_SOURCE_CODE_COMMIT=${source_code_commit}" \
  "${AF_READINESS_JOB}")"
readonly readiness_job_id="${submission%%;*}"
readonly resume_manifest="${AF_OUTPUT_ROOT}/resume_submission_manifest_${summary_job_id}.json"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_code_commit "${source_code_commit}" \
  --arg retry_audit_path "resume_audits/$(basename "${retry_audit}")" \
  --arg retry_audit_sha256 "$(sha256sum "${retry_audit}" | awk '{print $1}')" \
  --arg prepare_job_id "${prepare_job_id}" \
  --arg persistence_job_id "${persistence_job_id}" \
  --arg sindy_job_id "${sindy_job_id}" \
  --arg pysr_job_id "${pysr_job_id}" \
  --arg summary_job_id "${summary_job_id}" \
  --arg readiness_job_id "${readiness_job_id}" \
  '{
    schema_version: "phase-b-public-baseline-full-resume-submission-1",
    submitted_at_utc: $submitted_at_utc,
    source_code_commit: $source_code_commit,
    retry_audit_path: $retry_audit_path,
    retry_audit_sha256: $retry_audit_sha256,
    pysr_prepare_job_id: $prepare_job_id,
    persistence_job_id: $persistence_job_id,
    sindy_job_id: $sindy_job_id,
    pysr_job_id: $pysr_job_id,
    summary_job_id: $summary_job_id,
    common_readiness_job_id: $readiness_job_id,
    test_data_opened: false,
    private_reference_opened: false
  }' >"${resume_manifest}.tmp"
mv "${resume_manifest}.tmp" "${resume_manifest}"

cat "${retry_audit}"
echo "DELTA_RETRY_PERSISTENCE_JOB=${persistence_job_id}"
echo "DELTA_RETRY_SINDY_JOB=${sindy_job_id}"
echo "DELTA_RETRY_PYSR_JOB=${pysr_job_id}"
echo "DELTA_RETRY_SUMMARY_JOB=${summary_job_id}"
echo "DELTA_RETRY_COMMON_READINESS_JOB=${readiness_job_id}"
echo "DELTA_RETRY_MANIFEST=${resume_manifest}"
