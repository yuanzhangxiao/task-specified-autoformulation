#!/bin/bash
# Submit sealed predictive and common symbolic evaluation after development freeze.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_DEVELOPMENT_ROOT:=${AF_WORK}/phase_b/public-baselines-full-v1}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/public-baselines-full-v1-postfreeze}"
: "${AF_PREDICTIVE_CONCURRENCY:=32}"
: "${AF_POSTFREEZE_SHARD_COUNT:=24}"
: "${AF_POSTFREEZE_CONCURRENCY:=12}"

readonly development_submission="${AF_DEVELOPMENT_ROOT}/submission_manifest.json"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data" >&2; exit 2; }
[[ -f "${development_submission}" ]] || {
  echo "missing development submission manifest: ${development_submission}" >&2
  exit 2
}
: "${AF_DEVELOPMENT_READINESS_JOB_ID:=$(jq -er '.common_readiness_job_id' "${development_submission}")}"
for value in "${AF_PREDICTIVE_CONCURRENCY}" "${AF_POSTFREEZE_SHARD_COUNT}" "${AF_POSTFREEZE_CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid concurrency value" >&2; exit 2; }
done
readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  echo "use a new AF_OUTPUT_ROOT; frozen evaluations are never overwritten" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"
readonly evaluator_code_commit="$(git rev-parse HEAD)"
[[ "${evaluator_code_commit}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "cannot resolve evaluator commit" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "post-freeze submission requires a clean tracked worktree" >&2
  exit 2
}
readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_DEVELOPMENT_ROOT=${AF_DEVELOPMENT_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_EVALUATOR_CODE_COMMIT=${evaluator_code_commit},AF_POSTFREEZE_SHARD_COUNT=${AF_POSTFREEZE_SHARD_COUNT}"

prepare_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterok:${AF_DEVELOPMENT_READINESS_JOB_ID}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-classical-freeze-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-classical-freeze-%j.err" \
    --export="${common_export}" \
    scripts/hpc/phase_b_public_baseline_postfreeze_prepare.slurm
)"
readonly prepare_job_id="${prepare_submission%%;*}"
predictive_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterok:${prepare_job_id}" \
    --array="0-359%${AF_PREDICTIVE_CONCURRENCY}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-classical-test-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-classical-test-%A_%a.err" \
    --export="${common_export}" \
    scripts/hpc/phase_b_public_baseline_predictive_test.slurm
)"
readonly predictive_job_id="${predictive_submission%%;*}"
postfreeze_last_index="$((AF_POSTFREEZE_SHARD_COUNT - 1))"
postfreeze_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterok:${prepare_job_id}" \
    --array="0-${postfreeze_last_index}%${AF_POSTFREEZE_CONCURRENCY}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-classical-free-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-classical-free-%A_%a.err" \
    --export="${common_export}" \
    scripts/hpc/phase_b_public_baseline_common_postfreeze.slurm
)"
readonly postfreeze_job_id="${postfreeze_submission%%;*}"
predictive_summary_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterok:${predictive_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-classical-test-summary-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-classical-test-summary-%j.err" \
    --export="${common_export}" \
    scripts/hpc/phase_b_public_baseline_predictive_summary.slurm
)"
readonly predictive_summary_job_id="${predictive_summary_submission%%;*}"
common_finalize_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterok:${postfreeze_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-classical-common-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-classical-common-%j.err" \
    --export="${common_export}" \
    scripts/hpc/phase_b_public_baseline_common_finalize.slurm
)"
readonly common_finalize_job_id="${common_finalize_submission%%;*}"
readiness_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterok:${predictive_summary_job_id}:${common_finalize_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-classical-ready-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-classical-ready-%j.err" \
    --export="${common_export}" \
    scripts/hpc/phase_b_public_baseline_postfreeze_readiness.slurm
)"
readonly readiness_job_id="${readiness_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg evaluator_code_commit "${evaluator_code_commit}" \
  --arg development_readiness_job_id "${AF_DEVELOPMENT_READINESS_JOB_ID}" \
  --arg prepare_job_id "${prepare_job_id}" \
  --arg predictive_job_id "${predictive_job_id}" \
  --arg postfreeze_job_id "${postfreeze_job_id}" \
  --arg predictive_summary_job_id "${predictive_summary_job_id}" \
  --arg common_finalize_job_id "${common_finalize_job_id}" \
  --arg readiness_job_id "${readiness_job_id}" \
  '{
    schema_version: "phase-b-public-baseline-postfreeze-submission-1",
    submitted_at_utc: $submitted_at_utc,
    evaluator_code_commit: $evaluator_code_commit,
    development_readiness_job_id: $development_readiness_job_id,
    final_model_prepare_job_id: $prepare_job_id,
    predictive_test_job_id: $predictive_job_id,
    common_postfreeze_job_id: $postfreeze_job_id,
    predictive_summary_job_id: $predictive_summary_job_id,
    common_finalize_job_id: $common_finalize_job_id,
    readiness_job_id: $readiness_job_id,
    test_data_opened_at_submission: false,
    private_reference_opened: false,
    oracle_derivatives_used: false,
    oracle_latent_states_used: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "DELTA_FINAL_MODEL_FREEZE_JOB=${prepare_job_id}"
echo "DELTA_PREDICTIVE_TEST_JOB=${predictive_job_id}"
echo "DELTA_COMMON_POSTFREEZE_JOB=${postfreeze_job_id}"
echo "DELTA_PREDICTIVE_SUMMARY_JOB=${predictive_summary_job_id}"
echo "DELTA_COMMON_FINALIZE_JOB=${common_finalize_job_id}"
echo "DELTA_POSTFREEZE_READINESS_JOB=${readiness_job_id}"
echo "DELTA_POSTFREEZE_MANIFEST=${submission_manifest}"
