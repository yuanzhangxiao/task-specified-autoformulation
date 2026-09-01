#!/bin/bash
# Freeze and confirm an ACES-selected proposer operating point on Delta A40s.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_SOURCE_PLAN:=${AF_REPO_ROOT}/configs/phase_b_proposer_transport_calibration_v2.json}"
: "${AF_PRIMARY_ANALYSIS:?AF_PRIMARY_ANALYSIS must name the copied ACES v2 analysis}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/proposer-transport-confirmation-v2-delta-a40x4}"
: "${AF_CALIBRATION_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_transport_calibration_120b.slurm}"
: "${AF_ANALYSIS_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_transport_confirmation_analysis.slurm}"

readonly generated_config="${AF_OUTPUT_ROOT}/selected_confirmation_plan.json"
readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -f "${AF_SOURCE_PLAN}" ]] || { echo "missing source plan: ${AF_SOURCE_PLAN}" >&2; exit 2; }
[[ -f "${AF_PRIMARY_ANALYSIS}" ]] || { echo "missing ACES analysis: ${AF_PRIMARY_ANALYSIS}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data overlay" >&2; exit 2; }
[[ -d "${AF_TARGET_CONTRACT_ROOT}" ]] || { echo "missing target contracts" >&2; exit 2; }
[[ -f "${AF_CALIBRATION_JOB}" ]] || { echo "missing GPU job" >&2; exit 2; }
[[ -f "${AF_ANALYSIS_JOB}" ]] || { echo "missing analysis job" >&2; exit 2; }
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_proposer_transport_confirmation.py \
  --source-plan "${AF_SOURCE_PLAN}" \
  --source-analysis "${AF_PRIMARY_ANALYSIS}" \
  --output-config "${generated_config}" \
  --primary-platform aces-h100x2 \
  --confirmation-platform delta-a40x4
"${AF_PYTHON}" scripts/prepare_phase_b_proposer_transport_calibration.py \
  --config "${generated_config}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}"

readonly task_count="$(wc -l <"${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | tr -d ' ')"
((task_count > 0)) || { echo "confirmation task plan is empty" >&2; exit 2; }
readonly task_max=$((task_count - 1))
readonly common_export="ALL,AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_CALIBRATION_CONFIG=${generated_config}"
calibration_submission="$(
  sbatch --parsable \
    --array="0-${task_max}%2" \
    --output="${AF_REPO_ROOT}/logs/phase-b-proposer-confirm-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-proposer-confirm-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CALIBRATION_JOB}"
)"
readonly calibration_job_id="${calibration_submission%%;*}"
analysis_submission="$(
  sbatch --parsable \
    --dependency="afterany:${calibration_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-proposer-confirm-analysis-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-proposer-confirm-analysis-%j.err" \
    --export="ALL,AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_PRIMARY_ANALYSIS=${AF_PRIMARY_ANALYSIS}" \
    "${AF_ANALYSIS_JOB}"
)"
readonly analysis_job_id="${analysis_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg calibration_job_id "${calibration_job_id}" \
  --arg analysis_job_id "${analysis_job_id}" \
  --arg handoff_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/cross_cluster_handoff.json" | awk '{print $1}')" \
  --arg primary_analysis_sha256 "$(sha256sum "${AF_PRIMARY_ANALYSIS}" | awk '{print $1}')" \
  '{
    schema_version: "phase-b-proposer-cross-cluster-submission-1",
    submitted_at_utc: $submitted_at_utc,
    primary_platform: "aces-h100x2",
    confirmation_platform: "delta-a40x4",
    calibration_job_id: $calibration_job_id,
    analysis_job_id: $analysis_job_id,
    handoff_sha256: $handoff_sha256,
    primary_analysis_sha256: $primary_analysis_sha256,
    test_data_opened: false,
    scientific_judge_called: false,
    parameter_fitting_performed: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "DELTA_PROPOSER_CONFIRMATION_JOB=${calibration_job_id}"
echo "DELTA_PROPOSER_CONFIRMATION_ANALYSIS_JOB=${analysis_job_id}"
echo "DELTA_SUBMISSION_MANIFEST=${submission_manifest}"
