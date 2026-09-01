#!/bin/bash
# Freeze and submit the GPT-OSS-120B proposer output-budget calibration.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/proposer-transport-calibration-v1}"
: "${AF_CALIBRATION_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_proposer_transport_calibration_v1.json}"
: "${AF_CALIBRATION_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_transport_calibration_120b.slurm}"
: "${AF_ANALYSIS_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_transport_calibration_analysis.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data overlay" >&2; exit 2; }
[[ -d "${AF_TARGET_CONTRACT_ROOT}" ]] || { echo "missing target contracts" >&2; exit 2; }
[[ -f "${AF_CALIBRATION_CONFIG}" ]] || { echo "missing calibration config" >&2; exit 2; }
[[ -f "${AF_CALIBRATION_JOB}" ]] || { echo "missing GPU job" >&2; exit 2; }
[[ -f "${AF_ANALYSIS_JOB}" ]] || { echo "missing analysis job" >&2; exit 2; }
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_proposer_transport_calibration.py \
  --config "${AF_CALIBRATION_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}"

readonly common_export="ALL,AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_CALIBRATION_CONFIG=${AF_CALIBRATION_CONFIG}"
calibration_submission="$(
  sbatch --parsable \
    --array=0-5%2 \
    --output="${AF_REPO_ROOT}/logs/phase-b-proposer-cal-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-proposer-cal-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CALIBRATION_JOB}"
)"
readonly calibration_job_id="${calibration_submission%%;*}"
analysis_submission="$(
  sbatch --parsable \
    --dependency="afterany:${calibration_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-proposer-cal-analysis-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-proposer-cal-analysis-%j.err" \
    --export="ALL,AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}" \
    "${AF_ANALYSIS_JOB}"
)"
readonly analysis_job_id="${analysis_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg calibration_job_id "${calibration_job_id}" \
  --arg analysis_job_id "${analysis_job_id}" \
  --arg config_sha256 "$(sha256sum "${AF_CALIBRATION_CONFIG}" | awk '{print $1}')" \
  '{
    schema_version: "phase-b-proposer-transport-calibration-submission-1",
    submitted_at_utc: $submitted_at_utc,
    calibration_job_id: $calibration_job_id,
    analysis_job_id: $analysis_job_id,
    config_sha256: $config_sha256,
    test_data_opened: false,
    scientific_judge_called: false,
    parameter_fitting_performed: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "PROPOSER_CALIBRATION_JOB=${calibration_job_id}"
echo "PROPOSER_CALIBRATION_ANALYSIS_JOB=${analysis_job_id}"
echo "SUBMISSION_MANIFEST=${submission_manifest}"
