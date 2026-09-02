#!/bin/bash
# Freeze and submit the GPT-OSS-120B proposer calibration on ACES H100s.

set -euo pipefail

: "${PROJECT:?PROJECT is unset; log in through ACES and load the project environment}"
: "${SCRATCH:?SCRATCH is unset; log in through ACES and load the scratch environment}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_PUBLIC_DATA_ROOT:=${PROJECT}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/proposer-transport-calibration-v1-aces-h100x2}"
: "${AF_CALIBRATION_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_proposer_transport_calibration_v1.json}"
: "${AF_PREREQUISITE_ANALYSIS:=}"
: "${AF_VLLM_IMAGE:=${PROJECT}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_VLLM_IMAGE_URI:=docker://vllm/vllm-openai:v0.27.1}"
: "${AF_HF_HOME:=${SCRATCH}/huggingface-cache}"
: "${AF_IMAGE_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_transport_calibration_aces_image.slurm}"
: "${AF_CALIBRATION_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_transport_calibration_aces_h100.slurm}"
: "${AF_ANALYSIS_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_transport_calibration_analysis_aces.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data overlay: ${AF_PUBLIC_DATA_ROOT}" >&2; exit 2; }
[[ -d "${AF_TARGET_CONTRACT_ROOT}" ]] || { echo "missing target contracts: ${AF_TARGET_CONTRACT_ROOT}" >&2; exit 2; }
[[ -f "${AF_CALIBRATION_CONFIG}" ]] || { echo "missing calibration config" >&2; exit 2; }
if [[ -n "${AF_PREREQUISITE_ANALYSIS}" && ! -f "${AF_PREREQUISITE_ANALYSIS}" ]]; then
  echo "missing prerequisite analysis: ${AF_PREREQUISITE_ANALYSIS}" >&2
  exit 2
fi
for script in "${AF_IMAGE_JOB}" "${AF_CALIBRATION_JOB}" "${AF_ANALYSIS_JOB}"; do
  [[ -f "${script}" ]] || { echo "missing job script: ${script}" >&2; exit 2; }
done
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_HF_HOME}"
cd "${AF_REPO_ROOT}"
prepare_arguments=(
  --config "${AF_CALIBRATION_CONFIG}"
  --output-root "${AF_OUTPUT_ROOT}/frozen"
  --public-data-root "${AF_PUBLIC_DATA_ROOT}"
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}"
)
if [[ -n "${AF_PREREQUISITE_ANALYSIS}" ]]; then
  prepare_arguments+=(--prerequisite-analysis "${AF_PREREQUISITE_ANALYSIS}")
fi
"${AF_PYTHON}" scripts/prepare_phase_b_proposer_transport_calibration.py \
  "${prepare_arguments[@]}"
readonly task_count_raw="$(wc -l <"${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl")"
readonly task_count="${task_count_raw//[[:space:]]/}"
if [[ ! "${task_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid frozen task count: ${task_count}" >&2
  exit 2
fi
readonly task_max="$((task_count - 1))"

readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_CALIBRATION_CONFIG=${AF_CALIBRATION_CONFIG},AF_VLLM_IMAGE=${AF_VLLM_IMAGE},AF_VLLM_IMAGE_URI=${AF_VLLM_IMAGE_URI},AF_HF_HOME=${AF_HF_HOME}"
image_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --output="${AF_REPO_ROOT}/logs/proposer-cal-image-%j.out" \
    --error="${AF_REPO_ROOT}/logs/proposer-cal-image-%j.err" \
    --export="${common_export}" \
    "${AF_IMAGE_JOB}"
)"
readonly image_job_id="${image_submission%%;*}"
calibration_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --array="0-${task_max}%2" \
    --dependency="afterok:${image_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-proposer-cal-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-proposer-cal-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CALIBRATION_JOB}"
)"
readonly calibration_job_id="${calibration_submission%%;*}"
analysis_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --dependency="afterany:${calibration_job_id}" \
    --output="${AF_REPO_ROOT}/logs/proposer-cal-analysis-%j.out" \
    --error="${AF_REPO_ROOT}/logs/proposer-cal-analysis-%j.err" \
    --export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}" \
    "${AF_ANALYSIS_JOB}"
)"
readonly analysis_job_id="${analysis_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg image_job_id "${image_job_id}" \
  --arg calibration_job_id "${calibration_job_id}" \
  --arg analysis_job_id "${analysis_job_id}" \
  --arg config_sha256 "$(sha256sum "${AF_CALIBRATION_CONFIG}" | awk '{print $1}')" \
  --arg prerequisite_analysis_sha256 "$(jq -r '.prerequisite.analysis_sha256 // ""' "${AF_OUTPUT_ROOT}/frozen/freeze_manifest.json")" \
  --arg platform "aces-h100x2" \
  '{
    schema_version: "phase-b-proposer-transport-calibration-submission-1",
    submitted_at_utc: $submitted_at_utc,
    platform: $platform,
    image_job_id: $image_job_id,
    calibration_job_id: $calibration_job_id,
    analysis_job_id: $analysis_job_id,
    config_sha256: $config_sha256,
    prerequisite_analysis_sha256: (
      if $prerequisite_analysis_sha256 == ""
      then null
      else $prerequisite_analysis_sha256
      end
    ),
    test_data_opened: false,
    scientific_judge_called: false,
    parameter_fitting_performed: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "ACES_IMAGE_JOB=${image_job_id}"
echo "ACES_PROPOSER_CALIBRATION_JOB=${calibration_job_id}"
echo "ACES_PROPOSER_CALIBRATION_ANALYSIS_JOB=${analysis_job_id}"
echo "ACES_SUBMISSION_MANIFEST=${submission_manifest}"
