#!/bin/bash
# Freeze and submit compute-matched public-only D3 pilot tasks on ACES H100s.

set -euo pipefail

: "${PROJECT:?PROJECT is unset}"
: "${SCRATCH:?SCRATCH is unset}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${PROJECT}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_BASELINE_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_baseline_pilot_v1.json}"
: "${AF_PROMPT_OVERLAY_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_prompt_overlay_v3.json}"
: "${AF_PROPOSER_PLAN:=${AF_REPO_ROOT}/configs/phase_b_proposer_transport_calibration_v2.json}"
: "${AF_PROPOSER_ANALYSIS:?AF_PROPOSER_ANALYSIS must name the passing ACES proposer analysis}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/public-baseline-pilot-v1}"
: "${AF_VLLM_IMAGE:=${PROJECT}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_VLLM_IMAGE_URI:=docker://vllm/vllm-openai:v0.27.1}"
: "${AF_HF_HOME:=${SCRATCH}/huggingface-cache}"
: "${AF_IMAGE_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_proposer_transport_calibration_aces_image.slurm}"
: "${AF_D3_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_aces_d3.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/d3_submission_manifest.json"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -f "${AF_PROPOSER_ANALYSIS}" ]] || { echo "missing proposer analysis" >&2; exit 2; }
[[ -f "${AF_D3_JOB}" ]] || { echo "missing D3 job" >&2; exit 2; }
[[ ! -e "${submission_manifest}" ]] || {
  echo "D3 submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_HF_HOME}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_public_baseline_pilot.py \
  --config "${AF_BASELINE_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --prompt-overlay-config "${AF_PROMPT_OVERLAY_CONFIG}" \
  --proposer-transport-plan "${AF_PROPOSER_PLAN}"
"${AF_PYTHON}" scripts/freeze_phase_b_baseline_llm_operating_point.py \
  --baseline-plan "${AF_BASELINE_CONFIG}" \
  --proposer-plan "${AF_PROPOSER_PLAN}" \
  --proposer-analysis "${AF_PROPOSER_ANALYSIS}" \
  --output "${AF_OUTPUT_ROOT}/frozen/d3_llm_operating_point.json"

readonly d3_indices="$(
  jq -r 'select(.method == "d3_native_no_tools") | .task_index' \
    "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | paste -sd, -
)"
[[ -n "${d3_indices}" ]] || { echo "baseline plan has no D3 tasks" >&2; exit 2; }
readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_CALIBRATION_CONFIG=${AF_PROPOSER_PLAN},AF_VLLM_IMAGE=${AF_VLLM_IMAGE},AF_VLLM_IMAGE_URI=${AF_VLLM_IMAGE_URI},AF_HF_HOME=${AF_HF_HOME}"
image_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --output="${AF_REPO_ROOT}/logs/baseline-d3-image-%j.out" \
    --error="${AF_REPO_ROOT}/logs/baseline-d3-image-%j.err" \
    --export="${common_export}" \
    "${AF_IMAGE_JOB}"
)"
readonly image_job_id="${image_submission%%;*}"
d3_submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --array="${d3_indices}%2" \
    --dependency="afterok:${image_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-baseline-d3-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-baseline-d3-%A_%a.err" \
    --export="${common_export}" \
    "${AF_D3_JOB}"
)"
readonly d3_job_id="${d3_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg image_job_id "${image_job_id}" \
  --arg d3_job_id "${d3_job_id}" \
  --arg task_indices "${d3_indices}" \
  --arg operating_point_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/d3_llm_operating_point.json" | awk '{print $1}')" \
  '{
    schema_version: "phase-b-public-baseline-submission-1",
    submitted_at_utc: $submitted_at_utc,
    platform: "aces_h100x2",
    image_job_id: $image_job_id,
    job_id: $d3_job_id,
    task_indices: $task_indices,
    operating_point_sha256: $operating_point_sha256,
    test_data_opened: false,
    private_reference_opened: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "ACES_BASELINE_D3_IMAGE_JOB=${image_job_id}"
echo "ACES_BASELINE_D3_JOB=${d3_job_id}"
echo "ACES_BASELINE_D3_MANIFEST=${submission_manifest}"
