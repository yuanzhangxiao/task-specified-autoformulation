#!/bin/bash
# Freeze and submit the one-H100 function-stage sign-contract probe on ACES.

set -euo pipefail

: "${SCRATCH:?SCRATCH is unset}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/staged-sign-contract-v1-aces-h100x1}"
: "${AF_SIGN_CONFIG:=${AF_REPO_ROOT}/configs/staged_sign_contract_probe_v1.json}"
: "${AF_VLLM_IMAGE:=${PROJECT:-${SCRATCH}}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_HF_HOME:=${SCRATCH}/huggingface-cache}"
: "${AF_COMPUTE_CACHE_ROOT:=${SCRATCH}/autoformalism-runtime-cache/staged-sign-contract}"
: "${AF_IPC_TMP_ROOT:=${SCRATCH}/af-ipc}"
: "${AF_WORKER_JOB:=${AF_REPO_ROOT}/scripts/hpc/staged_sign_contract_aces.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
readonly submission_intent="${AF_OUTPUT_ROOT}/submission-intent"
module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
for path in "${AF_PYTHON}" "${AF_SIGN_CONFIG}" "${AF_VLLM_IMAGE}" "${AF_WORKER_JOB}"; do
  [[ -e "${path}" ]] || { echo "missing sign-contract input: ${path}" >&2; exit 2; }
done
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}
[[ ! -e "${submission_intent}" ]] || {
  echo "submission intent already exists; inspect the scheduler before retrying: ${submission_intent}" >&2
  exit 2
}
[[ -z "$(git -C "${AF_REPO_ROOT}" status --porcelain)" ]] || {
  echo "sign-contract checkout must be clean before freezing" >&2
  exit 2
}
mkdir -p "${AF_OUTPUT_ROOT}/logs" "${AF_HF_HOME}" "${AF_COMPUTE_CACHE_ROOT}" "${AF_IPC_TMP_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/staged_sign_contract_campaign.py freeze --config "${AF_SIGN_CONFIG}" --output "${AF_OUTPUT_ROOT}/plan.json"
mkdir "${submission_intent}"
jq -n --arg created_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg commit "$(git rev-parse HEAD)" --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/plan.json" | awk '{print $1}')" '{schema_version:"staged-sign-contract-submission-intent-1",status:"sbatch_pending_or_uncertain",created_at_utc:$created_at_utc,commit:$commit,plan_sha256:$plan_sha256}' >"${submission_intent}/intent.json"
readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_VLLM_IMAGE=${AF_VLLM_IMAGE},AF_HF_HOME=${AF_HF_HOME},AF_COMPUTE_CACHE_ROOT=${AF_COMPUTE_CACHE_ROOT},AF_IPC_TMP_ROOT=${AF_IPC_TMP_ROOT},AF_TENSOR_PARALLEL_SIZE=1"
submission="$(sbatch --parsable --account="${AF_ACES_ACCOUNT}" --output="${AF_OUTPUT_ROOT}/logs/staged-sign-contract-%j.out" --error="${AF_OUTPUT_ROOT}/logs/staged-sign-contract-%j.err" --export="${common_export}" "${AF_WORKER_JOB}")"
readonly job_id="${submission%%;*}"
printf '%s\n' "${job_id}" >"${submission_intent}/job_id.txt"
temporary="${submission_manifest}.tmp"
jq -n --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg job_id "${job_id}" --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/plan.json" | awk '{print $1}')" --arg commit "$(git rev-parse HEAD)" '{schema_version:"staged-sign-contract-submission-1",submitted_at_utc:$submitted_at_utc,platform:"aces-h100x1",job_id:$job_id,plan_sha256:$plan_sha256,commit:$commit,parameter_fitting_performed:false,scientific_judge_called:false,test_data_opened:false,private_reference_opened:false}' >"${temporary}"
mv "${temporary}" "${submission_manifest}"
echo "ACES_STAGED_SIGN_CONTRACT_JOB=${job_id}"
echo "ACES_STAGED_SIGN_CONTRACT_ROOT=${AF_OUTPUT_ROOT}"
