#!/bin/bash
# Freeze and submit the one-seed GPT-OSS-20B construction gate on Delta.

set -euo pipefail

readonly af_user="${USER:?USER is unset}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v2}"
: "${AF_MECHANISM_SPEC_ROOT:=${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/incremental-construction-pilot-v2-20b-delta-a40x1-smoke1}"
: "${AF_CONSTRUCTION_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_incremental_construction_pilot_v2_20b.json}"
: "${AF_VLLM_IMAGE:=${AF_PROJECT}/containers/vllm-openai-v0.27.1.sif}"
: "${AF_HF_HOME:=${AF_PROJECT}/huggingface-cache}"
: "${AF_COMPUTE_CACHE_ROOT:=${AF_WORK}/autoformalism-runtime-cache/incremental-construction-20b}"
: "${AF_TENSOR_PARALLEL_SIZE:=1}"
: "${AF_ARRAY_SPEC:=0}"
: "${AF_GPU_ACCOUNT:=bibo-delta-gpu}"
: "${AF_CPU_ACCOUNT:=bibo-delta-cpu}"
: "${AF_WORKER_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_incremental_construction_delta.slurm}"
: "${AF_SUMMARY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_incremental_construction_summary.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
[[ "${AF_TENSOR_PARALLEL_SIZE}" == "1" ]] || { echo "the Delta 20B gate requires tensor parallelism 1" >&2; exit 2; }
for path in "${AF_PYTHON}" "${AF_CONSTRUCTION_CONFIG}" "${AF_VLLM_IMAGE}" "${AF_WORKER_JOB}" "${AF_SUMMARY_JOB}"; do
  [[ -e "${path}" ]] || { echo "missing incremental-construction input: ${path}" >&2; exit 2; }
done
for directory in "${AF_PUBLIC_DATA_ROOT}" "${AF_TARGET_CONTRACT_ROOT}" "${AF_MECHANISM_SPEC_ROOT}"; do
  [[ -d "${directory}" ]] || { echo "missing incremental-construction directory: ${directory}" >&2; exit 2; }
done
[[ ! -e "${submission_manifest}" ]] || { echo "submission manifest already exists: ${submission_manifest}" >&2; exit 2; }

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_HF_HOME}" "${AF_COMPUTE_CACHE_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_incremental_construction_pilot.py \
  --config "${AF_CONSTRUCTION_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --mechanism-spec-root "${AF_MECHANISM_SPEC_ROOT}"
readonly task_count_raw="$(wc -l <"${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl")"
readonly task_count="${task_count_raw//[[:space:]]/}"
[[ "${task_count}" == "6" ]] || { echo "incremental-construction plan must contain six tasks: ${task_count}" >&2; exit 2; }

readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_MECHANISM_SPEC_ROOT=${AF_MECHANISM_SPEC_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_VLLM_IMAGE=${AF_VLLM_IMAGE},AF_HF_HOME=${AF_HF_HOME},AF_COMPUTE_CACHE_ROOT=${AF_COMPUTE_CACHE_ROOT},AF_TENSOR_PARALLEL_SIZE=${AF_TENSOR_PARALLEL_SIZE},AF_SKIP_MODULE_LOAD=true"
worker_submission="$(sbatch --parsable --account="${AF_GPU_ACCOUNT}" --array="${AF_ARRAY_SPEC}" --output="${AF_REPO_ROOT}/logs/incremental-construction-20b-%A_%a.out" --error="${AF_REPO_ROOT}/logs/incremental-construction-20b-%A_%a.err" --export="${common_export}" "${AF_WORKER_JOB}")"
readonly worker_job="${worker_submission%%;*}"
summary_submission="$(sbatch --parsable --account="${AF_CPU_ACCOUNT}" --partition=cpu --dependency="afterany:${worker_job}" --output="${AF_REPO_ROOT}/logs/incremental-construction-20b-summary-%j.out" --error="${AF_REPO_ROOT}/logs/incremental-construction-20b-summary-%j.err" --export="${common_export}" "${AF_SUMMARY_JOB}")"
readonly summary_job="${summary_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg worker_job "${worker_job}" \
  --arg summary_job "${summary_job}" \
  --arg array_spec "${AF_ARRAY_SPEC}" \
  --arg compute_cache_root "${AF_COMPUTE_CACHE_ROOT}" \
  --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/plan.json" | awk '{print $1}')" \
  '{schema_version:"phase-b-incremental-construction-submission-1",submitted_at_utc:$submitted_at_utc,platform:"delta-a40x1",requested_gres:"gpu:a40:1",requested_time:"00:45:00",requested_cpus_per_task:8,requested_memory:"64G",compute_cache_root:$compute_cache_root,array_spec:$array_spec,worker_job:$worker_job,summary_job:$summary_job,plan_sha256:$plan_sha256,parameter_fitting_performed:false,scientific_judge_called:false,test_data_opened:false,private_reference_opened:false}' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "DELTA_INCREMENTAL_CONSTRUCTION_JOB=${worker_job}"
echo "DELTA_INCREMENTAL_CONSTRUCTION_SUMMARY_JOB=${summary_job}"
echo "DELTA_INCREMENTAL_CONSTRUCTION_ROOT=${AF_OUTPUT_ROOT}"
