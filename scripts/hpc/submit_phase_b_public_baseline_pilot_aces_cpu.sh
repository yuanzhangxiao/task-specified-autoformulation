#!/bin/bash
# Freeze and submit public-only SINDy/PySR pilot tasks on ACES CPUs.

set -euo pipefail

: "${PROJECT:?PROJECT is unset}"
: "${SCRATCH:?SCRATCH is unset}"
: "${AF_ACES_ACCOUNT:=156264627414}"
: "${AF_REPO_ROOT:=$(pwd)}"
: "${AF_PYTHON:=${AF_REPO_ROOT}/.venv/bin/python}"
: "${AF_GCCCORE_MODULE:=GCCcore/13.2.0}"
: "${AF_PYTHON_MODULE:=Python/3.11.5}"
: "${AF_PUBLIC_DATA_ROOT:=${PROJECT}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_BASELINE_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_baseline_pilot_v1.json}"
: "${AF_PROMPT_OVERLAY_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_prompt_overlay_v3.json}"
: "${AF_PROPOSER_PLAN:=${AF_REPO_ROOT}/configs/phase_b_proposer_transport_calibration_v2.json}"
: "${AF_OUTPUT_ROOT:=${SCRATCH}/phase_b/public-baseline-pilot-v1}"
: "${AF_CPU_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_aces_cpu.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/cpu_submission_manifest.json"
module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data" >&2; exit 2; }
[[ -f "${AF_CPU_JOB}" ]] || { echo "missing CPU job" >&2; exit 2; }
[[ ! -e "${submission_manifest}" ]] || {
  echo "CPU submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_public_baseline_pilot.py \
  --config "${AF_BASELINE_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --prompt-overlay-config "${AF_PROMPT_OVERLAY_CONFIG}" \
  --proposer-transport-plan "${AF_PROPOSER_PLAN}"

readonly cpu_indices="$(
  jq -r 'select(.platform == "aces_cpu") | .task_index' \
    "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | paste -sd, -
)"
[[ -n "${cpu_indices}" ]] || { echo "baseline plan has no CPU tasks" >&2; exit 2; }
submission="$(
  sbatch --parsable \
    --account="${AF_ACES_ACCOUNT}" \
    --array="${cpu_indices}%4" \
    --output="${AF_REPO_ROOT}/logs/phase-b-baseline-cpu-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-baseline-cpu-%A_%a.err" \
    --export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE},AF_PYTHON_MODULE=${AF_PYTHON_MODULE},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}" \
    "${AF_CPU_JOB}"
)"
readonly job_id="${submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg job_id "${job_id}" \
  --arg task_indices "${cpu_indices}" \
  --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/plan.json" | awk '{print $1}')" \
  '{
    schema_version: "phase-b-public-baseline-submission-1",
    submitted_at_utc: $submitted_at_utc,
    platform: "aces_cpu",
    job_id: $job_id,
    task_indices: $task_indices,
    plan_sha256: $plan_sha256,
    test_data_opened: false,
    private_reference_opened: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "ACES_BASELINE_CPU_JOB=${job_id}"
echo "ACES_BASELINE_CPU_MANIFEST=${submission_manifest}"
