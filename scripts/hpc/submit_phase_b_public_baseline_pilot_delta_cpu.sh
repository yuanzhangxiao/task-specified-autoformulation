#!/bin/bash
# Freeze and submit public-only SINDy/PySR pilot tasks on Delta CPUs.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_BASELINE_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_baseline_pilot_delta_cpu_v1.json}"
: "${AF_PROMPT_OVERLAY_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_prompt_overlay_v3.json}"
: "${AF_PROPOSER_PLAN:=${AF_REPO_ROOT}/configs/phase_b_proposer_transport_calibration_v2.json}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/public-baseline-pilot-delta-cpu-v1}"
: "${AF_JULIA_DEPOT:=${AF_WORK}/julia-depot-pysr-1.5.9}"
: "${AF_PREPARE_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_delta_pysr_prepare.slurm}"
: "${AF_CPU_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_delta_cpu.slurm}"
: "${AF_SUMMARY_JOB:=${AF_REPO_ROOT}/scripts/hpc/phase_b_public_baseline_pilot_delta_summary.slurm}"

readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data" >&2; exit 2; }
for script in "${AF_PREPARE_JOB}" "${AF_CPU_JOB}" "${AF_SUMMARY_JOB}"; do
  [[ -f "${script}" ]] || { echo "missing job script: ${script}" >&2; exit 2; }
done
"${AF_PYTHON}" -c \
  'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pysr") else 2)' \
  || {
    echo "PySR is missing; install with: ${AF_PYTHON} -m pip install -e '${AF_REPO_ROOT}[pysr]'" >&2
    exit 2
  }
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_JULIA_DEPOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_phase_b_public_baseline_pilot.py \
  --config "${AF_BASELINE_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --prompt-overlay-config "${AF_PROMPT_OVERLAY_CONFIG}" \
  --proposer-transport-plan "${AF_PROPOSER_PLAN}"

readonly sindy_indices="$(
  jq -r 'select(.platform == "delta_cpu" and .method == "sindy") | .task_index' \
    "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | paste -sd, -
)"
readonly pysr_indices="$(
  jq -r 'select(.platform == "delta_cpu" and .method == "pysr") | .task_index' \
    "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | paste -sd, -
)"
[[ -n "${sindy_indices}" && -n "${pysr_indices}" ]] || {
  echo "baseline plan lacks SINDy or PySR tasks" >&2
  exit 2
}
readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_JULIA_DEPOT=${AF_JULIA_DEPOT}"
prepare_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --output="${AF_REPO_ROOT}/logs/phase-b-pysr-prepare-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-pysr-prepare-%j.err" \
    --export="${common_export}" \
    "${AF_PREPARE_JOB}"
)"
readonly prepare_job_id="${prepare_submission%%;*}"
sindy_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --array="${sindy_indices}%3" \
    --output="${AF_REPO_ROOT}/logs/phase-b-baseline-sindy-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-baseline-sindy-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CPU_JOB}"
)"
readonly sindy_job_id="${sindy_submission%%;*}"
pysr_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --array="${pysr_indices}%3" \
    --dependency="afterok:${prepare_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-baseline-pysr-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-baseline-pysr-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CPU_JOB}"
)"
readonly pysr_job_id="${pysr_submission%%;*}"
summary_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterany:${sindy_job_id}:${pysr_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-baseline-summary-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-baseline-summary-%j.err" \
    --export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}" \
    "${AF_SUMMARY_JOB}"
)"
readonly summary_job_id="${summary_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg prepare_job_id "${prepare_job_id}" \
  --arg sindy_job_id "${sindy_job_id}" \
  --arg pysr_job_id "${pysr_job_id}" \
  --arg summary_job_id "${summary_job_id}" \
  --arg sindy_task_indices "${sindy_indices}" \
  --arg pysr_task_indices "${pysr_indices}" \
  --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/plan.json" | awk '{print $1}')" \
  '{
    schema_version: "phase-b-public-baseline-submission-1",
    submitted_at_utc: $submitted_at_utc,
    platform: "delta_cpu",
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

echo "DELTA_PYSR_PREPARE_JOB=${prepare_job_id}"
echo "DELTA_SINDY_JOB=${sindy_job_id}"
echo "DELTA_PYSR_JOB=${pysr_job_id}"
echo "DELTA_BASELINE_SUMMARY_JOB=${summary_job_id}"
echo "DELTA_BASELINE_MANIFEST=${submission_manifest}"
