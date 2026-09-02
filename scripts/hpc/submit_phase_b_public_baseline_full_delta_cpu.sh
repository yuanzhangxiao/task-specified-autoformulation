#!/bin/bash
# Freeze and submit the complete public-only classical baseline matrix on Delta.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_BASELINE_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_baseline_full_delta_cpu_v1.json}"
: "${AF_PROMPT_OVERLAY_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_prompt_overlay_v3.json}"
: "${AF_PROPOSER_PLAN:=${AF_REPO_ROOT}/configs/phase_b_proposer_transport_calibration_v2.json}"
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
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data" >&2; exit 2; }
for script in "${AF_PREPARE_JOB}" "${AF_CPU_JOB}" "${AF_SUMMARY_JOB}" "${AF_READINESS_JOB}"; do
  [[ -f "${script}" ]] || { echo "missing job script: ${script}" >&2; exit 2; }
done
for value in "${AF_PERSISTENCE_CONCURRENCY}" "${AF_SINDY_CONCURRENCY}" "${AF_PYSR_CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid array concurrency: ${value}" >&2; exit 2; }
done
"${AF_PYTHON}" -c \
  'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pysr") else 2)' \
  || {
    echo "PySR is missing; install with: ${AF_PYTHON} -m pip install -e '${AF_REPO_ROOT}[pysr]'" >&2
    exit 2
  }
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  echo "use a new AF_OUTPUT_ROOT; completed runs are never silently overwritten" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${AF_JULIA_DEPOT}"
cd "${AF_REPO_ROOT}"
readonly source_code_commit="$(git rev-parse HEAD)"
[[ "${source_code_commit}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "cannot resolve the source commit" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "full baseline submission requires a clean tracked worktree" >&2
  exit 2
}
"${AF_PYTHON}" scripts/prepare_phase_b_public_baseline_pilot.py \
  --config "${AF_BASELINE_CONFIG}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --prompt-overlay-config "${AF_PROMPT_OVERLAY_CONFIG}" \
  --proposer-transport-plan "${AF_PROPOSER_PLAN}"

readonly persistence_indices="$(
  jq -r 'select(.platform == "delta_cpu" and .method == "persistence") | .task_index' \
    "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | paste -sd, -
)"
readonly sindy_indices="$(
  jq -r 'select(.platform == "delta_cpu" and .method == "sindy") | .task_index' \
    "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | paste -sd, -
)"
readonly pysr_indices="$(
  jq -r 'select(.platform == "delta_cpu" and .method == "pysr") | .task_index' \
    "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | paste -sd, -
)"
[[ -n "${persistence_indices}" && -n "${sindy_indices}" && -n "${pysr_indices}" ]] || {
  echo "full baseline plan lacks persistence, SINDy, or PySR tasks" >&2
  exit 2
}

readonly common_export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_JULIA_DEPOT=${AF_JULIA_DEPOT}"
prepare_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-pysr-prepare-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-pysr-prepare-%j.err" \
    --export="${common_export}" \
    "${AF_PREPARE_JOB}"
)"
readonly prepare_job_id="${prepare_submission%%;*}"
persistence_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --array="${persistence_indices}%${AF_PERSISTENCE_CONCURRENCY}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-persistence-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-persistence-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CPU_JOB}"
)"
readonly persistence_job_id="${persistence_submission%%;*}"
sindy_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --array="${sindy_indices}%${AF_SINDY_CONCURRENCY}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-sindy-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-sindy-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CPU_JOB}"
)"
readonly sindy_job_id="${sindy_submission%%;*}"
pysr_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --array="${pysr_indices}%${AF_PYSR_CONCURRENCY}" \
    --dependency="afterok:${prepare_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-pysr-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-pysr-%A_%a.err" \
    --export="${common_export}" \
    "${AF_CPU_JOB}"
)"
readonly pysr_job_id="${pysr_submission%%;*}"
summary_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterany:${persistence_job_id}:${sindy_job_id}:${pysr_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-summary-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-summary-%j.err" \
    --export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}" \
    "${AF_SUMMARY_JOB}"
)"
readonly summary_job_id="${summary_submission%%;*}"
readiness_submission="$(
  sbatch --parsable \
    --account=bibo-delta-cpu \
    --dependency="afterok:${summary_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-full-readiness-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-full-readiness-%j.err" \
    --export="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_SOURCE_CODE_COMMIT=${source_code_commit}" \
    "${AF_READINESS_JOB}"
)"
readonly readiness_job_id="${readiness_submission%%;*}"

temporary="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_code_commit "${source_code_commit}" \
  --arg prepare_job_id "${prepare_job_id}" \
  --arg persistence_job_id "${persistence_job_id}" \
  --arg sindy_job_id "${sindy_job_id}" \
  --arg pysr_job_id "${pysr_job_id}" \
  --arg summary_job_id "${summary_job_id}" \
  --arg readiness_job_id "${readiness_job_id}" \
  --arg persistence_task_indices "${persistence_indices}" \
  --arg sindy_task_indices "${sindy_indices}" \
  --arg pysr_task_indices "${pysr_indices}" \
  --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/plan.json" | awk '{print $1}')" \
  '{
    schema_version: "phase-b-public-baseline-full-submission-1",
    submitted_at_utc: $submitted_at_utc,
    source_code_commit: $source_code_commit,
    platform: "delta_cpu",
    pysr_prepare_job_id: $prepare_job_id,
    persistence_job_id: $persistence_job_id,
    sindy_job_id: $sindy_job_id,
    pysr_job_id: $pysr_job_id,
    summary_job_id: $summary_job_id,
    common_readiness_job_id: $readiness_job_id,
    persistence_task_indices: $persistence_task_indices,
    sindy_task_indices: $sindy_task_indices,
    pysr_task_indices: $pysr_task_indices,
    plan_sha256: $plan_sha256,
    test_data_opened: false,
    private_reference_opened: false
  }' >"${temporary}"
mv "${temporary}" "${submission_manifest}"

echo "DELTA_PYSR_PREPARE_JOB=${prepare_job_id}"
echo "DELTA_PERSISTENCE_JOB=${persistence_job_id}"
echo "DELTA_SINDY_JOB=${sindy_job_id}"
echo "DELTA_PYSR_JOB=${pysr_job_id}"
echo "DELTA_BASELINE_SUMMARY_JOB=${summary_job_id}"
echo "DELTA_COMMON_READINESS_JOB=${readiness_job_id}"
echo "DELTA_BASELINE_MANIFEST=${submission_manifest}"
