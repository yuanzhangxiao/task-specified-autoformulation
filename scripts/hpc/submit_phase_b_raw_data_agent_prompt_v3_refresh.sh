#!/bin/bash
# Freeze and submit only the 30 GPT-5.6 calls affected by public prompt v3.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/raw-data-agent-fitted-prompt-v3-refresh-v1}"

readonly config="${AF_REPO_ROOT}/configs/raw_data_agent_fitted_model_prompt_v3_refresh_v1.json"
readonly overlay_config="${AF_REPO_ROOT}/configs/phase_b_public_prompt_overlay_v3.json"
readonly source_full_config="${AF_REPO_ROOT}/configs/raw_data_agent_fitted_model_full_v1.json"
readonly job="${AF_REPO_ROOT}/scripts/hpc/phase_b_raw_data_agent_prompt_v3_refresh.slurm"
readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing prompt-v3 data root" >&2; exit 2; }
[[ -f "${config}" ]] || { echo "missing refresh config" >&2; exit 2; }
[[ -f "${overlay_config}" ]] || { echo "missing prompt-overlay config" >&2; exit 2; }
[[ -f "${source_full_config}" ]] || { echo "missing source full-agent config" >&2; exit 2; }
[[ -f "${job}" ]] || { echo "missing refresh Slurm job" >&2; exit 2; }
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -rsp "Paste OPENAI_API_KEY: " OPENAI_API_KEY
  echo
  export OPENAI_API_KEY
fi

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/prepare_raw_data_agent_prompt_v3_refresh.py \
  --config "${config}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --overlay-config "${overlay_config}" \
  --source-full-config "${source_full_config}"

submission="$(
  sbatch --parsable \
    --output="${AF_REPO_ROOT}/logs/phase-b-raw-v3-refresh-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-raw-v3-refresh-%A_%a.err" \
    --export="ALL,OPENAI_API_KEY,AF_REPO_ROOT=${AF_REPO_ROOT},AF_PYTHON=${AF_PYTHON},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT}" \
    "${job}"
)"
readonly job_id="${submission%%;*}"
readonly temporary_manifest="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg job_id "${job_id}" \
  --arg public_data_root "${AF_PUBLIC_DATA_ROOT}" \
  --arg output_root "${AF_OUTPUT_ROOT}" \
  --arg plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/plan.json" | awk '{print $1}')" \
  --arg task_plan_sha256 "$(sha256sum "${AF_OUTPUT_ROOT}/frozen/task_plan.jsonl" | awk '{print $1}')" \
  '{
    schema_version: "raw-data-agent-prompt-v3-refresh-submission-1",
    submitted_at_utc: $submitted_at_utc,
    job_id: $job_id,
    public_data_root: $public_data_root,
    output_root: $output_root,
    plan_sha256: $plan_sha256,
    task_plan_sha256: $task_plan_sha256,
    task_count: 30,
    test_data_opened: false
  }' >"${temporary_manifest}"
mv "${temporary_manifest}" "${submission_manifest}"

echo "GPT56_PROMPT_V3_REFRESH_JOB=${job_id}"
echo "SUBMISSION_MANIFEST=${submission_manifest}"
