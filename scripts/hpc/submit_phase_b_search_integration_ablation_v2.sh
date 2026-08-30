#!/bin/bash
# Submit the prompt-v3, deterministic-target judge integration ablation.

set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine user}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_TARGET_CONTRACT_ROOT:=${AF_REPO_ROOT}/configs/target_eval/phase_b_v1}"
: "${AF_PROMPT_OVERLAY_CONFIG:=${AF_REPO_ROOT}/configs/phase_b_public_prompt_overlay_v3.json}"
: "${AF_HIDDEN_AUDIT:=${AF_WORK}/phase_b/hidden-contract-audit-v2/hidden_contract_audit.json}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/search-integration-ablation-v3}"
: "${AF_VLLM_PORT:=8000}"

readonly config="${AF_REPO_ROOT}/configs/phase_b_search_integration_ablation_v2.json"
readonly search_job="${AF_REPO_ROOT}/scripts/hpc/phase_b_search_integration_ablation_v2_120b.slurm"
readonly evaluation_job="${AF_REPO_ROOT}/scripts/hpc/phase_b_search_ablation_evaluation_v2.slurm"
readonly submission_manifest="${AF_OUTPUT_ROOT}/submission_manifest.json"

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing prompt overlay: ${AF_PUBLIC_DATA_ROOT}" >&2; exit 2; }
[[ -d "${AF_TARGET_CONTRACT_ROOT}" ]] || { echo "missing target contracts: ${AF_TARGET_CONTRACT_ROOT}" >&2; exit 2; }
[[ -f "${config}" ]] || { echo "missing config: ${config}" >&2; exit 2; }
[[ -f "${search_job}" ]] || { echo "missing search job: ${search_job}" >&2; exit 2; }
[[ -f "${evaluation_job}" ]] || { echo "missing evaluation job: ${evaluation_job}" >&2; exit 2; }
[[ -f "${AF_HIDDEN_AUDIT}" ]] || { echo "missing hidden audit: ${AF_HIDDEN_AUDIT}" >&2; exit 2; }
[[ -f "${AF_HIDDEN_AUDIT}.sha256" ]] || { echo "missing hidden audit digest" >&2; exit 2; }
[[ ! -e "${submission_manifest}" ]] || {
  echo "submission manifest already exists: ${submission_manifest}" >&2
  exit 2
}

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"

readonly expected_hidden_audit_sha256="$(jq -r '.hidden_contract_audit.sha256' "${config}")"
readonly actual_hidden_audit_sha256="$(sha256sum "${AF_HIDDEN_AUDIT}" | awk '{print $1}')"
[[ "${actual_hidden_audit_sha256}" == "${expected_hidden_audit_sha256}" ]] || {
  echo "hidden audit differs from the frozen plan" >&2
  exit 2
}
(
  cd "$(dirname "${AF_HIDDEN_AUDIT}")"
  sha256sum -c "$(basename "${AF_HIDDEN_AUDIT}").sha256"
)

"${AF_PYTHON}" scripts/prepare_phase_b_search_integration_ablation.py \
  --config "${config}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --target-contract-root "${AF_TARGET_CONTRACT_ROOT}" \
  --prompt-overlay-config "${AF_PROMPT_OVERLAY_CONFIG}"

readonly common_export="ALL,AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_TARGET_CONTRACT_ROOT=${AF_TARGET_CONTRACT_ROOT},AF_VLLM_PORT=${AF_VLLM_PORT}"
judge_submission="$(
  sbatch --parsable \
    --array=0-5%2 \
    --output="${AF_REPO_ROOT}/logs/phase-b-search-v3-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-search-v3-%A_%a.err" \
    --export="${common_export}" \
    "${search_job}"
)"
readonly judge_job_id="${judge_submission%%;*}"

no_judge_submission="$(
  sbatch --parsable \
    --dependency="afterany:${judge_job_id}" \
    --array=6-11%2 \
    --output="${AF_REPO_ROOT}/logs/phase-b-search-v3-%A_%a.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-search-v3-%A_%a.err" \
    --export="${common_export}" \
    "${search_job}"
)"
readonly no_judge_job_id="${no_judge_submission%%;*}"

evaluation_submission="$(
  sbatch --parsable \
    --dependency="afterany:${no_judge_job_id}" \
    --output="${AF_REPO_ROOT}/logs/phase-b-search-v3-eval-%j.out" \
    --error="${AF_REPO_ROOT}/logs/phase-b-search-v3-eval-%j.err" \
    --export="ALL,AF_SEARCH_ROOT=${AF_OUTPUT_ROOT},AF_PUBLIC_DATA_ROOT=${AF_PUBLIC_DATA_ROOT},AF_HIDDEN_AUDIT=${AF_HIDDEN_AUDIT}" \
    "${evaluation_job}"
)"
readonly evaluation_job_id="${evaluation_submission%%;*}"

temporary_manifest="${submission_manifest}.tmp"
jq -n \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg judge_search_job_id "${judge_job_id}" \
  --arg no_judge_search_job_id "${no_judge_job_id}" \
  --arg evaluation_job_id "${evaluation_job_id}" \
  --arg public_data_root "${AF_PUBLIC_DATA_ROOT}" \
  --arg target_contract_root "${AF_TARGET_CONTRACT_ROOT}" \
  --arg vllm_endpoint "http://127.0.0.1:${AF_VLLM_PORT}" \
  --arg hidden_audit "${AF_HIDDEN_AUDIT}" \
  --arg hidden_audit_sha256 "${actual_hidden_audit_sha256}" \
  '{
    schema_version: "phase-b-search-integration-submission-2",
    submitted_at_utc: $submitted_at_utc,
    judge_search_job_id: $judge_search_job_id,
    no_judge_search_job_id: $no_judge_search_job_id,
    evaluation_job_id: $evaluation_job_id,
    public_data_root: $public_data_root,
    target_contract_root: $target_contract_root,
    vllm_endpoint: $vllm_endpoint,
    initial_no_judge_cache_policy: "required_cache_hit_no_provider_fallback",
    hidden_audit: $hidden_audit,
    hidden_audit_sha256: $hidden_audit_sha256,
    queue_time_policy: "derived separately from Slurm Submit and Start timestamps"
  }' >"${temporary_manifest}"
mv "${temporary_manifest}" "${submission_manifest}"

echo "JUDGE_SEARCH_JOB=${judge_job_id}"
echo "NO_JUDGE_SEARCH_JOB=${no_judge_job_id}"
echo "ABLATION_EVALUATION_JOB=${evaluation_job_id}"
echo "SUBMISSION_MANIFEST=${submission_manifest}"
