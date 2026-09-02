#!/bin/bash
#SBATCH --job-name=phase-b-classical-ready
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${AF_OUTPUT_ROOT:?AF_OUTPUT_ROOT is required}"
readonly final_freeze="${AF_OUTPUT_ROOT}/final-model-freeze/final_model_freeze.json"
readonly predictive="${AF_OUTPUT_ROOT}/predictive-summary/predictive_test_summary.json"
readonly common="${AF_OUTPUT_ROOT}/common-evaluation/final_evaluation_summary.json"
readonly adapter="${AF_OUTPUT_ROOT}/adapted/source_adapter_manifest.json"
readonly submission="${AF_OUTPUT_ROOT}/submission_manifest.json"
for path in "${final_freeze}" "${predictive}" "${common}" "${adapter}" "${submission}"; do
  [[ -f "${path}" ]] || { echo "missing evaluation artifact: ${path}" >&2; exit 2; }
done
jq -e '
  .status == "frozen_before_test_or_private_evaluation" and
  .model_count == 360 and
  .symbolic_subject_request_count == 240 and
  .test_data_opened == false and
  .private_reference_opened == false
' "${final_freeze}" >/dev/null
jq -e '
  .status == "complete" and .task_count == 360 and
  .test_data_opened_after_freeze == true and
  .private_reference_opened == false
' "${predictive}" >/dev/null
jq -e '
  .status == "complete" and .request_count == 240 and
  .adapted_count == 240 and .failed_count == 0 and
  .test_data_opened == false
' "${adapter}" >/dev/null
jq -e '.record_count == 240 and .source_completion_rate == 1' \
  "${common}" >/dev/null

readonly accounting="${AF_OUTPUT_ROOT}/slurm_accounting.psv"
readonly accounting_job_ids="$(jq -er '[
  .final_model_prepare_job_id,
  .predictive_test_job_id,
  .common_postfreeze_job_id,
  .predictive_summary_job_id,
  .common_finalize_job_id
] | join(",")' "${submission}")"
sacct -P -j "${accounting_job_ids}" \
  --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,NodeList \
  >"${accounting}"

readonly destination="${AF_OUTPUT_ROOT}/common_evaluation_readiness.json"
readonly temporary="${destination}.tmp"
jq -n \
  --arg created_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg final_model_freeze_sha256 "$(sha256sum "${final_freeze}" | awk '{print $1}')" \
  --arg predictive_summary_sha256 "$(sha256sum "${predictive}" | awk '{print $1}')" \
  --arg common_summary_sha256 "$(sha256sum "${common}" | awk '{print $1}')" \
  --arg source_adapter_manifest_sha256 "$(sha256sum "${adapter}" | awk '{print $1}')" \
  --arg slurm_accounting_sha256 "$(sha256sum "${accounting}" | awk '{print $1}')" \
  '{
    schema_version: "phase-b-public-baseline-common-evaluation-readiness-1",
    status: "ready_for_private_hidden_evaluation",
    created_at_utc: $created_at_utc,
    final_model_count: 360,
    common_symbolic_record_count: 240,
    predictive_reference_methods: ["persistence"],
    common_structural_methods: ["sindy", "pysr"],
    final_model_freeze_sha256: $final_model_freeze_sha256,
    predictive_summary_sha256: $predictive_summary_sha256,
    common_summary_sha256: $common_summary_sha256,
    source_adapter_manifest_sha256: $source_adapter_manifest_sha256,
    slurm_accounting_sha256: $slurm_accounting_sha256,
    test_data_opened_after_freeze: true,
    private_reference_opened: false,
    oracle_derivatives_used: false,
    oracle_latent_states_used: false,
    weighted_overall_score_defined: false
  }' >"${temporary}"
mv "${temporary}" "${destination}"
(cd "${AF_OUTPUT_ROOT}" && sha256sum common_evaluation_readiness.json > common_evaluation_readiness.json.sha256)
jq . "${destination}"
