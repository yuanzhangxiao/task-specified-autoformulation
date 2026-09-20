#!/bin/bash
#SBATCH --job-name=ext-baseline-prep
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
readonly af_user="${SLURM_JOB_USER:-${USER:?}}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_SYMBOLIC_FREEZE:=${AF_WORK}/phase_b/public-baselines-full-v1}"
# The completed Phase-B D3 models live at the pilot root; the full
# campaign writes to d3-native-vllm-120b-v1 and is still running.
: "${AF_D3_CAMPAIGN_ROOT:=${AF_WORK}/phase_b/d3-native-pilot-v1}"
: "${AF_HIDDEN_AUDIT:=${AF_WORK}/phase_b/hidden-contract-audit-v2/hidden_contract_audit.json}"
: "${AF_SOL_FREEZE_MANIFEST:=${AF_WORK}/phase_b/raw-agent-deterministic-evaluation-v1/frozen/raw_agent_freeze_manifest.json}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/external-baseline-evaluation-v1}"
: "${AF_EVALUATOR_CODE_COMMIT:?AF_EVALUATOR_CODE_COMMIT is required}"
: "${AF_CHAIN_INPUTS_DIGEST:?AF_CHAIN_INPUTS_DIGEST is required}"

readonly plan="${AF_REPO_ROOT}/configs/external_baseline_frozen_test_evaluation_v2.json"
cd "${AF_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${AF_EVALUATOR_CODE_COMMIT}" ]] || {
  echo "evaluator checkout changed after submission" >&2
  exit 2
}
export PYTHONPATH="${AF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
readonly actual_digest="$(bash scripts/hpc/external_baseline_inputs_digest.sh)"
[[ "${actual_digest}" == "${AF_CHAIN_INPUTS_DIGEST}" ]] || {
  echo "chain inputs changed after submission" >&2
  exit 2
}
[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
mkdir -p logs "${AF_OUTPUT_ROOT}"

# --authorize-execution refuses a draft plan, so a proposal cannot open test data.
"${AF_PYTHON}" scripts/prepare_external_baseline_frozen_test_evaluation.py \
  --config "${plan}" \
  --symbolic-development-freeze "${AF_SYMBOLIC_FREEZE}" \
  --d3-campaign-root "${AF_D3_CAMPAIGN_ROOT}" \
  --hidden-audit "${AF_HIDDEN_AUDIT}" \
  --raw-agent-freeze-manifest "${AF_SOL_FREEZE_MANIFEST}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --authorize-execution

# Re-hash every recorded source immediately before adaptation.
"${AF_PYTHON}" scripts/prepare_external_baseline_frozen_test_evaluation.py \
  --config "${plan}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --verify

"${AF_PYTHON}" scripts/export_phase_b_frozen_subjects.py \
  --requests "${AF_OUTPUT_ROOT}/frozen/source_adapter_requests.jsonl" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --output-root "${AF_OUTPUT_ROOT}/adapted"

"${AF_PYTHON}" scripts/prepare_external_baseline_frozen_test_evaluation.py \
  --config "${plan}" \
  --output-root "${AF_OUTPUT_ROOT}/frozen" \
  --seal-execution-record \
  --adapted-root "${AF_OUTPUT_ROOT}/adapted" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --evaluator-commit "${AF_EVALUATOR_CODE_COMMIT}" \
  --chain-inputs-digest "${AF_CHAIN_INPUTS_DIGEST}" \
  --maximum-trajectory-wall-time-seconds "${AF_MAX_TRAJECTORY_SECONDS:-300}"

jq -e '.execution_authorized == true and .test_data_opened == false' \
  "${AF_OUTPUT_ROOT}/frozen/external_baseline_freeze.json" >/dev/null
cat "${AF_OUTPUT_ROOT}/adapted/source_adapter_manifest.json"
