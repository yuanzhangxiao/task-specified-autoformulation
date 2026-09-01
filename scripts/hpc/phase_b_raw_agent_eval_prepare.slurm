#!/bin/bash
#SBATCH --job-name=phase-b-raw-eval-prep
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/phase-b-raw-eval-prep-%j.out
#SBATCH --error=logs/phase-b-raw-eval-prep-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:?}}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_HISTORICAL_RAW_ROOT:=${AF_WORK}/phase_b/raw-data-agent-fitted-v1}"
: "${AF_REFRESH_RAW_ROOT:=${AF_WORK}/phase_b/raw-data-agent-fitted-prompt-v3-refresh-v1}"
: "${AF_HIDDEN_AUDIT:=${AF_WORK}/phase_b/hidden-contract-audit-v2/hidden_contract_audit.json}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/raw-agent-deterministic-evaluation-v1}"

readonly plan="${AF_REPO_ROOT}/configs/phase_b_raw_agent_deterministic_evaluation_v1.json"
readonly full_protocol="${AF_REPO_ROOT}/configs/raw_data_agent_fitted_model_full_v1.json"
readonly refresh_protocol="${AF_REPO_ROOT}/configs/raw_data_agent_fitted_model_prompt_v3_refresh_v1.json"
readonly freeze_root="${AF_OUTPUT_ROOT}/frozen"
readonly adapted_root="${AF_OUTPUT_ROOT}/adapted"

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing prompt-v3 public data" >&2; exit 2; }
mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"

"${AF_PYTHON}" scripts/prepare_phase_b_raw_agent_deterministic_evaluation.py \
  --config "${plan}" \
  --full-protocol-config "${full_protocol}" \
  --prompt-refresh-config "${refresh_protocol}" \
  --historical-root "${AF_HISTORICAL_RAW_ROOT}" \
  --refresh-root "${AF_REFRESH_RAW_ROOT}" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --hidden-audit "${AF_HIDDEN_AUDIT}" \
  --output-root "${freeze_root}"

"${AF_PYTHON}" scripts/export_phase_b_frozen_subjects.py \
  --requests "${freeze_root}/source_adapter_requests.jsonl" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --output-root "${adapted_root}"

cat "${adapted_root}/source_adapter_manifest.json"
