#!/bin/bash
#SBATCH --job-name=phase-b-final-eval-pilot
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/phase-b-final-eval-pilot-%j.out
#SBATCH --error=logs/phase-b-final-eval-pilot-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_PROJECT}/phase_b/inputs/public}"
: "${AF_PRIVATE_DATA_ROOT:=${AF_REPO_ROOT}/data_raw}"
: "${AF_AUTOFORMALISM_ROOT:=${AF_WORK}/phase_b/final-evaluation-pilot-v1/autoformalism-search}"
: "${AF_RAW_AGENT_ROOT:=${AF_WORK}/phase_b/raw-data-agent-fitted-v1}"
: "${AF_HIDDEN_AUDIT:=${AF_WORK}/phase_b/hidden-contract-audit-v2/hidden_contract_audit.json}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/final-evaluation-pilot-v1/evaluation}"
: "${AF_MAX_TRAJECTORY_SECONDS:=300}"

readonly plan="${AF_REPO_ROOT}/configs/phase_b_final_evaluation_pilot_v1.json"
readonly mechanism_root="${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1/specs"
readonly freeze_root="${AF_OUTPUT_ROOT}/frozen"
readonly adapted_root="${AF_OUTPUT_ROOT}/adapted"
readonly postfreeze_root="${AF_OUTPUT_ROOT}/postfreeze"
readonly hidden_root="${AF_OUTPUT_ROOT}/hidden"
readonly final_root="${AF_OUTPUT_ROOT}/final"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

[[ -x "${AF_PYTHON}" ]] || { echo "missing Python: ${AF_PYTHON}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public data: ${AF_PUBLIC_DATA_ROOT}" >&2; exit 2; }
[[ -d "${AF_PRIVATE_DATA_ROOT}" ]] || { echo "missing private data: ${AF_PRIVATE_DATA_ROOT}" >&2; exit 2; }
[[ -d "${mechanism_root}" ]] || { echo "missing mechanism specs: ${mechanism_root}" >&2; exit 2; }

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}"
cd "${AF_REPO_ROOT}"

"${AF_PYTHON}" scripts/prepare_phase_b_final_evaluation_pilot.py \
  --config "${plan}" \
  --autoformalism-root "${AF_AUTOFORMALISM_ROOT}" \
  --raw-agent-root "${AF_RAW_AGENT_ROOT}" \
  --hidden-audit "${AF_HIDDEN_AUDIT}" \
  --output-root "${freeze_root}"

"${AF_PYTHON}" scripts/export_phase_b_frozen_subjects.py \
  --requests "${freeze_root}/source_adapter_requests.jsonl" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --output-root "${adapted_root}"

"${AF_PYTHON}" scripts/evaluate_phase_b_postfreeze.py \
  --subjects "${adapted_root}/frozen_evaluation_subjects.jsonl" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --output-root "${postfreeze_root}" \
  --maximum-trajectory-wall-time-seconds "${AF_MAX_TRAJECTORY_SECONDS}"

"${AF_PYTHON}" scripts/evaluate_phase_b_hidden_subspace.py \
  --subjects "${postfreeze_root}/postfreeze_subjects.jsonl" \
  --public-data-root "${AF_PUBLIC_DATA_ROOT}" \
  --private-data-root "${AF_PRIVATE_DATA_ROOT}" \
  --mechanism-config-root "${mechanism_root}" \
  --output-root "${hidden_root}" \
  --maximum-trajectory-wall-time-seconds "${AF_MAX_TRAJECTORY_SECONDS}"

"${AF_PYTHON}" scripts/assemble_phase_b_final_evaluation.py \
  --subjects "${hidden_root}/hidden_subjects.jsonl" \
  --mechanism-config-root "${mechanism_root}" \
  --source-outcomes "${adapted_root}/source_adapter_outcomes.jsonl" \
  --output-root "${final_root}"

"${AF_PYTHON}" scripts/summarize_phase_b_final_evaluation_pilot.py \
  --freeze-manifest "${freeze_root}/pilot_freeze_manifest.json" \
  --source-outcomes "${adapted_root}/source_adapter_outcomes.jsonl" \
  --records "${final_root}/final_evaluation_records.jsonl" \
  --output-root "${final_root}"

cat "${final_root}/pilot_endpoint_report.md"
