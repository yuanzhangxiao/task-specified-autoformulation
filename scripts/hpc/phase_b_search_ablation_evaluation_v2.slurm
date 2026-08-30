#!/bin/bash
#SBATCH --job-name=phase-b-search-v2-eval
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/phase-b-search-v2-eval-%j.out
#SBATCH --error=logs/phase-b-search-v2-eval-%j.err

set -euo pipefail

readonly af_user="${SLURM_JOB_USER:-${USER:-}}"
: "${af_user:?cannot determine the Slurm user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/inputs/public-prompt-v3}"
: "${AF_PRIVATE_DATA_ROOT:=${AF_REPO_ROOT}/data_raw}"
: "${AF_SEARCH_ROOT:=${AF_WORK}/phase_b/search-integration-ablation-v3}"
: "${AF_HIDDEN_AUDIT:=${AF_WORK}/phase_b/hidden-contract-audit-v2/hidden_contract_audit.json}"
: "${AF_OUTPUT_ROOT:=${AF_SEARCH_ROOT}/evaluation}"
: "${AF_MAX_TRAJECTORY_SECONDS:=300}"

readonly plan="${AF_SEARCH_ROOT}/frozen/plan.json"
readonly submission_manifest="${AF_SEARCH_ROOT}/submission_manifest.json"
readonly mechanism_root="${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1/specs"
readonly freeze_root="${AF_OUTPUT_ROOT}/frozen"
readonly search_audit_root="${AF_OUTPUT_ROOT}/search-audit"
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
[[ -f "${plan}" ]] || { echo "missing frozen search plan: ${plan}" >&2; exit 2; }
[[ -f "${submission_manifest}" ]] || { echo "missing submission manifest" >&2; exit 2; }
[[ -f "${AF_HIDDEN_AUDIT}" ]] || { echo "missing hidden audit: ${AF_HIDDEN_AUDIT}" >&2; exit 2; }
[[ -d "${AF_PUBLIC_DATA_ROOT}" ]] || { echo "missing public overlay: ${AF_PUBLIC_DATA_ROOT}" >&2; exit 2; }
[[ -d "${AF_PRIVATE_DATA_ROOT}" ]] || { echo "missing private data: ${AF_PRIVATE_DATA_ROOT}" >&2; exit 2; }

mkdir -p "${AF_REPO_ROOT}/logs" "${AF_OUTPUT_ROOT}" "${search_audit_root}"
cd "${AF_REPO_ROOT}"

readonly judge_job="$(jq -r '.judge_search_job_id' "${submission_manifest}")"
readonly no_judge_job="$(jq -r '.no_judge_search_job_id' "${submission_manifest}")"
sacct -P -n -j "${judge_job},${no_judge_job}" \
  --format=JobIDRaw,Submit,Start,End,ElapsedRaw,State,ExitCode,AllocCPUS,AllocTRES,MaxRSS,NodeList \
  >"${search_audit_root}/slurm_accounting.psv" || true

"${AF_PYTHON}" scripts/summarize_phase_b_search_integration_ablation.py \
  --plan "${plan}" \
  --search-root "${AF_SEARCH_ROOT}" \
  --output-root "${search_audit_root}"

"${AF_PYTHON}" scripts/freeze_phase_b_search_ablation_sources.py \
  --plan "${plan}" \
  --search-root "${AF_SEARCH_ROOT}" \
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

cat "${final_root}/final_evaluation_summary.md"
cat "${search_audit_root}/search_integration_audit.md"
