#!/bin/bash
#SBATCH --job-name=phase-b-raw-eval-final
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/phase-b-raw-eval-final-%j.out
#SBATCH --error=logs/phase-b-raw-eval-final-%j.err

set -euo pipefail
readonly af_user="${SLURM_JOB_USER:-${USER:?}}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/raw-agent-deterministic-evaluation-v1}"
readonly mechanism_root="${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1/specs"
mkdir -p "${AF_REPO_ROOT}/logs"
cd "${AF_REPO_ROOT}"
"${AF_PYTHON}" scripts/merge_phase_b_hidden_subspace.py \
  --subjects "${AF_OUTPUT_ROOT}/postfreeze/postfreeze_subjects.jsonl" \
  --input-root "${AF_OUTPUT_ROOT}/hidden" \
  --output-root "${AF_OUTPUT_ROOT}/hidden"
"${AF_PYTHON}" scripts/assemble_phase_b_final_evaluation.py \
  --subjects "${AF_OUTPUT_ROOT}/hidden/hidden_subjects.jsonl" \
  --mechanism-config-root "${mechanism_root}" \
  --source-outcomes "${AF_OUTPUT_ROOT}/adapted/source_adapter_outcomes.jsonl" \
  --output-root "${AF_OUTPUT_ROOT}/final"
"${AF_PYTHON}" scripts/summarize_phase_b_raw_agent_deterministic_evaluation.py \
  --freeze-manifest "${AF_OUTPUT_ROOT}/frozen/raw_agent_freeze_manifest.json" \
  --source-outcomes "${AF_OUTPUT_ROOT}/adapted/source_adapter_outcomes.jsonl" \
  --records "${AF_OUTPUT_ROOT}/final/final_evaluation_records.jsonl" \
  --resource-ledger "${AF_OUTPUT_ROOT}/frozen/source_resource_ledger.jsonl" \
  --output-root "${AF_OUTPUT_ROOT}/final"
cat "${AF_OUTPUT_ROOT}/final/raw_agent_endpoint_report.md"
