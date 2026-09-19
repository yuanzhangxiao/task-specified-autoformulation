#!/bin/bash
#SBATCH --job-name=ext-baseline-final
#SBATCH --account=bibo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
readonly af_user="${SLURM_JOB_USER:-${USER:?}}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/external-baseline-evaluation-v1}"
: "${AF_EVALUATOR_CODE_COMMIT:?AF_EVALUATOR_CODE_COMMIT is required}"
: "${AF_CHAIN_INPUTS_DIGEST:?AF_CHAIN_INPUTS_DIGEST is required}"
readonly mechanism_root="${AF_REPO_ROOT}/configs/mechanism_eval/phase_b_v1/specs"
readonly combined="${AF_OUTPUT_ROOT}/combined_source_outcomes.jsonl"

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
"${AF_PYTHON}" scripts/merge_phase_b_hidden_subspace.py \
  --subjects "${AF_OUTPUT_ROOT}/postfreeze/postfreeze_subjects.jsonl" \
  --input-root "${AF_OUTPUT_ROOT}/hidden" \
  --output-root "${AF_OUTPUT_ROOT}/hidden"

# One outcome per planned identity: adapted rows plus withheld missing and
# evaluator-unsupported rows. Assembly must see the whole roster.
cat "${AF_OUTPUT_ROOT}/adapted/source_adapter_outcomes.jsonl" \
    "${AF_OUTPUT_ROOT}/frozen/withheld_source_outcomes.jsonl" > "${combined}"
readonly planned="$(jq -r '.expected_source_count' \
  "${AF_OUTPUT_ROOT}/frozen/external_baseline_freeze.json")"
readonly actual="$(wc -l < "${combined}")"
[[ "${planned}" == "${actual}" ]] || {
  echo "combined outcomes ${actual} differ from planned ${planned}" >&2
  exit 2
}

"${AF_PYTHON}" scripts/assemble_phase_b_final_evaluation.py \
  --subjects "${AF_OUTPUT_ROOT}/hidden/hidden_subjects.jsonl" \
  --mechanism-config-root "${mechanism_root}" \
  --source-outcomes "${combined}" \
  --output-root "${AF_OUTPUT_ROOT}/final-evaluation"

# The shared summary counts every non-adapted row as a source failure; this
# report keeps missing, adaptation-failed and evaluator-unsupported distinct.
"${AF_PYTHON}" scripts/summarize_external_baseline_evaluation.py \
  --roster "${AF_OUTPUT_ROOT}/frozen/external_baseline_roster.jsonl" \
  --outcomes "${combined}" \
  --records "${AF_OUTPUT_ROOT}/final-evaluation/final_evaluation_records.jsonl" \
  --output-root "${AF_OUTPUT_ROOT}/final-evaluation"

echo "== external-baseline roster report (primary) =="
jq . "${AF_OUTPUT_ROOT}/final-evaluation/external_baseline_report.json"
echo "== shared assembler summary (legacy: pools methods and counts"
echo "   missing/unsupported rows as source failures) =="
cat "${AF_OUTPUT_ROOT}/final-evaluation/final_evaluation_summary.md"
