#!/bin/bash
# Stable code identity for this chain: its launchers, Python entry points, the
# rebuttal modules they call directly, and the frozen plan. It is deliberately
# not a transitive closure of every imported module, and not a repository-wide
# scan; HEAD equality alone would miss staged, untracked, or post-submission
# edits, while unrelated local artifacts must not block an evaluation checkout.
set -euo pipefail
: "${AF_REPO_ROOT:?AF_REPO_ROOT is required}"
cd "${AF_REPO_ROOT}"
readonly files=(
  "src/autoformalism/rebuttal/external_baseline_freeze.py"
  "src/autoformalism/rebuttal/final_evaluation.py"
  "src/autoformalism/rebuttal/final_evaluation_adapters.py"
  "src/autoformalism/rebuttal/postfreeze_evaluation.py"
  "src/autoformalism/rebuttal/phase_b_hidden_subspace.py"
  "scripts/prepare_external_baseline_frozen_test_evaluation.py"
  "scripts/summarize_external_baseline_evaluation.py"
  "scripts/export_phase_b_frozen_subjects.py"
  "scripts/evaluate_phase_b_postfreeze.py"
  "scripts/merge_phase_b_postfreeze.py"
  "scripts/evaluate_phase_b_hidden_subspace.py"
  "scripts/merge_phase_b_hidden_subspace.py"
  "scripts/assemble_phase_b_final_evaluation.py"
  "${AF_EVAL_PLAN:-configs/external_baseline_frozen_test_evaluation_v2.json}"
  "scripts/hpc/external_baseline_eval_prepare.slurm"
  "scripts/hpc/external_baseline_eval_postfreeze.slurm"
  "scripts/hpc/external_baseline_eval_merge.slurm"
  "scripts/hpc/external_baseline_eval_hidden.slurm"
  "scripts/hpc/external_baseline_eval_finalize.slurm"
  "scripts/hpc/submit_external_baseline_evaluation_delta.sh"
)
for file in "${files[@]}"; do
  [[ -f "${file}" ]] || { echo "chain input missing: ${file}" >&2; exit 2; }
done
# Hash contents only. sha256sum prints the path it was given, so an
# absolute plan path would otherwise produce a different identity from the
# same file named relatively.
sha256sum "${files[@]}" | awk '{print $1}' | sha256sum | awk '{print $1}'
