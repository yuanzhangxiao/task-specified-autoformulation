# Rebuttal experiment status

## Protocol decisions

- The objective study compares the requested ratio objective directly with the
  additive weighted-sum/MAP objective. It does not treat the current
  validation-first selector as an experimental arm.
- Structural stability is evaluated over the six registered benchmarks. No B4
  or T4 benchmark expansion is required.
- Private hidden trajectories may be used only by the explicit post-selection
  hidden-mechanism evaluator. They remain unavailable to proposal, judging,
  fitting, pruning, or selection.

## Implemented

- Development-only checkpoint indexing and candidate-pool export:
  `scripts/index_experiment_artifacts.py`.
- Ratio versus scaled weighted-sum ranking and frozen selection manifest:
  `scripts/analyze_ratio_vs_map.py`.
- Post-freeze refit and test evaluation:
  `scripts/evaluate_frozen_objective_selections.py`.
- Opt-in `--forbid-latent-states` proposer instruction and deterministic
  validation rule.
- Public graph-based mechanism coverage and structural-validity evaluation:
  `scripts/evaluate_mechanisms.py` and `configs/mechanism_eval/`.
- Evaluation-only affine hidden-mechanism metric with positive- or signed-scale
  policies: `scripts/evaluate_hidden_mechanisms.py`.
- Checkpoint learning-curve and fit/structure Pareto extraction:
  `scripts/analyze_learning_curves.py`.
- Alpha-normalized edge and term stability analysis:
  `scripts/analyze_structural_stability.py`.
- Explicit safe adversarial mutation recipes, repeated judge execution, and
  paired metrics: `scripts/run_adversarial_judge.py` and
  `scripts/analyze_adversarial_judge.py`.
- Deterministic construction and validation of 28 adversarial pairs across
  four hard-tier contexts: `scripts/build_adversarial_pairs.py`.
- Resumable, shardable adversarial scoring and lossless shard merging:
  `scripts/run_adversarial_judge.py` and
  `scripts/merge_adversarial_scores.py`.
- Compact baseline/ablation, LLM-family, objective, learning-curve, and
  structural-stability tables: `scripts/build_rebuttal_secondary_tables.py`.
- Frozen-structure derivative-fast-path versus generic-rollout fitting
  comparison: `scripts/run_fitting_fast_path_ablation.py`.
- Opt-in `--disable-derivative-fit-fast-path` fitting ablation.

## Pending experiment artifacts

- Score the 28 validated adversarial pairs with GPT and Gemini, three repeats
  per valid/adversarial candidate (336 calls total), then merge and analyze the
  four independent shards.
- Complete the one missing no-persistent-latent run
  (`benchmark6`, hard, seed 2) and consolidate all 12 runs.
- Finish the frozen-candidate fitting ablation and summarize accuracy/runtime.

## Expected output locations

- `artifacts/rebuttal/candidate_pool/`
- `artifacts/rebuttal/mechanisms/`
- `artifacts/rebuttal/objectives/`
- `artifacts/rebuttal/no_latent/`
- `artifacts/rebuttal/adversarial/`
- `artifacts/rebuttal/learning_curves/`
- `artifacts/rebuttal/structural_stability/`
- `artifacts/rebuttal/hidden_mechanisms/`
- `artifacts/rebuttal/fitting_ablation/`

## Current limitations

- Proposer mechanism tags were not originally constrained to canonical IDs;
  aliases therefore require a documented pool-level audit.
- Nonlinear multi-path regulatory signs are exported for blinded manual review
  instead of being guessed by the deterministic evaluator.
- Plot rendering is intentionally separated from metric extraction; the CSV
  outputs are the authoritative analysis artifacts.
- The deterministic structural predicates detect graph/mechanism failures but
  intentionally do not substitute for the independent LLM judge stress test.
- Live adversarial scoring requires API credentials and is therefore executed
  on the credentialed experiment machines rather than during local analysis.
