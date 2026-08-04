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
- Opt-in `--disable-derivative-fit-fast-path` fitting ablation.

## Pending experiment artifacts

- Consolidate distributed full, NoJudge, and family checkpoints into an input
  directory for candidate-pool indexing.
- Audit actual mechanism tags in the indexed pool and extend public tag aliases
  where proposers used equivalent noncanonical labels.
- Run the 12 no-persistent-latent hard-tier experiments: four representative
  benchmarks by seeds 0--2.
- Author and deterministically validate 28 adversarial pairs: seven mutations
  in each of four representative hard contexts.
- Export aligned candidate/reference hidden values from the sealed evaluation
  data. Raw private trajectories must not be copied into ordinary artifacts.
- Run fixed-candidate fitting ablations only for candidates confirmed eligible
  for derivative regression.

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
- Adversarial pair recipes require benchmark-specific component choices. The
  framework does not invent mutations automatically from prose.
