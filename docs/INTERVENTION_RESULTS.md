# Phase A3 frozen-intervention results

## Experiment

The version-1 private suite evaluates hard-tier frozen models on four shifts per
benchmark: input extrapolation with multiple events; shifted initial conditions
and a longer horizon; simulator-parameter changes with sparse observations; and
dense observations with deterministic measurement noise. Structures and fitted
parameters were frozen before private reference generation and were never
refitted during evaluation.

The cohort contains five seeds each for Full, D3, and LLM-feature-SINDy; five
planned seeds for No-judge; and the prespecified three seeds for No-latent.
Three No-judge Benchmark 5 runs never completed their historical final refit, so
that cell has two evaluable seeds. They remain unresolved in the hashed cohort
manifest rather than being treated as successful models.

Generated results are in
`artifacts/rebuttal/interventions/phase_a3_batch_v1/`:

- `cohort_manifest.json`: authoritative paths and SHA-256 hashes;
- `evaluations.json` and `evaluations.csv`: per-model/per-case outcomes;
- `summary.csv`: seed aggregates;
- `summary.md`: the complete failure-aware Markdown table.

## Main findings

- All 172 evaluations for the 43 available frozen models simulated
  successfully. Completion here means intervention-time numerical success, not
  historical run availability.
- On Benchmark 5 input extrapolation, Full has the lowest mean target NMSE
  (`7.43e-4`), followed by D3 (`9.86e-4`).
- Benchmark 5 parameter changes are substantially harder. D3 has the lowest
  mean target NMSE (`0.111`); Full is more variable (`0.271 ± 0.271`). This
  suggests sensitivity of some selected latent structures or parameters to
  kinetic shifts and motivates the Phase A5 fitting/parameter-robustness study.
- On Benchmark 6, LLM-feature-SINDy has the lowest mean target NMSE in three of
  four cases. Differences among methods are small for input extrapolation and
  shifted-initial/long-horizon cases, while parameter shifts separate them more
  clearly.
- Five-percent measurement noise dominates all methods similarly because the
  causal one-step protocol supplies noisy lagged target history. This case
  measures robustness of the complete prediction protocol, not only equation
  quality.
- No single method dominates every intervention endpoint. These results should
  be reported as a vector by shift type rather than collapsed into one headline
  score.

## Interpretation cautions

The degradation ratio divides intervention NMSE by each model's historical
in-distribution test NMSE. Ratios below one are possible because some frozen
interventions are easier than the original held-out protocol; the ratio is not
a matched-pair causal effect. Raw MSE and train-scale NMSE are the primary
cross-method endpoints.

D3 is evaluated using its native teacher-forced one-slot update without a
sampling-interval multiplier. Other equation models use their original
continuous-time causal rollout semantics. This preserves the operational
definition of each method rather than retrospectively redefining D3 as a
continuous solver.

The current table measures observed-target generalization and simulation
failure. Hidden-state alignment and qualitative direction/timing correctness
were subsequently added, together with an increasing-noise sweep and faithful
Dalla Man interventions. The complete Phase A3 synthesis and improvement plan
are in `docs/PHASE_A3_INTERVENTION_CONCLUSION.md`; this file remains the record
of the initial B5/B6 target-only milestone.
