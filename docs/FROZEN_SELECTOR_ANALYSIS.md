# Frozen selector analysis summary

## Scope

The Phase A2.5/A6 analysis compares 37 prespecified selector settings across 87
judged run-level candidate pools. It uses validation NMSE, historical judge
scores, and post-pruning additive term counts only. It does not read test
metrics, hidden trajectories, or private mechanism references.

The full generated development report is written to
`artifacts/rebuttal/frozen_selectors_v1/frozen_selector_report.md`; CSV outputs
contain every run-level selection, the sensitivity surface, leave-one-benchmark-
out results, and judge-category diagnostics.

## Main development findings

- The historical judge has limited leverage because many run-level candidates
  share similar scores and the validation winner is often retained.
- An epsilon-constrained selector with a 20% candidate-level tolerance changed
  29 of 87 selections. Among changed choices, the median validation-NMSE ratio
  was 1.032 and the median judge gain was 0.040. Its global 90th-percentile
  validation ratio was 1.066.
- A normalized weighted selector with judge weight 0.5 and no explicit sparsity
  weight changed 19 selections. Among changed choices, the median validation
  ratio was 1.032 and median judge gain was 0.050.
- Adding a modest sparsity weight can slightly reduce complexity across all
  runs, but the median changed-selection term difference is zero. The present
  candidate pool therefore provides weak evidence for selecting a precise
  sparsity coefficient.
- Leave-one-benchmark-out application of the prespecified development rule
  selected the 20% epsilon policy in every fold. Held-out development medians
  stayed near the validation-only choice, but this consistency reflects the
  rule and historical judge scores—not proof of superior private structural
  recovery or test performance.
- `constraint_compliance` and `mechanism_state_adequacy` are strongly associated
  with the aggregate judge score. `data_causal_consistency` is high with little
  variance, consistent with substantial overlap with deterministic validation.

## Interpretation

No selector is declared scientifically superior at this stage. The frozen
development evidence identifies three candidates for post-freeze evaluation:

1. validation-only production selection;
2. normalized weighted selection in the moderate judge-weight region, with a
   small sparsity-weight sensitivity analysis;
3. epsilon-constrained selection with tolerances from 5% to 20%.

Final comparison must report observed test error, private structural validity,
hidden error where defined, term count, and refit failure separately. A lower
test error alone should be described as lower test error, not a universally
better model.

Changing final selection on the frozen pool does not change proposal history.
Using a new objective for beam membership or proposer feedback is a new
end-to-end algorithm and requires a separate prospective confirmation study.

## Post-freeze multidimensional confirmation

The development analysis froze three policies before joining any new test or
private-reference outcome:

- production validation-only selection;
- normalized weighted selection with judge weight 0.5 and sparsity weight 0.1;
- epsilon-constrained selection with a 20% candidate-level tolerance.

The changed-selection manifest has SHA-256
`9239e5675d4451560382b4c1174c8004cb9f87fd13e27aac209ff363b3a5ecf4`.
It contains 47 changed policy choices representing 39 distinct structures.

| Policy | Complete | Changed complete | Geometric test NMSE | Mean structural validity | Geometric hidden NMSE | Mean terms |
|---|---:|---:|---:|---:|---:|---:|
| Validation only | 87/87 | 0/0 | 0.001624 | 0.7179 | 0.5878 | 4.977 |
| Weighted (0.5, 0.1) | 81/87 | 12/18 | 0.002193 | 0.6873 | 0.6088 | 4.966 |
| Epsilon (20%) | 77/87 | 19/29 | 0.001886 | 0.6940 | 0.5936 | 5.046 |

Among successfully refitted changed selections, weighted selection had a test-
NMSE geometric ratio of 2.250 relative to production (95% bootstrap CI
1.165--6.457) and won 4/12 pairs. Epsilon selection had a ratio of 1.038
(0.943--1.172) and won 8/19. Each alternative Pareto-dominated production in
three complete run-level comparisons; production dominated each alternative in
six. Remaining comparisons were equal or traded endpoints.

The operational failures are part of the result: six weighted-policy runs and
ten epsilon-policy runs lost completion because the alternative selected
structure failed the frozen train-plus-validation refit. Failed structures
still received structural-validity scores, but no fitted hidden trajectory was
invented.

These results do **not** support replacing production validation-only selection
with either tested policy. The weighted policy is predictively worse and less
reliable; the epsilon policy is approximately neutral on successful-pair test
error but less reliable and does not improve structural validity, hidden error,
or complexity. Future judge work should improve the measurement instrument
before increasing its selection influence. The existing production experiments
remain the primary results.
