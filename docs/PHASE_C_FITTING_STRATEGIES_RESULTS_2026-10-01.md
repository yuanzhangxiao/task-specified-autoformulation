# Phase C fitting milestone 3: Delta summary review

This review covers the submitted 72-row summary for
`phase-c-fitting-strategies-1`. See [the protocol](PHASE_C_FITTING_STRATEGIES.md)
for objectives, starts, accuracy thresholds and numerical budgets. No fitting,
model selection, test-data access or production-default change is performed here.

## Provenance and scope of verification

- Summary file SHA-256:
  `97c1ddcfdddc659798d193e5af4d228a9c5a5a3d07b56eaecaaed2811cae71b6`.
- Plan identity:
  `65a0a6db305e19b1d6a6d1ffcb0ac20f4574f46694368627d6e36b36bbe0c466`.
- The plan matches the earlier `review-20261001-212706.tar.gz` preparation
  archive exactly. Its plan/input seals and input-content binding were verified.
  All 72 task IDs and common-start hashes match that plan. Group counts, recovery
  counts and elapsed totals reproduce from the rows.
- This attachment contains the summary only. The individual backend/replay seals,
  mesh traces, process logs and qualification records for the completed run have
  not yet been inspected. Conclusions about individual solver failures remain
  provisional until that review.

All 72 tasks have terminal records: 71 complete independent replays and one
`no_complete_replay` (CSTR hard, start 2, multiple shooting). Campaign completion
means accounting is complete; it does not mean all accuracy tests passed.

## Matched results

Every strategy has three starts per case and the same 180-second total fitting
ceiling. Each column below reports passes, not a selected best start.

| Strategy | Four controls: output, parameter and latent recovery | CSTR easy: output accuracy | CSTR hard: output accuracy | Total output passes | Median fitting seconds on the 12 controls |
|---|---:|---:|---:|---:|---:|
| Rollout only | 12/12 | 3/3 | 3/3 | 18/18 | 18.1 |
| Collocation then rollout | 12/12 | 2/3 | 1/3 | 15/18 | 24.4 |
| Direct collocation | 12/12 | 0/3 | 0/3 | 12/18 | 45.2 |
| Adaptive multiple shooting | 11/12 | 0/3 | 0/3 | 11/18 | 69.2 |

Control recovery requires both split NMSEs <= `1e-6`, maximum parameter relative
error <= 1%, and both latent-trajectory NMSEs <= `1e-4`, with independent solver
agreement. CSTR output accuracy uses NMSE <= `0.01` on both splits and the same
solver-agreement check; it is not a parameter-recovery certificate.

The failed shooting control is fast/slow start 2: training NMSE `4.668e-5`,
validation NMSE `4.290e-5`, maximum parameter error 43.2%, and latent NMSE about
0.0137. Three mesh-stage timeouts are recorded. A visually small output error can
still accompany a substantially inaccurate latent/parameter estimate.

## CSTR output errors and fitted coefficients

Rollout-only endpoints are substantially below the declared output threshold:

| Tier | Start | Training NMSE | Validation NMSE | Maximum coefficient/initial relative error |
|---|---:|---:|---:|---:|
| Easy | 0 | 3.156e-8 | 1.457e-8 | 0.148% |
| Easy | 1 | 6.564e-14 | 3.482e-14 | 0.000215% |
| Easy | 2 | 1.973e-11 | 1.003e-11 | 0.00481% |
| Hard | 0 | 1.844e-7 | 6.081e-8 | 1.36% |
| Hard | 1 | 5.159e-6 | 1.754e-6 | 6.77% |
| Hard | 2 | 3.792e-4 | 1.835e-4 | 35.8% |

The last column is an additional post-selection diagnostic computed from the
summary's fitted vector and the matching frozen input's reference vector as
`max(abs(fitted-reference)/abs(reference))`. Every reference entry is nonzero.
It does not alter the campaign's CSTR pass criteria or supply truth to fitting.

In hard start 2, `jacket_exchange` is 0.96324 versus 1.5 (35.8% error), and
`exchange` is 1.90293 versus 2.5 (23.9%). Shared initial concentration is 0.25353
versus 0.26193 (3.21%), and initial jacket temperature is 344.434 versus 347.566
(0.90% on the absolute-temperature scale). That relative-temperature percentage
is not an invariant measure of error under a change of temperature origin.

Thus joint coefficient/initial optimization can produce accurate output rollouts
on these cases. It has not uniformly recovered CSTR parameters to 1%, nor does
this establish global identifiability or accurate unobserved trajectories.
The remaining error could reflect incomplete optimization or weakly constrained
parameter combinations; the summary cannot distinguish those explanations.

The hybrid's unsuccessful easy start 1 has validation NMSE 0.00799 but training
NMSE 0.01553, so it correctly fails the requirement that both be <= 0.01.
Direct collocation's CSTR validation errors span 0.142 to 1.197. Shooting spans
0.143 to 7.189 among its five completed CSTR replays, plus one unavailable replay.
These are not merely marginal misses of the accuracy threshold.

## Interpretation and next diagnostic

1. **Rollout-first is the leading development strategy in this comparison.**
   This is centered/scaled, bounded, sensitivity-based joint fitting with useful
   checkpoint retention and training stopping rules, not the older directional
   polling refinement. It receives no free collocation initialization or hidden
   reference values. Its success suggests first testing this path more broadly
   before requiring collocation for every candidate.
2. **Keep the other strategies available.** Direct collocation recovers all 12
   controls. This experiment compares solver/discretization/budget bundles; it
   does not establish that collocation or shooting is intrinsically inferior.
   Extra node variables, mesh construction and per-stage time allowances are
   possible cost sources, to be checked against backend records.
3. **Read timeouts at both levels.** Four rollout-only CSTR endpoints hit the
   outer wall limit but retain accurate checkpoints; all three hard starts do so.
   All six hybrid CSTR fits hit that limit. Shooting reports no outer budget
   exhaustion, yet records 23 mesh-stage timeouts across its 18 tasks. Two hard
   tasks stop with `native_stage_unavailable`. This is not evidence that they
   converged or had adequate numerical time. Null residual-call and mesh counts
   on interrupted outer workers remain unknown, not zero.
4. **Next inspect existing artifacts before increasing budgets.** Obtain the
   completed review archive from the existing inspector. Examine training cost
   versus elapsed time, checkpoint selection, native construction/solve times,
   continuity/ODE defects and the two unavailable shooting stages. Check the
   actual CSTR sensitivity/refinement logs and independent replay diagnostics.
   Determine whether longer fitting still improves hard CSTR coefficients and
   whether mesh work is failing numerically or simply receiving too little time.
5. **The next experiment should test robustness, not promote a default yet.**
   After that diagnosis, freeze broader generic starts and controlled noisy/sparse
   observations for the identifiable, correct-skeleton controls; compare a
   rollout-first strategy against explicitly budgeted fallbacks. Keep parameter,
   latent, initial-value and held-out output recovery separate. No such new
   campaign is implemented or submitted by this review.

The current evidence is limited to six correct-equation cases, three starts each,
noiseless controls and finite budgets. It does not address construction errors,
arbitrary latent dimension, general noisy-system identification or test-set
performance. Scaling, sensitivities and checkpoint retention were combined;
their separate causal contributions are not isolated here.
