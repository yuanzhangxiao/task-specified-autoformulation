# Construction portfolio search: returned results and next fitting work

1 November 2026. Review of the five-model campaign at commit
`878104c25eeb0023ce37cda012081249c8f5ac20`.

## Conclusion

Models 10/14 improve substantially, but neither has completed optimization.
Models 12/15 admit modestly better fits at a slower insulin-action timescale,
within the same equations. Their diagnostic profile searches did not adjust
most available parameters. Model 03 recovers one previously unresolved
derivative direction, with three still unresolved.

There is still no basis for telling construction that these equation families
have been fitted to a demonstrated optimum. The next work should address
remaining algebraic redundancy, expensive sensitivities, and inefficient
nuisance optimization before drawing that conclusion.

Every new score below is **training reconstruction NMSE**. No new causal-prefix
forecast, validation or test performance was evaluated. In particular, older
prefix scores cannot be attached to the new fitted endpoints.

## Authentication

- Input archive: `/Users/yuanzhangxiao/Downloads/inspection.tar.gz`.
- Archive SHA256:
  `de2931d1a2d2b26672fc0c65fcdc4602b2daaadb3b52a14a37c54620cb453eab`.
- Protocol: `construction-portfolio-search-1`.
- Plan SHA256:
  `6cddde26ba8ae9b91b5e086f3bde345b4e9e625fadeba397bbc4bd11394b4a9a`.
- Five expected terminals and all five summary rows reconstruct exactly from
  their operations; no timing reconciliation tolerance was needed.
- Source projections and launcher/source hashes match the committed campaign.
- The 341 original summary/receipt files authenticated before and after review
  are unchanged. No report was regenerated over historical results.

Local evidence is under
`output/construction-portfolio-search-review-20261101/`:
`archive-provenance.json`, `independent-receipt-audit.json`,
`timescale-review.json`, the original summary, and per-candidate operations.
No saved experiment was refitted; no remote execution or LLM calls were
performed in this review. Operational `complete` means the bounded attempts have recorded
outcomes; it does not mean numerical convergence.

## Models 10 and 14: real improvement, unfinished optimization

| Model | Supplied-start training NMSE | Retained training NMSE | Gp / I / U NMSE | Accepted steps |
| --- | ---: | ---: | --- | ---: |
| 10 | 128,915.8388 | 1.749667 | 0.92925 / 1.01053 / 3.30923 | 43 |
| 14 | 129,334.5955 | 1.689073 | 0.60902 / 0.99985 / 3.45835 | 44 |

Both searches exhausted their 1,800-second numerical allowance. Their final
accepted steps still improved NMSE by approximately 0.02374 and 0.001675,
respectively. These remain poor fits, but the evidence does not establish that
the equation families cannot fit better. Independent original-equation replay
agrees with their residuals within 1.93e-10 and 2.25e-10.

The main diagnostic bottleneck is the augmented sensitivity solve. Each
retained-endpoint diagnostic and last-accepted-endpoint diagnostic spent about
600 seconds on its first sensitivity evaluation, then timed out. No local
finite-difference checks began, and no stationarity assessment was produced.
For each of these two models, the best and last accepted parameter dictionaries
are identical. Thus approximately 1,200 seconds per model was spent attempting
the same numerical prerequisite twice.

There are also remaining exact coefficient combinations in these nonlinear
models: three coefficients multiplying the same meal-rate term, two insulin
gains in the action balance, two action-decay gains, and the product
`af_process_gain_1_0 * Vmax_ins` in the saturating response. Seven outer gains
were already fixed, but these combinations remain in the 29-coordinate problem.
They are candidates for further **certified** reduction: prove equivalence and
preserve original domains, including zero/product branches, before use.

The saturation is `Vmax_ins * Gp**2 / (Km_ins**2 + Gp**2)`.
Model 10's negative fitted `Km_ins` is not by itself a wrong effect: this
parameter is squared. Small fitted magnitudes may warrant numerical diagnosis,
but this review does not establish that they caused the sensitivity timeout.

## Models 12 and 15: a useful direction, incomplete profiles

For a state balance with negative self coefficient, the timescale chart writes
the same equation as a relaxation toward its driven response. Changing the
insulin-action timescale changes how quickly that response is reached, without
adding a state or changing the equation family.

The following are fixed-timescale controls and partial nuisance refits. The
control changes only the chosen timescale in the new chart; the equilibrium
gains and other chart coordinates stay at their saved values. This necessarily
changes the corresponding original rate coefficients together.

| State whose timescale is fixed | Timescale in native time units | Model 12: control → partial refit | Model 15: control → partial refit |
| --- | ---: | --- | --- |
| Insulin | Saved (~1e-11) | .564509 → .564509 | .564608 → .564608 |
| Insulin | .001 | .564512 → .564512 | .564611 → .564611 |
| Insulin | .1 | .564773 → .564773 | .564867 → .564867 |
| Insulin | 1 | .566248 → .566247 | .566290 → .566289 |
| Insulin | 10 | .578797 → .578541 | .577817 → .577789 |
| Insulin action | Saved (~1e-11) | .564509 → .564509 | .564608 → .564608 |
| Insulin action | .001 | .564509 → .564509 | .564609 → .564609 |
| Insulin action | .1 | .564548 → .564548 | .564653 → .564653 |
| Insulin action | 1 | .564427 → .564384 | .564589 → .564587 |
| Insulin action | 10 | **.559336 → .556566** | **.559515 → .558998** |

All 20 profile endpoints independently replay against the original equations.
Action timescale 10 improves the aggregate objective even before nuisance
refitting. The best partial refits improve on the incumbents by about 1.4% and
1.0%. These are feasible improvements, not evidence that this timescale is
identified or biologically correct. The main incumbents intentionally remain
unchanged because the profiles were diagnostic rather than selection runs.

The improvement also involves output tradeoffs. Gp NMSE worsens from
.451761 to .481015 in model 12 and from .449880 to .506924 in model 15, while U
improves from .594302 to .548172 and from .596433 to .540664. The fitted insulin
timescale becomes still smaller, about 4.19e-14 and 2.45e-13. Thus the profile
does not resolve stiffness or establish that all outputs benefit.

Every profile exhausted its 120-call cap before completing one Powell
iteration. Only 4–6 of 40/39 free nuisance coordinates actually varied; none of
the 32 training initial roots varied. Numerical time was roughly 69–72 seconds
per profile, below the 120-second allowance. The limiting factor was the call
allocation, not the numerical wall deadline. These results cannot be treated
as converged parameter profiles, likelihood intervals or confidence estimates.

Endpoint-history handling now distinguishes retained and last-accepted points
correctly. The latter have 53/47 accepted history points. Nevertheless, the
nominal projected gradients of 5.97e-10 / 1.33e-10 have uncertainty upper bounds
of 5.82e-5 / 2.32e-4, and coefficient histories remain unstable. Both models
therefore remain `optimization_unresolved`.

The dominant derivative uncertainty is in the meal-rate parameters
`k_A_decay` (12) and `k_abs` (15), rather than the fast insulin/action
coordinates. Their local finite-difference discrepancies decrease roughly
fourfold per step-halving. Further smaller-local-step checks are justified
there, subject to the usual roundoff and convergence tests.

## Model 03: precision coverage

| Coordinate | Result at tighter integration tolerances |
| --- | --- |
| 8: insulin-input coefficient | Inconclusive; derivative is near numerical precision |
| 10: `k_Vmax` | Timed out before completing the required local stencil |
| 19: train007 action initial value | Passed; relative errors about 1.17e-5 / 1.58e-5 |
| 22: train010 action initial value | Timed out before completing the required local stencil |

Directions 10/22 each used their 240-second allowance and completed five
stencil scales, but lacked the final required probe. Direction 8's derivative
RMS (roughly 1.3–2.7e-8) is comparable to its numerical precision estimate.
This is not a proof of structural unidentifiability. Empty `failed_indices`
does not imply all directions passed: incomplete attempts are separate.
Whole-model search remains unqualified.

## Accounting addendum

Historical receipts must remain immutable. The following corrections should
be explicit when reporting costs and implemented prospectively:

- Models 12/15 each performed **1,200 attempted, started and completed profile
  residual evaluations**, with zero failed calls. Their summary's search-call
  counters incorrectly show zero because profile records emit `calls_*` while
  aggregation expects `search_calls_*`.
- Three timescale-chart qualification calls per model are omitted from the
  aggregate verification aliases.
- Nested derivative-check calls in retained/accepted endpoint diagnostics
  (174 per diagnosis for 12; 170 for 15) are not included in the aggregate
  verification counters either. Future accounting should label nested scopes
  and avoid double-counting shared prerequisites.
- Recorded wall/CPU timing includes this work. The defect does not change
  fitted parameters, losses, or independent replay results.

## Recommended next bounded milestone

1. Standardize operation call counters and reuse identical numerical
   prerequisites at exactly matching model/data/parameters/chart/settings.
   Keep endpoint-specific history interpretation separate. Preserve all old
   receipts and publish additive accounting corrections.
2. For 10/14, certify the remaining coefficient reductions and check equation,
   domain and rollout equivalence. Diagnose and qualify endpoint sensitivities
   before allocating another long continuation. Resume from the verified best
   endpoints once the derivative path is usable.
3. For 12/15, exploit affine dependence on the 32 trajectory initial roots at
   fixed shared parameters. A certified bounded least-squares inner problem
   could reduce fixed-timescale numerical search from 40→8 and 39→7 dimensions.
   Verify equivalence to direct rollout, including observation maps, root
   domains, loss weights and any coupled initialization constraints; do not
   apply unconstrained elimination where those conditions fail. Compare the
   existing and action-timescale-10 seeds with matched budgets, preserve the
   incumbents, and report per-target tradeoffs.
4. Complete the narrowly targeted precision work: the missing 03 probes and
   the two dominant meal-rate directions in 12/15. Reuse valid receipts rather
   than replaying entire finished stencils; further shrinkage is not assumed
   to improve a direction already at the roundoff floor.

This is still numerical fitting work. Structural feedback can describe the
observed residual patterns and unresolved fitting limitations now, but should
not claim that nonlinear interactions, new inputs or new states are necessary
on the basis of these unfinished optimizations.

## Local verification and changed files

This review adds only this Markdown communication; implementation files and
historical artifacts remain unchanged. The five relevant pytest modules
(campaign, launcher, nonlinear search, timescale chart, report reconciliation)
passed **119 tests**, including synthetic rollout/resume smoke coverage. One
existing SciPy warning notes an ignored `jac` argument for an explicit solver.
`ruff check .` still reports the same 37 unrelated findings in untracked
`analysis/claude/` files. No broad fitting-performance claim follows from those
tests; the scientific limitations are the ones stated above.
