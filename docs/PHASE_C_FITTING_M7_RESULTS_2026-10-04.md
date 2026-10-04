# Phase C M7: mesh size, derivative cost and restart results

Review of `review-20261004-072922.tar.gz`, following the
[M7 diagnostic](PHASE_C_FITTING_NUMERICAL_DIAGNOSTIC.md). There are **36 saved
fitting backends, 35 finalized evaluations, and one independent-replay timeout**.
The supplied summary correctly reports an incomplete campaign. Basin passes all
18 fits. Alien prediction, coefficient and initial recovery remain two of three
starts for each rollout arm, and zero verified successes for the native NLP arms.

The diagnostic identifies actionable costs, without establishing a new successful
default: coarsening enables native convergence; exact integrated Hessians are
expensive; checkpoint diagnostics consume substantial solve time; one generic
restart does not rescue the stalled rollout start. This review changes documents
only. No campaign fit, independent endpoint replay, benchmark or production default
is changed.

## Provenance and verification

- Archive SHA256:
  `e9aa0eeb09f0f0483d3d42725026b8e43da4c19d094647cbc896e4fefaa1ff6a`.
- Ignored extraction: `artifacts/fitting-m7-review-20261004`.
- Experiment commit: `51b95cbca97a278e8700ab1464fb74806a720e26`.
- Plan SHA256:
  `65eb262dd2031bad6a7302e2d8255f58ddce2a857815d88597597a299e7ddf4a`.
- Inputs SHA256:
  `97aacd4d137ad3631b11798795ad857d4febcae5780565d351277806119adfc6`;
  identical to M6.
- Delta jobs: prepare `22647554`, fit array `22647555`, report `22647556`.
- All 147 sealed JSON records verify; the summary reproduces exactly. All 36
  worker payloads and backend identities reconcile with their common starts.
  All 35 finalized rows reconcile with their saved independent replay and
  recomputed coefficient/initial errors.
- The frozen source digest matches the preserved M7 release bundle. The current
  checkout additionally has an unrelated, untracked
  `search/construction_ledger.py`, so its all-Python source digest differs. None
  of the bundled source files differs; that local file is not part of this review.
- All 35 finalized endpoints have complete independent Radau/DOP853 replays;
  maximum normalized solver disagreement is 8.227e-6, below the 1e-4 gate.
- Reference replay and local sensitivity qualification pass. Alien's fitted
  block has rank 18 and normalized singular ratio 0.006420. This is a local
  result at the supplied reference, not global or practical identifiability.
- No LLM calls or test access. Validation/reference values do not enter the
  fitting payload, checkpoint selection or restart decision.

## Accuracy and recovery

The unchanged alien problem fits 13 dynamic factors and five shared hidden
initials. Correct equations, internal couplings and nonlinear shapes are supplied
to anchor latent coordinates. This is conditional recovery under that assistance,
not recovery of every generator coefficient. No hidden trajectories are fitted
against reference trajectories.

Alien prediction passes require train and validation NMSE <=0.01 plus solver
agreement. Dynamic recovery requires every free coefficient within 1%; shared
initial recovery requires maximum absolute error <=0.01. The three pass counts
coincide in the finalized rows below. All planned starts remain in denominators.

| Alien arm | Verified prediction / coefficient / initial passes | Validation NMSE, start 0 | Start 1 | Start 2 |
|---|---:|---:|---:|---:|
| Fixed dense collocation | 0/3 each | 0.437585 | 0.769574 | 0.907542 |
| Fixed reduced collocation | 0/3 each; one evaluation unavailable | 0.304437 | 0.877640 | unavailable |
| Fixed shooting, exact Hessian | 0/3 each | 0.626631 | 0.831025 | 0.604866 |
| Fixed shooting, limited-memory Hessian | 0/3 each | 0.996544 | 0.542132 | 0.604866 |
| Rollout continuation | 2/3 each | 7.78e-16 | 9.69e-15 | 0.121072 |
| Rollout with triggered restart | 2/3 each | 7.78e-16 | 9.69e-15 | 0.121959 |

The two successful rollout starts retain exactly the M6 successful parameter
vectors. Maximum dynamic relative errors are 5.38e-7 and 1.15e-6, respectively;
maximum initial absolute errors are 4.00e-7 and 6.40e-7. Start 2 still has maximum
coefficient relative error above 10 and initial absolute error above 2. Accurate
prediction and parameter recovery are both attainable, but optimization is not
reliable across starts.

Basin passes prediction and dynamic recovery in all 18 runs. Worst validation
NMSE is 4.914e-9 and worst maximum coefficient relative error is 0.0002610
(0.02610%). There are no free hidden initials in this control. Dense and reduced
collocation use the same basin mesh and produce identical retained vectors.

## Coarsening helps optimization, but does not solve recovery

Alien collocation falls from **115,218 to 33,906 variables**, a 70.6% reduction.
All 9,616 original observation residuals remain. Forcing interpolation is
preserved; the four densely varying schedules prevent reaching the soft
6,000-variable target. This is a conservative mesh reduction, not observation
subsampling.

Dense solves reach recorded iterations 84, 88 and 97 before native timeouts.
Reduced solves all return `Solve_Succeeded`, at 147, 146 and 313 iterations.
The third native solve succeeds even though its later full-fit screening and
independent replay encounter time limits. Native success and endpoint availability
must be reported separately.

The reduced solutions are not accurate models. Saved late collocation objectives
are approximately 0.225, 0.608 and 0.628, already far from the attainable near-zero
fit. This is not merely an independent integrator disagreeing with an otherwise
excellent observation fit. The optimizer has moved into poor parameter regions.

There are also serious discretization concerns. The selected start-2 checkpoint
has a tiny nodal equality defect (3.51e-7) but maximum off-node residual indicator
about 768. Its fitted `decay_x5` is 1262.8 while other decays are nearly zero;
`input_x0` is about 190,741. Input fidelity alone cannot make a coarse state
polynomial accurate for such fitted timescales. The indicator is a diagnostic
residual, not a certified global error bound.

The deployed endpoint is chosen by complete training rollouts, not by minimum
nodal defect or IPOPT status. In starts 0 and 1 it is an earlier checkpoint with
better rollout performance, not necessarily the final native iterate. Do not
combine the final iterate's convergence flag with a different checkpoint's
prediction score and call the selected vector a converged solution.

## Shooting: exact derivatives are a measured bottleneck

Shooting has only 2,304 variables, yet remains slow. One initial-point derivative
profile gives these ranges across starts:

| Quantity | Exact-Hessian shooting | Limited-memory shooting |
|---|---:|---:|
| Explicit NLP graph construction | 5.9--6.1 s | 4.9--6.2 s |
| Objective evaluation | 1.64--1.79 s | 1.67--1.70 s |
| Constraint Jacobian evaluation | 23.1--26.0 s | 23.5--24.4 s |
| Exact Lagrangian Hessian evaluation | 99.9--110.9 s | not requested |
| Last recorded iteration before timeout | 2--3 | 13--15 |

Exact Hessian generation additionally takes 13--14 seconds; its evaluation is
the larger cost. Limited-memory permits more iterations and improves start 1,
but worsens start 0 and leaves start 2 at the original retained vector. None
passes recovery. Avoiding second derivatives is useful computational evidence,
not a demonstrated accuracy improvement across these starts. First derivatives
and checkpoint processing still matter.

These profiles are measured at one initial point, with unit constraint
multipliers, and charged to the native budget. They are not average costs over
the optimization, nor a controlled end-to-end speedup estimate.

## Checkpoint bookkeeping is too expensive

At the last dense-collocation progress record, cumulative checkpoint work takes
**364--387 seconds out of 661--663 seconds elapsed** (55--59%). This includes
extracting node states, calculating off-node indicators for every trajectory and
serializing the retained history. It is not simply file writing. The callback
currently takes a detailed snapshot whenever at least one second has passed
since the previous one, in addition to iteration-based triggers.

Reduced solves spend about 101--217 seconds in that same checkpoint routine;
native callback accounting also reports substantial cost. The successful basin
control is too small to expose this overhead clearly. We should preserve useful
intermediate states while making routine callbacks cheap: retain minimal bounded
primal records, and compute expensive trajectory indicators only for a few
selected checkpoints or when a mesh decision actually needs them. Final and
best-known records must still survive interruption.

## Restart was exercised, but did not help this start

The start-2 arms have **exactly identical first-phase training histories and best
vectors**: 41 completed calls, ending at training NMSE 0.124176709. Improvement
over the last ten evaluations is 0.2391%, below the frozen 0.5% threshold. Both
arms record the same trigger; only the restart arm uses the new generic draw.

Within the remaining total budget:

- Continuation completes another 32 calls, improving training NMSE to
  0.123999298 and validation NMSE to 0.121071610.
- The restart completes another 33 calls, improving its new starting loss from
  0.883886 to 0.129566417. That is worse than the first-phase incumbent, which is
  correctly retained. Its independent validation NMSE is 0.121959440.

Both whole fits reach the 900-second guard. These are counts of recorded completed
evaluations; interrupted in-flight work is not claimed as fully accounted calls.
The summary's partial records omit the decision, but `fit/diagnostic.json` retains
it. A null summary decision therefore does not mean that no restart occurred.
The restart ends in a similar region: `decay_x0` about 0.797 and `output_gain_1`
about 0.0011, versus reference decay about 0.0686. This does not prove that all
restarts would fail or that the region is a certified local minimum.

## The missing row is a reporting/replay failure

Task **31**, `alien_hard_s2_fixed_reduced_collocation`, has a valid saved
`backend.json`, a retained parameter vector, and a complete training screen
(NMSE 0.565126933). Its native solve returns success; the whole fitting process
later exhausts its 900-second budget while screening further checkpoints.

After fitting, the independent replay reaches its separate deadline during
**DOP853** integration. An uncaught `TimeoutError` exits before `replay.json` and
`result.json` are saved. The extreme fitted decay is consistent with an expensive
explicit replay, but this log alone does not quantify all causes of its cost.
The recorded training screen already fails the accuracy threshold; validation
and independent solver agreement remain unavailable.

Do not refit all 36 jobs or silently extend the evaluation allowance. The bounded
follow-up should catch replay timeouts, preserve any completed trajectory records,
and finalize this as unavailable evaluation, with the original backend untouched.
An explicitly labeled post-hoc replay can be added if needed, but must not replace
the frozen-budget outcome or grant another optimization budget.

## Recommended next milestone

Keep the same equations, data, all starts and all available strategies. Prioritize
a small numerical-engineering milestone before adding harder benchmarks:

1. Make detailed checkpoints sparse and cheap; compare against the existing
   policy with unchanged starts/meshes/budgets. Measure optimization versus
   bookkeeping time explicitly and preserve interruption recovery.
2. Finalize failed independent evaluations as explicit outcomes; retain restart
   decisions and completed-call counts in partial reports.
3. Keep limited-memory shooting as the cheaper derivative option under study.
   Profile/reduce repeated integration and first-derivative work before expecting
   a larger shooting campaign to help. The current implementation integrates
   each original sample interval separately within a window.

After that, test a bounded mesh-accuracy continuation or different training-only
initialization/parameterization, rather than blindly adding optimizer seconds.
Coarse elements need error checks at the fitted vector; refinement should follow
an adequately optimized coarse solve, not merely a timeout. Bounds must come from
public restrictions or a declared numerical policy, not hidden reference values.
One unsuccessful restart does not warrant discarding restart methods, but there
is no evidence here for promoting this particular policy.

## Review checks and changed files

The 67 focused fitting tests pass. The six-arm small-control smoke passes fitting,
independent replay and exact completed-result resume. Repository-wide Ruff reports
39 unrelated findings: 37 in `analysis/claude/` and two in untracked construction
ledger code/tests. This review changes no Python code. The
full suite passed at the experiment commit (3,886 passed, eight skipped); it is
not rerun for this documentation-only review.

Changed files: this review, `PHASE_C_FITTING_PLAN.md`, and `PHASE_C_START_HERE.md`.
Raw experiment artifacts remain untracked. Follow-up implementation is proposed,
not included in this results review.
