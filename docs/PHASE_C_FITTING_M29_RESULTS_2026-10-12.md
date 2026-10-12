# M29 results: successful public handoff, structural limitations and flat directions

Reviewed `review-20261012-214207.tar.gz` on October 12, 2026.
All eleven entries are accounted for: eight return independently verified fitted
models, one has no usable numerical endpoint, and two retain their original
construction failures. All three known-structure controls recover accurately.
None of the six available archived constructions gives an accurate fit.

The public-request boundary preserves the qualified fitter's recovery on the
controls. The constructed cases expose both equation problems and a genuine
optimizer edge case: profiling can eliminate every output-relevant degree of
freedom while leaving irrelevant coordinates for the outer optimizer. We should
fix this and stationary-poor-fit handling before promoting the adapter.

These are **September 30 archived constructions**, selected as Full/seed 0 for
every cell, not Astra's latest construction pipeline. This experiment is not a
comparison against the old fitter, nor evidence that current construction changes
have failed. No model equations were repaired in this review.

## Provenance and audit

- Protocol: `phase-c-public-fitting-integration-1`; [runbook](PHASE_C_FITTING_PUBLIC_INTEGRATION.md).
- Source commit: `b268427c881b23b5ff0db5db483c4441cefd4552`.
- Source-tree SHA256: `8defff8ab3d0d8d5187e6bbfa3ddee27b2363142516bfe823eb8dfcae65b47f4`.
- Plan SHA256: `6e97f4740bfca695fa0c34c90955bbed2c099e8867c5b4f2cf379ea60377b083`.
- Input SHA256: `cce09c7f7b25573cd7c0c1549b15534d78a18f0e1fff631ebae8cf0148517744`.
- Archive SHA256: `e420fe6999c3fd9c308798c09e4385b113f7ededd64f1730ca0fc6f521d31809`.
- Delta jobs: preparation `22853009`, array `22853010`, report `22853011`.
- Output root: `/work/hdd/bibo/yxiao2/phase_c/fitting-public-integration-v1`.
- Runtime: Python 3.12.13, NumPy 2.3.5, SciPy 1.16.3, CasADi 3.7.2,
  Pydantic 2.12.5. One CPU per task, three concurrent tasks, no GPUs or LLM calls.

All 334 sealed JSON artifacts validate, and rebuilding the report reproduces the
supplied summary exactly. Source identity matches the committed implementation;
the archived input equals the frozen local export. All nine available requests
and training splits match their fitting preparations. Lowered candidate identities
match recompilation through the existing compiler. Backend hashes, fitted handoffs
and evaluator parameter hashes agree. Known-control coefficient errors recompute
exactly. No validation data or reference coefficients enter the fitting inputs;
test data remain unopened.

All 82 recorded fitting operations are terminal: 78 complete and four unavailable
training checks. There are nine completed independent evaluator attempts for
eight fitted endpoints; the known basin control uses its reserved tighter retry.
The constructed coupled-basin case returns no parameters for evaluation. All
eleven cost records report complete accounting, including unsuccessful work.

## Prediction and coefficient recovery

Scores below are the independent evaluator's primary-solver NMSE, after its
two-solver agreement check. Fitting minutes exclude the separate evaluator and
public-request setup.

| Known-structure control | Train NMSE | Validation NMSE | Maximum dynamic coefficient relative error | Fit minutes |
|---|---:|---:|---:|---:|
| Alien-device hard | 8.36e-15 | 1.91e-14 | 0.000122% | 7.01 |
| Coupled basin | 1.56e-16 | 2.65e-16 | 0.00000703% | 2.00 |
| Nonlinear CSTR hard | 3.39e-13 | 1.21e-13 | 0.001807% | 4.66 |

All three reach the strict training target in their first search block. Maximum
hidden-initial absolute errors are 8.27e-7 for alien and 7.56e-5 for CSTR, in their
respective physical units. Basin has no fitted hidden initials. Joint local
sensitivity ranks are respectively 18/18, 2/2 and 9/9. These are local diagnostics,
not global uniqueness or calibrated confidence. Restart generation differs from
M28, so timing differences are not a matched method comparison.

| Archived constructed case | Train NMSE | Validation NMSE | Fit minutes | Outcome |
|---|---:|---:|---:|---|
| Dalla Man T1 easy | 0.6532 | 0.5568 | 12.95 | Usable, poor fit |
| Dalla Man T1 hard | 0.6600 | 0.5288 | 2.46 | Usable, poor fit |
| Dalla Man T2 easy | — | — | 0 | Original construction failure |
| Dalla Man T2 hard | 0.9842 | 1.0069 | 11.01 | Usable, poor fit |
| CSTR easy | 1.0363 | 0.4342 | 60.37 | Usable, poor fit |
| Alien-device easy | — | — | 0 | Original construction failure |
| Coupled basin | — | — | 0.21 | No finite search rollout at any start |
| Independent basin | 2.6330 | 0.8338 | 3.25 | Usable; exact within-model optimum remains poor |

Total recorded fitting time is 103.92 task-minutes. Independent evaluation adds
9.86 minutes, and setup adds about 5.44 seconds. Total is 113.88 task-minutes,
not scheduler elapsed time or billed service units. Search starts 781 residual
evaluations and completes 775; failed/incomplete calls remain charged. CSTR easy
alone consumes 58% of fitting time without becoming accurate.

## What the equations explain

Parameter names below are shortened for readability. Signs and dependencies are
those of the compiled requests; these are equation inspections, not new judge
certificates. Poor NMSE alone does not prove structural infeasibility.

### Independent basin: the exact best fit cannot drain

After expanding processes, the observed equation is

\[
\dot h=g_0\,u_{\rm down}(t)/A_{\rm down},\qquad g_0\ge0.
\]

The proposed outflow process never enters this balance. A second state integrates
the same inflow but never affects the measured depth. Thus nonnegative inflow can
only increase depth, and depth is constant when inflow stops. The supplied
trajectories visibly drain. Neither changing initial-map coefficients nor more
optimizer budget can create that missing dependence.

There are seven fitted variables, but only one affects the output. For trajectory
\(r\), define

\[
a_{ri}=A_r^{-1}\int_{t_{r0}}^{t_{ri}}u_r(t)\,dt,
\quad b_{ri}=y_{ri}-y_{r0}.
\]

The public input is piecewise linear, so trapezoidal integration is exact at its
knots. The observed initial depth is fixed to its initial observation. The one
channel's shared training normalization cancels from the least-squares minimizer:

\[
g_0^*=\max\left(0,\frac{\sum_{ri}a_{ri}b_{ri}}{\sum_{ri}a_{ri}^2}\right)
=0.08996330802043.
\]

The retained gain is 0.08996330802066. The analytic minimum training NMSE is
2.63299484367035, reproducing the reported score. Evaluating this same coefficient
on validation gives 0.83375028643123. This is a training-only closed-form audit,
with no new ODE optimization or validation-based parameter choice. It establishes
the within-model optimum for this particular case; it does not establish optimality
for the other constructed models.

The numerical sensitivity rank is 1/7, and the six inactive columns are exactly
zero. The fitter already records alternative parameter vectors with unchanged
loss. However, its recommended action remains `diversify_or_extend_search`, which
is unhelpful once the analytic output problem has been solved.

### CSTR: the jacket term has the opposite physical effect

The compiled model is

\[
\dot T=g_f C_f(T_f-T)+g_r CT-g_j(T_j-T),\qquad g_f,g_r,g_j\ge0.
\]

When the jacket is cooler than the reactor, \(T_j<T\), the final term is positive:
it heats the reactor. The fitted nonnegative domain preserves this wrong operator;
the fitter has not flipped a positive coefficient negative. It instead makes both
\(g_r\) and \(g_j\) approximately 1.11e-16, essentially disabling reaction and
jacket exchange, leaving \(g_f\approx0.001229\). The fitted model has full local
parameter sensitivity rank, which does not make its physical structure correct.

### Coupled basin: discharge creates water and unstable feedback

The submitted balances add the transfer to both upstream and downstream states,
and add outlet/overflow flows instead of subtracting them:

\[
\dot h_u=u_u+q_{\rm transfer}+q_{\rm overflow},\qquad
\dot h_d=u_d+q_{\rm transfer}+q_{\rm out}.
\]

The proposed outlet is \(q_{\rm out}=A_d(h_d-c_d)\), producing positive feedback
in depth. The depth balances also omit conversion of volumetric flow by basin
area. These errors are present in the submitted equations. All four starts fail
their first search evaluation with nonfinite rollout calculations; none completes
a search residual. This is not evidence that the initial RHS itself is nonfinite,
or that every conceivable parameter vector fails. The known basin control fits
accurately on its correctly specified equations.

### Dalla Man: omitted dynamics and non-identifiable latent scale

T1 easy reduces to \(\dot G=k_1 r_{\rm meal}+k_2G\); T1 hard to
\(\dot G=r_{\rm meal}-gG\). Both have only observed glucose as a state. They lack
separate meal-absorption or insulin-action states. They can produce a glucose
response, but parameter fitting cannot add an independently delayed mechanism.
The observed glucose state itself still has memory; these are not memoryless
models. Their poor scores do not by themselves prove global best-fit bounds.

T2 hard does include an insulin-action state:

\[
\dot G=k_m r_{\rm meal}-k_i I_a,\quad
\dot I=k_u u_I-I/\tau_I,\quad
\dot I_a=k_1 I-I_a/\tau_a,
\quad I_a(0)=a+bI(0)+cU_{ii}(0).
\]

Its declared process \(U=k_2I_a\) is never used by a state equation or observation.
Consequently \(k_2\) cannot be fitted from the targets. There is also an exact
positive scale symmetry: multiply \(I_a,k_1,a,b,c\) by \(s>0\) and divide \(k_i\)
by \(s\); observed \(G,I\) remain unchanged. A weak numerical singular value is
not needed to establish these particular dependencies. Additional near-flat
directions may reflect the fitted point and the available experiment, rather than
further global symmetries; the measured local rank is 6/10.

The fitted insulin time constant grows to approximately 600,561 minutes. The
aggregate score hides a particularly poor insulin fit: training/validation NMSE
is 1.456/1.531 for insulin versus 0.513/0.482 for glucose. This equation only
forces insulin through the external insulin input and otherwise decays. This is
an equation/data limitation to inspect, not a claim that the public task explicitly
requires a glucose-to-insulin edge. Identifiability and prediction quality must
be reported separately.

## Fitter issue: fully profiled output, irrelevant outer coordinates

Independent basin is routed to exact terminal-output profiling. The inner bounded
least-squares solve estimates the only output-relevant gain. All six remaining
outer coordinates have zero influence on the profiled output. Instead of returning
the solved inner problem, the outer trust-region optimizer attempts a flat search.
Each of the four starts reports `nonfinite optimizer coordinate reconstruction`,
with SciPy divide-by-zero/invalid-operation warnings, after one profile call.

The joint fallback recovers the correct within-model optimum, but uses most of
each remaining block: the whole case takes 158 search evaluations. In several
fallback blocks, projected gradients are already near machine precision and the
loss does not improve. Absolute gradient and function-reduction termination were
disabled to protect high-accuracy coefficient recovery in earlier milestones;
that choice now wastes work on a stationary poor fit.

This is a fitter defect even though the final gain is correct. It is distinct from
the missing outlet equation. Fixing the optimizer should make this result cheaper
and clearer; it should not be expected to improve the model's best possible NMSE.

## Recommended next milestone

Keep the same frozen requests and budgets, with the three successful known
controls retained as regression cases. Do not begin by increasing fitting budget
or changing these archived equations.

1. **Handle exact output-inactive coordinates and fully profiled problems.** Use
   symbolic dependency analysis through shared processes, state dynamics,
   observation maps and initializers. Freeze only coordinates certified irrelevant
   to the fitted outputs, preserve their declarations/values in the returned model,
   and mark them unestimated. If profiling leaves no active outer variables,
   verify and return the bounded inner solution without invoking an outer solver.
   A small numerical Jacobian column alone must not authorize removal. This is
   optimization bookkeeping, not scientific pruning of the candidate.
2. **Stop stationary poor searches without claiming a correct model.** Combine
   scaled projected-gradient, step, loss-progress and numerical-resolution checks;
   retain bounded distinct restarts where optimality is only local. Preserve the
   strict accurate-fit path and coefficient-recovery checks. Expose separate
   outcomes for accurate recovery, stationary poor fit, exhausted search and
   unavailable rollout. Exact solved subproblems can justify stronger stopping
   than heuristic plateaus.
3. **Return actionable construction feedback with provenance.** Report certified
   inactive parameters and missing output paths, per-target fit failures and
   observed numerical failures. Keep scientific sign diagnoses explicitly separate
   from deterministic fitting certificates. Share the equation findings in this
   document with the construction work; no agent was messaged in this review.

Compare search calls, total fitting/verification time, retained training/validation
scores and known-control coefficient/initial errors. Correct behavior should remove
the independent-basin optimizer failures and redundant search while preserving
its poor but optimal fit. CSTR and coupled-basin repairs require construction work;
this fitter milestone should not silently edit their signs. Production integration,
noise-aware tolerances and broader structural coverage remain subsequent work.

## Review verification

The 49-test local preparation suite passes, matching 49 passes in the archived
Delta preparation log. The public-rollout smoke checks a two-output model, shared
initializer, independent evaluation and unchanged deterministic resume. No benchmark
model was refitted locally. Repository-wide `ruff check .` reports the same 37
pre-existing issues in unrelated `analysis/claude` files. This review changes only
documentation; no next experiment is implemented or submitted.
