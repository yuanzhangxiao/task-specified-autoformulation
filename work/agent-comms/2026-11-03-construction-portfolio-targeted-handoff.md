# Targeted parameter study: implementation handoff

3 November 2026. Implements the follow-up recommended in
`2026-11-03-construction-portfolio-checkpoint-results.md`.
Commands and resources are in
`docs/PHASE_C_CONSTRUCTION_PORTFOLIO_TARGETED.md`.

## Scope and comparison

The new `construction-portfolio-targeted-1` campaign consumes the four complete
authenticated checkpoint-continuation records. Source plan, projected records,
training data and numerical settings are pinned. It projects only immediate
evidence into a compact intake, without copying the nested historical chain.
The original source roots, parameters and receipts are unchanged.

The 20 CPU tasks are:

- Model 12: free warm continuation and four fixed decay values, zero, 0.1/T,
  1/T and 3/T. Other shared parameters and training initial roots are refitted.
- Model 15: original versus meal-flux coordinates from each of three matched
  positive starts (saved rate, 0.1/T and 1/T), plus a separate exact original
  zero-absorption control.
- Models 10 and 14: four fixed saturation scales each (saved, 0.1S, S and 10S),
  with residual-only nuisance fitting.

T is the maximum training trajectory duration, 300. S is the median positive
absolute observed training Gp, 172.6891611318373. These rules and the roster are
predetermined; no private equations, validation/test scores or future fit
outcomes select them. Every source equation and original parameter domain is
preserved. There are no model revisions, LLM calls or production-default changes.

## Certified positive meal coordinates

For the exactly recognized reservoir/output block

`A' = q*r - k*A`, with glucose contribution `+k*A`,

the positive branch permits `B=k*A` and `g=k*q`, hence

`B' = g*r - k*B`, with glucose contribution `+B`.

The certificate checks restricted polynomial identities, exclusive symbol uses,
the independent trajectory initializers and nonnegative unbounded domains. It
rejects unsupported algebra, extra uses, constraints, dependent initials or
incompatible bounds. Parameter names only nominate the pattern to check; they
do not establish the proof.

The implementation transforms the shared-parameter outer search from `(k,q)`
to `(k,g)` and uses a logarithmic positive k coordinate. It continues to use
the original ODE and original-root bounded least-squares profiler. Every trial
reconstructs `q=g/k`, including when g stays constant and k changes. The
conditional root optimum is the same feasible problem because each original
nonnegative initial A0 corresponds to B0=k*A0. Scaled B0 values are recorded,
not added as a second set of independently fitted variables. A finite inverse
and original-domain checks remain mandatory.

The alternative positive starts preserve g and every B0. Both coordinate arms
receive precisely the same reseeded original parameter dictionary and the same
numerical/call allowances. This tests coordinates and initialization within the
same equation family, not the effect of giving one arm a better starting model.

At k=0, finite original parameters imply g=0 and all B0=0. The positive chart
rejects that point, and the zero control runs the original equations with k
held at zero. It cannot introduce a nonzero flux or initial contribution at
zero rate by taking diverging original q/A0 values. No artificial positive
physical lower bound is introduced by the chart.

## Held coordinates, reporting and failure behavior

Residual-only search now optionally accepts explicit fixed coordinates. Values
must be finite, named coordinates inside the original feasible domain. They are
held in the seed and omitted from the balanced sweep and Powell directions.
All-fixed controls perform one evaluation; they do not imply parameter confidence.
Historical calls without this option retain their previous search behavior.

The original incumbent is verified against its historical NMSE separately from
the intentionally changed diagnostic seed. Qualification failures preserve the
incumbent and do not trigger alternative models or renewed budgets. Returned
trial endpoints must preserve held values and pass independent original-equation
verification. Improved verified endpoints may replace the source under the
existing numerical-error guard; ties and worse trials do not.

Reports distinguish the trial's verified loss from the best retained loss.
A worse fixed-parameter trial must not inherit the source incumbent's better
loss in a parameter-profile plot. Trial losses are feasible upper bounds on
profile minima. Budget-limited nuisance optimization cannot exclude a parameter
region or produce a confidence interval. Neither optimizer success nor exact
conditional root fitting alone establishes full convergence.

Source costs remain inherited records, not new physical work. New operation
costs are separate; nested logical-view counts must not be added as disjoint
backend solves. Each numerical stage has an immutable, nonrenewable reservation.
Progress uses compact checkpoints; a hard interruption never grants a fresh
search allowance. Terminal/report reconciliation remains receipt-derived.

## Resources and remote execution

Profiled tasks: 1,800 numerical seconds and 2,000 objective attempts each.
Saturation tasks: 600 numerical seconds and 250 attempts each. Qualification
and verification retain their separately recorded budgets. The largest stage
reservation sum is 4,020 seconds per task.

Delta workers request one CPU, 8 GB, at most two concurrent tasks and a four-hour
scheduler ceiling. The dependent report has thirty minutes. The source root is
`/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-checkpoint-v1`; the new root is
`/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-targeted-v1`.
The source bundle contains code only. Existing saved experiments are consumed
on Delta; neither the source data nor the numerical results are committed.

## Verification and limits

The combined relevant suite passes 404 tests (374 existing/campaign/launcher
tests plus 30 chart/targeted-campaign tests). This includes actual-ODE controls,
same-start profile equivalence, all trajectory initial transformations,
zero-stratum rejection, invalid-expression/domain rejection, fixed-value
retention, failed qualification, interruption/resume, authenticated source
projection and immutable report reconstruction.

On the latest saved model 15, chart round trips at all three positive starts
give identical residuals. Preserving B0 during reseeding has maximum discrepancy
4.44e-16. The same-start profiled NMSE is 0.43845982962519514, with a
basis/full-rollout discrepancy of 1.85e-14. These are numerical equivalence
checks, not evidence of improved fitting.

An initial deliberately short local verification allowance of thirty seconds
expired during model 10's independent original replay. It retained the verified
historical incumbent and launched no search. A separate bounded preflight uses
up to 120 seconds for independent verification and three objective calls for
search; this does not change the Delta campaign's 300-second verification budget
or reuse the spent local operation. Its results are recorded separately.

All six subsequent bounded preflights passed qualification and independent
original-equation replay: model 10 at saved saturation, model 12 at zero decay,
model 14 at the training glucose scale, and model 15 in the saved positive
chart, the 1/T positive chart and the exact original zero branch. Every search
stopped at its three-attempt cap. All terminal reports reconstruct exactly.
The largest retained-trial solver prediction discrepancy was 1.27e-9.
Evidence: `output/portfolio-targeted-preflight-_w5ta2f9/smoke.json`.
The unchanged source incumbents were retained in all six cases. Model 12's
zero-decay trial was numerically tied with its very small positive decay.
No fitting superiority or parameter-region exclusion follows from these short
checks.

Changed Python files pass Ruff. Repository-wide `ruff check .` still reports
the same 37 unrelated existing findings under `analysis/claude`. Bash syntax,
CLI help, source/runtime verification and receipt-only reporting also pass.

Remaining limitations: no long optimization has yet been run with these new
controls; the coordinate change cannot ensure global optimality or
identifiability; a single nuisance start can leave a profile value unresolved;
new retained endpoints still need causal-prefix evaluation before generalization
claims. No remote login or cluster submission was performed.
