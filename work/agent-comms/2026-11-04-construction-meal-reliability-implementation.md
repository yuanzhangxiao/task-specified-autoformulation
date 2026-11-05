# Meal fitting reliability: implementation and Delta handoff

4 November 2026. One opt-in fitting milestone following the returned joint
profiling experiment. No construction changes, data/prompt changes, LLM calls,
validation/test access, or remote submission were performed here.

## Why this milestone

The ten returned joint/root-only comparisons were numerical ties. Their small
one-sided slopes were often labeled step-sensitive even though all evaluations
were finite. For a smooth loss,

\[
\frac{F(q+h)-F(q)}h=F'(q)+\frac h2F''(q)+O(h^2).
\]

Ordinary curvature at a nearly stationary point can therefore make the secant
change substantially when the step is halved. This is not by itself evidence
that the objective is unreliable. Conversely, successful coordinate probes do
not exclude descent involving several coordinates together.

The fixed-rate experiments also left four useful, predetermined alternative
endpoints. Their meal rate was previously constrained. This campaign releases
that constraint rather than repeating another fixed-rate experiment.

## Numerical implementation

`profile_curvature_audit.py` is separate from the historical diagnostic. It
uses frozen optimizer coordinates, with logarithms for strictly positive
rate/timescale coordinates and physical scaling for others. Profiled and fixed
coordinates are excluded. Every sample re-solves the certified joint bounded
least-squares problem.

Centered first and second differences are evaluated at two small scales;
feasible one-sided stencils cover parameter bounds. Tensor products of the
first-difference stencils measure actual mixed loss curvature. The profiled
residual Jacobian is also measured, with per-column two-scale checks and SVD.
Its Gauss–Newton matrix is reported separately because

\[
\nabla^2 F=\frac2N J^TJ+\frac2N\sum_i r_i\nabla^2 r_i
\]

contains a second term when residuals are nonzero. Positive Gauss–Newton
curvature alone does not exclude a saddle. Negative chart curvature alone
does not establish feasible constrained descent either; bounds, gradient,
active-set switches and actual tested losses matter.

Paired residuals are transient in memory. Loss differences use
`mean((r-r0)*(r+r0))` to reduce cancellation. The separate reporting resolution
remains conservative and is not presented as a rigorous integration-error
bound. Missing stencils and changing bounded-inner active sets defer stronger
smooth interpretations.

The audit tests bounded coupled directions against the actual objective. These
finite candidate steps are explicitly distinguished from derivative checks.
Strictly improving candidates survive partial or interrupted audits, but only
an independent original-equation replay can make one eligible for campaign
retention. No local audit certifies stationarity, global optimality, parameter
confidence or a required equation revision.

## Five-task campaign

Source: `construction-meal-joint-profile-v1`, frozen plan
`7ccff974fa6d1564f0ccf0c5b231e837c8e6abd2de68db3b6b4a33eb3d8d8c61`.
Only its five joint-arm records are consumed, in frozen order with explicit
record digests. Source receipts remain unchanged.

1. Warm endpoint: qualification, replay and curvature audit; no new Powell search.
2. Four former fixed-rate endpoints: own saved coefficients/initials, rate
   released, unrestricted joint-profile search, replay, then curvature audit.

Each trial retains its own verified endpoint and the common warm incumbent
independently. Reports distinguish source trial, search trial, post-audit trial
and global retained fit. No failed or worse revision discards either incumbent.
An interrupted operation cannot silently renew its numerical reservation.

Each restart receives 1800 numerical-active search seconds, at most 2000
attempts, and a 2700-second overall search limit. Each audit receives 360
numerical-active seconds, at most 220 profile calls and a 600-second overall
limit. Separate replay/setup reservations bring the campaign totals to 3060
seconds for the warm task and 6180 seconds per restart. Delta uses one CPU and
8 GB per task, concurrency two, four-hour worker limits, and a thirty-minute
dependent report.

The launcher authenticates code, runtime, plan, worker and scheduler receipts.
Confirmed submissions are reused; ambiguous replies stop automatic resubmission.
Inspection works from any working directory through the saved repository path.

## Verification

- 330 tests passed across the new numerical/campaign/launcher components and
  adjacent profiling, coordinate, checkpoint, termination and campaign regressions.
- Scoped Ruff, shell syntax, CLI help and diff checks passed.
- Repository-wide Ruff still reports the same 37 unrelated existing findings
  in `analysis/claude`; those files were left unchanged.
- The actual production source authenticated and prepared five tasks locally.
  Verification, empty report reconstruction, inspection archive and launcher
  resource/runtime identity checks passed. No jobs were submitted by this smoke.
- Bounded real-endpoint numerical preflights completed under the final source.
  Both completed all 12 axial and 30 mixed derivative stencils, with no failed
  profile calls. Independent Radau replay of original equations agreed within
  `3.3e-12` in training NMSE.

| Saved start | Profile calls | Baseline NMSE | Best materially better audited point | Audit wall time |
| --- | ---: | ---: | ---: | ---: |
| Warm | 177 | 0.423866885099 | None | 109.8 s |
| Formerly fixed `1/T` | 181 | 0.457389535550 | 0.455793553387 | 110.9 s |

The second point improves its own start after releasing the rate, but remains
worse than the common warm incumbent. No outer optimizer was run locally.
The warm Hessian and inner active set were stable across scales, while one
centered gradient remained step-sensitive; the implementation therefore deferred
the combined smooth local conclusion. Neither absence of detected descent nor
this remaining step sensitivity establishes a global optimum or equation defect.
The bounded audit is ready for the five-task Delta campaign; unrestricted
restart outcomes remain unmeasured.

## Changed files and operation

New numerical module and its tests; a transient residual accessor in the joint
profiler; new research campaign, policy and tests; Delta CLI/worker and launcher
tests; runbook; project context and design updates; this handoff. Generated
preflights and the code bundle are untracked output artifacts.

Use `docs/PHASE_C_CONSTRUCTION_MEAL_RELIABILITY.md` for upload, submission,
status, inspection and download commands. New root:
`/work/hdd/bibo/yxiao2/phase_c/construction-meal-reliability-v1`.
The code-only archive is
`output/construction-meal-reliability-delivery-20261104/construction-meal-reliability-code.tar.gz`.

The resulting scores are full-training reconstructions with fitted training
initial roots. They do not establish causal prefix inference on new trajectories
or held-out intervention accuracy. An unsuccessful set of restarts also cannot
prove that the equation family has reached its global best fit.
