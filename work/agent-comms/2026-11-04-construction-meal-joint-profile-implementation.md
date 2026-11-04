# Joint shared-forcing and initial-root profiling

4 November 2026. Implementation of the next fitting milestone, following the
meal-scale recovery review. No construction, scientific-judge calls, benchmark
changes or held-out evaluation are included.

## Numerical change

At fixed remaining dynamics, the saved model's outputs are jointly affine in
the shared meal gain `g=k_abs*k_abs_rate`, shared auxiliary drift
`s=k_Uii_term_12_0_244d72574a`, and its 32 training initial roots. A global weighted
14,448-by-34 design includes g and s once across all sixteen trajectories and
separate initial-root blocks. The drift is an existing term, not a proposed
scientific repair. Original equations, signs and domains remain unchanged.

`prefix_meal_joint_profile.py` certifies affine forcing, independence of the
state matrix/readouts/initializers from the two profiled coefficients, identity
prior lifts and preserved bounds. Symbolic forcing columns and an exact affine
propagator avoid subtracting nearly equal complete trajectories. Column-scaled
BVLS supplies rank, active-bound and normalized KKT diagnostics. Positive k has
the finite inverse q=g/k. Exact zero remains a separate excluded stratum.

The existing root-search API gains an opt-in profiler factory. Its historical
default is unchanged. The new arm removes g and s from the outer search, leaving
six warm coordinates or five when meal decay is fixed. Root-only controls retain
eight or seven outer coordinates.

## Endpoint evidence

All five authenticated saved starts passed both conditional solves and fresh
independent original-equation Radau replay. Joint design ranks were 34/34,
maximum normalized KKT residual 5.91e-13, maximum basis/full-rollout discrepancy
8.37e-14, and maximum independent NMSE discrepancy 3.27e-12. The conditional rank
does not establish joint identifiability with the outer parameters free.

| Seed | Root-only NMSE | Joint NMSE |
| --- | ---: | ---: |
| warm-flux | 0.423866958337 | 0.423866885105 |
| fixed-0p001 | 0.426489814386 | 0.426489814382 |
| fixed-0p01 | 0.426269576741 | 0.426269576741 |
| fixed-0p1 | 0.424601448721 | 0.424601381729 |
| fixed-1 | 0.457389602578 | 0.457389537135 |

These are tiny changes. All joint solves put s at its original lower bound of
zero. The goal of the next experiment is to test whether eliminating this tiny
boundary coordinate and gain/root coupling improves outer optimization, not to
claim that the fixed-endpoint solve has already materially improved the model.

Local evidence lives in
`output/construction-meal-joint-profile-delivery-20261104/preflight.json`.
No outer optimization was performed in that preflight.

## Diagnostic change

`profile_endpoint_probe.py` examines feasible outer directions at two small
step sizes, re-solving the same conditional inner problem. Units are frozen at
the endpoint using the optimizer's scaling rule; log coordinates apply to
positive rates/timescales. Fixed parameters and profiled variables are excluded.
This gives evidence of remaining coordinate descent, step-resolution limitations,
numerical failure, or no resolved coordinate descent. It never certifies joint
stationarity, global optimality, coefficient confidence or equation inadequacy.

The reporting resolution is not a rigorous numerical error bound. Curvature
alone can make finite-step slopes disagree, so step-resolution limitations are
separate from numerical integration failures. Attempts, actually started calls,
completed calls and failures are distinct; an expired budget cannot claim a
profile call that never executed. Interrupted diagnostics retain partial evidence
and cannot renew their reservations.

One real warm diagnostic completed 25/25 profiles in 51.6 wall seconds, with no
failed evaluations and no improvement over the conditional baseline. Eight
directions were step-sensitive. This does not establish convergence. Its saved
raw evidence precedes only a reporting-label/counter clarification; the final
implementation distinguishes that outcome as `step_resolution_limited`.

## Delta experiment

`construction-meal-joint-profile-1` authenticates source plan
`5c0b0ffd13b1f3256234f5587e899f3eaadd0088d14cb23d2ba1c3bb6ec13955`.
Ten tasks compare both arms from each of the five identical saved starts.
Each receives 1800 numerical search seconds, at most 2000 attempts, and 900
seconds of persistence headroom. Setup, qualification, original-equation replay
and a 180-second endpoint diagnostic have separately recorded costs. Total
reserved work is 5520 seconds/task under a four-hour Slurm ceiling.

One CPU/8 GB per task, concurrency two; no GPU or LLM. Source and destination
roots are separate. Global warm and constrained-rate incumbents are retained
independently; worse, invalid or interrupted new work cannot erase either.
Diagnostics cannot promote new points without independent verification.
Source/runtime identity, operation journals, scheduler receipts and reporting
remain immutable and deterministic. Source preparation/verification and empty
report/inspection smoke passed for all ten tasks without fitting/submission.

Commands and interpretation are in
`docs/PHASE_C_CONSTRUCTION_MEAL_JOINT_PROFILE.md`. No remote jobs were submitted
by the assistant. This remains training reconstruction with fitted training
initials; it supplies no new causal-prefix or intervention performance claim.

## Files and checks

New fitting profiler and endpoint diagnostic; opt-in root-profiler hook;
campaign/config; Delta CLI/worker; four focused test files; runbook and context
updates. All success/failure, domain/scope, finite-lift, retention, resume,
ambiguous-submission and reporting checks belong to this numerical milestone.
Repository-wide Ruff retains the existing 37 unrelated findings under
`analysis/claude`; changed implementation files pass scoped lint.
The final focused pytest run passes 225 tests. CLI-help with the documented
PYTHONPATH, shell syntax, production-source prepare/verify and report/inspection
smoke checks also pass. The portable bundle is built from the committed source,
with per-file checksums and a verified numerical source identity.
