# Saved construction parameters: independent output verification

This is a separate, opt-in follow-up to the returned eight-model ACES fitting
qualification. It does not change the original campaign, fitted coefficients,
initialization roots, prefix rule, benchmark, or selection policy.

## Which fitter actually ran

The returned source identity is
`5747e11830f6fa3c4672f74d0039d3b045a4a52fcd851f0fb4ec97f44b22fde0`.
It matches the repository's scientific source before this supplement, including
Orion's current scoped-initialization implementation. The later M47 material at
`fe78880` added qualification documentation/results, not a newer fitter that was
missing from this run.

The executed route was `construction_fit_qualification` -> `prefix_execution`
-> `prefix_initialization`: joint bounded fitting of shared coefficients/map
parameters and training-trajectory latent initials, analytic forward
sensitivities, guarded relative optimizer scaling, and independent DOP853/Radau
checks. The execution-accounting and multiple-root support were present.

This route is direct rollout fitting, with one start and a fixed eight-minute
validation prefix. It does not automatically enable all strategies developed in
other qualification routes: certified affine/terminal-output profiling,
collocation recovery, restart portfolios, or the M47 duration-selection rule.
Those policies require explicit qualification for this scoped-initializer
interface. Thus "current source" does not mean "every experimental fitting
strategy is active."

All eight constructions are T2-easy candidates, not eight different benchmarks.
Their 16 training trajectories and considerably larger parameter vectors differ
from the small known-equation controls used in recent fitting qualifications.
Successful controls do not establish convergence on these constructions.

## Why verify the saved endpoints

All eight searches exhausted their 600-second numerical-active allowance. Seven
then exhausted the independent 60-second verification allowance during the first
DOP853 pass with sensitivities; Radau was never reached. One candidate completed
both verification passes. This separates an unfinished verification from a poor
training fit, and from a claim that optimization has found the best possible fit.

The supplement takes each exact `search.best.parameters` dictionary, including
all fitted trajectory roots. It verifies membership in a completed historical
search evaluation and binds the original problem, data, numerical policy, and
operation receipts. It snapshots the public inputs and historical receipts byte
for byte into a new directory. Derived historical validation summaries may
reconstruct with relative/absolute roundoff tolerance `1e-12`; their saved values
are retained. Parameters, input bytes, receipts, and endpoint hashes are exact.

## New numerical work

Each candidate receives two state/output-only passes, DOP853 and Radau, over all
16 training trajectories. No sensitivity integration, rank calculation,
optimization, new validation-initial-state inference, LLM call, test access,
selection, or promotion occurs. The solvers retain `rtol=1e-9`, `atol=1e-11`, and
the original normalized-residual agreement tolerance `1e-6`.

A new, explicitly recorded allowance is 300 numerical-active seconds and 360
inclusive wall seconds per candidate. This is a supplemental budget, not a
retrospective reinterpretation of the original 60-second allowance. Hardware and
the changed verification workload prevent a clean speedup claim from timing
alone. Per-trajectory method, wall/CPU time, and available solver counters are
retained to support profiling.

Only agreeing finite trajectories receive training NMSE and per-target scores.
All eight endpoints remain in the report, including failed verification. The
summary reports median and unscaled MAD with the available/total denominator.
Rank is explicitly unassessed. Old validation scores remain labeled historical;
verification does not invent the seven missing validation scores.

## Checkpointing, execution, and inspection

`construction_fit_verification.py` owns the numerical check.
`construction_verification_campaign.py` owns frozen source provenance and durable
operation reservations. Once an operation is reserved, an interrupted resume
cannot renew its allowance or start it again. Partial reports include reserved
budget and observed progress, with unknown totals left unknown.

`verify_construction_parameters.py` provides `prepare`, `run`, `report`, `inspect`,
and CPU-only ACES submission. The launcher submits an eight-task array with at
most four concurrent single-CPU tasks and an `afterany` report. Unconfirmed Slurm
replies retain their intents; the launcher cannot silently resubmit them.

The portable delivery contains the inspected source snapshot. Use its
`run_verify.sh run` wrapper in a short ACES CPU allocation. The default output is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-construction-verification-v1`.
After completion, `run_verify.sh inspect` writes `inspection.tar.gz`; the readable
table is `SUMMARY.md`, and machine-readable results are `report/summary.json`.

## Fitting work to qualify separately

The first local saved-endpoint run completed all eight operations: seven passed
independent output agreement; topology seed 2 / proposal seed 0 exhausted 300
seconds in DOP853 after 11 of 16 trajectories (the twelfth started). Radau did not
start for that candidate. The largest observed normalized-residual solver
difference among passing candidates is `4.462e-7`, below the frozen `1e-6` limit.

Verified training NMSE has median **1.802080**, unscaled MAD **0.749723** (7/8
available); all seven agree closely with their retained search errors. Insulin
training NMSE remains approximately 1.79–1.85 across those seven candidates.
Verification recovers available evidence; it does not improve fit quality.
The new operations used 1,102.801 CPU seconds and 1,107.762 summed operation-wall
seconds, with zero optimization/LLM calls. They were executed in two local CPU
processes, not on ACES; these wall seconds are not campaign elapsed time. The
executed source and its plan/runtime hashes are preserved alongside the local
report at `output/saved-parameter-verification-20261021/REPORT.html`. Subsequent
source hardening checks historical numerical-policy equality without changing
the numerical verifier or these saved endpoints.

See `PHASE_C_CONSTRUCTION_FITTING_HANDOFF.md`.
Priorities are independent output qualification, cost-aware sensitivity/rank
diagnostics, exact redundant-coordinate reduction with original constraint
preservation, and a controlled comparison of optimization strategies on the
same frozen constructions. A larger budget or more starts should follow that
profiling rather than obscure the current failure mode.
