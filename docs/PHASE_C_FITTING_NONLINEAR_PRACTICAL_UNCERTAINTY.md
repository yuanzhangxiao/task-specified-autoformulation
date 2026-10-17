# M41: practical uncertainty with a nonlinear nuisance rate

## Purpose and scope

M39/M40 qualified parameter-support calculations when every unknown entered the
predicted readings linearly. M41 keeps a known equation structure and adds an
unknown decay rate. It asks whether a broad gain profile reflects limited data
information or unsuccessful optimization of the other parameters.

This is an isolated synthetic qualification, not a benchmark result or a change
to production fitting, construction, benchmark prompts or datasets. No LLM calls,
GPUs, validation selection or test data are involved. The ODE is linear in its
state, but its integrated response is nonlinear in the unknown decay rate. A
fully nonlinear state equation remains a later qualification.

## Frozen experiment

The physical and observation equations are

\[
\dot x=-k x+g u,\qquad x(0)=x_0,\qquad y=x+b t+\epsilon,
\qquad\epsilon\sim\mathcal N(0,\sigma^2).
\]

All three acquisitions share the unknown gain, decay rate, preparation and
sensor drift. Their constant inputs are `1-delta`, `1`, and `1+delta`. Each has
17 readings from time 0 to 4. The raw acquisition, calibration, clock conversion,
known noise scale and preparation provenance are frozen before fitting.

- Input separation: `0.5`, `0.02`, `0`.
- Known noise standard deviation: `0.01`, `0.2`.
- Three paired noise seeds, shared by the direct and profiled fitting methods.
- Eighteen acquisitions and **36 fitting/diagnostic tasks**.
- Domains: `g in [0.05,3]`, `k in [0.02,1.5]`, `x0 in [0.01,2]`,
  `b in [-2,2]`.
- Generic start: `g=0.45`, `k=1.1`, `x0=0.35`, `b=0`, with two predetermined
  restarts under one total search allowance.

The generator uses `g=1.3`, `k=0.65`, `x0=0.4`, `b=0.06`. These values and clean
trajectories are evaluator-only records. They are never fitter inputs, restart
choices, stopping criteria or profile-support thresholds.

Unlike M39, identical inputs need not produce exact confounding. The response is

\[
y(t)=x_0e^{-kt}+\frac{g u}{k}(1-e^{-kt})+bt.
\]

Its curvature can separate the decay rate, gain and drift. At the generating
point the joint sensitivity has rank four at all three separations. The two
smallest separations may therefore yield similar information. If the trajectory
starts at equilibrium, however, the exponential transient disappears and the
ratio `g/k` alone does not identify those two coefficients. The experiment must
report what the data support rather than assume that zero separation is always
unidentifiable.

## Fitting and diagnostic sequence

1. Run the existing qualified measurement fitter with direct rollout, or with
   eligible affine gain/initial/drift profiling and an outer search over `k`.
   Both methods retain guarded coordinates, fixed restart budgets, useful
   checkpoints and independent rollout verification.
2. Seal the fitted endpoint before the uncertainty diagnostic. A diagnostic that
   finds a better solution reports that finding without silently replacing the
   endpoint or claiming the original optimization was solved.
3. Profile gain: fix `g`, reoptimize `k`, `x0` and `b`. At fixed `(g,k)`, solve
   the bounded weighted least-squares problem for `(x0,b)`. Compare against a
   slice that freezes all three nuisance parameters.
4. Check the nonlinear nuisance search against a separate dense one-dimensional
   scan of the declared `k` domain, refining detected basins and including the
   boundaries. Check the integrated endpoint against the exact exponential
   response. Agreement is numerical evidence, not proof of global optimality.
5. Only after the fit and diagnostic are frozen, evaluate coefficient error and
   clean physical/reading error against the synthetic generating records.

The loss is half the sum of squared residuals divided by known variances.
Objective-rise thresholds `0.5` and `2` are diagnostic choices. The corresponding
support is the set of gain values whose reoptimized loss lies within that rise
of the retained loss, conditional on successful endpoint and search checks.
These ranges are **not calibrated confidence intervals**. Sparse profile grids,
parameter-bound clipping and unresolved nuisance searches must stay visible.
No single interval is assumed to capture a potentially disconnected support set.

This first nonlinear milestone reports **sampled support and crossing brackets**,
not the continuous support widths produced by M39/M40's affine calculation.
`width` remains unavailable. The span of accepted samples is a resolution-limited
quantity: a single accepted sample does not mean zero uncertainty. The global
gain grid is supplemented with local sensitivity-based samples; that sensitivity
is a sampling heuristic, not an assumption that the whole profile is quadratic.
Boundaries between samples and any missed narrow components remain unresolved.

## Accounting and continuation

The fitting allowance is 60 search seconds, 15 independent-verification seconds
and 200 optimizer calls, shared across the initial start and two restarts.
Diagnostics have a separate 60-second, 15,000-evaluation allowance with bounded
per-operation reservations and record primary/reference work,
CPU time, elapsed time and persistence overhead. The `numerical_*` subcategory
times evaluated responses and bounded affine solves, not all optimizer driver,
oracle-construction or curvature work. Use total diagnostic time for overall
overhead, and do not add the nested subcategories to that total. Slurm wall time also includes
imports, provenance verification and artifact writes. It is not a target fitting
duration.

Plan, raw inputs, runtime, source identity and per-task artifacts are immutable.
Completed work is reused on resume. An interrupted operation or exhausted budget
must remain visible; restarting the command does not erase charged work. Slurm
receipts are journaled before dependency submission, and uncertain submission
outcomes are not automatically resubmitted.

Summary output is deliberately compact: detailed node records and raw evidence
remain in the per-task results included in the review archive.

## Delta execution

Use a fresh pinned code directory and a fresh output root. Existing M39/M40
results are not needed: this is a new acquisition with a new model structure.
The delivered archive is built from a committed tree and includes `SOURCE_COMMIT`
and `SHA256SUMS`.

```bash
cd "$AF_CODE"
export AF_NONLINEAR_ROOT=/work/hdd/bibo/yxiao2/phase_c/nonlinear-practical-uncertainty-v1
export AF_CONCURRENCY=2
bash scripts/hpc/submit_phase_c_nonlinear_practical_delta.sh
```

The wrapper prints the code/output paths and preparation milestones before
submission. Default resources are one CPU, 16 GB and a 15-minute scheduler limit
per array task, with two concurrent tasks; no GPU queue is involved. The actual
optimizer and diagnostic budgets are much shorter than the scheduler limit.

After completion, use the same code directory:

```bash
bash scripts/hpc/inspect_phase_c_nonlinear_practical_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/nonlinear-practical-uncertainty-v1
```

This prints task/diagnostic status and creates a timestamped review archive.
Missing or failed tasks remain missing or failed in the report.

## Exit criteria

- Direct/profiled fitting uses the same frozen acquisitions and allowances.
- Independently evaluated objective values agree within the stated tolerance,
  or the disagreement explicitly blocks a stronger parameter-support conclusion.
- Nonlinear nuisance-search failure cannot masquerade as narrow uncertainty.
- Boundary limitation, disconnected/sparsely resolved support and competing
  nuisance solutions are retained without claiming unique physical recovery.
- Raw data and fitted endpoints are preserved, with deterministic resume and
  separate diagnostic cost accounting.
- No probability coverage, automatic model-versus-data attribution or general
  nonlinear global-optimality claim is introduced.

## Local verification, 17 October 2026

The M41 suite passed 69 tests. After adding a separate interruption-reservation
reporting check, all 17 campaign tests passed (70 distinct M41 tests in total).
Another 103 existing measurement-fit, sensor-drift and affine-profile regression
tests passed. Scoped Ruff, shell syntax and whitespace checks pass.

Four end-to-end smoke tasks exercised both fitting methods at high separation/
low noise and zero separation/high noise. All four endpoints and nonlinear
profiles qualified numerically. Following the final reporting/checkpoint edits,
the direct high-separation and profiled zero-separation cases were rerun from
fresh roots with the final source; both completed. Terminal resume preserved
all result/journal bytes and modification times, and repeated reports agreed.
The incomplete smoke roster correctly reports missing tasks and unknown full
campaign totals. Smoke artifacts are uncommitted under
`output/phase-c-fitting-m41-verification/`.

Whole-checkout Ruff reports 37 existing findings in unrelated `analysis/claude`
files. A full pytest attempt was bounded at 180 seconds and did not complete;
it recorded setup errors outside the focused M41 suite during concurrent work.
A first-failure follow-up passed 246 tests and skipped three before being stopped
at 131 seconds without reproducing those setup errors. The repository-wide
suite is not claimed to pass. These unrelated files were not changed.

Delta results are pending; local smoke tests are not a substitute for the frozen
36-task campaign. Continuous profile-width certification and calibrated
probability coverage remain unclaimed.
