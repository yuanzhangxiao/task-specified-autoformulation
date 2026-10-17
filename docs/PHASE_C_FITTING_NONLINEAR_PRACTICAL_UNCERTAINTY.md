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
- Eighteen acquisition sets (three input levels each) and **36 fitting/diagnostic tasks**.
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

## Delta result review, 17 October 2026

Returned archive: `review-20261017-224009.tar.gz`, SHA-256
`9fdbf6d22f26122b10e0b39c67105ff5102197405aa43068979bc00d827952f3`.
Frozen plan: `04bc18eab8f78a3b2b8f07d91c6d4bff1ecb5d1cb54d018b2cf31bb0b02285de`.
The local review verified 939 top-level sealed JSON records, reproduced the
summary exactly from the saved stages, and preserved all 1,675 extracted file
contents. Review artifacts are uncommitted under
`output/phase-c-nonlinear-practical-m41-delta-review/`; the separate read-only
audit is `output/phase-c-nonlinear-practical-m41-review-audit.json`.

All **36 fitted endpoints** pass the independent rollout and numerical endpoint
qualification. All **18 paired objectives** agree, with maximum absolute
difference `5.97e-13`; the largest paired parameter difference is `5.34e-8`.
No diagnostic found a better endpoint beyond tolerance. All local joint
sensitivities have rank four. These are finite numerical checks, not global
optimality, coefficient-accuracy or confidence-probability certificates.

### Recovery and practical uncertainty

Below are direct-arm medians over three noise seeds. Profiled fitting gives
essentially the same endpoints. Parameter errors are **absolute**, calculated
after sealing the fit/diagnostic; the compact summary's grouped parameter-error
medians are signed and are not used as accuracy statistics here. Reading RMSE
compares the complete sensor prediction `x+b*t` with clean generating readings;
physical RMSE compares the inferred `x` with the generating physical trajectory.
Neither clean trajectory is available during fitting or diagnostics.

| Input separation | Noise sigma | Gain error | Rate error | Drift error | Reading RMSE | Physical RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.01 | 0.00242 | 0.00412 | 0.00208 | 0.00302 | 0.00455 |
| 0.02 | 0.01 | 0.00502 | 0.01420 | 0.01226 | 0.00291 | 0.02651 |
| 0 | 0.01 | 0.00607 | 0.02675 | 0.01934 | 0.00316 | 0.04310 |
| 0.5 | 0.2 | 0.05403 | 0.07692 | 0.04073 | 0.05947 | 0.08851 |
| 0.02 | 0.2 | 0.17949 | 0.27931 | 0.20465 | 0.05748 | 0.43501 |
| 0 | 0.2 | 0.52509 | 0.39938 | 0.76218 | 0.05469 | 1.74172 |

At fixed noise, the measured signal remains comparably predictable while the
inferred physical trajectory becomes much less reliable. In
`d0_sigma0p2_s2__direct`, the retained solution has gain `2.90842`, rate
`0.0843224` and drift `-2` at its lower bound, compared with generating values
`1.3`, `0.65`, and `0.06`. Its clean reading RMSE is `0.05259`, but its clean
physical RMSE is `4.85108`. Drift and physical dynamics compensate in the
measured signal. Reaching the same optimum with two algorithms does not remove
that ambiguity.

The training-only profile exposes the problem without consulting truth:
at objective rise 2, this example accepts sampled gain values from `0.935` to
the upper domain limit `3`, with explicit clipping and active nuisance bounds.
Across the three zero-separation seeds, median accepted-sample span grows from
`0.03502` at noise `0.01` to `1.96667` at noise `0.2`. Every continuous `width`
remains null: these are sampled extents, not certified interval endpoints or
calibrated confidence intervals. Four acquisitions/eight arms have upper-gain
domain clipping at rise 2. Frozen-nuisance slices sometimes accept just one
sample; zero sampled span must not be described as zero uncertainty.

### Two partial diagnostics: reference search, not fitting failure

**34/36 profiles complete**. Both partials are acquisition
`d0p5_sigma0p01_s1`, one per fitting method. Each resolves 39 of 40 gain nodes.
There are 1,400 qualified nodes and two unresolved nodes across the campaign.
No call failed or was interrupted, and these partials did not exhaust their
budgets.

At gain `2.3116666667`, the multistart nuisance optimizer finds rate
`1.4953772541` with objective `3902.6138770`; the reference scan returns the
upper rate bound `1.5` with objective `3903.1646077`. Its grid descends all the
way to the boundary, but a minimum lies inside the last grid interval. The
reference currently refines sampled interior basins and does not refine the
boundary-adjacent intervals. Checking the boundary alone misses this minimum.

A separate cheap scalar check using saved training inputs, an independently
assembled exponential basis and unscaled least squares finds rate
`1.4953772949`, objective `3902.6138770`, agreeing with the primary result to
`4.3e-11`. This used 12 scalar evaluations including bracket checks, without
rerunning a fit or changing saved artifacts. The problematic node has objective
rise about `3874`, far outside the support thresholds `0.5` and `2`. It does not
invalidate the fitted endpoint; the formal support summaries are nevertheless
withheld under the current all-nodes-qualified rule.

### Measured overhead and next step

| Stage | CPU seconds | Summed task wall seconds |
| --- | ---: | ---: |
| Fitting | 130.599 | 379.293 |
| Profiles | 189.221 | 444.513 |
| Evaluator-only scoring | 0.071 | 1.732 |

Profiles add **144.9% of fitting CPU time** in this small control. Persistence
callbacks consume `203.192` wall seconds, or `45.7%` of diagnostic wall time.
The evaluated numerical-work subcategory consumes `95.951` wall seconds; both
are nested in the total, which also includes other diagnostic work. Summed task
wall times are not campaign elapsed time. Profiled fitting uses `60.704` CPU
seconds versus `69.895` for direct fitting (13.2% less here), without a general
speedup claim. Uninstrumented inner-solver counters remain unknown: compact
observed zeros must not be read as measured zero work.

The next focused change should refine both boundary-adjacent reference cells,
test lower/upper near-boundary and exact-boundary optima, then supplement only
the two partial diagnostics with a new source identity and separately recorded
costs. Preserve all fitted endpoints and original evidence. Increasing the
wall budget alone cannot repair this search omission. Subsequently refine the
objective-threshold crossings adaptively and reduce persistence overhead before
qualifying larger nonlinear nuisance blocks. The current result supports
training-only ambiguity feedback, not automatic model/data attribution, a
confidence percentage, or a claim that general nonlinear fitting is solved.

This documentation-only review reran all 70 M41 tests successfully and passed
scoped Ruff and whitespace checks. Reconstructing the full returned report was
the relevant artifact smoke check and changed no source artifact bytes. Whole
checkout Ruff still reports the same 37 unrelated `analysis/claude` findings;
the full repository pytest suite was not rerun for this documentation update.
