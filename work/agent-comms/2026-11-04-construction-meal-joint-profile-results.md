# Returned joint meal/drift/root profiling results

4 November 2026. Review of the completed matched CPU experiment. No new fitting,
LLM calls, data changes or held-out evaluation were performed during this review.

## Authentication

- Archive: `/Users/yuanzhangxiao/.codex/attachments/3e88ab1a-e3ae-4ab1-ab70-2ae8fe7c9082/inspection.tar.gz`.
- Archive bytes: 1,457,414; SHA256:
  `3857b53dcd1c3a533d94bd79aa8f86d494f49a1c88e65842ddc8cf49e5a44998`.
- All 264 unique archive members were checked for safe relative paths and
  ordinary file/directory types before extraction.
- Protocol: `construction-meal-joint-profile-1`; implementation `bf32c4f`.
- Plan SHA256:
  `7ccff974fa6d1564f0ccf0c5b231e837c8e6abd2de68db3b6b4a33eb3d8d8c61`.
- Source identity matches the local committed implementation:
  `6aec0733ababd48506ec5c5c8a080447166701a6bfcafa0a72f7d7daf0d0491e`.
- Evidence directory:
  `output/construction-meal-joint-profile-review-20261104/`.

All ten terminals reconstruct from their eight completed operation receipts
each. Every summary row and paired initial-projection comparison reconstructs
exactly. The stored original-equation independent replay checks pass: maximum
loss discrepancy 3.27e-12, maximum prediction discrepancy 8.26e-10. These are
authenticated returned numerical checks, not fresh integrations in this review.
Zero test-data access and zero LLM calls remain recorded.

## Joint profiling is correct, but does not improve these warm fits

The table uses each arm's independently retained same-rate `trial_endpoint`,
not its unrestricted global incumbent. The four fixed families retain the warm
free-rate model globally; their top-level retained metrics must not be interpreted
as fixed-rate results. Tiny improvements below the comparison threshold preserve
the earlier endpoint, so raw search-best losses can differ in their last digits.

| Start | Root-only trial NMSE | Joint trial NMSE | Search active seconds, root/joint |
| --- | ---: | ---: | ---: |
| Warm | 0.423866885109 | 0.423866885096 | 184 / 200 |
| 0.001/T | 0.426489814386 | 0.426489814386 | 178 / 190 |
| 0.01/T | 0.426269576741 | 0.426269576741 | 172 / 179 |
| 0.1/T | 0.424601381729 | 0.424601381728 | 157 / 156 |
| 1/T | 0.457389537712 | 0.457389535549 | 158 / 472 |

Every matched pair is numerically tied under the declared tolerance. The largest
gap is 2.16e-9, smaller than the approximately 4e-9 comparison threshold.
The small gains over the inherited warm, fixed 0.1/T and fixed 1/T source points,
about 6.5e-8 to 7.3e-8, mostly come from putting the existing tiny positive drift
exactly at zero.
Joint conditional projection already obtains essentially all of this gain.

The global 34-variable bounded block is healthy. Across saved joint audits,
rank is 34/34; normalized KKT residuals are below 7.55e-13 and basis/full-rollout
discrepancies below 1.05e-13. The two shared coefficients are represented once
across all trajectories. This establishes conditional numerical reliability,
not joint identifiability or global optimality of the outer problem.

All ten searches stop by the optimizer criterion, with no exhausted budget,
invalid trial or failed evaluation. There are 1,971 search evaluations in total.
Nine runs stop after one Powell iteration; fixed-1 joint uses three. Root-only
optimizer terminals can reintroduce drift around 5e-9 and become worse than the
retained zero-drift checkpoint. Retention works correctly; joint profiling removes
that terminal artifact. Qualified optimizer stops still do not certify stationarity.

## Costs

Search numerical activity totals 2045.23 seconds; search persistence totals
12.37 seconds (0.60% of search wall time). The endpoint diagnostics use 248
successful calls and 272.35 summed operation wall seconds. Across all operations,
wall time is 2543.43 seconds, excluding queue/report time; overlapping historical
costs must not be added to this total. Per-arm new operation wall totals are
1088.61 seconds for root-only and 1454.81 seconds for joint profiling.

Joint profiling takes about 1.2 seconds per evaluation versus 0.9 seconds for
root-only. It often reduces the number of outer calls but does not improve
runtime in this experiment. Persistence is no longer the bottleneck. The extra
inner linear solve is useful for numerical control, not a demonstrated speedup.

## What the endpoint warning actually means

All 248 endpoint evaluations succeed. None of the sampled points improves its
profiled baseline. Of the feasible/blocked directions, 72 are `step_sensitive`,
47 show no resolved descent, and five are blocked by the drift lower bound.
All ten aggregate results are consequently `step_resolution_limited`.

The current one-sided slope consistency rule is deliberately conservative, but
ordinary curvature can trigger it near a perfectly smooth minimum. For a local
coordinate q and objective F,

```text
[F(q+h)-F(q)]/h = F'(q) + (h/2) F''(q) + O(h^2).
```

When F'(q) is small, halving h approximately halves this one-sided slope.
Even F(q)=q^2 at q=0 exhibits that behavior. It is not evidence that integration
or computed derivatives have failed.

Recombining the saved symmetric samples gives centered gradients and diagonal
curvature estimates without new evaluations. They are consistent with small
gradients and positive coordinate curvature. For the warm joint point, estimated
coordinatewise quadratic improvements are at most about 1.6e-11; for fixed-1
joint, below 9.8e-10. These lie below the declared loss-comparison resolution.
This is a diagnostic inference, not a certificate or a trustworthy prediction
along coupled directions. The retained calculations are in
`review-centered-differences.json` beside the extracted evidence.

Coordinate tests alone cannot rule out a coupled descent direction or another
basin. At nonzero residual, the positive-semidefinite Gauss--Newton matrix also
cannot by itself exclude negative curvature of the actual objective.

## Scientific status and recommended next milestone

The warm model still has Gp/I/U NMSE about 0.1932/0.7346/0.3438, respectively.
The meal-memory timescale remains about 1414.73 minutes against a 300-minute
training horizon. The fitted major coefficients are effectively unchanged.
The substantial insulin mismatch survives conditional gain/root optimization.
That is evidence of a persistent mismatch at these fitted endpoints, not proof
that the equation family cannot fit better.

Stop identical blanket warm continuations. The next bounded reliability
milestone should:

1. Use centered gradients and curvature-aware step refinement, with bound-aware
   alternatives and explicit active-set stability, to distinguish true descent
   from ordinary curvature and numerical resolution limits.
2. Inspect coupled outer directions using the profiled residual Jacobian and
   actual loss probes; permit a small verified correction only when an improvement
   exceeds the measured comparison resolution. Do not substitute a Gauss--Newton
   rank check for an actual-objective stationarity/curvature assessment.
3. Release k from the four saved fixed-rate endpoints. These provide predetermined
   different unrestricted starts without inventing ground-truth parameters.
   Preserve the warm incumbent and assign equal explicit budgets. The existing
   fixed-rate runs are constrained upper bounds, not these unrestricted restarts.

If that bounded check still finds no material improvement, report the persistent
training residual patterns to construction with the tested optimization scope
and unresolved global uncertainty. No theorem of model inadequacy or parameter
confidence percentage should be inferred. This review does not implement or
submit that next milestone.

## Verification and records

The focused joint-profiler and endpoint-probe suite passes 24 tests. Full-repo
Ruff still reports the same 37 unrelated existing findings under `analysis/claude`.
The review authentication and saved-sample calculations are local output
artifacts; only documentation is committed. All historical receipts are untouched.
