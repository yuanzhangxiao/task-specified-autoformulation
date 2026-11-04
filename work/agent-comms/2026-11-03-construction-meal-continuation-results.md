# Model-15 meal continuation: returned results

3 November 2026. Review only; no fitting code or historical results changed.

## Provenance and verification

- Received `/Users/yuanzhangxiao/Downloads/inspection (1).tar.gz`, 670,330 bytes.
- Archive SHA256: `be45007f787177222171ad536f907c93e93f1022a9e7d262c49710198fd1c20d`.
- Protocol: `construction-meal-continuation-1`.
- Plan SHA256: `f2bf759f4b34ff303fbdf05408cb1576bdd1d7b97b22c65d5db1a1b63f180434`.
- Implementation: `9ff72ba9eb27aa75f57fa8dbfd3294412b9dd611`.
- Local evidence: `output/construction-meal-continuation-review-20261103/`.

All 128 archive members were checked for safe unique paths and regular-file/
directory types before extraction. All six sealed terminals reconstruct exactly
from operation receipts and match the supplied summary. The frozen source
identity matches the current checkout. Every trial endpoint passes independent
original-equation replay; the maximum prediction discrepancy is 8.11e-10.
The local `review-authentication.json` records the reconciliation.

These are training reconstructions with fitted training initials, not new
causal-prefix forecasts. No validation/test data or LLM calls were used. All
stationarity, convergence and parameter-confidence certifications remain false.

## Results

T=300 is the maximum training horizon. The inherited endpoint has NMSE
0.4266471542730798.

| Control | Verified trial NMSE | Meal decay timescale 1/k | Termination |
| --- | ---: | ---: | --- |
| Free warm continuation | **0.423867303091** | **1421.66 = 4.74T** | Overall wall limit |
| Fixed 0.001/T | 0.426489875347 | 300,000 | Numerical wall limit |
| Fixed 0.01/T | 0.426269638539 | 30,000 | Qualified feasible optimizer return |
| Fixed 0.1/T | 0.425224094783 | 3,000 | Unqualified optimizer terminal |
| Fixed 1/T | 0.521512048455 | 300 | Unqualified optimizer terminal |
| Exact zero | 3.969897553175 | Not a positive-rate reservoir | Qualified feasible optimizer return |

The 1/T and zero trials retain the better inherited endpoint. Their trial losses
must remain separate from that retained score. A qualified feasible optimizer
return establishes a valid evaluated point, not stationarity or global optimality.

## A better fit no longer requires the previous nearly infinite memory

The warm continuation improves the source by **0.6516%**. More significantly,
it leaves the previous almost non-decaying meal-state regime without a fixed-rate
constraint. Its parameters are:

```text
k = 0.0007034052664225309
q = 45.36967193276078
g = k*q = 0.03191326617336642
1/k = 1421.6555487079715
```

Original meal initials A0 now range from about 1,381 to 3,587, compared with
roughly 4e10–9e10 previously. Scaled initials B0=k*A0 range from 0.97156 to
2.52282. The homogeneous meal-memory component decays by about 19.0% over
T=300, versus essentially zero at the previous endpoint. The fitted insulin
and insulin-action timescales are 229.57 and 32.33.

This establishes a better feasible finite-memory fit in the unchanged model
family. It does not identify physiological latent quantities or establish that
1422 is the correct timescale. The horizon remains shorter than this fitted
timescale and full numerical adequacy is not certified.

| Output | Source NMSE | Warm continuation NMSE |
| --- | ---: | ---: |
| Gp | 0.1990620521 | 0.1931963251 |
| I | 0.6912813615 | 0.7345345424 |
| U | 0.3895980492 | 0.3438710418 |

The aggregate improvement still trades better glucose/utilization reconstruction
for worse insulin reconstruction. The last four accepted objectives decrease
from 0.42387947 to 0.42387014, 0.42386806 and 0.42386730 before the wall limit.
Further numerical work may help; these results do not justify blaming the
remaining insulin mismatch on its equation form.

## Fixed-rate controls have a transformed-coordinate scaling defect

Every positive-rate start correctly preserves physical g=k*q=0.02589255 and
every B0=k*A0. However, its numerical optimizer scale is not preserved.
`MealCoordinates.units()` multiplies the old original-q unit (approximately
9.8071e8, inherited from the prior near-zero-rate fit) by the new rate.

| Control | Starting g | Optimizer unit for g |
| --- | ---: | ---: |
| Warm | 0.02589255 | 0.02589255 |
| 0.001/T | 0.02589255 | 3,269.04 |
| 0.01/T | 0.02589255 | 32,690.40 |
| 0.1/T | 0.02589255 | 326,904.04 |
| 1/T | 0.02589255 | 3,269,040.42 |

Thus physically comparable starts acquire gain steps roughly 1e5–1e8 times
larger. The 0.1/T and 1/T initial positive gain probes raise original q to
approximately 9.8e8, giving objectives of order 5.6e14 and 2.5e16. Each search
later accumulates 37 basis/full-rollout pairing failures and terminates after
149 attempts with no accepted Powell iteration. Their verified best g stays
exactly at its starting value. The precision checks correctly reject unreliable
trials; relaxing those checks would not fix the optimizer geometry.

The smaller-rate controls are also affected by this scaling, even though they
make more progress. For example, the 0.01/T terminal gain displacement is only
about 5.4e-9 in optimizer coordinates, close to the optimizer's coordinate
tolerance despite a material physical scale mismatch. Its raw success cannot
be treated as proof of an optimized profile minimum.

This is an implementation defect in the numerical scaling of reseeded controls,
not a change to their equations or a corrupt retained endpoint. The prior
three-attempt preflights exercised initial evaluation and the first coordinate
probes, but stopped before reaching the gain coordinate. They verified replay
and retention without covering the characteristic gain step. Future preflight
coverage must include the transformed gain and the full intended initial sweep.

## The termination fix correctly withholds false success

Both 0.1/T and 1/T report raw optimizer success at a nonfinite reported objective.
Their terminal original q is about 2.08e16. The new ledger classifies both as
unqualified and excludes their invalid callbacks from accepted histories.
Their finite verified best checkpoints survive. No retention corruption was
found, and no extra numerical calls were added by terminal qualification.

The 0.01/T and zero returns have valid paired terminal evaluations. The zero
run nevertheless demonstrates why this is weaker than finding the best fit:
the previous targeted campaign already has an independently verified zero-rate
endpoint at **2.652436316871**, below the new 3.969897553175. Both belong to the
same observed-output family; canonicalizing q/A0 at zero preserves those outputs.
The current run instead makes the insulin-action timescale approximately
5.46e33 and all its fitted initial roots zero. Its result cannot certify a
zero-rate minimum or prove that the corresponding family cannot fit better.

## Computation and persistence

| Control | Completed / attempted profiles | Valid / invalid callbacks | Numerical seconds | Persistence seconds |
| --- | ---: | ---: | ---: | ---: |
| Warm | 1692 / 1693 | 9 / 0 | 1505.38 | 414.62 |
| 0.001/T | 1844 / 1845 | 9 / 0 | 1800.00 | 92.54 |
| 0.01/T | 1632 / 1633 | 9 / 0 | 1479.55 | 37.09 |
| 0.1/T | 112 / 149 | 0 / 1 | 162.76 | 4.98 |
| 1/T | 112 / 149 | 0 / 1 | 152.53 | 4.55 |
| Zero | 1141 / 1141 | 7 / 0 | 1074.11 | 181.56 |

These are search-stage timings. Persistence is a subset of wall time, not an
additional charge. Other setup/verification costs remain separate in the report.
The warm run makes 426 writes, spending **21.59% of its wall time** in persistence
and reaching the 1920-second overall cap with 294.62 numerical seconds unused.
Callback CPU is only 3.37 seconds, consistent with persistence latency dominating
this overhead; this alone does not diagnose a site-wide filesystem outage.

## Recommended next milestone

1. Correct the transformed gain unit under reseeding. A certified source-chart
   unit or a directly defined effective-coordinate scale should remain comparable
   when g and B0 are preserved. Define and test a finite zero-g fallback; retain
   the original parameter domains and numerical precision gates.
2. Add scale-invariance and characteristic-step regression controls, plus
   preflights that actually exercise the gain direction. Recover the fixed
   positive-rate fits with corrected scaling before interpreting their profile.
   Preserve all current best endpoints and give new work separate allowances.
3. Continue the new 0.423867303091 warm endpoint with measured persistence
   overhead accounted for. Reducing forced writes would require an explicit
   bound on potentially lost sub-tolerance improvements; it must not silently
   weaken durable-incumbent guarantees. Extra wall headroom must also be explicit.

Keep exact zero distinct and use its known better feasible incumbent if it is
studied further. Do not launch another equation-revision recommendation from
these unfinished or poorly scaled searches. A later possible improvement is to
profile the shared meal forcing gain jointly with trajectory roots using one
across-trajectory bounded least-squares problem; that is not implemented here.

## Review validation

All six source/terminal reconciliations and summary comparisons are exact.
The focused meal-chart, root-profile, termination, campaign and launcher suite
passes **149 tests**. These existing tests do not cover the newly identified
reseed-scale invariance defect; that regression is part of the recommended fix.
Repository-wide Ruff still reports the same 37 unrelated existing findings
under `analysis/claude`. This review performs no refitting, model revisions,
remote submissions, new LLM calls or test-data access.
