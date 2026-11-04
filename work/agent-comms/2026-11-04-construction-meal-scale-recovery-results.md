# Meal-scale recovery: results and next numerical milestone

4 November 2026. Review of returned results only; no new fitting campaign,
model changes, LLM calls or held-out evaluation.

## Provenance

- Archive: `/Users/yuanzhangxiao/.codex/attachments/45387f5f-85bd-4c11-a149-5eada68a159b/inspection.tar.gz`.
- Bytes: 877,534; SHA256:
  `940fb8165a4b81587107d924ed098f2afaa69907acc50ec702c61d50982ba633`.
- All 124 unique members were checked for relative safe paths and regular
  file/directory types before extraction.
- Protocol: `construction-meal-scale-recovery-1`.
- Plan SHA256: `5c0b0ffd13b1f3256234f5587e899f3eaadd0088d14cb23d2ba1c3bb6ec13955`.
- Source identity matches implementation commit `8d027c72b6543f3e9ca249780404f6fe75b60327`.
- Evidence: `output/construction-meal-scale-recovery-review-20261104/`.

All five terminals reconstruct exactly from the seven operation receipts per
task. Every supplied summary row equals its reconstructed row. The saved
independent original-equation replays pass: maximum prediction difference
8.26e-10, loss difference below 3.27e-12. These are authenticated stored numerical
checks, not fresh integrations performed during this review. All historical
source and new endpoint/retention identities remain intact.

## Fit results

T=300 is the maximum public training horizon. Compare fixed-rate trial errors,
not retained global errors, which may still reference the free-rate incumbent.

| Trial | Source NMSE | New verified NMSE | Relative improvement | Stop |
| --- | ---: | ---: | ---: | --- |
| Free-rate warm continuation | 0.423867303091 | **0.423866958337** | 0.0000813% | Numerical limit |
| 0.001/T | 0.426489875347 | 0.426489814386 | 0.0000143% | Qualified optimizer stop |
| 0.01/T | 0.426269638539 | 0.426269576741 | 0.0000145% | Qualified optimizer stop |
| 0.1/T | 0.425224094783 | **0.424601448721** | 0.1464% | Qualified optimizer stop |
| 1/T | 0.521512048455 | **0.457389602578** | 12.2955% | Numerical limit |

`search_no_meaningful_improvement` for the fixed-rate tasks means that they did
not beat the better inherited **free-rate** incumbent. It does not mean that
their own constrained fit failed to improve: the 1/T trial improved substantially.
The report's separate trial endpoints and comparisons resolve this distinction.

The correction worked numerically. Every initial sweep completed; there were
zero invalid trials and zero basis/full-rollout pairing failures. Only the warm
and 1/T runs lost one final evaluation to the numerical deadline. Gain units are
0.02543–0.03191, rather than the earlier 3,269–3,269,040. Both formerly blocked
fixed-rate searches now progress normally. Nevertheless, this recovery also
changes starts and time accounting, so the exact 12.3% improvement is not an
isolated causal estimate of the scale change alone.

## Warm fit is effectively plateauing

The warm fit gains only 3.45e-7 NMSE after another 1,800 numerical seconds.
Its parameters remain in the same regime:

- Meal rate k=0.000706847394249; timescale 1/k=1414.73=4.72T.
- Meal gain q=45.19372; effective gain g=kq=0.03194506.
- Meal initials A0 span 1369.55–3572.97; B0=kA0 spans 0.9681–2.5255.
- Insulin and insulin-action timescales are 228.95 and 32.33.
- The unforced meal component decays about 19.1% over T, close to the source fit.

The fixed 0.1/T fit (timescale 10T) is only about 0.17% worse than the best warm
fit. Almost non-decaying fits are about 0.6% worse. This is a shallow set of
attained errors, not a certified profile minimum or confidence interval for k.
No fixed rate can be excluded from an unfinished nuisance fit.

| Trial | Gp NMSE | I NMSE | U NMSE |
| --- | ---: | ---: | ---: |
| Warm | 0.193199 | 0.734590 | 0.343811 |
| 0.1/T | 0.195598 | 0.714593 | 0.363614 |
| 1/T | 0.209992 | 0.874135 | 0.288041 |

The faster meal control improves disposal at the cost of insulin reconstruction.
The overall error remains substantial. These training reconstructions use fitted
training initials; they do not establish causal-prefix or intervention accuracy.

## Inner solves are reliable; outer convergence is not certified

All retained per-trajectory root solves have rank 2/2, no zero columns, and
normalized KKT residuals at most 6.11e-14 against approximately 4.4e-9 tolerances.
Their retained basis/full-rollout discrepancies are below 7.88e-14. This supports
conditional initial-root optimization, not joint parameter identification.

The two smallest-rate controls stop after only one accepted Powell iteration
(208 and 200 evaluations). Their retained improvement is essentially the
nonnegative auxiliary drift coefficient `k_Uii_term_12_0_244d72574a` moving from
about 5.3e-9 to exactly zero, followed by root profiling. Their optimizer terminals
move that coefficient back near 5.3e-9 and are about 6.1e-8 worse than the retained
best. Retention is correct: a feasible optimizer terminal does not erase the
better evaluated checkpoint.

This drift coordinate still has numerical unit 1 while its fitted value is
near 1e-9 and Powell's coordinate tolerance is 1e-8. The stop therefore does not
supply a strong bound-aware stationarity certificate. The other qualified
optimizer stop is 0.1/T after 11 iterations. Warm/1T have 8/12 valid callbacks
and stop on numerical limits. All five explicitly withhold stationarity,
convergence and parameter-confidence certification.

## Persistence is no longer the bottleneck

| Trial | Completed / attempted profiles | Numerical seconds | Persistence seconds | Writes |
| --- | ---: | ---: | ---: | ---: |
| Warm | 1934/1935 | 1800.00 | 39.14 | 258 |
| 0.001/T | 208/208 | 195.45 | 1.28 | 11 |
| 0.01/T | 200/200 | 187.54 | 0.47 | 11 |
| 0.1/T | 1861/1861 | 1791.05 | 51.70 | 333 |
| 1/T | 1931/1932 | 1800.02 | 29.51 | 440 |

Persistence takes 122.09 of 5896.16 search wall seconds (2.07%). The warm run
uses its complete numerical allowance; its old 415-second persistence blockage
is absent. Headroom, write counts and filesystem latency all changed, so the
entire reduction must not be attributed to headroom alone. New operation totals
are 6013.40 summed wall seconds and 5830.25 CPU seconds, excluding queue/report
and final sealing. Callback time is nested within these totals, not additive.
Historical overlapping source costs remain separate.

## Recommended next milestone: certify and solve a larger linear block

Do not spend another identical warm continuation or infer equation inadequacy
from a plateau alone. At fixed remaining dynamics, the saved equations support
exact joint profiling of **shared meal gain g, shared auxiliary drift s, and all
32 training initial roots**. The relevant equivalent equations are

```text
B' = g*r - k*B
G' = r - C - X + B
I' = r + a*G - lambda*I + b*u
X' = c*I - d*X
C' = s
```

For fixed eta=(k,a,lambda,b,c,d), weighted observed residuals are affine in the
shared g/s and trajectory roots z_i=(B_i(0),X_i(0)):

```text
residual_i = offset_i(eta) + h_i(eta)*g + ell_i(eta)*s + D_i(eta)*z_i.
```

Stack all trajectories into one bounded least-squares problem. There are 14,448
scalar observations and **34 fitted linear coordinates**: two common coefficients
plus 32 local initials. This differs from independently fitting a gain for every
trajectory, which would change the model. The positive chart has finite original
lift q=g/k and A_i(0)=B_i(0)/k; exact zero remains separate. Bounds and existing
initializer/readout dependencies must be included in the certificate.

First test this at the five frozen endpoints: certify the affine dependence,
check rank/active bounds and KKT, compare against root-only projection at identical
remaining coefficients, and replay in the original equations. No worse verified
endpoint may replace an incumbent. Then test bounded outer optimization and
bound-aware stationarity checks on the remaining coordinates. Shared insulin
forcing b is another potential linear coefficient but can remain fixed in this
first block pilot. The outer problem remains nonconvex; an inner convex optimum
is not a global model-fit optimum or a parameter-confidence percentage.

This is a concrete way to remove the newly observed tiny-boundary direction and
the gain/root coupling before deciding whether the proposer should revise the
function forms. No implementation of this proposed milestone is included here.

## Review checks

The focused meal-coordinate, root-profiling and termination suite passes
**90 tests**. Repository-wide Ruff still reports the same 37 unrelated existing
findings under `analysis/claude`. This review changes only this Markdown report.
