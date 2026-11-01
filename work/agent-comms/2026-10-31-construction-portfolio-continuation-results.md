# Construction portfolio continuation: returned evidence and report repair

31 October 2026. This review performs no new fitting, LLM calls or test-data
access. It concerns the five-model follow-up to the 23-model construction
portfolio, using unchanged equations, domains and training/prefix contracts.

## Provenance and report failure

- Archive: `construction-portfolio-continuation-raw.tar.gz`.
- SHA256: `8d90f4f3bebc48cde507806861f86eba548cecdd6c8bb8ecbb36c9a6b94ca538`.
- Delta root: `/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-continuation-v1`.
- Code: `d940830409e1ca5572a7a930ea4a250f4effc0e5`.
- Plan: `e92e3bf12f64431d6db74c812f849396747ed98bcb0a30ac02a60a47b3491ea1`.
- Jobs: run `23376624`, report `23376625`.
- Local extraction: `output/construction-portfolio-continuation-review-20261031`.

All five terminals authenticate against their operation reservations, progress
files, nested probes and receipt hashes. Before the reporting correction, four
terminals reproduce exactly. Candidate 03 differs only in
`cost.callback_cpu_seconds_observed` and `cost.callback_cpu_seconds_total`:
`34.210724816999615` versus `34.21072481699962`, about `7.1e-15` seconds.
All scientific fields, statuses and operation hashes reproduce exactly.

The worker accumulates timings in execution order; the reader loads operation
directories alphabetically. Ordinary floating-point addition changes its final
bit with that order. The strict terminal comparison therefore rejected intact
results. The actual run used Python 3.12.13; this defect does not require an older
Python version because the callback accumulator uses repeated `+=`.

The correction is limited to this campaign's reporting. Derived timing fields
now use sorted operands and `math.fsum`. Historical timing discrepancies may
be reconciled only within an addition-roundoff bound calculated from the same
authenticated, nonnegative timing operands. Original terminals are retained.
The report records both terminal hashes and every reconciled field/value/bound.
All other fields, including metrics, parameters, statuses, counters and unknown
totals, remain exact. There is no numerical tolerance on scientific evidence.
Other historical campaigns retain their existing behavior.

On the returned archive, the canonical sums reproduce **all five original
terminals exactly**. No roundoff allowance is needed to accept these particular
terminals. The corrected report completes in about 105 seconds locally and
preserves every original terminal file hash.

## Numerical outcomes

| Candidate | Derivative qualification | Retained training NMSE before → after | Outcome |
| --- | --- | --- | --- |
| 03 | 24 passed, 4 inconclusive, all 28 examined | No new fit | Not yet search-qualified |
| 10 | 29/29 passed | No shared search in this milestone | Search-qualified |
| 14 | 29/29 passed | No new fit | Search-qualified |
| 12 | 41/41 passed | 0.564509061292 → 0.564509058786 | Negligible improvement; optimization unresolved |
| 15 | 40/40 passed | 0.564608325105 → 0.564608324964 | Negligible improvement; optimization unresolved |

The shared prerequisite repair worked. All nonlinear checks reached actual
state-only perturbations, with no failed verification calls or prerequisite
budget failures. Candidates 03/10/14 respectively completed 219/119/119 new
verification calls, including one shared Jacobian per model. Their operation
wall totals were approximately 4,028/1,787/2,252 seconds. These are call and
operation totals, not end-to-end queue times or independent trajectory counts.

Candidate 03's unresolved directions are indices 8, 10, 19 and 22:
`k_insulin_term_7_0_92f124a75e`, `k_Vmax`, and the insulin-action initial roots
for training trajectories 007 and 010. The first has derivative magnitude near
the finite-difference roundoff estimate. In the other directions, shrinking the
step often increases the discrepancy. These are inconclusive precision checks,
not demonstrated equation defects or evidence of nonidentifiability.

Models 12/15 completed bounded searches in approximately 272/233 seconds,
stopping on `xtol` without exhausting their 1,800-second allowances. Their fresh
prefix forecast NMSEs remained about 3.145 and 3.123 (slightly worse in the final
digits). Consequently, the tiny reconstruction improvements do not establish a
useful predictive gain. These are training diagnostics with shared coefficients
fitted on all training observations, not independent validation estimates.

## Why models 12 and 15 stagnate

The saved equations expose a fast-state limit. For the stable fitted insulin
and action blocks, write a balance as

`dz/dt = input_terms - decay*z = (equilibrium_input - z)/tau`,

where `tau = 1/decay` and each equilibrium input coefficient is its original
source coefficient divided by `decay`. This is an interpretation of these saved
stable points, not a global restriction on the declared model family.

| Candidate | Insulin timescale before → after | Action timescale before → after |
| --- | --- | --- |
| 12 | 8.93e-7 → 3.27e-11 | 4.80e-7 → 6.87e-11 |
| 15 | 5.34e-8 → 4.10e-11 | 1.09e-8 → 2.28e-11 |

The insulin/glucose equilibrium gain remains approximately 0.5762/0.5757 and
the action/insulin gain approximately 0.02221/0.02214. Thus source and decay
coefficients move together by thousands-fold while those ratios and observed
predictions barely move. The fit approaches an effectively instantaneous response
at the sampled times. This suggests a weakly informed fast limit. It is not a
new proof of exact redundancy, structural nonidentifiability or wrong equations.

Nominal projected gradients are small, but the corresponding uncertainty upper
bounds are approximately `5.82e-5` and `2.32e-4`, above the `1e-8` convergence
criterion. Also, the retained best points differ from the optimizer's last
accepted points. The searches saved 53/47 accepted-history points, but those
histories cannot certify progress at a different retained endpoint. This is
distinct from missing history logging. Both assessments correctly remain
`optimization_unresolved`.

## Recommended next milestone

1. Run bounded searches for the now-qualified 10/14 starts, retaining independent
   original-equation verification and checkpoint preservation.
2. Recheck only 03's four inconclusive directions with targeted precision/noise
   diagnostics. Do not enlarge perturbations beyond the local validity policy
   or interpret mere budget completion as derivative agreement.
3. For 12/15, investigate an equivalent time-constant/equilibrium-gain chart and
   profile the fast timescales. Preserve the original domains, account explicitly
   for zero/unstable branches, and make no arbitrary positive lower-time bound.
   Another identical warm continuation is unlikely to address the observed drift.
4. Independently verify and diagnose both the retained best point and the last
   accepted point, each with its own history. Preserve the best fit; do not attach
   another point's stopping evidence to it.

The reporting fix is implemented now. The next numerical campaign is a separate
milestone; no new searches were launched in this review. Parameter uncertainty,
optimization reliability and scientific mechanism validity remain distinct.

## Verification

The affected tests pass: 115 tests covering canonical timing, historical
roundoff, changed scientific fields, receipt tampering, interrupted unknown
totals, report reconstruction, immutable terminals and existing campaign/launcher
behavior. Changed files pass Ruff. Whole-repository Ruff reports 37 pre-existing
findings in unrelated analysis files. The complete repository pytest suite was
not run. The returned production receipts are also checked directly; corrected
summary and reconciliation evidence are retained beside the local extraction.
