# Phase C M5: formulation reuse and screened primal starts

Reviewed 2026-10-02 (Pacific/Honolulu). This closes the numerical comparison in
[the reuse diagnostic](PHASE_C_FITTING_REUSE_DIAGNOSTIC.md). This review changes
documentation only. No benchmark, fitting implementation or production default
is changed. All reported experiment metrics come from the supplied records;
no campaign fits or rollouts were rerun.

## Provenance and verification

- Archive: `review-20261002-235039.tar.gz`, extracted under
  `artifacts/phase-c-fitting-m5-review-20261002-235039` (untracked).
- Archive SHA256:
  `80f0a186fd3c33487f31aaf61b028a43753078ec0c0eea45a5d829dd277ce19d`.
- Experiment commit: `8d9a450762a57a923d1e334c8e48d3731a5914e0`.
- Plan SHA256:
  `22fc1680384d87b7ba048a4b9d54330f53db7d0fd866f4518a4c649aa178065b`.
- Delta jobs: prepare `22632658`, fit array `22632659`, report `22632663`.
- All 101 sealed records verify. All 24 result rows reconcile with saved worker
  payloads, common-start identities, backends, coefficient-error calculations,
  independent replay scores and summary aggregates.
- All six common records and both case payloads exactly match the sealed M3
  inputs, also used by M4. No fitted M3/M4 endpoint was imported.
- All 24 endpoints pass prediction and coefficient recovery. Independent
  Radau/DOP853 replay disagreement is at most `4.14066e-6`, below the `1e-4` gate.
- All 94 recorded training screens complete. There are no recorded snapshot
  rejections, partial endpoints or outer-budget exhaustions. Across 48 native
  solves, 42 report `Solve_Succeeded` and six `Solved_To_Acceptable_Level`.
- No test data or LLM calls were used. Screening and endpoint selection used
  training only; reference coefficients and validation were post-selection checks.

## Hard CSTR results

Seven dynamic coefficients and two shared hidden initials are fitted from three
generic starts. Every arm uses the same fixed collocation mesh and two native
solves. The first solve is always cold; the second is the diagnostic intervention.

| Second-solve policy | Prediction passes | Coefficient passes | Median validation NMSE | Coefficient error median / worst (%) | Mean fit seconds |
|---|---:|---:|---:|---:|---:|
| Rebuild; original start | 3/3 | 3/3 | 7.56e-10 | 0.1370 / 0.3771 | 295.6 |
| Reuse; original start | 3/3 | 3/3 | 7.56e-10 | 0.1370 / 0.3771 | 265.1 |
| Reuse; latest primal | 3/3 | 3/3 | 7.50e-10 | 0.1371 / 0.3725 | 172.2 |
| Reuse; screened primal | 3/3 | 3/3 | 7.50e-10 | 0.1371 / 0.3725 | 166.0 |

Coefficient error is the maximum absolute relative error across dynamic
coefficients in each fit; the table then takes its median and maximum over starts.
All coefficients pass the frozen 1% threshold. Hidden initials are excluded from
that statistic: worst absolute concentration/jacket-temperature errors are
`3.62369e-4` / `0.06012 K` for the cold arms and `2.49492e-4` / `0.04981 K` for
the warm-start arms. CSTR latent-trajectory recovery was not separately scored.

Time is mean fitting wall time, including construction, screening and process
overhead, excluding separate endpoint replay. Group `seconds` in the raw summary
is a sum over three fits. Timings are descriptive observations from separate jobs,
not a hardware-controlled speed benchmark.

The smaller saturating-shape control also passes prediction, coefficients and
latent recovery on all 12 runs. Median validation NMSE is approximately `1.08e-9`;
the worst maximum dynamic-coefficient error is `0.004589%`. Mean fit times in the
same arm order are 25.0, 19.6, 17.7 and 17.4 seconds.

## What formulation reuse establishes

For all six matched case/start pairs, rebuilding and reusing the formulation
produce exactly equal retained parameters and independent train/validation scores.
Native iteration counts are also identical. Reusing the mathematical problem
does not itself import the previous solution or change the result in these tests.

Hard-CSTR formulation time falls from 13.45 seconds for two builds to 6.50 seconds
for one build. This directly removes approximately seven seconds of repeated
construction. Total observed fitting time falls by 10.3% (295.6 to 265.1 seconds),
but that entire difference cannot be attributed to expression construction:
lazy derivative/solver setup remains in native time, and job timing varies.
Neither solver factorization reuse nor a general speedup factor is established.

This rejects the interpretation that M4's two hard-CSTR failures demonstrated a
fundamental problem with formulation caching. M4 additionally changed initial
solver settings, short solve chunks, primal/dual transfer, mesh progression and
screening allowances. M5 does not identify which of those caused the regressions.

## What warm starts establish, and what they do not

Hard-CSTR native iteration counts are:

| Start | First solve, all arms | Second solve, both cold arms | Second solve, both primal-transfer arms |
|---|---:|---:|---:|
| 0 | 162 | 162 | 6 |
| 1 | 73 | 73 | 5 |
| 2 | 180 | 180 | 9 |

The supplied primal includes coefficients, hidden initials and collocation states.
Previous multipliers are not imported. A good full primal point makes the repeated
solve much cheaper; mean second-solve native time is about 111 seconds for cached
cold solves, versus 8--9 seconds for primal transfer. The screened arm's total
time is 37.4% below cached cold and 43.8% below rebuilding, in this two-solve
diagnostic. These are not speedups relative to doing one adequate solve and stopping.

Crucially, every first solve already reaches success or an acceptable solution.
All first-solve endpoints have complete low-error training rollouts and dynamic
coefficient errors below 1%. The screen selects the latest checkpoint in all six
matched pairs. Every retained first-solve checkpoint passes its screening checks;
the fallback is never exercised. Screened and latest-primal arms therefore have
identical second starts, native iteration counts and final parameters. Their
small timing differences are not evidence that screening improves optimization.

The experiment supports the usefulness of starting from a good solution. It does
**not** test the user's concern that a poor checkpoint can trap a later solve, or
establish that the screening rule prevents this. Feasibility alone is not enough:
M4 already contained a nearly dynamically consistent but poorly fitting endpoint.
Conversely, a temporarily inconsistent trajectory might still be a useful start.
The current screen is a conservative policy whose general utility remains open.

Since the first solve is already adequate, a production policy should be able to
stop after a training/feasibility gate and independent verification. Its stopping
decision cannot use the private coefficient errors calculated in this review.
The small second-solve coefficient changes are not uniformly beneficial: start 1
slightly worsens maximum coefficient error while improving output error.

## Recommended next milestone

Close the reuse isolation question and retain all fitting strategies. Carry
in-memory formulation caching into subsequent development with an explicit choice
of start and training-based stopping; do not promote the older bundled M4 reuse
policy. These results do not justify replacing rollout fitting or the hybrid.

Next qualify harder fixed-equation cases, as already planned: retain the known
basin case as a regression control and introduce genuinely challenging coupled
latent dynamics in alien-device. Audit coefficient/initial identifiability and
remove latent-scale redundancies before interpreting coefficient errors. Preserve
the benchmark observations; use generic starts independent of private truth.

Within that harder comparison, include predeclared difficult starts or early
checkpoints that actually exercise the screening veto and fallback. Compare
screened reuse, unconditional reuse and original-start fallback under equal
budgets, including screening cost. Report all starts and both prediction and
coefficient/initial accuracy. Do not choose difficult checkpoints by consulting
private coefficient errors. No further repetition of this already-successful
two-solve CSTR experiment is necessary to proceed.

## Review verification

In addition to the archive checks above, the 43 focused fitting/reuse tests pass.
The four-arm local control smoke passes, including independent replay and exact
resume. Repository-wide `ruff check .` still reports the same 37 pre-existing
findings in unrelated `analysis/claude/` scripts; this review changes no Python.
The results note, fitting plan and Phase C handoff are the only changed files.
