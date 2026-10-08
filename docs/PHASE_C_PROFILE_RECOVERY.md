# M20: finish saved-profile verification before further scaling

Protocol `phase-c-profile-recovery-1`. This is a new, separately budgeted CPU
diagnostic of all 24 completed M19 endpoints. It does not change historical runs,
production fitting defaults, construction, benchmark data or prompts. There are
no LLM calls, GPUs or test-data access. The original M19 run remains frozen.

## What this milestone must resolve

M19's scoring recovery completed all 24 results. Coefficient recovery within 1%
improved from 15/24 to 16/24 and initial-value recovery within 0.001 from 13/24 to
15/24. Six fast/slow six-state endpoints have verified alternative-vector
witnesses, but every one of the 862 in-domain profile checks exhausted its time
allowance. Two ordinary six-state solutions still have 323% coefficient error.
Those retrospective errors motivate this diagnostic; they do not choose starts,
profiles or stopping decisions inside it.

M20 first verifies every saved profile candidate. It then spends bounded extra
optimization only on training-selected unresolved profiles and endpoints whose
training loss remains above the numerical target. It distinguishes verification
of a feasible candidate from convergence of the nuisance minimization and from
any claim about parameter determination.

## Frozen export and information boundary

Source: `review-20261008-084818.tar.gz`, original M19 plan
`318cdc03bc281a226a76d4f66f08a44166ab62f54b0d7a642eb9fc5a4fce3738`.
The portable export contains all 24 endpoints and all 862 saved profile candidates,
plus the two out-of-domain points. The exporter verifies source results, completed
backends, fitting identities, sealed profile grids and checkpoint operation
locations. Each imported progress record receives a content digest. No endpoint
is removed based on prediction or parameter error.

The numerical worker receives an explicit `RecoveryInput` schema: candidate
equations, public training arrays, training-derived coordinates, bounds, generic
start, retained incumbent, original profile grid and saved parameter vectors.
Validation and reference values are separate evaluator inputs. They are read for
scoring only after the new backend and selected vector are sealed. Schema tests
reject extra evaluation fields; campaign tests check the selection-before-scoring
boundary. All original parameter and initializer domains remain unchanged.

## Equation-derived affine propagation

The new solver symbolically certifies that the supplied dynamics have the form

\[
\dot x=A(\theta)x+B(\theta)u+c(\theta),
\]

with no explicit time dependence and no state/input dependence in A, B or c.
Coefficients may be nonlinear functions of parameters. No benchmark-specific
coefficient values, state dimensions or coupling matrix enter the solver.
Nonlinear or time-varying dynamics are rejected, not locally approximated.

Over each public piecewise-linear input interval, augment the state with the
input, its constant interval slope and a constant one. The resulting autonomous
linear system advances by `expm(G * dt)`. Every sampled forcing interval is
retained, without rounding its duration. Initial states use the existing causal
initialization rules; hidden initial parameters remain fitted globals.

Parameter sensitivities use the Fréchet derivative of the matrix exponential,
including derivatives of the initial state. This is a derivative of the actual
affine rollout, not a finite-difference approximation or a regression against
estimated latent trajectories. It uses SciPy's
[expm_frechet](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm_frechet.html).
Matrix exponentials still have floating-point error; “exact” refers to the
constant-coefficient interval solution, not arbitrary precision.

Independent verification evaluates the same saved parameters through both the
matrix-exponential solver and the existing DOP853 solver of the original
equations. **Neither verification call computes a full sensitivity matrix.**
Verification requires both normalized prediction and loss agreement under M19's
declared tolerance rules. A final joint coefficient/initial sensitivity analysis
is computed once at the selected vector. Production integrators are unchanged.

## Ordered computation and budgets per endpoint

1. Recheck the incumbent and up to three saved M19 reliability alternatives.
2. Recheck all 24 or 48 frozen profile points with surviving candidates. Retain
   explicit domain-limited/missing/unresolved records. Each independent check
   receives 30 seconds, separately from optimization.
3. Freeze a shortlist of at most **four** profiles, at most one per parameter.
   Points already demonstrating a materially different near-equivalent solution
   at tolerance 1e-12 need no refit. Among remaining verified points, try those
   closest to the retained training loss first; unavailable points sort last.
   Stable profile index breaks ties. No coefficient truth or validation ranking
   participates.
4. Each shortlisted profile receives **60 seconds / 200 complete training
   residual evaluations**, shared by two nuisance starts: its saved candidate
   (or the current verified point) and the original generic start. Its coordinate
   remains fixed. All other coefficients and hidden initials remain free. A
   separate 30-second verification checks the best result. A failed/worse refit
   does not discard a better verified saved candidate.
5. If the best verified training loss is above M19's numerical target 1e-18,
   run two free joint searches: the best verified vector and the original generic
   start. Each receives **120 seconds / 300 evaluations**, then its own independent
   check. This targets observable fitting difficulty rather than selecting only
   endpoints known retrospectively to have wrong coefficients.
6. Select using the worse of the two independent training losses. Compute one
   final joint sensitivity matrix. Seal the backend before separate prediction,
   coefficient and initial-value evaluation at `after_saved` and `selected`.

The scalar target is a numerical accuracy target for these noiseless controls,
not a universal tolerance for experimental data. The original M19 profile grid
is preserved. If the selected vector moves, the report says so; it does not
pretend the old grid certifies the new center.

Both original method labels receive this same follow-up optimizer. The `arm`
field describes their M18 provenance, not a new comparison of joint fitting
against variable projection. Nuisance/free searches here are joint bounded
least squares with affine-rollout derivatives.

## Interpretation and accounting

The three numerical loss-increase tolerances remain 1e-12, 1e-10 and 1e-8.
A materially different independently verified near-equivalent solution is a
constructive ambiguity witness even if its optimizer stopped early. A high-loss
candidate is only an upper bound on the profile minimum. Verification alone
cannot exclude a better nuisance solution.

`profiles.verified` counts checked candidates, **not converged profile minima**.
This milestone does not certify full-grid minimization or global exclusion and
therefore cannot promote an endpoint to `locally_supported_on_tested_grid`.
It can report weak constraint, numerical disagreement or incomplete search.
There are no probability/95% labels and no coverage claim. Incorrect-skeleton
and noisy-data calibration remain future work.

Each check and optimization operation records its wall time, process CPU time,
allowance, actual dataset rollout calls and checkpoint. Matrix/sensitivity calls
and independent ODE checks can be distinguished by their operation stage; one
call covers the whole training split, not one trajectory or RHS evaluation.
Setup, saved-candidate verification, profile refits, free searches and final
sensitivity are reported separately. Original M19 costs are retained under
`source_cost`; they are never charged again. Retrospective scoring has separate
timing. This is additional-budget recovery, not a matched algorithm comparison.

Completed operations are reused exactly. Interrupted operations remain unresolved
with explicit incomplete accounting and their original allowance charge; resume
does not reset their budgets. New operations may continue. Task locks exclude
concurrent writers. Submission journals refuse to repeat an uncertain scheduler
submission. Source/runtime/plan identities are checked on resume.

## Verification and scope

Equation-derived propagation and joint derivatives were checked against DOP853
and finite differences on all four controls, including latent initial parameters.
Tests reject nonlinear/time-varying dynamics, changed fixed profile coordinates,
changed source training data and evaluator fields. They cover bounded shortlist
selection, failed verification, exact resume, post-seal scoring and idempotent CPU
submission.

A deliberately shortened six-state fast/slow smoke verified all 48 saved profile
candidates in 23.4 local seconds; its complete numerical work took 28.6 seconds.
It allowed only one six-call nuisance fit and two six-call free searches. These
are verification/resume smoke measurements, not prospective Delta performance
or coefficient-recovery results. The full campaign has not been run locally.

Final verification: **4,297 passed, eight skipped** in the full primary-checkout
suite (the skips require unavailable Torch); 45 focused tests and 17 tests in the
portable bundle passed. A complete controller smoke, including both independent
postfit scoring stages, finished and resumed with an identical sealed result.
Changed-file Ruff and `git diff --check` passed. Repository-wide Ruff still has
37 pre-existing unrelated findings.

## Completed Delta results — 2026-10-08

Reviewed `review-20261008-214938.tar.gz`, SHA-256
`9ca09a8d2bd4d71b476067b6e95b806e39d8eb2fd2de0313e4cd5a7c9eab472f`.
Plan: `88263866402e621e39d381e4d8ec472e36161128d5353dc91abd5049b5213c53`.
All 24 endpoints completed with complete cost accounting. All 2,615 sealed JSON
records passed their content checks; inputs match the delivered frozen export.
Result/backend/finished identities agree, and regenerating the report reproduces
every saved summary field. This review performed no new fitting or test access.

### Recovery and where it happened

| Evaluation gate | M19 retained | After saved checks | M20 selected |
|---|---:|---:|---:|
| Train and validation NMSE <= 1e-6, independently checked | 23/24 | 23/24 | 23/24 |
| Every dynamic coefficient within 1% | 16/24 | 16/24 | 21/24 |
| Every hidden initial within 0.001 absolute error | 15/24 | 15/24 | 15/24 |

These are retrospective accuracy gates. Reference values did not guide fitting,
candidate selection or stopping. Original arm names identify historical sources;
all endpoints received the same M20 optimizer, so this is not a new comparison
of joint versus profiled fitting.

All five new coefficient recoveries are among the six fast/slow six-state
endpoints. Their coefficient recovery rises from 1/6 to 6/6. Median maximum
relative coefficient error falls from **4.832% to 0.858%**; the final range is
0.839–0.934%. Validation NMSE falls to 9.20e-19–9.70e-19. All six final searches
stop at the training NMSE target 1e-18, rather than at a time limit.

All six selected fast/slow solutions come from `rescue-checks/0`: the free joint
rollout search started at the best verified current vector. Profile nuisance
refits first improved four of those starting vectors, but did not produce any
of the five new 1% crossings. The subsequent joint search produced those
crossings. There is no matched direct-from-M19 control, so the result does not
establish that the preceding profiles were necessary or unnecessary. A test that
omits them under a matched budget is the appropriate efficiency comparison.

Initial-value recovery counts hide a real but insufficient improvement. All six
fast/slow endpoints reduce their maximum initial error; the median falls from
0.06939 to 0.009979, with final errors 0.00978–0.01101. The largest residual error
is in `init_x5_value`, whose reference value is 0.1. Thus roughly 10% error remains
in that hidden initial despite prediction errors near 1e-18.

Not every coefficient gets closer to its reference as loss falls. Fast/slow
six-state seed 0, historically profiled, goes from **0.000889% to 0.934%** maximum
coefficient error while improving its initial error from 0.04012 to 0.01101 and
lowering prediction error. This directly illustrates compensation between
coefficients and initials; the aggregate recovery count is not a guarantee of
monotonic parameter accuracy.

### Remaining failures and uncertainty

Two ordinary six-state historical profiled endpoints (seeds 0 and 1) remain near
323% maximum coefficient error and validation NMSE 3.675e-7. Their warm joint
searches terminate on the small-step `xtol` condition without escaping that
region. Their generic-start searches reach the 300-call limit at worse losses.
Other starts of the same control recover the reference, so these failures are
not evidence that recovery is impossible or that the parameters are structurally
unidentifiable. They remain failures of the current bounded search strategy.

The fast/slow three-state historical joint seed 2 remains poor (validation NMSE
0.001620, maximum coefficient error 1,700%). Its new selected loss is slightly
worse than its M19 incumbent; the reason is the verification issue below.

At the strictest declared loss-increase tolerance 1e-12, eight endpoints have
verified materially different near-equivalent alternatives and 16 remain
`search_incomplete`. The eight are the six fast/slow six-state endpoints and the
two unsuccessful ordinary six-state endpoints. In the latter two, the witness
concerns a hidden initial near the retained poor local solution; it does not
certify uncertainty around the unknown global optimum. At 1e-10 and 1e-8 the
witness counts rise to 14 and 18, respectively.

For the fast/slow six-state selected solutions, the scaled joint sensitivity
matrix has condition number 2.06e8–2.49e8 and numerical rank 11/12 at the declared
relative cutoff. Its weakest direction combines the fastest decay with hidden
initials. This is evidence of poor conditioning at the given data and numerical
scale, not proof of exact structural nonidentifiability. The witnesses remain
useful even though retrospective coefficient accuracy improved. Neither the
witness category nor `search_incomplete` is a probability of correctness.

### Verification and boundary findings to address next

All **862** saved in-domain profile checks finished; none timed out. Of these,
835 passed the strict numerical agreement rule immediately. One further profile
became verified after a nuisance refit, giving **836 verified, 26 unresolved** and
two out-of-domain points. These are verified candidate vectors, not 836 certified
profile minima.

All 26 unresolved checks have finite outputs. They fail the absolute loss
agreement requirement 1e-13, with discrepancies 1.09e-13–3.21e-12. Their maximum
normalized prediction disagreement is at most 4.16e-9, below the prediction
agreement cutoff 1e-7. Thus the remaining issue is numerical agreement at the
chosen profile tolerance, not missing output or exhausted verification time.

The same strict rule rejects the fast/slow three-state seed-2 incumbent: its two
training losses differ by 4.83e-13. M20 then selects a strictly verified profile
refit with a worse actual training loss, 0.00174787 versus the source 0.00174500;
validation rises from 0.00161547 to 0.00162039. The historical record is intact,
but the new selection policy needs separate notions of **usable retained fit**
and **verified evidence at a requested uncertainty tolerance**. A failed stringent
uncertainty check should trigger bounded tighter verification or an unresolved
label, not force replacement of a previously usable incumbent. Do not loosen the
profile threshold retroactively to manufacture an uncertainty certificate.

Five of the 192 nuisance-start attempts also stop with `parameter vector outside
declared domain`. Each affected operation still retains a result from its other
start. The archive does not save the rejected vector, so boundary roundoff is a
hypothesis to test, not an established explanation. Record the offending vector
and domain distance; any numerical projection must be limited to demonstrated
roundoff and must not relax declared scientific bounds.

### Computation and recommended follow-up

| M20 stage | Summed task wall time |
|---|---:|
| Verify all saved profiles | 15.12 min |
| 96 nuisance refits and their checks | 47.23 min |
| 18 free joint searches and their checks | 10.74 min |
| Setup, incumbent/alternative checks, final sensitivities | 2.10 min |
| Total numerical work | **75.20 min** |
| Separate retrospective scoring | 8.80 min |

Numerical CPU time is 53.11 minutes and the counted complete-training rollout
attempts total 17,822. These are sums over tasks, not elapsed campaign duration.
All six successful fast/slow warm joint searches, including their checks, took
90.39 summed seconds; their preceding profile work is additional and cannot be
silently omitted from the campaign cost. M19 required 7.66 summed task-hours,
but did different search/verification work, so the totals are not a controlled
solver speedup measurement.

The next bounded experiment should distinguish **finishing an already good fit**
from **escaping an incorrect local solution**:

1. Repair the verification/retention distinction and diagnose the five domain
   failures. Preserve every historical artifact and count all new computation.
2. Compare direct warm joint affine-rollout polishing with exact profiling of
   input gain and hidden initials at fixed matrix coefficients, starting from
   the same frozen endpoints and using matched budgets. Keep the saved profiles
   as diagnostics; avoid paying for a full profile grid before every recovery.
3. Add a declared training-only portfolio of genuinely different feasible starts
   for unresolved searches, with bounded trials and retention of useful existing
   fits. Repeating an already unsuccessful generic start is not enough. Report
   basin recovery, numerical targets and coefficient/initial errors separately.

These are recommendations, not an implemented M21 or production change. All four
controls still have correct linear skeletons and noiseless observations. The
results do not establish statistical confidence, nonlinear-benchmark performance,
or robust recovery from arbitrary initializations.

Review verification: 27 relevant existing tests passed; report-regeneration smoke
and `git diff --check` passed. Repository-wide Ruff still reports the same 37
unrelated findings. Only this runbook and the fitting plan changed; no numerical
implementation or experiment artifact was modified or committed.

## Delta commands

Upload `transfers/phase-c-fitting-m20.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. The bundle includes code, tests, the frozen
configuration and the self-contained M19 export. No access to the original M19
directory is needed by the new worker.

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=/work/hdd/bibo/yxiao2/phase_c/code/fitting-m20
mkdir -p "$AF_CODE"
tar -xzf /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m20.tar.gz -C "$AF_CODE"
bash "$AF_CODE/scripts/hpc/submit_phase_c_profile_recovery_delta.sh"
BASH
```

Default root: `/work/hdd/bibo/yxiao2/phase_c/profile-recovery-v1`.
There are 24 one-CPU tasks, six concurrent, each with a one-hour scheduler ceiling
covering the worst permitted default check/search/scoring allowances. Jobs stop
when their work finishes. Existing `AF_PYTHON` and `AF_CASADI_ROOT` environments
are reused. Set `AF_CONCURRENCY` only to change scheduling, not numerical budgets.

After completion:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m20/scripts/hpc/inspect_phase_c_profile_recovery_delta.sh
```

The inspector regenerates `summary.json` and a dated review archive, showing
before/after recovery, verified profile counts, confidence categories and costs.
Missing results remain explicit. No remote sessions or jobs are launched by the
local coding agent.
