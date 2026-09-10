# Fitter initialization diagnostics and start portfolio

This milestone compares the existing sensitivity routes with a training-only
two-start policy. It retains the fixed synthetic equations, input trajectories,
noise recipes, initial conditions, parameter domains, collocation mesh, and
numerical verification from `FITTER_COLLOCATION_SENSITIVITY.md`. Production fitter
defaults, proposer contracts, benchmark data, and finalized prompts are unchanged.
There are no LLM calls, test data, or private benchmark evaluations.

## Motivation from the paired Delta results

The previous C+S route recovered all 12 cases and was faster than ordinary
sensitivity fitting in ten. Two separated-system replicate-2 runs were exceptions:

- The clean initializer hit the 150-iteration ceiling after 22.2 seconds. Its
  parameters were discarded; ordinary-start fallback eventually recovered.
- The noisy initializer converged with `k=0.00322` and `tau_f≈98003`, almost
  suppressing feedback over the 24-unit horizon. Refinement needed 62 residual
  calls, versus 21 from the ordinary start.

A local replay of that noisy case found initial observed-training NMSE about
0.002830 for the collocation parameters and 88.41 for the ordinary parameters.
The collocation node objective and fresh-rollout objective agreed closely at
that point. Thus an initial-loss-only selector would have chosen the slower
start. The proposed policy compares loss *after a short refinement* instead.

These are opened development fixtures. This experiment tests the policy; it is
not a pristine holdout or a production-default promotion.

## Initializer diagnostics

Protocol `fitter-methods-3` enables a callback that records each IPOPT iteration's:

- iteration number and elapsed time;
- normalized sum-of-squares node objective;
- largest raw violation of the NLP constraints, including parameter bounds;
- parameter values and whether they are finite and within their exact domains.

The constraint diagnostic mixes the model's physical coordinates and is not a
scaled KKT residual or an identifiability measure. No dual-optimality claim is
made. It diagnoses progress rather than certifying an accepted model.

Every eligible iterate is atomically saved as `last_finite.json`, alongside
`iterations.json`. A limit or solver failure still returns `success=false` and
`parameters=null`; the saved point is explicitly separate and remains
`rollout_verified=false`. If the child hits its hard wall deadline, the parent
retains the most recent checkpoint after terminating the child. Stale snapshots
are cleared before a new initializer starts. Finite points outside parameter
bounds are logged but are not retained as eligible starts.

Only the seven parameter values are reused. Estimated collocation node states
are discarded. The mesh, tolerances, and 150-iteration limit are unchanged, so
this experiment first measures progress before increasing the iteration budget.

## Four comparison routes

| Route | Behavior |
| --- | --- |
| S | Ordinary broad start, followed by sensitivity fitting. |
| C+S | Converged collocation start, followed by sensitivity fitting; ordinary fallback if initialization fails. |
| Portfolio-12 | Compare starts after at most 12 calls/30 seconds per pilot. |
| Portfolio-24 | Compare starts after at most 24 calls/60 seconds per pilot. |

All routes using collocation receive the same shared initializer checkpoint.
For Portfolio, the second start is the converged initializer, or the last finite
in-domain iterate if initialization failed. Its first sensitivity residual call
performs a fresh ODE rollout on every training trajectory from fixed zero initials.
Collocation node values and node loss never enter the decision. Invalid rollouts
do not become eligible best evaluations. If no distinct collocation point exists,
Portfolio uses ordinary sensitivity fitting directly with the remaining budget.

A first local smoke showed that 12 calls still favored the slow collocation
start on both previously troublesome cases. The 24-call policy is therefore
an explicit additional development arm, not a replacement for the short pilot.
Both use the same total budget; longer pilots spend more of it before selection.

For two distinct starts:

1. Refine the ordinary start for at most **12 calls/30 seconds**, or **24 calls/60 seconds** in the long-pilot arm.
2. Refine the collocation start under the same pilot limits.
3. Compare the lowest finite, integration-valid training residual cost reached
   by each pilot. The lower cost wins. Ties within
   `1e-12*max(1,abs(cost_a),abs(cost_b))` retain the ordinary pilot.
4. If the winning endpoint has converged, stop. Otherwise restart sensitivity
   least squares from its best parameter vector using the remaining budget.
5. Retain the best valid training evaluation across both pilots and continuation,
   so a failed or worse continuation cannot discard a better pilot result.

The policy does not select by initial loss, fractional improvement, validation,
clean-reference accuracy, parameter distance, or knowledge of the generating
parameters. A short pilot is still an imperfect predictor of a better eventual
solution; this is the reason for the comparison rather than an assumed guarantee.
On failed-initializer cases, Portfolio also tests saved-iterate reuse, while the
C+S control retains its previous strict fallback. These cases must be interpreted
separately when attributing any improvement to pilot selection versus iterate reuse.

## Budget and resume accounting

Each logical fit has **600 seconds total**, including the original shared
initializer charge of at most 120 seconds, and a maximum of **150 residual calls**
for the sensitivity portfolio. For each policy, both pilots and continuation consume that common
call allowance. The controls retain their 150 optimizer-function-evaluation
ceiling; actual residual calls are reported separately. There are no
finite-difference refinement arms in this matrix.

Each completed pilot/continuation has its own identity-checked `stage.json`,
initial vector, first-rollout loss, best valid point, call count, elapsed time,
and optimizer report. Resume reuses completed stages and charges their original
time and calls. Only an interrupted stage restarts from its frozen starting vector;
the optimizer's internal iteration state is not serialized. Native initializer
processes and outer workers retain hard wall limits. Final independent replay
checks are separately budgeted, as in v2.

`total_fit_seconds` is the initializer charge plus all stage durations, including
the losing pilot. It excludes reference generation, final replays and outer
setup/reporting. Physical shared-initializer work occurs once, while its original
duration is charged to all three logical routes that use collocation. Portfolio also records current
invocation elapsed time so cached resume is distinguishable from charged time.
`fit.nfev` is summed when all stages report it; the authoritative total is
`actual_residual_calls`. Per-stage convergence and Jacobian diagnostics remain
inside each stage report rather than being assigned to a different selected point.

## Frozen matrix and reporting

`configs/fitter_methods_v3.json` fixes the same moderate/separated systems, noise
fractions 0/0.03, and three seeds 20260909/20260910/20260911. Clean replicates vary
only the start; noisy replicates vary both start and noise realization, paired
across routes. All parameters and node initials are independent of reference
quality. Four training inputs and two validation inputs span 24 time units with
sample step 0.2. All three model initial states remain zero.

There are **2 guards + 12 shared initializers + 48 fits = 62 array tasks**, plus a
summary job. Each task uses one CPU and 8 GB, with concurrency two and no GPU.
The conservative sum of worker ceilings is about 12.2 CPU-hours, plus summary;
actual use depends on early convergence and failures. ACES is not needed for this
fitter diagnostic.

The numerical guards check independent references and sensitivities at truth and
all broad starts. Every final fit undergoes fresh production Radau, BDF, and
tighter-Radau replays on both development splits. `complete` means numerical
verification; output recovery separately requires clean-reference training and
validation NMSE <=1e-4. Neither reference score enters optimization or selection.
Known admissible parameter aliases remain diagnostic rather than recovery gates.

`summary.md` includes the aggregate and per-fit results, the initializer progress
table, and the two pilot losses and chosen route. In the v3 tables, `Init failed`
records native initializer nonconvergence; Portfolio may still use a saved
unconverged iterate. `compact.json` retains exact selection and initializer
diagnostics. `summary.json` and `details.md` retain detailed records. Missing and
failed tasks remain in denominators. Initializer traces are in each initializer's
`initializer/iterations.json`; pilot residual and iteration logs are under each
portfolio fit's `portfolio/<stage>/attempt-<n>/` directory.

## User-run Delta commands

Use the exact pinned commit and existing-directory-safe checkout command in the
handoff. Reuse the previous `autoformalism-v21` Python and dedicated
`fitter-methods-v1-deps` CasADi 3.7.2 installation. No package change is needed.

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-fitter-methods-v3 && bash scripts/hpc/submit_fitter_methods_v3_delta.sh
```

The launcher submits guards, shared initializers, fits, and the summary in order.
Repeating submission returns the recorded job IDs. Partial submission stops for
queue reconciliation; it does not create replacement jobs automatically. Source,
configuration, candidate and runtime drift are rejected on resume.

When complete, paste:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-methods-v3/summary.md
```

To regenerate the current report without submitting jobs:

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-fitter-methods-v3 && PYTHONPATH=src:/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps /projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python scripts/run_fitter_methods.py summarize --output /work/hdd/bibo/yxiao2/phase_b/fitter-methods-v3
```

## Local verification before Delta

The full test suite passed: 1,183 tests, with three optional Torch tests skipped
because Torch is not installed. Ruff, shell syntax and diff checks passed.
Regression tests cover native iteration-limit recovery, stale-checkpoint rejection,
training-only selection, combined call/time accounting, staged resume, both pilot
policies, and a simulated Delta submission with duplicate-submission protection.

Both full-size numerical guards passed at the generating parameters and all
three broad starts. Full-size development smokes used the two previously
troublesome separated-system replicate-2 cases:

| Noise | 12-call pilot winner | 24-call pilot winner | Long-policy clean validation NMSE |
| ---: | --- | --- | ---: |
| 0 | Collocation saved iterate | Ordinary | 2.01e-21 |
| 0.03 | Converged collocation | Ordinary | 2.1242e-5 |

The ordinary pilots converged in 22 and 21 calls respectively, while each
collocation pilot used all 24 calls without reaching the ordinary pilot's loss.
The long policies used 46 and 45 total residual calls, including both starts.
Independent production replays passed, and resuming a completed fit reused its
checkpoint. The clean failed initializer retained its last finite parameter
iterate for a successful fresh training rollout.

These runs verify implementation and expose pilot-length sensitivity. They do
not establish a runtime improvement: the policy pays for initialization and a
losing pilot, and local elapsed times under concurrent tests cannot be compared
directly with the previous Delta timings. The full paired Delta matrix is the
next comparison; both pilot lengths remain frozen in it.

## Remaining limitations

The adapter still requires supported smooth expressions, global parameters,
fixed numeric initials, and free rollouts. It does not yet implement the full
prediction contracts of arbitrary proposer-generated candidates. A portfolio can
reduce time available to a promising start or select the wrong start after a short
pilot; a converged pilot may also be a poor stationary point. Three replicates
are a development robustness screen. Any wider adoption requires a separate
comparison on frozen proposer candidates under their actual public contracts.
