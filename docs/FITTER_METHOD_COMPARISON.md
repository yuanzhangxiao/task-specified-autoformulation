# Fitter method comparison, milestone 1

This is an opt-in, CPU-only experiment on three fixed synthetic equation families.
It follows the successful active-dynamics recovery experiment. It changes neither
production fitting defaults nor Sol's proposer, contracts, benchmark data or prompts.
The user submits the jobs on Delta. ACES remains available for proposal experiments.

## Feasibility: exploit coefficients without assuming a linear trajectory fit

For a model written as

```
x' = F(t, x, u, theta)
y  = H(t, x, u, theta)
```

having coefficients of fixed functions can make the local parameter partials very
cheap. For example, if `F = Phi(t,x,u) theta`, then `F_theta = Phi`. This remains
true when those fixed functions are nonlinear in the state, such as `tanh(x)`.
It does not make the fitted trajectory linear in the parameters. Even
`x' = -a*x` gives `x(t) = x(0)*exp(-a*t)`.

For a trajectory least-squares residual, the required Jacobian includes the
parameter-induced change in every latent state. With fixed, parameter-independent
initial conditions, the forward sensitivity equations are

```
S = dx/dtheta
S' = F_x S + F_theta,      S(0) = 0
J_y = H_x S + H_theta
```

The implementation constructs exact local derivatives by algorithmic differentiation
of the restricted expression graph, then numerically integrates these equations.
It avoids subtracting two nearby ODE solutions for each parameter derivative.
The resulting trajectory derivatives still have integration error; they are not
closed-form trajectory solutions. With `n` states and `p` parameters this integrates
`n*(1+p)` quantities. It can be slower for large systems; this is measured rather
than assumed to win.

Linearity must be checked after expanding named algebraic processes. In the older
three-latent-state fixture, `v01` depends on fitted coefficients and is substituted
into `f' = k*v01**2/(1+v01**2) - f/tau_f`. The expanded dynamics are not jointly affine
in the original parameters. Time constants also occur reciprocally. Replacing
`1/tau` by a rate can linearize an individual decay term, but does not remove that
feedback dependence and changes the parameter coordinates and bounds.

The probe therefore emits a conservative symbolic certificate of RHS and joint
RHS/observation affinity, plus the expanded equations and local partials. Failure
to certify is not a proof that no separable block exists. This milestone does not
constrain the LLM's function vocabulary or automatically reparameterize its model.

## Comparison matrix

| Method | Applicable cases | What changes |
| --- | --- | --- |
| `production_fd` | All | Existing interpreted Radau trajectory fitting, scaled finite differences |
| `compiled_fd` | All | CasADi expression graph and exact ODE state Jacobian; same outer finite differences |
| `forward_sensitivity` | All | Same compiled engine, augmented forward sensitivity solve and residual Jacobian |
| `derivative_init` | Fully observed affine control | Bounded linear matching to smoothed time derivatives, then production trajectory refinement |
| `integral_init` | Fully observed affine control | Bounded linear matching of short-window increments and RHS integrals, then the same refinement |
| `weak_init` | Fully observed affine control | Bounded weak-form matching with compact test functions, then the same refinement |
| `shooting_init` | Two latent cases | Adaptive CVODES multiple shooting with continuity constraints, then the same refinement |
| `collocation_init` | Two latent cases | Implicit two-stage Radau IIA collocation, then the same refinement |

The compiled finite-difference arm is needed to distinguish the benefit of the
compiled integration engine from that of the outer parameter Jacobian. Its engine
also supplies the exact ODE state Jacobian, so its comparison with production is
not a pure expression-evaluation microbenchmark.

Every initializer competes against the same broad start and uses only the training
split. Initialization and refinement share one 600-second allowance. Failed
initialization falls back to the original vector with the remaining budget. No
validation score chooses an initializer or parameter vector. Matching-stage costs
are retained for diagnosis but are not compared numerically with trajectory costs.

### Cases and data

1. **Fully observed affine control:**
   `x' = -a*x + b*tanh(x) + c*u01 + d`, `v01 = x`, `x(0)=0`.
   Truth: `a=1.2, b=0.5, c=0.8, d=-0.4`. The state dynamics are nonlinear in `x`
   and affine in all four parameters; the output is the observed state itself.
2. **Moderate hidden dynamics:** the earlier signed-offset, three-latent-state
   recovery fixture, unchanged equations and generating parameters.
3. **Separated hidden dynamics:** the same fixture with the previous separated
   time scales, including `tau=0.08`.

The two hidden fixtures expose only `u01` and `v01` to fitting. Their reference
latent states are saved for numerical guards but never passed to initializers or
fitting objectives. Node states in shooting/collocation are optimization variables,
not supervised labels; initial states remain fixed at zero. Final evaluation uses
only the returned global parameters in a fresh causal open rollout, discarding all
fitted node states.

Four training and two validation inputs are reused from active recovery, over
`[0,24]` with observation step `0.2`. There are no test trajectories or benchmark
files. Each case has clean observations and independent additive Gaussian noise
with SD `0.03` times the clean training-output SD. The noise recipe and NumPy version
are frozen: SHA-256 of the seed, case, noise index and trajectory ID supplies the
random seed. All methods receive identical observations for each case/noise pair.
Only training observations define fitting normalization. No noise level is tuned.

One broad start per case is drawn before fitting: positive parameters are log
uniform over `[0.25,4]`, and the signed offset is uniform over `[-2,2]`. The seed is
`20260909`, independently of truth. The two hidden cases share the same vector.
This is a new start, not the previous active-recovery experiment's seed `20260910`.

There are **32 fits**: `6*2` on the observed control and `5*2*2` on latent cases,
plus **three guard tasks** and one summary job. One start and one nonzero noise
realization provide a screening experiment, not a robustness or statistical
superiority claim. Parameters remain in their original physical coordinates and
retain the production role-derived domains.

### Initializer details

The observed-state methods all use a fixed seven-point, degree-three Savitzky-Golay
smoother. Derivative matching drops three endpoint samples. Integral and weak
matching use eight-interval windows with stride four and trapezoidal quadrature.
The weak test function is `phi(q)=q^2*(1-q)^2`, zero at both window boundaries;
integration by parts moves the derivative from observed `x` onto known `phi`.
The bounded linear solve retains nonnegative gains and a signed offset. Smoothing,
quadrature error, errors in the regressors and correlated window errors remain;
this is not an implementation of SIMODE's full statistical methodology or WENDy's
errors-in-variables correction.

Multiple shooting places nodes at every input-sample interval and propagates with
CVODES at `rtol=1e-7, atol=1e-9`, preserving linear input interpolation. Collocation
uses Radau IIA nodes `1/3,1`, with rows `(5/12,-1/12)` and `(3/4,1/4)` of its Butcher
matrix. It is order three and stiffly accurate; it is not the old fixed-RK4
initializer. Both enforce continuity and fit global parameters and intermediate
states with IPOPT. Their maximum initialization allowance is 120 wall seconds,
including the child process startup. The parent can kill a native solver stalled
inside a callback. Iteration checkpoints and initializer outcomes are retained.

## Numerical guards, final evaluation and interpretation

Every case must pass its own guard before any of its fits can run:

- Independently written Radau and DOP853 references at tight tolerances agree;
  hidden memory states also agree with their closed-form linear-filter solutions.
- The production model at the generating vector passes full train/validation
  replays with Radau, BDF and tighter Radau, and has negligible reference error.
- At truth and the broad start, compiled predictions agree with production.
  Forward sensitivities agree with central differences at both `1e-4` and `1e-5`
  scaled steps. Times and discrepancies are reported at each anchor.

All fits use the corrected segmented-input integration and the existing scaled
finite-difference policy (`1e-4`, physical-coordinate scale floor 1) where applicable.
Final parameters are replayed by the production interpreter with Radau and BDF
at `1e-9/1e-11` and refined Radau at `1e-11/1e-13`. Agreement is checked pointwise
using training scale. Each split/solver replay has a separate 30-second allowance.

The summary reports initializer success, optimizer termination, residual call
counts, initializer plus refinement time, parameter errors, observed-data NMSE,
and clean-reference NMSE. Clean scores are diagnostic only and never enter fitting.
`complete` means the returned model passed numerical replay checks; it does not
mean parameter recovery, a good noisy-data fit, or optimizer convergence.
Inspect each of these outcomes separately. Failures and timeouts stay in the table.

Per-task elapsed time includes setup and verification; `total_fit_seconds` reports
initializer plus optimizer time, excluding shared dataset/graph setup and final
replays. The full worker deadline additionally bounds overhead. CasADi solver counts
are recorded for the compiled arms; residual calls count complete training residual
evaluations, including sensitivity solves where used, not just SciPy's `nfev`.

The restricted symbolic adapter rejects non-global parameters, nonfixed initial
states, lagged-target reset semantics, state constraints, negative/variable powers,
and nonsmooth/domain-restricted functions. It retains the existing denominator
guard. Derivatives at clipping/branch boundaries are not smooth scientific
derivatives; invalid numerical trials are reported. This adapter is not a general
replacement for the production interpreter or the full public model contract.

## Checkpoints, resources and commands

Preparation freezes the plan, candidate files, starts, method matrix, package
source hashes, launcher hashes and runtime versions. CasADi is pinned to 3.7.2.
References and replay arrays are content-hashed. Completed tasks, initializers and
replays are reused. An interrupted optimizer restarts deterministically from its
original or cached initialized vector; a cached initializer is charged its original
time. This does not claim bitwise optimizer continuation or identical timeout
endpoints across machines. Terminal failures are not silently rerun or deleted.

Delta requests one CPU, 8 GB per task and at most two concurrent array elements.
There are no GPUs. The hard worker ceilings are 360 seconds per guard and 840
seconds per fit, about **7.8 CPU-hours** if all 35 tasks hit their ceilings, plus the
small summary. Slurm allocations allow 10 minutes per guard, 20 per fit and 5 for
the summary, including process startup. Queue wait and actual duration may differ.

Use a clean isolated checkout at the commit supplied with this milestone. The
launcher refuses source/runtime drift, blocks duplicate submissions, and reports a
partial submission for queue reconciliation instead of submitting replacements.
These commands are intended to be pasted as single lines in the Delta terminal:

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-v21 && git fetch origin codex/fitter-method-comparison && git worktree add --detach ../autoformalism-fitter-methods-v1 FETCH_HEAD
```

The final handoff supplies an exact-commit, existing-directory-safe version of that
checkout command. Keep the established fitting environment unchanged; install the
one additional package into a dedicated dependency directory:

```bash
/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python -m pip install --no-deps --target /projects/bibo/yxiao2/venvs/fitter-methods-v1-deps casadi==3.7.2
```

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-fitter-methods-v1 && bash scripts/hpc/submit_fitter_methods_delta.sh
```

After completion, paste this summary; it includes each guard and fit's diagnostics:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-methods-v1/summary.md
```

Missing tasks can be inspected without resubmitting by rerunning the summary:

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-fitter-methods-v1 && PYTHONPATH=src:/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps /projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python scripts/run_fitter_methods.py summarize --output /work/hdd/bibo/yxiao2/phase_b/fitter-methods-v1
```

Do not delete the submission manifest or intent directory to rerun failed work.
Review the cause and freeze a new versioned run if algorithm or budget changes are
needed. `AF_ARRAY_CONCURRENCY=1` is supported for reducing simultaneous CPU use.

## Local verification before cluster submission

The focused tests exercise analytic decay sensitivities, hidden algebraic feedback,
initializer eligibility and bounds, actual CVODES/IPOPT and collocation solves,
observation-only latent reconstruction, a hard initializer timeout, freeze tampering,
resume, successful sensitivity-based recovery and duplicate/partial submission.

All three full-size numerical guards passed locally. The largest relative
sensitivity-versus-central-difference discrepancy across both steps and both anchors
was approximately `1.96e-6`; the guard threshold is `1e-3`. This verifies derivative
construction at these points. It is not an end-to-end fitting speedup result, and
Delta reruns the guards in its own pinned numerical environment.

Full regression verification: **1,162 passed, 3 skipped** (the optional Torch
baselines are unavailable in this environment). Preparation, supervised guard
execution, completed-task resume, summary generation, shell syntax and Ruff also
passed locally. The 32-fit experiment is left for user submission.

## Follow-up decisions after reviewing this run

1. **If sensitivities pass and improve time/accuracy:** combine them with the best
   supported initializer, then repeat multiple starts and noise realizations.
   Compare CPU time to a predeclared accuracy threshold, not only terminal NMSE.
2. **If integral/weak matching improves the observed control:** extend to certified
   observed affine blocks. For partially observed models, profile coefficients only
   conditional on a valid latent reconstruction or joint trajectory formulation;
   never substitute unavailable latent truth. Separability should be derived from
   the expanded graph. Do not force the proposer to emit only linear functions.
3. **If shooting/collocation helps latent cases:** test mesh refinement and harder
   time-scale separation before using it in model search. A low collocation loss
   alone cannot establish a correct rollout. Dense sensitivity growth may motivate
   sparse solvers or adjoint gradients for larger models; those are a later test.
4. **Then test frozen proposer candidates with Sol:** keep the scientific candidate,
   public data contract, fitting route and numerical verdict separate. Exact
   derivative-label regression is eligible only when the public dataset actually
   supplies those labels. It is a separate route from estimating time derivatives
   from noisy observations. The current synthetic open-rollout probe must not be
   installed as a blanket replacement for causal one-step benchmark semantics.

The first run tests method families using one implementation framework, not a
package leaderboard. CasADi provides Python AD, CVODES and NLP building blocks;
SIMODE remains a useful reference for separable integral matching. PySINDy's weak
features are useful for observed-state sparse discovery but do not directly solve
our arbitrary fixed latent-model fitting problem. A GEKKO implementation would be a
second implementation of dynamic optimization, not a distinct first-screen arm.
Package ports and a full separable nonlinear integral-matching estimator should
follow evidence from this controlled comparison.

References: [CasADi documentation](https://web.casadi.org/docs/),
[SIMODE paper](https://arxiv.org/abs/1807.04202),
[PySINDy weak/integral features](https://pysindy.readthedocs.io/en/stable/examples/12_weakform_SINDy_examples/example.html).
