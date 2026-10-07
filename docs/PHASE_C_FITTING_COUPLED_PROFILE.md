# M16: exact profiling through coupled linear state equations

Protocol: `phase-c-coupled-profile-1`.
Configuration: `configs/phase_c_fitting_coupled_v1.json`.
This is an opt-in numerical qualification, separate from the 28-cell benchmark
release and from production fitting. M15 polishing remains a separate experiment.

## Scope and mathematical guarantee

The first qualification extends terminal-output profiling to an entire linear
state block with feedback. For outer parameters beta and inner parameters g,
require

\[
\dot x=A(t,u;\beta)x+B(t,u;\beta)g+d(t,u;\beta),\qquad
x(0)=q_0(\beta,y_0)+V_0(\beta,y_0)g.
\]

The observed initial data y0 are supplied causally; latent initial parameters
are shared across training trajectories. For fixed beta, linear superposition gives

\[
x(t)=q(t)+V(t)g,\quad
\dot q=Aq+d,\quad \dot V=AV+B.
\]

Off-diagonal entries of A propagate effects between states in both directions.
Coupling therefore does not prevent profiling. Parameters that enter A are
nonlinear outer unknowns: including them in the inner block would invalidate
superposition. The implementation verifies joint affinity symbolically using the
restricted compiler. It rejects nonlinear state dynamics, inner parameter
products, and unsafe observation mappings rather than silently approximating them.

This implementation supports one identity-observed state. Stacking its sampled
predictions over all training runs produces `prediction = offset + design @ g`.
With training normalization fixed, bounded least squares determines g at each
outer point. The profiled residual Jacobian includes the derivative of this inner
solution, including the nonzero-residual correction on a fixed active set. Bounds
and parameter roles remain the original ones. Changes of the active set can make
the Jacobian nonsmooth; a rank-deficient design is explicitly unavailable.

Affine latent initials are now included in g, unlike M14's terminal-output
implementation. Their fitted values remain frozen during validation. No latent
trajectories, derivatives, reference coefficients or validation values enter the
optimizer. "Exact" refers to algebraic separability, not zero numerical integration
error or a guarantee of globally optimizing beta. All input interpolation corners
remain integration boundaries, and original-equation replay checks the result.

## Controlled comparison

Use two standalone, noiseless controls with the same correct equation structure:

\[
\dot y=z,\qquad \dot z=-az-by+cu(t),\qquad z(0)=z_0.
\]

| Control | Reference a | Reference b | Reference c | Reference z0 |
|---|---:|---:|---:|---:|
| Coupled linear | 0.8 | 1.3 | 1.6 | 0.4 |
| Coupled fast–slow | 18 | 2 | 2 | 0.4 |

Only y and the public input u are observed. Each case has three training and two
validation trajectories, each with 121 observations over [0,12], and continuous
piecewise-linear inputs with one-unit knots. Initial observed y differs across
trajectories. This is a feedback system: y affects z and z affects y. The fast–slow
case has two distinct decay timescales, while the first is a damped oscillator.
These are qualification controls, not claims of difficulty comparable to the
six-state alien-device benchmark.

The fixed equation y'=z anchors the latent scale. Under ideal continuous
observations, z=y' and subtraction of two candidate y'' equations gives a linear
regression in a,b,c. Full column rank of [-y',-y,u] establishes uniqueness, and
z0=y'(0). The evaluator checks this excitation condition using its private witness;
those derivatives and latent values never enter fitting. This argument does not
establish good finite-sample conditioning, which the experiments must measure.

Both methods use all three reproducible generic starting vectors, independent of
the reference coefficients. Starts are the same across cases and methods:

- `rollout_only`: optimize a,b,c,z0 together by scaled TRF and forward sensitivities.
- `coupled_profiled_rollout`: optimize a,b; jointly solve c,z0 by bounded least
  squares at every outer evaluation.

There are **12 tasks**, with no best-start selection or removal of failed starts.
The profiled representation integrates six basis states and their derivatives,
versus two physical states in the direct method. Fewer nonlinear unknowns do not
necessarily imply less integration work. Compare elapsed fitting time, actual
residual calls, integration evaluations, predictions, coefficient errors and
initial-value errors; do not infer speed from call counts alone.

Each task receives 300 seconds, including setup and up to 60 seconds reserved for
training certification, and at most 300 actual optimizer residual calls. Both use
mean training NMSE <=1e-8 and maximum trajectory NMSE <=1e-7 as stopping targets.
Independent Radau and DOP853 rollouts of the original equations must each meet the
prediction gates and agree within normalized 1e-5. The independent evaluator has
120 seconds after selection freezes. Its accuracy gate is train/validation NMSE
<=1e-6, coefficient recovery means every dynamic coefficient is within 1% of its
reference, and shared initial recovery means absolute error <=0.001. Reference
errors cannot trigger stopping or selection.

Completed tasks resume from sealed results without new fitting. Interrupted fits
retain available checkpoints without refreshing their budgets. Unknown child
termination blocks evaluation. Scheduler intents and receipts prevent duplicate
submission after an ambiguous scheduler response. All tasks remain in reports;
`complete` means the records exist, not that every numerical fit succeeded.

## Following milestones

**Nonlinear dynamics with conditionally linear coefficients.** For
`x'=f0(x,u;beta)+Phi(x,u;beta)g`, fixed trajectory nodes make a quadratic
soft-defect objective convex in g. Profiling this inner problem is exact for that
relaxed collocation objective, not for the original rollout loss. The trajectory
and outer problem remain nonconvex. Hard collocation equalities may instead be
infeasible at fixed nodes. Test whether this changes collocation's empirical
performance; earlier unprofiled failures do not settle the question. Retain
independent rollout certification and matched budgets.

**Convex nonquadratic inner objectives.** Convexity must hold in the chosen inner
variables after all dynamics, observation maps, domains and weights are included.
If predictions/residuals are affine in g and the feasible set C is convex, then

\[
\min_{g\in C}\sum_i w_i\rho_i(a_i^Tg+b_i)+\lambda R(g)
\]

is convex when weights are fixed and nonnegative, lambda>=0, and each rho_i and R
is convex. For twice-differentiable losses the Hessian is

\[
\nabla_g^2 F=\sum_i w_i\rho_i''(a_i^Tg+b_i)a_i a_i^T
+\lambda\nabla_g^2 R\succeq0.
\]

For nonsmooth losses, the same conclusion follows from convexity under affine
composition and nonnegative sums. Squared, absolute, Huber and quantile losses meet the loss condition;
L1/L2 penalties and linear equality/inequality constraints preserve convexity.
A Poisson negative log-likelihood is convex in an affine **positive mean** for
nonnegative counts; it requires a count-observation rationale, not a change of
loss merely for computational convenience. Student-t and redescending robust
losses are not globally convex in general. Parameter-dependent weights,
non-affine observation maps, nonlinear state domains or coefficient-dependent
transition matrices require separate analysis. A nonlinear RHS can be convex as
a function without its squared rollout residual being convex.

Convexity does not imply uniqueness, closed-form solution or a smooth profiled
outer objective. Strict convexity/full rank, active constraints, numerical inner
accuracy and appropriate derivatives must be examined before implementation.
Quadratic convex inner problems are already the best-supported starting point.

## Delta commands

Upload `phase-c-fitting-m16.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c/`.
The archive contains pinned source, tests, configuration and a standalone control
generator. It needs no previous campaign directory or downloaded fitted models.
It generates and seals controls once; it never modifies benchmark data.

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m16"
tar -xzf "$AF_BASE/phase-c-fitting-m16.tar.gz" -C "$AF_BASE/code/fitting-m16"
export AF_COUPLED_ROOT="$AF_BASE/fitting-coupled-v1"
bash "$AF_BASE/code/fitting-m16/scripts/hpc/submit_phase_c_fitting_coupled_delta.sh"
BASH
```

The default is six concurrent one-CPU tasks, 16 GB each, with the existing
conservative 35-minute scheduler wall limit. Actual fitting allowance remains
five minutes per task. No GPUs or LLM calls. Set `AF_CONCURRENCY` if necessary.

Inspect and create a review archive:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m16/scripts/hpc/inspect_phase_c_fitting_coupled_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/fitting-coupled-v1
```

Download the printed review archive. The inspector prints all case/seed/arm
records, prediction and coefficient metrics, and process accounting.

## Local qualification

Structural/numerical and orchestration tests check full feedback reconstruction,
original forward sensitivities, shared-initial features, bounded-profiled
Jacobians, rejection of unsafe blocks, all-start accounting, training-only payloads,
post-freeze evaluation, child termination and idempotent CPU submission. The
small real smoke runs both methods on coupled-linear seed 0 and checks terminal
resume without extra calls. It obtained independent validation NMSE 5.83e-10
(joint) and 3.60e-11 (profiled); maximum coefficient relative errors were 0.0145%
and 0.00201%. This is a correctness smoke, not the paired campaign's result.

Verification: 51 focused tests pass in both the checkout and portable archive;
full `pytest -q -n 4` passes 4,187 tests with eight optional-Torch skips. All changed
Python files pass Ruff. Repository-wide `ruff check .` still reports 37 existing
issues in unrelated untracked `analysis/claude/` files. Shell syntax, archive
checksums, twelve-task preparation and inspector/archive output were also checked.
