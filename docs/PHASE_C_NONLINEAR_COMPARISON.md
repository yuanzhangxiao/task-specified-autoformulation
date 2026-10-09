# M22: conditional coefficient fitting versus the strongest applicable rollout

Protocol: `phase-c-nonlinear-conditional-comparison-1`.
Configuration: `configs/phase_c_nonlinear_comparison_v1.json`.

## Question and scope

Does estimating auxiliary trajectories and conditionally fitting their linear
coefficients improve recovery or cost over our strongest applicable rollout
fitter on a known nonlinear benchmark skeleton?

The user clarified that the comparator must retain the improvements already
qualified in M14–M21. It is **not** an intentionally weakened joint optimizer.
Both arms retain exact profiling wherever the equations permit it, scaled
coordinates, analytical sensitivities, tight polishing, diverse starts,
independent verification, incumbent retention and the fitting assessment.

This first nonlinear comparison reuses the unchanged alien-device hard inputs
from M12/M15, including all three original generic parameter/initial starts.
There are six CPU tasks: three starts times two methods. There are no LLM calls,
new data generation, test observations, or changes to production fitting defaults.
The correct anchored equation skeleton is supplied to both methods; this is a
fitting qualification, not a discovery result. The frozen input content digest is
`ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09`.

The model has six coupled states, sixteen training trajectories and eighteen
unknowns: thirteen dynamic parameters plus five shared hidden initial values.
Known internal couplings and nonlinear shapes anchor the chosen latent
coordinates. The measured initial output comes from each trajectory's permitted
first observation. The public contract specifies shared reproducible hidden
preparation; we do not fit a separate arbitrary hidden initial vector per run.

## Two arms

| Arm | Conditional trajectory work | Rollout search |
|---|---|---|
| `best_rollout` | None | Strongest symbolically certified kernel |
| `conditional_then_best_rollout` | Up to 30% of each warm/trial search allowance | The same kernel, with the remaining allowance |

The common kernel selection is: exact affine-block profiling if the full system
qualifies; otherwise exact terminal-output profiling if that block qualifies;
otherwise joint rollout optimization. Symbolic exclusions are recorded. If an
otherwise valid profile solve fails numerically, a joint fallback can use only
the remaining time and call allowance of that stage.

For this alien-device skeleton, the terminal output state does not feed back
into the other five states. Its equation is linear in four output gains, and
those gains occur nowhere else. Fixing the other parameters and hidden initials
therefore gives an exactly affine output trajectory:

\[
\widehat y(\eta,\beta)=b(\eta)+A(\eta)\beta,
\qquad
\beta^*(\eta)=\arg\min_{\ell\leq\beta\leq h}
\|W[b(\eta)+A(\eta)\beta-y]\|^2.
\]

Both arms use bounded least squares for these four gains at every outer
evaluation, including the derivative of the profiled residual on a fixed active
set. The nonlinear search has fourteen coordinates, including the five hidden
initials. Unlike the linear controls, these hidden initials cannot in general be
profiled exactly through the nonlinear dynamics. Rank-deficient/nonfinite inner
solves are reported, not silently called successful.

## What trajectory-conditional fitting adds

With nodal trajectories fixed, many more coefficients enter the ODE residual
affinely even though their **integrated trajectories** do not. Symbolic analysis
certifies twelve such coefficients here. Only `input_scale` and the five fitted
hidden initials remain in the node/shape subproblem. Initializer parameters are
excluded from the linear block so that its affine certificate covers the first
interval as well as later ones.

Let `X` contain scaled auxiliary state values, `eta` the remaining parameters,
and `beta` the certified coefficient block. The conditional objective is

\[
L_\lambda(X,\eta,\beta)
=\frac{\|r_{\rm obs}(X)\|^2}{2N_{\rm obs}}
+\frac{\lambda\|r_{\rm Radau}(X,\eta,\beta)\|^2}{2N_{\rm defect}}.
\]

Observation residuals are normalized by training target scales; defects by
declared optimizer state scales. The two-stage Radau equations in each interval
of length `h` use stages at `1/3` and `1`:

\[
r_1=X_{1/3}-X_0-h(5f_{1/3}-f_1)/12,\qquad
r_2=X_1-X_0-h(3f_{1/3}+f_1)/4.
\]

Each trajectory's first boundary is its causal initializer evaluated at the
current fitted parameters. It is not a frozen hidden value. All observations
remain in the objective through the Radau interpolation polynomial.

For fixed `X, eta`, stacked residuals are affine in `beta`, so bounded least
squares solves their convex coefficient subproblem. The implementation then
holds those coefficients fixed and optimizes nodes, nonlinear shape and hidden
initials with IPOPT. It alternates at most four times, with up to 25 IPOPT
iterations per node/shape solve, and finishes with another coefficient solve.
Zero design columns preserve the previous coefficient and are reported;
rank-deficient conditional solves are not coefficient-identifiability claims.

This is **alternating conditional minimization**. It does not assert that a
nonlinear rollout is affine, and the linear block is not re-minimized inside
every IPOPT iteration. The nodes and their low collocation objective never
serve as predicted trajectories or certify a model.

## Mesh, reuse and selection

Two nested mesh levels use soft variable targets 36,000 and 54,000, with defect
penalties 100 and 10,000. Every piecewise-linear input corner is retained. A
minimum temporal resolution and eight separated high-slope/curvature observation
anchors per trajectory add boundaries; every original observation is still used.
The frozen seed-0 audit finds 39,114 and 56,442 total variables after anchors,
of which 39,102 and 56,430 enter the node/shape optimizer. These are still large
sparse problems; their measured cost is a principal outcome of the comparison.
The initial proposed 6,000/18,000 targets were rejected before launch because
mandatory input corners made their actual grids identical.

The first mesh starts from permitted initial values and interpolated observed
nodes; hidden nodes start as causal constants. The finer mesh can receive the
current coarser polynomial. Formulation graphs are cached across later trials,
but the first-level primal start is reset for each trial and duals are never
transferred from a previous basin. All graph construction costs are included in
the conditional allowance. No reference latent trajectories are available here.

Three quarters of that allowance are reserved for the node/shape work and one
quarter for full original-equation training rollouts of its endpoints and the
original starting vector. Only a finite complete rollout can choose a proposed
start for subsequent common rollout optimization. The best finite conditional
basin receives polishing even if its current rollout is worse than the original
start; otherwise an initially poor parameter/initial pair could prevent exploring
a useful new basin. The original vector is recorded as a control, and remains
protected by the common incumbent-retention rule. A conditional stage with no
finite new candidate falls back to its original start. Selecting a polishing
start never authorizes incumbent replacement, which requires independently
verified improvement after the rollout stage.

## Common recovery, budgets and assessments

1. Verify the original incumbent. Search from it for up to 600 seconds/600
   residual-Jacobian evaluations unless the declared numerical target is met.
2. If needed, evaluate all three deterministic domain-respecting diverse starts,
   each with 200 seconds/200 evaluations. Their proposed vectors are identical
   across methods and contain no reference information.
3. If still needed, give the best independently verified trial a further
   600 seconds/600 evaluations of the common rollout optimizer. Continue that
   distinct basin even when its short trial was worse than the incumbent, while
   preserving the incumbent until an actual verified improvement is found.

The search ceiling is 1,800 seconds per task. Conditional time is subtracted
from the warm/trial stage, not added to it. Continuation is rollout-only in both
arms. The call ceilings concern the rollout optimizer; additional conditional
candidate-check calls are counted separately within conditional time. Equal
ceilings do not imply identical consumed time, call counts or convergence.

Common rollout fitting uses centered/scaled bounded TRF, analytical forward
sensitivities, residual scaling, and a tighter numerical target rather than
stopping at merely small prediction error. For this noiseless nonlinear test
the mean training target is `1e-12` and the worst-trajectory target is `1e-11`.
These differ from the `1e-20` exact-linear-control target because we now integrate
nonlinear equations numerically. Neither threshold certifies coefficients and
neither is proposed as a universal rule for noisy data.

Every retained candidate is checked using original-equation DOP853 and Radau
rollouts (default relative/absolute tolerances `1e-10`/`1e-12`). Each check has
up to 120 seconds; a stricter `1e-12`/`1e-14` retry is allowed when the numerical
uncertainty check requires it. A useful incumbent survives a stricter-check
failure. Replacement requires improvement of the two-integrator loss interval.
No success or optimizer-convergence flag alone can promote a candidate.

Final scaled joint sensitivity, including hidden initials, receives another
120 seconds. The M21 assessment separates fit quality, numerical reliability,
search status, weak parameter information and recommended action. Verified
alternative vectors are recorded. Full profile-likelihood grids are not
minimized in M22, as in M21; no calibrated probability or global uniqueness is
claimed. Setup, conditional work, rollout search, verification, sensitivity and
retrospective scoring have separate accounting. Independent checks can be a
material part of runtime; the Delta wall request is 90 minutes, not a fixed
optimization duration.

## Separation, checkpointing and report

The optimizer takes a strict allowlist: request, training data, coordinate
scales, original start and incumbent. It has no validation/reference fields.
Fitting decisions and backend artifacts are sealed before retrospective scoring
opens validation and ground-truth parameters. Scoring after warm recovery and
at the final retained point reports prediction, coefficient and hidden-initial
accuracy. Its gates are inherited unchanged from the frozen nonlinear inputs:
train/validation NMSE at most 0.01, all dynamic parameters within 1% relative
error, and all hidden initials within 0.01 absolute error. Actual continuous
errors remain available; these gates are not fitting stopping criteria.

Operations record their identity, allowance, progress and terminal artifact.
Completed operations resume exactly. An interrupted operation is reported and
charged its whole allowance rather than restarted for free; later stages can
proceed with the last verified incumbent. Conditional primal checkpoints are
saved as hashed numeric archives, independently of final selection. Submissions
use intent/receipt journals and refuse to repeat an ambiguous scheduler reply.
Budgets are cooperative solver deadlines, with measured elapsed time charged;
individual library/setup calls can slightly overrun. An interrupted accounting
interval is explicitly incomplete. `complete` means the protocol finished, not
that its coefficients are known correct.

## Delta commands

Upload `phase-c-fitting-m22.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c`, then:

```bash
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m22"
tar -xzf "$AF_BASE/phase-c-fitting-m22.tar.gz" -C "$AF_BASE/code/fitting-m22"
bash "$AF_BASE/code/fitting-m22/scripts/hpc/submit_phase_c_nonlinear_comparison_delta.sh"
```

The default uses `bibo-delta-cpu`, one CPU and 16 GB per task, at most three
concurrent array tasks. It reuses the existing Python and CasADi environments;
`AF_PYTHON`, `AF_CASADI_ROOT`, `AF_ACCOUNT`, `AF_CONCURRENCY` and
`AF_NONLINEAR_COMPARISON_ROOT` can override their defaults. Run preparation tests
before the array through the submitted dependency. No GPU is requested.

Inspect and package the results with this command, which prints its exact
download path and works even when some tasks are still missing:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m22/scripts/hpc/inspect_phase_c_nonlinear_comparison_delta.sh
```

Default output: `/work/hdd/bibo/yxiao2/phase_c/nonlinear-conditional-v1`.
Do not infer coefficient recovery from training NMSE alone. Compare all six
results, failure counts and measured costs, including conditional setup overhead.

## Local qualification

Focused tests cover the symbolic partitions, bounded coefficient update,
initializer dependence, finite improving checkpoints, graph/primal separation,
common exact profiling, identical portfolios, retained incumbents, bounded
fallback, interrupted accounting, strict scoring boundary, and submission resume.
The independent two-state nonlinear smoke reaches training NMSE below `1e-10`
and maximum absolute parameter error below `1e-5` in both methods, with exact
completed resume. Its conditional candidate initially has worse rollout error
than the original start but recovers after common polishing. This validates
exploration with incumbent protection, not a conditional speedup.
The hard-case local audit only builds graphs and certifies structure; full
benchmark fitting is delegated to the six Delta tasks.

Verification: the full regression run passed 4,346 tests with eight optional
Torch skips. After the final conditional-basin selection refinement, all 21
focused tests and the two-arm numerical smoke passed again. Changed Python
files pass Ruff; repository-wide Ruff still reports 37 pre-existing findings in
the unrelated, untracked `analysis/` scripts. The portable package is also
tested independently of the checkout, including six-task preparation, incomplete
reporting and checksums. Benchmark comparison results remain pending.
