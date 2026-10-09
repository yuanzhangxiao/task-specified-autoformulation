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
reporting and checksums. Completed benchmark results are reviewed below.

## Completed Delta review: archive `review-20261009-081105`

Source commit: `9647a65d9bbdb784b33d791e6d1116d10af370a5`.
Plan: `89d3225646b010783bc8baa14b19b40279c7cb120da5f9bd239f3298fdb5d0db`.
Root: `/work/hdd/bibo/yxiao2/phase_c/nonlinear-conditional-v1`.
All six tasks completed with complete cost accounting. Regenerating the report
from the downloaded sealed results reproduces the supplied summary exactly;
plan/input identities, backend hashes and completed fitting artifacts agree.
No test observations or LLM calls were used. This is a completed experiment,
not six successful parameter recoveries.

### Recovery and cost

These are final independent retrospective scores, after fitting decisions were
sealed. Relative coefficient errors below are fractions, not percentages.

| Original start | Method | Train NMSE | Validation NMSE | Maximum relative dynamic-parameter error | Maximum absolute hidden-initial error | Fitting plus diagnostics, minutes |
|---|---|---:|---:|---:|---:|---:|
| 0 | Best rollout | 1.26e-16 | 3.13e-16 | 1.05e-7 | 4.52e-8 | 15.52 |
| 0 | Conditional then rollout | 0.4082 | 0.5138 | 8.72 | 1.49 | 47.52 |
| 1 | Best rollout | 0.1236 | 0.09238 | 8.68 | 2.41 | 44.36 |
| 1 | Conditional then rollout | 0.5077 | 0.6191 | 31.99 | 8.54 | 49.55 |
| 2 | Best rollout | 0.3787 | 0.4970 | 18.26 | 1.90 | 49.47 |
| 2 | Conditional then rollout | 0.3981 | 0.3391 | 12.05 | 1.89 | 50.60 |

Best rollout recovers prediction, all thirteen dynamic parameters and all five
hidden initials in **1/3** original starts; conditional then rollout in **0/3**.
The same counts already hold after the warm stage. The successful warm search
reaches the numerical target in 35 residual/Jacobian calls and 590.5 seconds.
This is the strongest applicable method: exact profiling of four terminal
output gains, with fourteen remaining nonlinear coordinates. It is not the
earlier unprofiled joint baseline.

The successful endpoint's full scaled sensitivity has rank 18/18 and condition
number about 1,078. Recovery is verified retrospectively here; neither this
rank nor a small training loss alone establishes global uniqueness. The five
unsuccessful final searches exhaust their search allowance. They do not prove
structural nonidentifiability, and budget termination is not evidence of a
converged local minimum. Multiple starts remain necessary, but this portfolio
and allocation are not yet robust enough. Seed 2 has lower validation NMSE in
the conditional arm, so the rollout arm does not dominate every individual
metric; neither seed-2 endpoint passes the recovery gates.

Aggregate elapsed operation times across the three tasks per method:

| Cost component | Best rollout, minutes | Conditional then rollout, minutes |
|---|---:|---:|
| Conditional trajectory/coefficient work, including screening | 0 | 17.11 |
| Rollout optimization | 69.87 | 72.91 |
| Independent verification | 38.58 | 56.78 |
| Final joint sensitivity | 0.90 | 0.86 |
| Total fitting plus diagnostics, including small setup cost | 109.35 | 147.67 |
| Additional retrospective reference/validation evaluation | 12.82 | 11.50 |

The conditional arm consumes 35.0% more fitting/diagnostic time in this run.
These are sums of task elapsed times, not elapsed campaign time or CPU time.
CPU totals are 101.48 and 137.84 minutes, respectively. Both arms had the same
search ceilings; the successful rollout task stopped early, and repeated
verification adds separately charged overhead. Therefore the aggregate cost
difference is not a pure estimate of conditional graph construction overhead.

### Why the conditional arm did not help here

**The auxiliary trajectories were not accurate physical solutions.** For start
0, the coarse conditional endpoint has nodal observation NMSE 0.0006215, but its
actual original-equation training rollout has NMSE 20.0749. The finer endpoint
has nodal NMSE 0.002657 and actual rollout NMSE 15.9316. Both are worse than the
original starting vector's rollout NMSE 1.2251. For start 1, the coarse/fine
rollout errors are 368.95/512.91, against 1.5921 at the original start.

The implementation deliberately polishes the best finite *new conditional*
candidate even when worse than the original vector, to explore a new basin
while protecting the incumbent. This policy was recorded before launch. It
used the warm search on much worse physical starts for seeds 0 and 1; in seed
0, direct polishing of the original vector was the successful route. This
does not justify always discarding worse starts, but it argues against replacing
the only original-start optimization attempt with one.

All 57 recorded node/shape solves stop at an iteration limit (39) or a callback
stop (18); none reports convergence. Each node solve has at most 25 iterations,
on approximately 39,000 or 56,000 variables. Convex coefficient least squares
does not make the coupled node/shape problem convex or ensure that alternating
updates have converged. For the seed-0 coarse checkpoint, mean squared scaled
defect is 1.17e-4 and maximum scaled defect is 0.099; it is not dynamically
exact despite a small observation residual. Raw increment defects also depend
on interval length, so their penalty scaling needs diagnosis before interpreting
smaller defects as better continuous dynamics.

**Screening was underbudgeted.** Ten of twelve conditional stages produce no
completed screening candidate: all nine short trials and the seed-2 warm stage.
Their thirty candidate rollouts time out. A short trial reserves about fifteen
seconds for three full-training rollouts, giving roughly five seconds each.
That is insufficient on this case. These stages fall back to their original
start after consuming conditional time. All three final conditional-arm
incumbents come from rollout continuation of those fallback trial branches,
not from a successfully screened conditional proposal.

This is an allocation defect in the pilot, not evidence that those unscored
candidates have infinite error. Together with unconverged node solves, it means
the experiment supports rejecting this implementation/configuration for default
use, but does not settle the broader value of trajectory-conditional profiling.

**Verification was too eager to demand ultimate precision.** Across both arms,
27 tighter verification retries all time out, consuming 54.01 minutes. Ordinary
independent rollouts already establish that the poor endpoints have large
errors. Three final assessments nevertheless recommend `verify_numerics` because
their absolute two-integrator loss discrepancies exceed the stringent 1e-13
threshold derived from the eventual 1e-12 fitting target. Their maximum
normalized prediction disagreements are below 7e-10, and loss discrepancies
are at most 1.49e-12. This uncertainty is immaterial to whether their NMSE is
0.38--0.51. It is still relevant to certifying an eventual near-zero residual.
Retention correctly keeps useful ordinary-verified incumbents when a tighter
retry fails; the problem is prioritization and repeated cost, not lost results.

### Recommended next diagnostic, not implemented by this review

Keep the current strongest profiled rollout as the reference. Preserve an
original-start rollout attempt in every strategy; experimental conditional
proposals should compete for an explicit additional/restart allocation within
the same total cap. Report this policy change separately from M22.

Before another full comparison, run a bounded diagnostic that:

1. Measures complete training-rollout cost and reserves enough time to finish
   screening at least one conditional proposal. Screen candidates sequentially,
   reuse known control scores, and skip conditional work when the remaining
   allowance cannot support a meaningful solve plus verification. Record this
   as a budget skip, not a failed scientific candidate.
2. Checks conditional coefficient recovery on diagnostic reference trajectories
   in a separate evaluator, then on the saved estimated trajectories. This can
   distinguish coefficient-subproblem correctness from trajectory-estimation
   failure. Reference trajectories must never initialize the actual fitting arm.
   Inspect normalized defects, mesh scaling and node-solve progress together.
3. Retains ordinary independent rollout checks for candidates, but requests
   costly ultimate-precision retries when near the numerical target or when a
   near-tie makes retention uncertain. A reliably poor fit should route to
   optimization rather than repeated sub-picounit loss verification. No accuracy
   certificate should be relaxed for the successful endpoint.

Do not promote conditional fitting, change benchmark data, or claim that fitting
is solved on nonlinear systems from this result. First remove the measurable
allocation problems, then compare recovery across the same frozen starts and
matched budgets. This review changes documentation only.

Review verification: regenerated report exactly matches the supplied summary;
paired portfolio vectors and the three fallback continuation origins were
checked. All 21 relevant pytest cases pass. `git diff --check` passes.
Repository-wide `ruff check .` still reports the same 37 unrelated findings in
untracked `analysis/` scripts; no implementation files changed in this review.
