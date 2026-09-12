# Fitter feasibility and model-feature diagnosis

This is an opt-in follow-up to physical-initialization v1. It keeps the equations,
scientific signs, allowed domains, and initialization contract unchanged. The
production/default adapter remains `recovery_policy="legacy"`.

## What the preceding results establish

All 48 synthetic fixed/fitted arms completed numerical replay. With fitted
initials, the three smooth families recovered 18/18 and the threshold family
recovered 4/6. Collocation reported convergence even for the two threshold
misses, so convergence does not establish recovery. Their initial trajectories
never activated `max(m-1,0)`; polling escaped but exhausted 240 evaluations.

All eight public arms failed. These are four frozen proposed models for one
anonymous-system benchmark task, not a representative sample of the full public
benchmark suite. All four fitted arms timed out in collocation, then attempted
one failed augmented rollout at the ordinary start. The legacy error penalty
had a zero Jacobian and triggered a native `gtol` stop. Final status correctly
remained `fit_failed`.

Candidate class is not a causal explanation. Even seed0_parent has a trivial
feasible constant trajectory with all nonnegative coefficients set to zero.
We must distinguish existence of feasible constraints, locating a useful fit,
solver setup/iteration costs, physical integration, and sensitivity integration.

## Implemented policies

`CollocationSensitivityConfig` adds:

- `recovery_policy`: `legacy`, `feasible`, or `branch_aware`;
- `collocation_diagnostics`: saves progress and descriptive model features;
- `recovery_max_starts`: maximum initial full-training screens (default 10);
- `recovery_probe_seconds`: per-screen time cap (default 10 seconds).

All policies use the same declared domains and physical initial boundary.
Collocation may start with evaluable but ODE-inconsistent latent nodes. It does
not require a successful physical starting rollout. Initial observations and
known latent boundaries cannot be changed to manufacture branch coverage.

### Collocation progress

The callback records objective, unscaled maximum constraint violation, the
unscaled Lagrangian-gradient maximum when available, decision-value magnitude,
and elapsed time. The gradient measure is our explicit diagnostic, not IPOPT's
scaled `inf_du`. `solver_called` and the first recorded iteration distinguish
pre-iteration setup from iterative progress. Counts of decision variables and
constraints and construction seconds are retained.

At most three finite, in-domain parameter checkpoints are retained: latest,
least constraint violation, and lowest objective among points with constraint
violation at most 1e-6. That threshold labels a restart candidate; it does not
certify a physical rollout or scientific plausibility. These checkpoints survive
the child process's normal 120-second timeout. Later node values are not reused
as predictions. Native IPOPT internal-state continuation is not implemented.

### Feasibility and refinement

`feasible` screens preserved C estimates, the ordinary start, slower-equation
starts, and fixed-seed scaled perturbations with state-only integration. Only
complete finite training rollouts can compete. Screens have a per-attempt cap
and use at most 40% of refinement wall time and one-third of evaluation capacity.
A valid converged smooth C start does not trigger unnecessary screens.
The `slower_equations` proposals scale equation parameters down while preserving
initial parameters. This slows the coefficient-only RHSs tested here; it is not
a guarantee for general expressions, particularly inverse time constants.
Every proposal must pass the same physical rollout screen.

The best screened point enters sensitivity refinement for smooth models. Failed
augmented integration raises a typed unavailable-derivative result: no constant
penalty/zero-Jacobian pair reaches the optimizer. A second screened point may be
tried, followed by exact-cost polling when needed. If no initial screen succeeds,
bounded polling can still search for feasibility. Partial progress is preserved;
there is no promise that a feasible point exists or will be found.

All state-only screens, augmented evaluations, verification calls inside local
fitting, and fallback polls share the same 180-second/240-evaluation budget.
Per-evaluation native integration remains protected by the worker supervisor.
The best finite full-training evaluation survives local optimizer failure.
Native local-solver status, available derivatives, numerical replay, and clean
synthetic recovery are reported separately. Independent final scoring/replay is
outside the fitting budget in every arm.

### Branch-aware starts

`branch_aware` adds branch margins from validated, process-expanded expressions:
`abs(a)` uses the sign of `a`; binary `min/max(a,b)` uses `a-b`; n-ary primitives
compare each argument with its competitors. Initial-map branches are included.
Extraction is capped at 32 comparisons and truncation is explicit.

The runtime first targets both sides by moving permitted parameters, including
fitted initial values. If a physical boundary cannot move, it can instead alter
later latent-node guesses while preserving the initial boundary and directly
observed node guesses. Up to two additional C starts share the remaining
initializer budget after the ordinary C attempt. This is a bounded exploratory
policy, not exhaustive enumeration of all branch combinations.

The start audit records requested sides and whether the initial side was reached;
node-shift records include sampled coverage. An unreached side is not certified
unreachable. Guessed-node coverage does not imply that the eventual physical
trajectory crosses a threshold. No branch constraint is added to the model, and
no test data, hidden labels, or validation fitting enters start selection.

## Controlled Delta comparison

The existing physical-initialization source supplies the four frozen public
problems, including their explicit fitted-initialization plans. Input hashes are
checked before copying; no source output or benchmark data are modified.

Thirteen synthetic cases are constructed independently:

| Cases | Purpose |
|---|---|
| Shared initial, noise 0 and .03 | Smooth initial-state recovery regression |
| Causal initial map, noise 0 | Boundary-map recovery regression |
| Known-zero generator, noise 0 | Unnecessary initial freedom regression |
| Threshold, two starts and two noise levels | Inactive versus active branch starts |
| Four-state linear coupling | More states and 15 equation weights |
| Four-state quadratic coupling | Superlinear feedback with attainable generating parameters |
| Same quadratic model, safer ordinary start | Only the starting weights change; observations identical |
| Same quadratic model, latent f coordinate multiplied by 100 | Coordinate scaling; observations and weights identical |
| Same quadratic model over a longer horizon | Horizon/mesh-size stress; both changes are reported |

The coupled controls have stable generating parameters, but the ordinary
all-.1 guesses can grow rapidly. No generating latent trajectories or weights are
provided to the fitter. The scale transform is a coordinate change in the
candidate equations; its reference observations remain exactly unchanged.
The linear/quadratic comparison changes the generating family too; it is a
controlled structural contrast, not identical-data causal attribution. Horizon
and mesh-size effects are not separated in the long-horizon control.

Every case compares `legacy` and `feasible`. Only the four genuinely piecewise
cases add `branch_aware`, avoiding redundant branch jobs for smooth models.
With the expected four public candidates this gives **38 fit tasks**:
30 synthetic and eight public. Each task requests one CPU, 16 GB, and a 30-minute
Slurm allocation. The array runs at most two tasks concurrently. No GPU or LLM
calls are made. A CPU smoke job gates the fit array; a file-only summary follows.

All arms receive 120 seconds for C and 180 seconds/240 evaluations for
refinement. Diagnostic logging is enabled in every arm. Comparisons include its
overhead; no claim of zero-overhead instrumentation is made. The branch arm can
spend unused C time on alternate starts but cannot obtain a new C budget.
The diagnostic legacy arm also enforces the hard evaluation cap on verification
calls inside local fitting. Its old failure transition is retained as the control;
the production legacy policy without diagnostics is unchanged.

## Reporting and interpretation

`summary.md` separates observed-target NMSE from clean validation NMSE. Synthetic
recovery requires independent Radau/BDF agreement and both clean train and
validation NMSEs <=1e-4. Public models have no hidden-reference recovery label.

`diagnostics.json` contains model size, coupled components, state powers, branch
expressions, initial Jacobian magnitude/eigenvalue diagnostics, C progress,
checkpoint parameters, state-only screens, augmented failures, and final selected
parameters. A positive initial eigenvalue is local evidence, not proof of global
instability. Different node scales, constraint violation, and long per-iteration
times should be assessed together before attributing a failure to a feature.

Use these distinctions:

- No first C iteration: investigate construction or native solver setup.
- Large persistent constraint residual: investigate feasibility/scaling.
- Small constraint residual but large objective: investigate fit geometry/start.
- State-only failure: the physical rollout is unavailable at that point.
- State-only success, augmented failure: investigate sensitivities/numerics.
- Verified poor fit: inspect model adequacy and remaining optimization budget.

No category by itself proves a scientific model defect.

## Run and resume

From the isolated pinned checkout, with the existing Python/CasADi dependencies:

```bash
export AF_REPO_ROOT="$PWD"
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/physical-initialization-v1
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-feasibility-v1
bash scripts/hpc/submit_feasibility_delta.sh
```

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-feasibility-v1/summary.md
```

Summary regeneration requires no numerical dependencies or allocation:

```bash
"$AF_PYTHON" -S scripts/run_feasibility_campaign.py summarize --output "$AF_OUTPUT_ROOT"
```

Completed fits and replays are reused under identical code/runtime/input hashes.
A worker killed without a completed fit is recorded as interrupted, never silently
given a fresh native optimizer budget. The stored C checkpoints remain available
for inspection. A deliberately fresh attempt requires a new output root.
For an incomplete task that already has a saved fit, preserve all checkpoints and
submit `sbatch --array=INDEX --export=ALL scripts/hpc/feasibility_delta.slurm run`
with `AF_CODE_COMMIT=$(git rev-parse HEAD)` exported. This completes missing replay
stages without refitting. The launcher refuses duplicate or ambiguous submissions.

## Remaining limits

This is a diagnostic and bounded recovery milestone, not a global optimizer.
First-sample branch targeting uses the first training trajectory as an anchor;
node starts are rebuilt for every training trajectory. Branch combinations and
state constraints outside the existing adapter contract remain unsupported.
Neither unit-invariant optimization nor automatic state rescaling is introduced;
the scale control measures whether that is the next required remedy. No active
proposer protocol is switched and no benchmark prompt is changed.

## Local verification

The final full suite passed 1,333 tests; three optional PyTorch tests were skipped
because PyTorch is unavailable in the local environment. Ruff and shell syntax
checks passed. The standalone smooth and inactive-threshold recovery smoke tests
achieved validation NMSEs of approximately 2.9e-16 and 2.5e-10, respectively.
A stubbed Slurm launcher check verified smoke gating, array/summary dependencies,
and duplicate-submission protection without submitting cluster jobs. Delta results
for the 38-arm comparison remain to be collected.
