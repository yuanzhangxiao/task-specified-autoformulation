# Phase C fitting M5: formulation reuse versus starting-point reuse

Protocol `phase-c-fitting-reuse-diagnostic-1`, agreed 2026-10-02.
This is one bounded fitting milestone. Astra's construction work, benchmark
releases, production defaults and all previous fitting arms remain unchanged.

## Question and comparison

The M4 `collocation_reuse` arm bundled graph retention, eight short native solves,
primal/dual transfer, different mesh-time allocation and a convergence gate before
mesh refinement. Its two hard-CSTR failures cannot identify graph retention as
the cause. See [the M4 review](PHASE_C_FITTING_M4_RESULTS_2026-10-02.md).

M5 uses two uninterrupted native solves on the **same fixed mesh**:

| Arm | Second formulation | Second starting point |
|---|---|---|
| `rebuild_cold` | Rebuild identical equations/mesh | Original generic full primal vector |
| `cache_cold` | Retain same Opti object and solver setup | Same original full primal vector |
| `cache_last_primal` | Retain | Latest finite, parameter-bounded checkpoint |
| `cache_screened_primal` | Retain | Best qualifying first-solve checkpoint, or original start |

The primal vector includes coefficients, shared hidden initials, mesh-boundary
states and internal collocation states. Every first solve is cold. Every solve
explicitly resets supplied constraint multipliers to zero and uses IPOPT's normal
initializer (`warm_start_init_point=no`). No previous dual vector, L-BFGS history
or full optimizer state is imported. Here “primal warm start” means changing the
user-supplied primal initial guess, not enabling IPOPT's primal/dual warm-start mode.
This follows [CasADi's explicit-start interface](https://web.casadi.org/docs/#the-opti-stack)
and [IPOPT's initialization options](https://coin-or.github.io/Ipopt/OPTIONS.html).

Reuse is in-memory within one task. Nothing is cached across different models,
meshes, processes or jobs. The first formulation's physical initial vector is
hashed; both cold solves receive that exact vector. Tests compare actual native
iteration counts and starting-vector identities on a small control. Rebuilding
uses the same variable ordering, coordinates, data, bounds and expressions.

`rebuild_cold` versus `cache_cold` isolates the cost of retaining the formulation
and solver setup while holding the user-supplied start fixed. The two warm-start
arms then change primal initialization only. This is a diagnostic, not an efficient
production allocation: repeating an identical cold solve deliberately spends
budget to measure reuse without better initialization.

## What qualifies as a useful checkpoint

Checkpoint preservation does not require native optimizer success. At most three
distinct points are retained: latest, least dynamically inconsistent, and best
feasible nodal fit (or best nodal fit if none is feasible). The fixed pool
feasibility threshold is 1e-6 in scaled dynamical defects.

For `cache_screened_primal`, a first-solve checkpoint must have:

1. A finite full primal vector and parameters inside their declared bounds.
2. A finite complete free rollout on **training**, with the same fitted initial
   rule and input schedule. A failed rollout or penalty vector is not a score.
3. Maximum scaled dynamical defect at most 1e-6.
4. At least 0.1% relative improvement over the original start's training NMSE.
   If the original rollout is unavailable, a complete qualifying rollout is
   sufficient; an unavailable baseline is not assigned zero error.

Choose the qualifying checkpoint with lowest training NMSE. Otherwise reset to
the original generic start. The threshold is a frozen numerical diagnostic
policy, not scientific certification or a globally optimal selection rule.
The test does not add new state-domain constraints. Existing equation/domain
choices are identical across arms; the warm-start gate explicitly checks parameter
bounds and scaled dynamical consistency.

Final endpoint selection compares complete training rollouts from both solves
and the original start, **even when a point is unsuitable for transferring its
inconsistent nodal trajectory**. Thus a bad second solve cannot overwrite an
already verified better endpoint. This does not guarantee better optimization:
a rejected checkpoint might have led to a better basin, and a good current
rollout does not guarantee the best latent trajectory or coefficient estimate.
Validation and reference coefficients are evaluator-only after selection.

## Frozen roster, input identity and budgets

Config: `configs/phase_c_fitting_reuse_diagnostic_v1.json`.

- Nonlinear-shape identifiable control and hard CSTR, three original generic starts.
- Four arms: **24 CPU tasks**, one CPU per fit, concurrency two by default.
- Exact sealed M3 observations, equation requests, coordinates and node guesses.
  These are also the matched inputs used by M4. No fitted M3/M4 endpoint is imported.
- Two native attempts, each at most 200 iterations and approximately 200 wall
  seconds (checked at native callbacks), also subject to a 200 CPU-second limit.
- Maximum 10 seconds per training rollout screen, at most seven planned screens.
  Every attempt is logged before execution, including unavailable/timed-out ones.
- A hard parent-process ceiling of 600 seconds includes startup, graph construction,
  screening, solver work and checkpoint writes. A stalled native call can consume
  the remaining total ceiling before a callback; the parent kills the process group.
- Independent endpoint replay has a separate 240-second allowance. It verifies
  train/validation predictions with Radau and DOP853; it cannot modify parameters.
- Reference/excitation qualification precedes the array. CSTR has a local
  sensitivity-rank gate, not a proof of global identifiability.

Report all starts and unavailable endpoints, not a best-of-three selection.
Coefficient and initial-value errors remain separate. One coarse fixed mesh need
not achieve M4's adaptive-mesh accuracy; the purpose is to isolate reuse decisions.
The small local smoke checks implementation and replay, not CSTR performance.
An endpoint marked `complete` has replayable predictions; native convergence,
budget exhaustion and coefficient accuracy are reported separately.

## Diagnostics and interruption behavior

Within each result's `fit/` directory:

- `attempt-N/formulation.json`: graph construction/configuration wall time,
  dimensions, cache flag and chosen start.
- `attempt-N/started.json`: initial full-primal hash, zero-dual policy and limits.
- `attempt-N/checkpoints.json`: preserved full primal vectors and nodal/defect facts.
- `attempt-N/checkpoint_rejections.json`: nonfinite, extraction and bound failures.
- `attempt-N/warm_start_decision.json`: training scores and each checkpoint's
  screened-policy veto reasons, even in the unconditional arm.
- `attempt-N/native.json`: return status, iterations, native wall time and checkpoints.
- `screens/NNN.json`: distinct started/completed/failed rollout-screen records.
- `best.json`: best verified training endpoint, written before later work.

Reported graph time covers expression construction and calling the solver
configuration API. Lazy derivative/solver initialization inside `solve()` remains
in native time. Do not call graph time “all compilation cost,” nor claim a
particular sparse-factorization cache is retained. Compare total time as well.
Summary groups report recorded native timing counts so missing timing is visible.

Completed tasks resume by reading sealed results without refitting. An interrupted
parent with a consumed budget is terminal, not automatically restarted. The outer
worker recovers saved endpoints and partial attempt metadata after a child timeout.
Other unstarted array tasks can proceed. Unknown scheduler replies retain their
submission intents and must be reconciled before resubmission.

## Local verification before release

- Full pytest: 3,848 passed, eight skipped because Torch is unavailable locally.
- Focused fitting/reuse tests: 43 passed, including an isolated portable-package run.
- Four-arm native-solver smoke: all four passed prediction and coefficient checks,
  independent replay and exact completed-result resume. The two cold arms produced
  identical errors with two versus one formulation builds.
- Portable input verification: all six common starts match the sealed M3 inputs.
- Mock scheduler: correct 24-task array and dependencies; a repeated submission
  makes no additional scheduler calls. No real jobs were submitted locally.
- Changed Python files pass Ruff. Repository-wide `ruff check .` reports 37
  pre-existing findings in unrelated `analysis/claude/` scripts.

## Delta launch and inspection

Upload `transfers/phase-c-fitting-m5.tar.gz` to your Delta home directory. It
contains committed source, selected tests, configuration and sealed matched inputs,
without old fit results or test data. From a Delta terminal:

```bash
bash <<'BASH'
set -euo pipefail
AF_ARCHIVE="$HOME/phase-c-fitting-m5.tar.gz"
AF_REV=$(tar -xOf "$AF_ARCHIVE" SOURCE_COMMIT)
AF_CODE="/projects/bibo/yxiao2/repos/phase-c-fitting-m5-${AF_REV:0:7}"
mkdir -p "$AF_CODE"
tar -xzf "$AF_ARCHIVE" -C "$AF_CODE"
export AF_ACCOUNT=bibo-delta-cpu AF_CONCURRENCY=2
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_reuse_delta.sh"
BASH
```

The launcher uses the existing Delta fitting environment and verifies SHA256SUMS.
Default output: `/work/hdd/bibo/yxiao2/phase_c/fitting-reuse-diagnostic-v1`.
Existing M3/M4 paths are not touched. If the portable code directory is your current
directory, inspect with:

```bash
bash scripts/hpc/inspect_phase_c_fitting_reuse_delta.sh
```

It prints scheduler state, refreshes the read-only summary and writes a timestamped
review archive to the campaign root for download. It performs no fitting.

## Next milestone: harder fixed-equation qualification

Keep rollout-only, original collocation, adaptive shooting and the hybrid. Keep
the historical reuse variant available as a diagnostic, without promoting it.
After the isolated M5 comparison, qualify coupled basins and alien-device using
immutable Phase C development arrays and correct equation templates. Their
reference rollouts are already qualified; parameter identifiability is not yet.
The known-rating, two-coefficient basin case already fit accurately in M1; keep
it as a regression control rather than claiming it is a newly difficult problem.
Alien-device's coupled nonlinear latent dynamics are the main new challenge.
Its reference witness currently fixes numerical coefficients; it is not yet a
free-parameter fitting template. In a hard-tier template that frees all latent
forcing amplitudes and tanh argument scales, a common transformation `z -> a*z`
can be offset by multiplying latent forcing amplitudes by `a` and dividing the
corresponding tanh argument scales by `a`, leaving the observed output unchanged.
Linear latent couplings are unchanged under common scaling; zero preparation is
unchanged too. An explicit scale convention or restricted identifiable parameter
block is needed before claiming recovery of individual private coefficients.
This is a qualification requirement, not a change to the benchmark observations.

First enumerate coefficient/initial parameters, remove known scale redundancies,
and audit local sensitivities under the actual training inputs. Separate exact
or analytically identifiable controls from merely locally full-rank stress cases.
Start with noiseless observations and generic, truth-independent starts; then
increase initialization distance, dynamical stiffness and observation noise in
separately declared comparisons. These harder experiments are not submitted by
M5. Dalla's accepted approximation of private gastric bookkeeping remains a
model-class question, not an exact-coefficient-recovery gate.
