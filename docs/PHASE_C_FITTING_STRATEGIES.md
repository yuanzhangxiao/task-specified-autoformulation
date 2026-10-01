# Phase C fitting milestone 3: compare trajectory transcriptions

Agreed 2026-10-01. This is an opt-in, known-equation fitting experiment owned by
Orion. Astra's construction work and production fitting defaults are unchanged.
It follows the [M2 recovery study](PHASE_C_IDENTIFIABLE_FITTING_RESULTS_2026-10-01.md):
all 27 small controls recovered, but that study gave every arm a collocation
initializer and therefore did not test whether collocation was necessary.

## Question and frozen comparison

Does a joint trajectory/parameter formulation help difficult latent fitting,
and which numerical strategy best uses a fixed budget? Keep direct collocation,
multiple shooting and ordinary rollout fitting available until this comparison
supplies evidence. No claim of a new optimization algorithm is made.

Protocol `phase-c-fitting-strategies-1`, config
`configs/phase_c_fitting_strategies_v1.json`, entry point
`scripts/phase_c_fitting_strategies.py`.

| Arm | Numerical work within one shared 180-second ceiling |
|---|---|
| `rollout_only` | Joint bounded TRF fitting of coefficients and shared latent initials, using forward sensitivities and training stopping rules. |
| `collocation_only` | Joint parameter/initial/trajectory NLP with two-stage Radau IIA; up to three meshes, without rollout parameter optimization. |
| `adaptive_shooting` | Joint parameters/initials and interval boundary states; CVODES integrates within intervals, IPOPT enforces continuity; up to three mesh/tolerance passes. |
| `collocation_rollout` | Existing scaled collocation initializer receives one-third of the budget, then the existing joint rollout refinement receives the remainder. |

Every arm starts with the same physical parameter vector. All trajectory-based
arms start from the same physical state guesses: measured training states where
available, otherwise the causal initial guess. A coarser shooting mesh samples
that same guess. No reference coefficients or latent trajectories initialize any
fit. All parameters and shared hidden initials are optimized jointly; observed
initial readings remain fixed. There is no validation-specific initialization.

Numerical coordinates, physical bounds and observation weights are identical
across arms. CSTR hard's shared initial concentration is nonnegative, justified
by its public concentration interpretation. No new path-state bounds or hidden
reference ranges are introduced. The existing guarded physical equations remain
unchanged. Derivatives through piecewise expressions are a numerical heuristic
at a crossing; independent exact-expression rollouts remain required.

The hybrid is a matched-budget version of the current approach. The three other
arms compare whole strategies, including their solvers and discretizations;
this is not an isolated test of just mesh adaptation or just collocation order.

## Roster and what identifiability means

Six cases, three fixed generic starts, four arms: **72 CPU fits**. Report all
starts; do not select a favorable seed.

- The M2 linear, nonlinear-restoring-force and fast/slow controls have
  `y'=z`, `z'=-a*z-b*y+c*u-d*y^3` (omit `d` in the linear case).
- A new saturating-stiffness control has
  `y'=z`, `z'=-a*z-b*y/(1+q*y^2)+c*u`. Here `q` changes the nonlinear
  shape of the equation; it is not a reparameterized linear amplitude.
- CSTR easy and hard reuse the **unchanged, sealed M1 Phase C development inputs**
  and known coupled equations. Easy provides auxiliary initial readings; hard
  jointly fits the shared hidden initial concentration and jacket temperature.
  Both are explicitly assisted numerical stress tests.

For the new control, multiplying the second equation by `1+q*y^2` gives

```
y'' = -a*y' + c*u - b*y - q*y^2*y'' - (a*q)*y^2*y' + (c*q)*y^2*u.
```

Full column rank of the six displayed regression functions uniquely determines
the lifted coefficients `(a,c,b,q,a*q,c*q)`, hence `(a,b,c,q)`. Also `z=y'`
fixes the latent scale and `z(0)=y'(0)`. This is a sufficient argument for ideal
continuous observations under the checked excitation condition. It does not prove
identifiability from arbitrary finite noisy samples. The evaluator verifies the
ideal witness and local output sensitivity rank; neither derivatives nor hidden
labels enter the fitting payload. The new witness's column-normalized singular
ratio is approximately 0.0245, and its reference output-Jacobian ratio is 0.0739.

**CSTR is not globally certified identifiable.** Require reference attainability
and full local numerical sensitivity rank before running it, then report output
accuracy separately. No failed CSTR parameter recovery is automatically called an
optimizer failure or evidence of nonidentifiability. A low output error alone
cannot certify its hidden states or parameters.

## Discretization, adaptation and acceptance

For every trajectory minimize the sum of squared observation residuals,
normalized by each channel's training standard deviation and the total number of
observed values. Keep **every original observation, exactly once**, at its original
weight. A newly added mesh point adds no data or observation weight.

Direct collocation jointly optimizes physical parameters, shared initialization
parameters, interval endpoints and internal Radau states. Its equality constraints
express the two-stage Radau integration equations. IPOPT iterates may temporarily
violate them. Multiple shooting can likewise temporarily violate continuity,
but its trajectory inside each interval is generated by numerical integration.
These are different ways to leave the globally consistent trajectory manifold
during optimization; collocation does not uniquely possess that freedom.

- Collocation starts at every sample/input knot. Later passes bisect the worst
  quarter of intervals whose **off-node** scaled ODE defect exceeds `1e-6`.
  Defects are evaluated at fractions 0.15, 0.55 and 0.85 of the interval, not at
  the two points where the NLP already enforces the equations. All states,
  including unobserved ones, contribute. The cap is four times the original
  interval count. This first implementation refines; it does not coarsen.
- Shooting starts with up to twelve uniform windows plus up to eight strong
  corners per observed/input channel. Every original input/sample knot remains
  an internal integration boundary, even when it is not a free shooting node.
  No input is replaced by a line across a longer window. CVODES adapts its internal
  steps using all state components. The outer mesh bisects high observed-residual
  or continuity-defect windows (threshold 0.05 in normalized units), capped as
  above. This is a heuristic for optimization difficulty, not a certified mesh
  error estimator. Relative integration tolerance tightens from `1e-5` to `1e-7`
  to `1e-9`; absolute tolerance is one-hundredth of that.
- Each NLP sees a fixed mesh. Between solves, interpolate the previous physical
  node trajectory to initialize the new variables and restore the causal initial
  boundary at the new parameter point. Do not reset nodes to measured values.
- Retain recent parameter/trajectory checkpoints, the least-defective point and
  the best feasible discretized point. A low collocation loss at inconsistent
  nodes is **not** an accepted fit. Only a finite, complete training free rollout
  can select the deployed parameters. No collocation/shooting node values become
  extra hidden initials during final replay.

Rollout refinement uses the M2 accuracy/stagnation stops. Mesh arms may stop when
training accuracy is reached; collocation may stop when its indicator is satisfied
and its NLP converged. Otherwise stop at the mesh-pass or time/call limit. These
stopping events are reported separately from independently verified accuracy.

## Budgets, checkpointing and independent scoring

The outer subprocess supervisor counts startup, graph construction, NLP work,
training-rollout screening and refinement against the **same 180-second wall
ceiling**. Screening and refinement share at most 180 complete-training residual
attempts; native NLP evaluations are reported separately and are not treated as
identical-cost calls. Each mesh pass reserves approximately one quarter of its
remaining allocation for training-rollout checkpoint screening. The hybrid does
not receive a free initializer outside its total budget.

The supervisor kills the process group on a wall timeout, including an active
native child, and records a bounded two-second cleanup grace separately through
actual elapsed time. This addresses the M2 initializer's delayed return; it is
not a promise of identical CPU progress under different scheduling loads.
Atomic incumbent files allow recovery of already verified training points. A
partial worker result explicitly marks incomplete accounting rather than
inventing an exact evaluation count. `mesh_stage_timeouts` reports exhausted
mesh subproblem allowances separately from the overall budget flag; the legacy
hybrid initializer retains its own diagnostic record in the backend.

After fitting, parameters are frozen. An independent Radau/DOP853 replay, with a
separate 240-second evaluation allowance, checks both development splits and
requires solver agreement within `1e-4` training standard deviations. Reference
qualification uses the same replay boundary and must achieve NMSE <= `1e-6`.

Synthetic-control success requires both output NMSEs <= `1e-6`, all parameter
relative errors <= 1%, and both latent-trajectory NMSEs <= `1e-4`. Private recovery
metrics are computed only after endpoint selection. CSTR's output threshold is
0.01; global parameter/latent recovery remains unclaimed. Report metrics even when
thresholds fail, and keep optimizer convergence, budget exhaustion and accuracy
as separate facts.

Plans bind all Python source, numerical package versions, input content, starts,
node guesses and configuration. Completed results are reused byte-for-byte.
An interrupted fit with no sealed backend result is recorded as interrupted;
it is not silently rerun with a fresh budget. If the backend was already sealed,
independent evaluation can resume. This is deterministic artifact/budget resume,
not portable serialization of IPOPT/CVODES internal state.

## Implementation and local verification

The new opt-in mesh/transcription modules reuse the restricted expression compiler,
causal initialization, coordinate maps, sensitivity oracle, M2 rollout refinement,
existing hybrid initializer and independent replay. A separate campaign is needed
because M2 only varied refinement after a shared initializer. No historical
campaign protocol is changed. No LLM, GPU, test data or construction stage is used.

Focused tests cover input-pulse preservation, observation counts, off-node hidden
defects, nonlinear-shape excitation, paired starts, native collocation/shooting,
training-only boundaries, hard time limits, checkpoint recovery, immutable resume,
source drift and uncertain scheduler receipts. The standalone four-arm smoke
also checks independent replay and exact completed-result resume. On the small
linear control all four arms pass output, parameter and latent recovery. This is
plumbing qualification; it does not establish performance on CSTR or the harder
controls.

Local verification on 2026-10-01:

- All 18 focused tests pass. All six reference/sensitivity gates pass, including
  the two CSTR tiers; preparation confirms the full 72-task roster.
- The portable four-arm smoke passes recovery and exact resume. Training/validation
  NMSEs are below `2.3e-10` in all four arms. Shooting retains a successful checkpoint
  even when a later mesh stage exhausts its allowance; that timeout remains visible.
- The full pytest run reports 3,807 passed, eight skipped, three failures and one
  setup error. All four unsuccessful tests rejected changing frozen source/runtime
  identities while implementation edits were still in progress; all pass when
  rerun with the source held fixed. The skipped tests require unavailable Torch.
- Changed Python files pass Ruff, shell launchers pass `bash -n`, and patch whitespace
  checks pass. Repository-wide `ruff check .` still reports 37 pre-existing issues
  under the unrelated `analysis/claude` directory.

## Delta commands

Upload the single portable archive `phase-c-fitting-m3.tar.gz` to Delta's home
directory. It contains the sealed existing CSTR inputs, source, focused tests,
configuration, `SOURCE_COMMIT`, and `SHA256SUMS`. No installation is required;
reuse the existing fitter Python and CasADi dependency directory.

```bash
bash <<'BASH'
set -euo pipefail
AF_ARCHIVE="$HOME/phase-c-fitting-m3.tar.gz"
AF_REV=$(tar -xOf "$AF_ARCHIVE" SOURCE_COMMIT)
AF_CODE="/projects/bibo/yxiao2/repos/phase-c-fitting-strategies-${AF_REV:0:7}"
mkdir -p "$AF_CODE"
tar -xzf "$AF_ARCHIVE" -C "$AF_CODE"
export AF_CAMPAIGN=/work/hdd/bibo/yxiao2/phase_c/fitting-strategies-v1
export AF_ACCOUNT=bibo-delta-cpu AF_CONCURRENCY=2
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_strategies_delta.sh"
BASH
```

Expected: one CPU preparation job, a 72-task CPU fitting array with concurrency
two, and a report job. Optimizer ceilings sum to 3.6 CPU-hours, plus qualification,
independent scoring, tests and overhead. A 20-minute scheduler allocation is not
extra optimization budget. The launcher records all scheduler intents/replies;
an uncertain response requires reconciliation instead of duplicate submission.

After completion (or to inspect partial progress):

```bash
bash <<'BASH'
set -euo pipefail
AF_REV=$(tar -xOf "$HOME/phase-c-fitting-m3.tar.gz" SOURCE_COMMIT)
AF_CODE="/projects/bibo/yxiao2/repos/phase-c-fitting-strategies-${AF_REV:0:7}"
export AF_CAMPAIGN=/work/hdd/bibo/yxiao2/phase_c/fitting-strategies-v1
bash "$AF_CODE/scripts/hpc/inspect_phase_c_fitting_strategies_delta.sh"
BASH
```

The inspector prints scheduler states, a grouped summary and the exact review
archive to download. It does not refit or submit anything. Keep all originals,
including failed/interrupted work and logs. A missing result is not zero error.

Exit: reconcile all 72 endpoints, compare recovery and actual time within the
same case/start, inspect mesh diagnostics and failed CSTR attempts, and choose
the next bounded change. Production integration requires a subsequent decision.
