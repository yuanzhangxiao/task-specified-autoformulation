# Fitting physical latent initial states

Local verification: the full repository run passed 1,318 tests with three
optional PyTorch tests skipped. After the final boundary-failure diagnostic
addition, all 60 focused fitting, initialization, smoothness and checkpoint
checks passed. Ruff and shell syntax checks passed. A stubbed scheduler verified
smoke/array/summary dependencies and duplicate-submission protection.

Standalone noiseless smoke: shared initial-value recovery gave train/validation
NMSE `3.12e-16 / 2.92e-16` and estimated the hidden initial value as `2.000000002`.
Causal-map recovery gave `1.39e-16 / 3.76e-16` and estimated `1 + 2*v01(0)`.
These are controlled checks, not results for the public candidate comparison.

This opt-in milestone fixes a missing optimization variable: previous C+S/C+P
experiments fitted parameters and later collocation states while holding the
physical initial state fixed. A zero-input trajectory cannot leave a zero
equilibrium just because the optimizer changes coefficients. Unknown initial
states must be represented as unknowns explicitly.

## Public initialization contract

Keep the two usual scientific choices. The new per-state reply schema is
`LatentInitializationReply` in `fitting/initialization.py`, with an accompanying
versioned prompt. The runtime selects the state and supplies allowed symbols.

```json
{"initial": {"mode": "value", "guess": 0.0}}
```

This introduces one **shared training-fitted** initial-state parameter. Zero is
only the optimizer's starting guess. Its domain is real by default. An explicit
scientific `role: "nonnegative_coefficient"` restricts a nonnegative initial
quantity; the runtime does not infer positivity from the state name.

```json
{
  "initial": {
    "mode": "map",
    "expression": "a + b*v01",
    "parameters": [
      {"name": "a", "role": "coefficient", "guess": 0.0},
      {"name": "b", "role": "coefficient", "guess": 0.1}
    ]
  }
}
```

Here `v01` means the measured value **at the first sample only**. Allowed symbols
are declared public initial target/auxiliary observations, initial input values,
numeric fixed covariates, time, and declared map-local parameters. Use public
channel names, not generated state aliases. Maps cannot read future observations,
private states, trajectory IDs, or other generated initial states. The existing
restricted parser and domain checks apply before compilation. Map-local parameter
names are deterministically namespaced as `init_<state>_<parameter>`; an equation
parameter with the same local spelling is not implicitly shared.

A rare explicit exception is
`{"initial":{"mode":"known","value":0,"justification":"..."}}`.
This preserves a known preparation/reset boundary. Zero input or zero observed
output alone is not a scientific justification for zero hidden states.

All directly observed state initials are bound to their own first measurements;
the proposer cannot optimize or override them. Identity observation mappings
determine which states are directly observed, even when their old `kind` field
says `latent`. A noninvertible or nonlinear observation is not silently inverted
to obtain a latent initial state.

## Runtime and compatibility

`LatentInitializationPlan(protocol="shared-latent-initialization-1")` is the
explicit opt-in. `apply_initialization_plan` returns a compiled candidate,
suggested parameter starts, and an audit. It adds global parameters referenced
by initial expressions, leaving RHSs and observation expressions unchanged.
It enables `ValidationContext.fitted_initialization` only for that compiled
model. Omitting the plan leaves old fixed initials and old validation behavior
unchanged; historical `fixed_value` records are never silently reinterpreted.

The fit adapter also accepts the plan directly. It persists the lowered
candidate, context, original plan, and scope audit in
`initialization_contract.json`. This is the model to replay with the returned
parameters. Initializer parameters are part of the ordinary global parameter
dictionary, not a trajectory-specific side table.

1. Collocation optimizes equation parameters, initializer parameters, and later
   node states. Its first boundary is `x(0)=I(initial_public_data; phi)`, not a
   fixed numeric constant. Infeasible but evaluable later-node guesses remain
   permitted.
2. Rollout refinement discards the later node values. It integrates from the
   same fitted boundary and optimizes the same combined parameter vector.
   Forward sensitivities now start with `S(0)=dI/d(theta,phi)`, including an
   identity block for directly parameterized latent initial values.
3. Smooth candidates use sensitivities. Piecewise candidates use the existing
   deterministic directional polling policy, including when the nonsmooth
   expression occurs only in the initialization map. Initial-map expressions
   participate in the smoothness audit.
4. Training and validation are freshly replayed with frozen parameters and the
   same boundary function. Initial observed measurements are allowed; no
   validation/test initial-state optimization occurs. Actual resolved physical
   initials are included in the fit report for both splits.

This milestone supplies the contract and adapter for Sol's next versioned
pre-fitting integration. It does not alter the frozen staged proposer prompt or
automatically relaunch ACES. Sol should select the new reply schema/prompt,
assemble the plan, version the cache identity, and pass the plan to the adapter.

## Controlled Delta experiment

`configs/physical_initialization_v1.json` pairs fixed and fitted latent initials
with identical equations, data, equation-parameter starts, and budgets. Both arms
use exact observed initial values. Each arm gets 120 seconds for collocation and
180 seconds for refinement; `ftol` is disabled for sensitivity refinement in both
arms. The existing polling method is unchanged.

Four synthetic families, two noise levels, and three starts give 48 fit tasks:

- A hidden reservoir with a nonzero shared initial state, including zero-input
  trajectories with nonzero output.
- A hidden initial state depending on initial observed output through a fitted
  affine map; validation includes new initial observations.
- A known-zero generating initial state, to check whether introducing an
  unnecessary degree of freedom damages held-out predictions.
- A threshold interaction `max(m-1,0)` with a nonzero hidden initial state.

Independent DOP853 reference generation uses no fitted graph. Noise is added to
later observations; the initial measured boundary is exact in these controls.
Clean reference outputs are stored separately and opened only after fitting.
Recovery requires verified replay and both clean NMSEs at most `1e-4`.

The four public candidates are imported from the existing
`piecewise-fitter-v1/problems` snapshots. Their original frozen hashes are verified, and
only their public train/validation data are copied into the new experiment.
The fitted arm explicitly treats old numeric latent initials as guesses for
shared unknowns. Existing analytic initials are retained; the experiment does
not invent causal maps for public candidates. This is a declared initialization
hypothesis, not a change to a finalized benchmark prompt. Four public cases add
eight tasks, for **56 fits total**, one CPU each, at most two concurrent tasks.

Every task saves its start marker, collocation result, fitted parameters and
independent Radau/BDF replay checks. Completed stages are reused. A killed native
fit without a saved fit is recorded as interrupted and is not silently rerun with
a fresh budget. A worker log and stage journal distinguish startup, fitting and
replay. Full native IPOPT/SciPy solver-state continuation is not supported.
The summary command uses only the Python standard library and saved JSON.

On Delta, from the isolated checkout at the pushed commit:

```bash
bash scripts/hpc/submit_initialization_delta.sh
```

The launcher first submits a ten-minute CPU smoke job, then submits the array
with an `afterok` dependency and a file-only summary with `afterany`. Defaults:

```text
AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/piecewise-fitter-v1
AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/physical-initialization-v1
```

Read the report:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/physical-initialization-v1/summary.md
```

To rebuild a summary without allocating a job:

```bash
/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python -S \
  scripts/run_initialization_campaign.py summarize \
  --output /work/hdd/bibo/yxiao2/phase_b/physical-initialization-v1
```

To resume a particular incomplete task without discarding checkpoints, export
the five `AF_*` variables above, set `AF_REPO_ROOT=$PWD` and
`AF_CODE_COMMIT=$(git rev-parse HEAD)`, then submit:

```bash
sbatch --array=TASK_INDEX --export=ALL scripts/hpc/initialization_delta.slurm run
```

Replace `TASK_INDEX` with an actual task index from `freeze.json`. Saved numerical
failures remain failures. Requesting a fresh numerical attempt requires a new
output directory so its budget and provenance remain explicit.

## Interpretation and remaining limits

This can remove a zero-equilibrium obstruction; it cannot guarantee that the
candidate equations explain the data. Collocation may still time out, and a bad
ordinary parameter start can still fail numerical integration. A shared fitted
initial state also assumes a shared preparation. When trajectories have different
unknown initial conditions, use a justified causal map or specify a different
inference protocol. No arbitrary per-trajectory hidden initials are fitted here.

The general expression grammar still has limits; this is not universal support
for every continuous piecewise formula. Polling belongs to established direct
search methods, but our finite-direction implementation is not full MADS.
NOMAD is a mature future comparison if needed:
https://nomad-4-user-guide.readthedocs.io/en/latest/Introduction.html
