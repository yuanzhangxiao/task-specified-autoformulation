# Final bounded fitting comparison

This is the last planned investigation of the difficult reference case before
freezing the fitter for the next pre-fitting phase. It does not launch follow-up
jobs, increase budgets after failures, or change production fitting defaults.
Poor fits and resource failures remain in the report. Any useful result is
reviewed before adoption; solver agreement alone is never called recovery.

## Frozen matrix

The source is the completed parameter-freedom campaign, preferably its memory
rerun. Only frozen inputs, ordinary starting guesses, and gate provenance are
imported. None of its fitted parameters, near starts, validation rankings or
collocation checkpoints is used to initialize this comparison.

For each of two reference families, run four routes:

| Family | Information supplied | Adjustable parameters |
| --- | --- | ---: |
| fixed_shapes | Correct skeleton, exact shape constants, known physical hidden boundaries | 26 |
| free_shapes | Correct skeleton and known physical hidden boundaries; generic shape guesses | 48 |

| Route | Search procedure |
| --- | --- |
| collocation_exact | Current mapped collocation with exact Hessian, then existing feasibility/sensitivity recovery |
| direct_multistart | Three independent sensitivity fits from the ordinary, seeded perturbed, and uniformly smaller parameter guesses |
| horizon_continuation | Sensitivity fits on 25%, 50%, then 100% of each training trajectory's original time span; parameters carry forward |
| collocation_limited | Same collocation/recovery route, mesh, starts and budget as exact, with limited-memory Hessian approximation |

The ninth run fits the smaller model below with exact collocation followed by
the existing recovery route. It is a model-adequacy experiment rather than a
recovery experiment for the generating skeleton. No near-start run is included.
These nine configurations are not independent estimates of population success
rates. The exact-Hessian controls are rerun at the new shared budget, not copied
from a differently budgeted historical fit.

## Smaller model

Let y = v01, z = (y - center)/S and U = u01/Su. Center, S and Su are frozen from
training observations only; S is the normal training output standard deviation,
and Su is max(training input standard deviation, 1). The state equations are:

```text
memory'   = -rate_m*memory + gain*U
feedback' = omega*z - rate_f*feedback
y'        = S*(a*z - b*z^3 - omega*feedback + h*memory + d*U + c)
```

The memory is input driven. The z/feedback pair provides persistent reciprocal
coupling, and the cubic term permits nonlinear feedback with saturation. Output
is the integrated observed state. All states are signed scaled quantities.
rate_m, rate_f, omega and b have positive rate-role domains; the other five
coefficients are signed. These are scientific design choices for this single
hand-specified hypothesis, not runtime-inferred universal sign rules.

There are nine equation parameters and four initial-map parameters. The observed
y(0) is exactly its initial measurement. Each hidden state has a shared fitted
map `offset + slope*z(0)`, frozen before validation. No hidden reference boundary,
trajectory identifier, validation-fitted initial condition, or future output
enters those maps. The frequency starting guess uses the median dominant Fourier
frequency of regularly sampled training outputs, with an inverse-span fallback.
It is adjustable and is not a recovered hidden frequency. The structure itself
is fixed before fitting and is not an LLM proposal or judge-certified model.

This public-initialization contract differs from the reference controls' known
hidden boundaries. Some supplied trajectories may have identical public inputs
and initial output but different hidden preparations. Any deterministic model
with these public initial maps must generate the same response for such a group.
The report therefore includes the training-only lower bound for indistinguishable
preparations. Failure of the smaller model cannot be attributed solely to model
size or the optimizer. No benchmark observations are altered to remove this issue.

## Equal bounded opportunity

Every fit allocation requests one CPU, 64 GB and two hours. At most two array
tasks run concurrently. Nine fits reserve at most 18 CPU-hours, plus a ten-minute
8-GB numerical gate and a ten-minute 2-GB summary job. No ACES/GPU job is needed.

Each route has 4,800 seconds for search plus full-training selection:

- Reserve 240 seconds and 12 residual calls for full-training candidate checking.
- Collocation routes receive 1,200 seconds for C and 3,360 seconds for recovery.
- Direct multistart receives three separate 1,520-second sensitivity allocations.
- Horizon continuation receives 912, 912 and 2,736 seconds on the three horizons.
- Search is capped at 108 rollout residual calls: 36 per direct/horizon stage,
  or 108 shared across the existing C recovery screen/refinement/polling stages.
  Collocation's node evaluations are separate, bounded by its time/iteration caps.
- State-only checks are capped at 60 seconds each; augmented sensitivity
  evaluations at 180 seconds. Unused stage allocations are not redistributed.

The default numerical tolerance, sparse sensitivity solver Jacobian, input
interpolation and full observation objective are retained. Reference C uses the
24,000-variable target with at least 120 intervals per trajectory. The three-state
model uses a 12,000-variable target with the same minimum resolution; this roughly
preserves time-point density while halving the state dimension. Actual mesh
diagnostics are saved. All observations remain in each full-horizon objective.

Prefixes keep original sample times, input values, interpolation segments and
physical initial conditions. They never shift a window's left boundary or fit
independent states at a later window start. All stages use the same full-training
normalization. Prefix objective values are never compared with full-horizon
objective values. Their parameter estimates must undergo full-training rollout
before joining the final candidate pool. The best finite full-training estimate
from any stage is retained even if a later stage or check fails.

Direct routes use the same guarded sensitivity oracle as the existing adapter.
Invalid first sensitivities fail explicitly; invalid later trials are rejected
without supplying a fabricated zero Jacobian. Direct multistart gives each
prespecified starting point an actual optimization allocation, rather than
ranking only its initial residual. The C routes retain existing bounded recovery,
including directional polling when sensitivities are unavailable. Route logs
make any such fallback explicit.

## Checkpoints, interpretation and stopping

Input, start, configuration and executable identities are immutable. Completed
stages are reused. An interrupted stage is charged its full allocation and is
not restarted; a saved finite incumbent can be retained before proceeding to
untouched later stages. No interrupted optimizer receives another iteration/time
budget. Final parameters are checkpointed before independent replay. A stage
failure is explicit, including if later work produces a useful fit. Submission
is idempotent; uncertain partial submissions require queue reconciliation.

Native IPOPT iteration and timing output is enabled for these C runs and saved
in `results/task_*/worker.log`. Native calls killed by the deadline may not print
their final timing table, so iteration checkpoints and the last phase remain
the evidence; no missing timing is invented. The production default remains
`collocation_hessian=auto` and silent solver logging. Explicit exact Hessians
are rejected for piecewise graphs requiring the first-order solver policy.

Separate BDF/Radau replays have 240 seconds each per split, after parameter
selection. Replay agreement means numerically consistent integration of the
fitted equations, not accurate observations or scientific correctness. Report
training/validation NMSE and all three prespecified descriptive bands:

- strict: both NMSEs at most 1e-4;
- good: both at most 0.01;
- practical: both at most 0.1.

All bands also require numerical replay agreement. They are review aids, not
claims of scientific adequacy, unique parameters or correct named latent states.
Training alone selects within a route. No cross-route winner is automatically
selected using validation. The private diagnostic never opens test data or
passes reference information to the proposer/judge.

After this batch: review any transferable improvement. If none works, park the
difficult case and freeze the chosen existing fitter configuration for pre-fitting
experiments, retaining this limitation. Even an improvement does not trigger an
automatic extension of this campaign.

## Delta commands

From the new clean pinned checkout:

```bash
export AF_REPO_ROOT="$PWD"
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-parameter-freedom-v1-mem64
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-final-alternatives-v1
export AF_ARRAY_CONCURRENCY=2
bash scripts/hpc/submit_final_fitter_delta.sh
```

The source original `fitter-parameter-freedom-v1` directory is also acceptable if
its immutable inputs and gate remain intact; no historical fitted result is used.
The gate runs all four numerical routes on a separate small control before the
fit array becomes eligible. A failed gate blocks all fits and remains in summary.

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-final-alternatives-v1/summary.md
cat /work/hdd/bibo/yxiao2/phase_b/fitter-final-alternatives-v1/diagnostics.json
```

Summary regeneration uses the standard library and runs safely on the login node:

```bash
"$AF_PYTHON" -S scripts/run_final_fitter.py summarize --output "$AF_OUTPUT_ROOT"
```
