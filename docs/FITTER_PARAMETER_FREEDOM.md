# Parameter freedom and optimization opportunity

The v3 reference preflight verified accurate fixed-parameter collocation and
consistent sparse sensitivities. Its six parameter fits exhausted both the
300-second collocation and 600-second refinement allocations. The selected
sensitivity stages used only 4–10 residual calls. These are budget-limited fits,
not evidence of convergence to poor local optima.

This milestone isolates fitting nonlinear scales/biases from fitting weights
and rates. It uses exactly the noiseless synthetic observations, forcing,
initial boundaries and generating family from the completed v3 reference
campaign. It never reads test observations or passes private reference facts to
the proposer or judge. It does not revise the public prompt or benchmark data.

## Six comparisons

For each ordinary/near **weight** start:

| Arm | Fitted parameters | Shape values at start | Shape values during fitting |
|---|---:|---|---|
| fixed_basis | 26 | Reference | Fixed at reference |
| free_shapes_matched | 48 | Reference | Fitted |
| free_shapes_original | 48 | Original v3 start | Fitted |

The first two arms start from identical physical models, with identical weights
and initial conditions. Both receive privileged reference shape information.
The original arms separately measure what a larger budget does for the original
v3 starts. All near starts are reference-assisted: the original near control
also starts with perturbed reference shapes. Every arm receives known
per-trajectory hidden initial values. An ordinary weight start is not a blind discovery of
the generating model: even the original arm is given its skeleton and initials.

The parameter partition is derived from the validated reference AST: parameters
inside function arguments are fixed to full-precision literals. The transformed
candidate is revalidated and CasADi must certify the remaining dynamics and
observation functions as affine in the remaining parameters. The frozen manifest
records all removed values and starting points. There is no Python evaluation
of expression text. This narrow classifier fails closed outside the expanded
reference family; it is not a general production parameter-role inference rule.

This is closer to the strict graph-meta-model formulation with fixed nonlinear
functions and fitted outer weights. It is still nonlinear jointly in unknown
state nodes and weights. It does not assert named latent-state identifiability.

## Budgets and numerical choices

Every arm receives one CPU, 16 GB, at most 20 minutes for collocation and
60 minutes for refinement. Refinement has a 120-residual-call cap, including
screening; the native optimizer can converge earlier. Screening is additionally
capped at 240 seconds, so increasing the refinement allowance cannot enlarge
its former 40% share indefinitely. Individual state/sensitivity limits remain
60/180 seconds. Budgets permit more optimization, not a guaranteed number of
accepted iterations.

The same sparse sensitivities, tolerances, exact smooth-collocation Hessian,
trust-region least squares, original input interpolation, and corrected reduced
state mesh are retained. All observations remain in the objective. A preflight
requires matching state meshes, matching starting rollouts and matching weight
sensitivity columns for the paired arms. The v3 fixed-node gate is required as
source evidence; its expensive solves are not repeated.

The production defaults and their standard budget ceilings are unchanged.
Larger budgets require an explicit extended_diagnostic configuration.

## Checkpoints, verification, and reporting

Parameter estimation is saved before production replay. Separate BDF and tighter
Radau replays use at most 240 seconds per method per split. Thus a replay timeout
cannot erase a completed parameter fit. Checkpoints include code, runtime,
input and start identities. Repeating submission returns existing job IDs.
An interrupted native fit or replay cannot receive a fresh budget on resume;
completed parameter estimates and completed replay checks are reused.

Workers and their process groups have a 110-minute outer limit; Slurm allocations
are 120 minutes. At most two fitting tasks run concurrently. The six fit jobs
therefore request at most 12 allocated CPU-hours, plus short smoke/gate/report
jobs. No GPU or login-node numerical fitting is needed.

Reports distinguish collocation acceptance, optimizer native success, replay
verification and strict synthetic recovery (both NMSEs <=1e-4). Practical NMSE
values remain visible when strict recovery is missed. Diagnostics include
symbolic setup, construction/solver phase, time to the first collocation callback,
recorded iterations, screening time, sensitivity-stage time and stopping reasons.
Time before the first callback can include native derivative compilation; it is
not interpreted as an optimization iteration. A phase still running at a hard
deadline remains explicit instead of being assigned a fabricated duration.

Per-task progress_summary.json includes training costs at evaluations and
optimizer iterations. Evaluation time sums exclude optimizer overhead; elapsed
callback times include it. No validation scores enter these progress decisions.

## Delta

From a clean pinned checkout of codex/fitter-parameter-freedom-v1:

```bash
export AF_REPO_ROOT="$PWD"
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-resolution-v3
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-parameter-freedom-v1
export AF_ARRAY_CONCURRENCY=2
bash scripts/hpc/submit_parameter_freedom_delta.sh
```

After completion:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-parameter-freedom-v1/summary.md
cat /work/hdd/bibo/yxiao2/phase_b/fitter-parameter-freedom-v1/diagnostics.json
```

Rebuild reports without numerical packages or another allocation:

```bash
"$AF_PYTHON" -S scripts/run_parameter_freedom.py summarize --output "$AF_OUTPUT_ROOT"
```

The training-trajectory evidence packet for the proposer is a separate
coordinated milestone. Adaptive meshes, new input interpolation, alternate
optimizers, and model simplification are intentionally not bundled into this
comparison.
