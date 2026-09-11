# Exact continuous piecewise functions in the fitter

The collocation adapter now accepts bare and composed `abs`, `min`, and `max`,
including `max(x-10,0)`, fitted thresholds, and nested clipping. Scientific
expressions are preserved exactly. There is no softplus substitution, smoothing
width, squared-rectifier replacement, or change to parameter signs/domains.
The restricted parser remains the execution boundary; this adds no Python
callbacks or arbitrary conditional language.

## Numerical routes

`CollocationSensitivityConfig.piecewise_policy` defaults to `allow` in this
opt-in adapter. `reject` reproduces the earlier conservative capability policy.
`SymbolicODE` itself still defaults to the earlier policy for legacy probes;
pass `allow_piecewise=True` when explicitly constructing the expanded graph.

The `piecewise_refinement` setting has three choices:

- `auto`: classical sensitivity refinement for the previously certified
  expressions; derivative-free directional polling when a bare piecewise
  primitive remains.
- `branch_sensitivity`: use the existing rollout least-squares optimizer with
  branch derivatives. This is an experimental nonsmooth heuristic.
- `directional_poll`: use exact full-training rollout costs without parameter
  derivatives, including on smooth models when explicitly selected.

At ties, the existing CasADi translation selects derivative zero for `abs(0)`
and averages the two branch derivatives for binary `min`/`max`. Multiple
arguments are folded left. These are documented primitive selections, not a
claim that automatic differentiation produces the classical or Clarke Jacobian
of every complete composite or trajectory map. The audit marks this distinction.
Ordinary transverse-crossing tests compare branch sensitivities with independently
perturbed production rollouts. General event, grazing and sticking sensitivity
certificates are not implemented.

The collocation initializer keeps the exact Radau constraints and uses IPOPT
with limited-memory curvature for piecewise graphs. This is explicitly a
heuristic: IPOPT's documented smoothness assumptions do not become satisfied
merely by supplying a generalized derivative. A failed initializer falls back
to the ordinary parameter start, and `rollout_or_observed` still permits node
initialization without a successful initial rollout.

Directional polling evaluates both the collocation and ordinary starts, then
positive/negative coordinate directions and deterministic rotating orthogonal
directions. Its initial wider search uses radii 1, 4 and 16; subsequent radii
expand after sufficient improvement and contract otherwise. Parameter scales
are frozen as `max(abs(ordinary_start),1)`. Feasible points stay inside the
existing bounds. The best feasible training evaluation is retained; penalties
for failed rollouts never become fitted parameters. All decisions use training
only. The state integrator may still use a local state Jacobian for its Newton
iterations; the polling optimizer requests no parameter sensitivities.

This bounded experimental direct search is **not MADS** and makes no global
optimality or nonsmooth-stationarity claim. `optimizer_success=false` is
intentional for polling: its stopping reason is a budget or searched-radius
limit. Numerical completion and trajectory recovery are reported separately.
It may cost substantially more rollouts than sensitivity fitting.

## Mesh, checkpoints and numerical verification

`collocation_mesh_substeps` adds 1–8 integration subintervals per observed
interval. It adds latent decision nodes but never interpolates extra observed
target losses. Initial states remain fixed/causal; intermediate node values
are discarded for final scoring.

Candidate/context, training identity, starts, configuration, source and numerical
dependencies bind the adapter checkpoint. The campaign additionally hashes all
input bytes and launcher files. Initializer results and completed arms are
reused. Polling checkpoints preserve candidate evaluation order, incumbent,
remaining queue, radii and cumulative time/evaluation budgets. Native IPOPT
iteration state is not resumable: an interrupted initializer falls back to the
ordinary start without a fresh initialization budget. An interrupted branch
refinement is recorded as incomplete rather than rerun with an extra budget.
An in-flight call lost to an external kill can be repeated; completed checkpointed
evaluations are retained. Native process workers have hard wall-clock limits.

Each returned fit is replayed from the original initial conditions by the
production evaluator. The campaign then independently checks tighter Radau and
BDF trajectories on train and validation. Agreement requires maximum prediction
difference below `1e-4` in training-standard-deviation units. This establishes
numerical agreement, not model adequacy. All normalization uses training data.
Clean synthetic references are opened for scoring only after parameters freeze.
The summary command reads saved records and never runs an ODE solver.

## Frozen Delta comparison

`configs/piecewise_fitter_v1.json` defines 24 synthetic paired jobs:

- four cases: threshold inside the ODE, threshold in an observation, fitted
  observation threshold, repeated nested clipping/absolute-value crossings;
- observation noise fractions 0 and 0.03;
- three fixed starts, including inactive and threshold cases.

Every job runs four arms: C+S and C+P, each at mesh subdivision 1 and 2. Here
C means the shared collocation initializer, S branch-sensitivity refinement,
and P derivative-free polling. Both refinements receive the same initializer
point at each mesh, the same original training observations, a 120-second
refinement ceiling and 240 evaluation ceiling. The shared initializer has a
45-second ceiling. Report its saved cost once per arm for fair total-budget
accounting; actual second-arm wall time naturally excludes the reused solve.

References are generated by independent explicit DOP853 equations, split at
every supplied-input knot. Synthetic generation has no benchmark loader or
latent labels. The optional existing `collocation-node-cases-1` bundle adds
one paired job per exported public candidate, without filtering by previous
fitting success. Every parent and committed revision in the bundle is retained.
The numerical-only campaign uses zero LLM calls and opens no test/private data.

The launcher uses Delta's `cpu` partition, account `bibo-delta-cpu`, one CPU,
16 GB and 50 minutes per paired job, with at most two jobs running together.
A 10-minute summary job depends on completion of the whole array. No ACES/GPU
job is required.

On Delta, in the clean checkout of this commit:

```bash
export AF_REPO_ROOT="$PWD"
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/collocation-node-cases-v1
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/piecewise-fitter-v1
bash scripts/hpc/submit_piecewise_delta.sh
```

Defaults use the existing Python environment
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python` and CasADi installation
`/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps`. The launcher refuses a dirty
checkout, changed frozen inputs, or an ambiguous duplicate submission.

```bash
cat /work/hdd/bibo/yxiao2/phase_b/piecewise-fitter-v1/submission.json
cat /work/hdd/bibo/yxiao2/phase_b/piecewise-fitter-v1/summary.md
```

For synthetic-only local verification, omit `--source`:

```bash
PYTHONPATH=src python scripts/run_piecewise_campaign.py prepare --output /tmp/piecewise-smoke
PYTHONPATH=src python scripts/run_piecewise_campaign.py run --output /tmp/piecewise-smoke --task-index 6
PYTHONPATH=src python scripts/run_piecewise_campaign.py summarize --output /tmp/piecewise-smoke
```

## Remaining boundaries

This milestone supports continuous piecewise expressions formed from the
existing accepted arithmetic/functions with finite local branches. It does not
promise support for every continuous mathematical function. Restricted
`sqrt`/`log` domains and negative/variable powers retain their earlier sensitivity
adapter restrictions. Existing guarded-division semantics remain unchanged and
are not a global continuity proof. General discontinuous resets, arbitrary
conditionals, state constraints and fitted latent initial states are outside this
adapter's current scope. Failures must be identified as numerical capability or
optimization outcomes, not scientific evidence against a threshold mechanism.

## Local verification

The full repository suite passed **1,303 tests**, with three optional PyTorch
skips. After a checkpoint-write ordering adjustment, all 21 focused adapter,
piecewise and campaign tests passed again. Ruff and shell/diff checks passed.
Tests cover exact primitive tie derivatives, a transverse ODE crossing against
independently perturbed rollouts, inactive-start recovery, two mesh resolutions,
failure-penalty exclusion, deterministic interrupted polling resume, frozen
input tampering, data separation and reference/production equivalence.

Twelve final campaign arms (three representative cases, two meshes, two methods)
passed independent Radau/BDF replay. In the fitted-threshold control starting
at `b=16`, C+S retained `b=16` with clean validation NMSE `1.27489`; C+P recovered
`b=10.0000094` with clean validation NMSE `3.58e-11` at both meshes. The true
threshold is 10. Both methods recovered the ODE-threshold control, and their
clean validation NMSEs were approximately `4.75e-6` in the noisy repeated-clipping
control. An earlier four-arm observation-threshold smoke also recovered the
known parameter. These are implementation controls, not evidence that polling
is generally faster or better on proposed benchmark models. The full frozen
Delta comparison remains to be run.

Primary background: [IPOPT's smoothness requirements](https://coin-or.github.io/Ipopt/),
[generalized ODE sensitivities](https://arxiv.org/abs/2201.03819), and
[mesh adaptive direct search](https://epubs.siam.org/doi/10.1137/040603371).
