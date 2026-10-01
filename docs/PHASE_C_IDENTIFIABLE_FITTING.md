# Phase C fitting milestone 2: jointly fit identifiable latent controls

Agreed 2026-09-30. Protocol `phase-c-identifiable-fitting-1`; configuration
`configs/phase_c_identifiable_fitting_v1.json`. Fitting owns this milestone;
construction remains separate. No benchmark releases, prompts, production
fitting defaults, test payloads or LLM calls change.

## What the earlier alternating variant actually did

There are distinct algorithms in the repository:

- The recent M1 joint collocation initializer optimizes state values at time
  nodes, dynamical parameters and the permitted initial-value parameters together
  in a constrained nonlinear problem. Trajectories and parameters must agree with
  each other, so collocation estimates both. It returns parameter checkpoints;
  fitted node values never substitute for the final free rollout.
- The earlier conditional/alternating experiment, in `conditional_optimizer.py`,
  solves linear outer coefficients with node trajectories and nonlinear shape
  parameters fixed. It then holds those linear coefficients fixed and uses IPOPT
  to update trajectories and nonlinear shape parameters **jointly**. It repeats
  these blocks with bounded inner iterations. Initial conditions are fixed in
  that earlier experiment; it does not establish joint initial-state recovery.
- Rollout refinement eliminates the trajectory-node optimization variables. Every
  parameter/initializer trial is integrated from the allowed observed boundary;
  the optimizer changes the coefficient and initializer vector to reduce complete
  training rollout residuals. The new controls use forward-sensitivity Jacobians
  and bounded trust-region reflective nonlinear least squares.

Thus the user's recollection of an alternating variant is correct. The other
block was indeed optimized jointly, and the overall procedure alternated; it
was not a one-step globally optimal solve. This milestone does not yet compare
joint versus alternating **collocation**. It first establishes joint recovery
and tests how to use the resulting checkpoints during **rollout refinement**.

## Three controls with a conditional uniqueness argument

Use noiseless synthetic experiments, independent of the 28-cell release:

\[
\dot y=z,\qquad \dot z=-az-by+cu-dy^3,\qquad y\text{ observed},\ z\text{ latent}.
\]

| Control | a | b | c | d | Purpose |
|---|---:|---:|---:|---:|---|
| Linear | 0.8 | 1.3 | 1.6 | absent | Basic joint coefficients/latent-initial recovery |
| Nonlinear | 0.6 | 0.8 | 1.4 | 0.7 | Nonlinear state dynamics and nonconvex rollout fitting |
| Fast/slow | 18 | 2 | 2 | 0.5 | Separated dynamical timescales and a short initial transient |

The unit coefficient in `y'=z` fixes the latent scale. Two parameter vectors
producing the same ideal continuous output must have the same `z=y'`. Subtracting
their equations gives a linear dependence among `(-y', -y, u, -y^3)`. If those
functions are independent on the training trajectories, all coefficient
differences vanish. The latent initial is then the right derivative `y'(0)`.
Omit the cubic column for the linear case.

We check this sufficient excitation condition on evaluator-only reference
trajectories, and separately check the rank and singular spectrum of the
sampled-output sensitivity matrix, including the unknown initial value. The
analytic argument concerns ideal continuous observations under the excitation
condition. Numerical rank is evidence for the chosen finite design, not a general
symbolic proof, a finite-sample global uniqueness theorem or noise robustness.
A regression test deliberately frees the gain in `y'=k*z`; the resulting scaling
ambiguity is detected as a deficient local sensitivity matrix.

All runs share unknown `z(0)=0.4`, as specified by the synthetic public preparation
contract. Initial y is observed and varies across runs. There are three training
trajectories and two validation trajectories, each sampled at 0.1 time units to
12. Public u is a continuous piecewise-linear sequence with knots every unit;
validation changes its phase and initial y. References are independently
integrated with every input corner as a boundary. No hidden trajectory,
observation derivative or reference coefficient is passed to fitting.

Starts use generic coefficients (1 for a/b/c, 0.3 for d) with seeded log-scale
perturbations, plus a signed latent-initial guess drawn from [-0.8, 0.8]. These
rules do not consult the reference vector. Positive coefficient domains belong
to the supplied control template; the latent state and its initial are signed.

## Matched comparison: 27 endpoints

Three controls × three numerical starts × three refinement arms:

| Arm | Refinement policy |
|---|---|
| `joint` | Screen saved pairs, then jointly optimize coefficients and initializer; ordinary native stopping plus budget limits |
| `joint_stopping` | Same, with training-accuracy and accepted-step stagnation stops |
| `conditional_stopping` | Same stops; bounded conditional block rescue of distinct saved pairs before final joint refinement |

Every case/start creates **one** shared collocation run. Its numerical coordinates,
physical node guesses and complete saved parameter pool are frozen and reused
across all arms. Observed nodes start from measured training samples; latent nodes
start at the initial guess. There is no time-limited rollout warm-up, so elapsed
compilation time cannot change the node starting vector as it did in M1.

Collocation retains the existing converged, best-feasible, least-violating and
latest points. An additional eight-entry pool retains its first sampled iterate
and recent iterates, sampling every five iterations, regardless of objective rank.
These are suggestions to replay, not certified solutions or a native solver
checkpoint. They store complete coefficient/initializer pairs, not independently
reusable latent node trajectories. Older/midway iterates can still be evicted;
this is a bounded approximation to an archive, not preservation of every point.

Conditional rescue retains the lowest-error screened pair and then chooses pairs
far apart in frozen scaled parameter coordinates. It considers unscreened and
poor-rollout pairs too. For at most three pairs it fits the initial-value block
with dynamical coefficients fixed, then coefficients with the resulting initial
fixed. Each block gets at most 12 residual calls. Complete training rollouts rank
the retained incumbent across every stage; joint refinement then uses the remaining
allowance. This tests the user's concern that a good block can be hidden by a poor
partner without treating arbitrary spliced pairs as verified solutions.

## Budgets, stopping and certification

The shared initializer has 60 seconds. Each arm receives up to 120 seconds and
180 actual full-training residual calls for screening, conditional blocks and
joint refinement combined. Screening consumes at most one third of calls and
one quarter of the time allowance. Conditional work ends by 65% of the total
allowance, reserving the remainder for joint work. Each rollout is also bounded.
Sensitivity integration accompanies a residual call; a matching Jacobian is
reused. These are evaluation allowances, not promises of equal CPU cost per call.
Report elapsed seconds alongside calls.

For accounting comparisons, charge the 60-second initializer allowance to every
arm, although the physical initializer is executed once per case/start. Three
seeds are repeated numerical starts, not independent scientific problems.

The two stopping arms stop when training NMSE <= 1e-10 **and** every training
trajectory's NMSE <= 1e-9, or after four accepted steps each show relative cost
improvement <= 1e-7 and scaled step norm <= 1e-5. Ordinary native gradient/step
termination and hard budgets also apply. An accuracy stop awaits independent
replay; a stall is not convergence to the correct solution. These criteria do
not inspect validation error or reference parameters.

The evaluator first confirms that the reference coefficients reproduce both
splits at NMSE <= 1e-8 and that Radau/DOP853 agree within 1e-4 training standard
deviations. After each arm ends, it freezes parameters and independently replays
both splits. Recovery passes only if:

1. both output NMSEs <= 1e-6 and the independent solvers agree within 1e-4;
2. maximum relative error across all coefficients **and the shared initial** <= 1%;
3. latent-trajectory NMSE <= 1e-4 on both splits, scaled by the reference training
   latent standard deviation.

Private recovery metrics are evaluator-only and cannot influence a checkpoint,
parameter update or stopping rule. Reports keep all failures and missing rows;
execution complete does not mean recovery passed. The fit stage never receives
validation arrays. Final validation uses the training-fitted shared initial.

## Resume and limitations

Input content, source, runtime, configuration, physical starts, nodes, initializer
pools and results are hashed. A completed endpoint is reused. If a backend finished
before interruption, scoring can resume without refitting. An interrupted native
optimizer is recorded or blocked; it never silently receives a fresh allocation.
This is deterministic artifact reuse, not serialization of IPOPT/TRF internals,
nor a guarantee of identical timed endpoints across machines.

The coefficients enter these vector fields linearly, although their joint
rollout fitting problem is nonlinear. Intrinsically nonlinear shape parameters
are not yet qualified by these controls.

These supplied equations are assisted controls. They do not establish fitting
success on discovered equations, noisy observations, many hidden states, arbitrary
initialization maps or model mismatch. Scaling and conditional optimization are
established techniques; empirical benefit and methodological novelty are separate
questions. CSTR is not called identifiable merely because a reference vector fits.
The optional `audit-cstr` command computes its local sensitivity evidence at the
reference vector and explicitly leaves structural identifiability uncertified.

Exit: inspect all 27 endpoints, failures, stopping reasons and paired costs;
establish reproducible joint recovery on these controls before broadening the
identifiable roster. If an arm fails, diagnose that failure first. No automatic
promotion into construction or production defaults occurs in this milestone.

## Local qualification and CSTR follow-up

The native linear-control smoke recovers all coefficients and the shared initial
(maximum relative parameter error 3.75e-9), with training/validation output NMSE
2.22e-18/2.16e-18 and latent NMSE below 2.4e-18. Artifact resume is identical.
The three reference controls pass independent replay and have column-normalized
local sensitivity minimum/maximum singular ratios 0.0888 (linear), 0.1055
(nonlinear), and 0.0141 (fast/slow). These checks qualify the controls; they are
not the pending 27-endpoint comparison or evidence of one arm's superiority.

The optional audit was also run on the existing M1 CSTR-hard training inputs,
using all seven dynamical coefficients plus two shared hidden initials at their
reference values. Training NMSE is 2.57e-17. All nine local sensitivity directions
have numerical rank: the column-normalized singular values range from 2.63238 to
0.006533, a ratio of 0.002482 and condition number about 403. The corresponding
normalized Gauss–Newton matrix has condition number about 1.62e5.

This does not exhibit an exact local scaling ambiguity of the simple `c*z` kind
at the reference. There is nevertheless considerable local coupling after column
normalization. Global uniqueness, other parameter regions and robustness to
noise remain unresolved. It would be premature to explain the M1 joint-fitting
failure as structural non-identifiability. The audit used the existing guarded
CSTR template's branch derivatives; it performed no parameter optimization and
opened no test data.

## Delta execution

Upload the supplied `phase-c-identifiable-fitting-m2.tar.gz` to home, extract into
a fresh durable project directory, verify `SHA256SUMS`, and run:

```bash
bash scripts/hpc/submit_phase_c_identifiable_delta.sh
```

Defaults use `/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`, the existing
CasADi dependency directory, and
`/work/hdd/bibo/yxiao2/phase_c/fitting-identifiable-v1`. Override `AF_PYTHON`,
`AF_CASADI_ROOT` or `AF_CAMPAIGN` only before the first submission. The launcher
prints exact paths and checks the SciPy callback API. It submits CPU preparation,
27 array tasks (concurrency 2), and a report. Fits request one CPU, 8 GB and 15
minutes of scheduler time; optimizer work is limited separately as above. No GPU.

The prepare job runs the focused tests, evaluator qualification and the nine
shared initializers. Saved scheduler intents prevent duplicate submission after
uncertain replies; inspect receipt files and `sacct` before recovery. Repeating
the unchanged successful launch reuses job IDs and the frozen campaign.

After completion:

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/fitting-identifiable-v1
jq '{status,expected,recorded,status_counts,recovery_passes,paired_comparisons}' "$AF_ROOT/summary.json"
tar -czf "$AF_ROOT/review.tar.gz" -C "$AF_ROOT" \
  plan.json inputs.json summary.json qualification common results submission_manifest.json logs
```

Download `review.tar.gz`. Preserve the original campaign; no cleanup is part of
this milestone. Optional CSTR diagnostic, in a CPU allocation with the same code:

```bash
"$AF_PYTHON" scripts/phase_c_identifiable_fitting.py audit-cstr \
  --inputs /path/to/previous/fitting-m1/inputs.json --root /path/to/new/cstr-audit
```

`--inputs` must be the actual sealed M1 input artifact, not a results archive or
benchmark private directory. The command reports local conditioning only, performs
no fitting, and does not access final test data.

## Verification record

The final native smoke repeats the linear-control recovery above and confirms
artifact reuse. The new suite has 19 passing tests covering excitation failure,
latent-scale ambiguity, physical node identity/shape/boundary checks, conditional
rescue, scaled Jacobians, stopping, inner versus overall time limits, bounded
checkpoint retention, source drift, budget-safe resume and scheduler receipts.
The 14 existing coordinate/qualification tests also pass.

The repository-wide `pytest -q -n 2` run reported 3,766 passed, eight skipped
(optional Torch dependency absent), and six failures while the shared checkout
was being edited. All six failing cases passed on rerun after those edits:
five construction-submission tests and the historical matched-refit resume test.
This records the actual run and rerun, rather than claiming an uninterrupted
whole-suite pass at the final snapshot. `ruff check` passes for every changed
Python file. `ruff check .` still reports 37 findings in the existing
`analysis/claude/{afload,make_figures,make_figures_final,make_tables}.py` files.
The Delta shell scripts pass Bash syntax checks. The live 27-fit study remains
pending user submission.
