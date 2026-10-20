# M46: multiple hidden initial values and prefix forecasts

## Question and scope

M43 qualified shared roots with dependent maps and trajectory-specific roots.
M44 tuned an observation-window duration with one hidden root. The returned M45
supplement recovered all three missing forecasts without fitting again. M46 asks
whether the same interface works when two unknown hidden initials must be
inferred from one observed channel, and whether the forecast works immediately
after the observation window.

This is an isolated CPU qualification of known equations. It does not change a
benchmark, proposer prompt, production fitting default or test-data contract.
Prediction accuracy and recovery of individual hidden states are separate outputs.
No LLM or GPU is used.

## Controls

The observed state is x, with y=x:

    dx/dt = -theta*x + z + r
    dz/dt = -z
    dr/dt = -lambda*r
    x_j(0) = y_j(0)
    z_j(0) = xi_j,z >= 0
    r_j(0) = xi_j,r >= 0

The shared unknown is theta. There are two independent hidden initial coordinates
per trajectory; neither is derived from the other. Training jointly estimates
theta and six roots from three trajectories: seven optimization coordinates.
Only the two roots are fitted on each new trajectory. Shared theta, model laws,
domains and training normalization remain frozen.

| Case | lambda | Meaning |
|---|---:|---|
| separated | 2 | Hidden contributions decay at clearly different rates. |
| near_confounded | 1.05 | Their measured responses are similar; separation can be noise-sensitive. |
| exact_confounded | 1 | Only the sum of the roots affects y; the two roots cannot be uniquely recovered. |

Each case has clean observations and additive Gaussian observation noise with
known standard deviation 0.01, and two predetermined training starts. This makes
12 array tasks. The two starts share the same acquisition and are not independent
noise replicates. All trajectories have 41 samples on [0,4], and the initial
observed value is exactly 0.4 even in the noisy cases.

Each task has three training, four duration-tuning and four fresh follow-up
trajectories. The follow-up set is synthetic development qualification, not a
released benchmark test split. Its trajectory identifiers and hidden initial
conditions differ from the tuning set. Generating roots and clean signals remain
in a separate evaluator payload.

## Frozen selection and evaluation

For each trained model:

1. Infer roots using prefixes of duration 0, 0.1, 0.2, 0.5, 1 and 2.
   Each prefix uses the same three declared starting pairs: (0.3,1.3),
   (1.3,0.3) and (1,1).
2. Choose among successful inferred roots by prefix observation loss only,
   breaking exact ties by start index. Future measurements do not choose roots.
3. Independently integrate with DOP853 and Radau before accepting an endpoint
   or forecast. Compare all starts on both the common future horizon t>2 and
   the half-unit interval immediately after each prefix.
4. A duration is adequate when the mean forecast NMSE across all four tuning
   trajectories is at most 0.01 and the maximum pairwise start-to-start forecast
   RMS difference, normalized by the frozen training scale, is at most 0.01
   on **both** horizons. These are declared experiment thresholds.
5. Require the entire grid and every start to be available. Seal the shortest
   adequate duration, or explicitly report unavailable/no adequate window.
6. Use only that sealed duration on the four fresh trajectories. Infer their
   roots from their prefixes with shared parameters frozen. Do not retune the
   duration or select between the two training starts using follow-up accuracy.

The immediate interval tests whether the model is useful as soon as it must
forecast; the common late interval permits comparisons on identical timestamps.
An own-suffix score is descriptive only. No single universal duration is declared:
selection is separate for each trained model. Configured cutoffs must exactly
match observation timestamps, preventing floating-point boundary ambiguity.

## Distinguishing failed optimization from ambiguity

Let A(t;theta) be the two measured responses to unit roots. With theta frozen,
prefix predictions have the exact form

    y_prefix = b(theta,y0) + A(theta) * [xi_z, xi_r]^T.

Two diagnostics are computed at the chosen prefix endpoint:

- **Conditional information:** SVD of the root sensitivity matrix, its rank and
  weak direction, in native root coordinates with frozen training output scaling.
  For positive known noise sigma, sigma/s_i describes the amplification of one
  measurement-noise unit along a singular direction of the physical matrix,
  excluding the exact initial reading. Null directions have no finite scale.
- **Conditional optimization check:** solve the two-variable nonnegative
  least-squares reference by enumerating its bounded faces. Compare its minimum
  loss with the retained optimizer endpoint. This can identify failure to solve
  the prefix problem, even when the dynamics integrators agree.

These diagnostics do not replace inferred roots or choose the duration.
The convex reference is conditional on fitted theta, not on ground-truth theta.
Analytic predictions and sensitivities are checked against the compiled numerical
model. Closed-form diagnostics reject models outside the exact declared controls.

After the duration is sealed, the evaluator reports root errors, root-sum error,
shared-coefficient error and clean forecast errors. In the exactly confounded
case, a large individual-root error can coexist with excellent forecasts because
the output depends only on their sum. Neither low forecast NMSE nor a solved
conditional optimization problem certifies unique hidden-state recovery.

Noise-amplification scales are not confidence intervals. They exclude shared
parameter uncertainty, nonnegative-boundary truncation and uncertain initial
observations. Near-confounded results are numerical evidence within this control,
not a general identifiability theorem.

## Execution and accounting

M46 adds an opt-in execution wrapper around the existing prefix fitter. It keeps
the numerical method and tolerances, while separating search and independent
verification budgets. Search receives 30 active seconds for training and 10 for
each prefix, with at most 100 oracle calls. Verification receives 10 and 5 active
seconds respectively. An inclusive 120-second search cap and 60-second
verification cap also charge synchronous checkpoint time.

Measured checkpoint time is excluded only from numerical-active time. It is
recorded separately and remains within total operation costs. Compilation/setup
count toward active and inclusive limits. A numerical search timeout may leave a
saved best to verify with its own allowance; checkpoint failure or overall expiry
does not authorize additional search or a successful result.

Diagnostics have their own numerical allowances, checkpoint timing and call
receipts. Attempt reservations, actual starts and completed computations remain
distinct. A returned unavailable diagnostic is not counted as completed.
Deadlines are cooperative: they cannot interrupt blocked filesystem calls; when
such calls return after expiry, further numerical work and qualification stop.

Durable operation receipts prevent repeated work on resume. Interrupted attempts
remain charged and cannot silently restart. Unknown partial counters remain
unknown; observed progress provides lower bounds. Portable reports reconstruct
scores, identities, operation rosters and selections. All terminal files existing
is reported separately from all tasks completing their numerical qualification.

Reported CPU/wall costs include setup and progress writes, exclude final receipt
sealing, report writing and scheduler waits, and include stage-level costs for
training, inference, forecasts and diagnostics. Timing subcategories are nested
within totals and must not be added again.

## Construction integration boundary

The local construction artifacts at
`output/initialization-graph-analysis-20261019/audit.json` and
`output/initialization-graph-delivery/preview-final/summary.json` contain dependency
declarations and deferred evidence. They do not provide a completed, executable
new-contract function model. M46 therefore explicitly reports
`actual_construction_export_qualified: false`.

The next integration gate requires one completed analytic construction export,
its permitted observations and sealed initializer graph. Do not invent missing
function laws, convert an old initializer silently, or treat successful dependency
declarations as successful fitting. The common interface remains
[observation-prefix initialization](PHASE_C_OBSERVATION_PREFIX_INITIALIZATION.md).

## Run on Delta

Upload `phase-c-fitting-m46.tar.gz` to the path in `AF_BUNDLE`. This experiment is
self-contained and needs no previous campaign snapshot.

```bash
bash <<'BASH'
set -euo pipefail
AF_BUNDLE=/work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m46.tar.gz
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/prefix-multiroot-m46.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_CODE"
cd "$AF_CODE"
export AF_PREFIX_MULTIROOT_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-multiroot-v1
export AF_ACCOUNT=bibo-delta-cpu
export AF_CONCURRENCY=2
bash scripts/hpc/submit_phase_c_prefix_multiroot_delta.sh
BASH
```

The launcher verifies source checksums and prints import/submission progress.
Preparation runs the relevant tests. The array has 12 tasks, one CPU and 16 GB
per task, concurrency two, with a 45-minute Slurm limit. This is an allocation
ceiling, not a target runtime. Uncertain submission replies stop for inspection.

After the report job finishes:

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-multiroot-v1
jq '{status, expected, recorded, all_results_recorded, status_counts,
     selection_counts, cost}' "$AF_ROOT/summary.json"
jq '[.rows[] | {task_id, status, selection, diagnostics_available,
  diagnostics_expected, followup: [.followup[]? | .metrics]}]' \
  "$AF_ROOT/summary.json"
AF_REVIEW="$AF_ROOT/review-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_ROOT" \
  plan.json inputs.json summary.json results
printf '%s\n' "$AF_REVIEW"
```

Keep unsuccessful tasks and all starts in the review archive. Review predictions,
hidden-root ambiguity, optimizer-reference gaps and diagnostic overhead together.
Remote outcomes are pending; local checks do not establish a universal prefix
policy or complete construction integration.

## Local qualification

The 173 focused initialization, execution, controls, campaign and launcher tests
pass. They cover successful forecasts, conditional-rank loss, noisy root
ambiguity, bounded reference optima, shared-parameter freezing, future-data
exclusion, complete denominators, checkpoint failure, timeouts, unknown partial
costs, exact resume and uncertain scheduler replies. Python 3.11 syntax and Bash
syntax are checked; the full repository suite is also run before delivery.
The extracted source bundle passes its actual prepare/verify commands, checksum
verification and the same 173-test worker preparation gate.

The full repository run finished with 7,544 passed, eight skipped for unavailable
Torch, and one source-identity failure in the historical paired-fit resume test.
That test overlapped a source edit: inspection found only the source digest and
its derived identities differed, with all model/data/profile fields preserved.
Its isolated rerun passed after the source stabilized. Ruff passes outside the
37 unchanged findings in untracked `analysis/claude/` files. No unrelated files
were repaired or committed.

Two full-grid local smoke tasks also complete, with byte-preserving resume:

| Task | Chosen duration | Follow-up immediate NMSE | Follow-up common NMSE | Conditional root rank |
|---|---:|---:|---:|---:|
| separated, clean, start 0 | 0.2 | 9.55e-19 | 1.64e-17 | 2 |
| exactly confounded, noisy, start 0 | 0.2 | 0.00172 | 0.00206 | 1 |

Both have all 28 chosen-endpoint diagnostic pairs available. The second task has
maximum root-vector spread 2.50 across starts on the follow-up trajectories, while
normalized prediction spread is approximately 1e-10. The rank-one result correctly
withholds unique-root recovery despite useful predictions. These two local cases
are smoke evidence, not the results of the complete twelve-task experiment.
