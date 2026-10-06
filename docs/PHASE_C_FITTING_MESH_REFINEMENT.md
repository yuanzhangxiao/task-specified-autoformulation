# M11: qualify coarse-to-fine collocation before testing initialization robustness

This is the next approved fitting milestone after the
[M10 review](PHASE_C_FITTING_M10_RESULTS_2026-10-05.md). M10's dense solve stayed
accurate near a previously training-fitted solution, while the reduced solve
converged with a 9.68% coefficient error. It compared fixed meshes, not a
coarse-to-fine transfer. This experiment tests discretization and transfer first;
generic-start recovery is the following milestone, not a claim of this one.

## Frozen comparison

Use the same alien-device hard equations, 16 training trajectories, four validation
trajectories, 13 dynamic factors and five shared latent initials. Import both
eligible M7 training-fitted starts through the verified M9/M10 export. Do not
regenerate arrays or select sources by coefficient or validation error. These two
starts are close to the same solution and are not independent recovery basins.
Correct equations and anchored coordinates remain supplied diagnostic assistance.

For each fitted start, run four independent resolutions and one continuation:

| Level | Requested variable target | Actual released variables | Intervals across training trajectories |
|---|---:|---:|---:|
| 0 | 6,000, subject to input/resolution floor | 33,906 | 2,824 |
| 1 | 60,000 | 59,994 | 4,998 |
| 2 | 90,000 | 89,994 | 7,498 |
| 3 | All observation times | 115,218 | 9,600 |

Every mesh retains every original observation in the objective with its original
weight. The state mesh is independent of the observation table: intermediate
observations use the Radau polynomial. Input interpolation corners are mandatory;
the complete supplied input function is unchanged. The first resolution is the
same conservative minimum-24-interval policy as M10. All four resolved meshes
are nested, verified before shipment without running a hard-case optimizer.

- **Independent ladder:** At each resolution, integrate the original fitted
  model to initialize boundary and Radau internal nodes. Fix its coefficients
  and shared initial values while solving for nodes; then release them. Comparing
  fixed stages isolates discretization error from parameter-search error.
- **Continuation:** Start identically at level 0. Transfer each converged final
  endpoint to the next resolution, including parameters, shared initials and
  the full previous Radau polynomial. Interpolation uses the internal node as
  well as boundaries; it does not linearly interpolate away the old polynomial.
  At the next level, first hold the transferred parameters fixed while projecting
  nodes, then release them. Only primal guesses transfer, not solver multipliers.

This yields ten tasks and sixteen released endpoints. Each stage must finish
with native success, finite complete final nodes and scaled discrete defect at
most `1e-6`; its released endpoint must also have a complete training rollout
before continuation advances. An unfinished solve does not trigger a larger NLP.
All native checkpoints remain available for later diagnostics.

This is a **controlled fixed refinement ladder**, not yet an automatic error-driven
mesh controller. Between-node ODE defects are recorded to inform that later
policy. A small defect at the enforced collocation nodes is not an accuracy
certificate. Dense observation-time collocation is the comparison endpoint,
not an assumption that no finer mesh could be useful.

## Keep continuation separate from model selection

The continuation path follows each converged released endpoint even when its
training rollout is worse than the supplied incumbent. Otherwise it would keep
restarting from M7 and fail to test transfer. Meanwhile the best complete training
rollout, including the supplied source, is always retained separately.

The report shows the released endpoint's prediction/coefficient/initial errors at
**every** resolution, not only the selected incumbent. After all of a task's
training decisions are sealed, independently replay each released endpoint and
the retained model using Radau and DOP853 on train/validation. Reference values
are used only for these post-fit coefficient/initial diagnostics. No evaluation
result returns to refinement, fitting or selection. No test data or LLM is used.

Expected interpretations:

1. Decreasing endpoint coefficient/initial error with mesh refinement supports a
   discretization explanation for M10's reduced-grid bias.
2. Continuation approaching the independent dense result supports useful
   coarse-to-fine transfer. Failure to do so identifies a path/initialization
   problem separately from mesh resolution.
3. Accurate incumbents retained without improved endpoints are explicitly labeled
   inherited success, not new optimization recovery.
4. Discretization accuracy does not establish robustness from generic starts.
   After this qualification, compare coarse-to-fine and rollout-only fitting from
   matched generic starts/budgets; perturbations around fitted starts are only an
   additional diagnostic. Keep the previously unsuccessful start in the roster.

## Limits, checkpointing and costs

Each task reverifies its supplied start with a complete Radau training rollout
(120 seconds including startup). Allow at most 180 seconds for initial node
integration, 250 seconds per fixed solve, 250 per released solve, and 120 for each
released endpoint's training screen. M10 observed complete Radau evaluations at
32--39 seconds; 120 seconds provides additional startup/publication headroom.
Post-fit independent replay has a separate 300 seconds per endpoint.

Each operation journals its start before launching. Completed operations and
sealed levels resume exactly. A started operation without a terminal receipt is
closed as interrupted, without refreshing its allowance. Unstarted later levels
can run only after qualified earlier levels. A completed backend can resume its
separate evaluation journal without rerunning fitting. Interrupted evaluations
close explicitly with available partial evidence and no fresh integration budget.

The M11 supervisor records timeout and cleanup separately. If a killed child has
not exited after ten seconds, preserve that fact and launch no subsequent solve
or evaluation. Existing M9/M10 supervisors and historical result files are not
modified. This diagnostic therefore avoids M10's uncaught second cleanup timeout
without silently rewriting its historical missing task.

Report native convergence, independent evaluation availability, prediction
accuracy and parameter recovery separately. Process wall-time totals exclude
parent mesh planning, post-fit evaluation and upstream fitting; do not call them
end-to-end cost. Upstream M7 costs are retained as provenance and must not be
summed repeatedly across tasks importing the same source. Independent and
continuation arms have different total work; this is not an equal-budget speed
comparison or a production fitter promotion.

## Delta commands

Upload `phase-c-fitting-m11.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c/`.
The bundle contains code and the existing sealed input export. It needs no new
clone, environment, dataset generation, GPU or API credential.

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m11"
tar -xzf "$AF_BASE/phase-c-fitting-m11.tar.gz" -C "$AF_BASE/code/fitting-m11"
export AF_CAMPAIGN="$AF_BASE/fitting-mesh-refinement-v1"
bash "$AF_BASE/code/fitting-m11/scripts/hpc/submit_phase_c_mesh_refinement_delta.sh"
BASH
```

Preparation runs the focused tests. Ten CPU tasks then use one CPU and 16 GB each,
with at most two running concurrently. The 90-minute scheduler limit accommodates
the four-level continuation and independent evaluation; it is a ceiling, not a
fixed fitting duration. A dependent report runs even after array failures and
lists missing tasks explicitly. Repeating the submission reuses recorded job IDs;
an uncertain scheduler reply stops for reconciliation rather than duplicating jobs.

Check and package results:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m11/scripts/hpc/inspect_phase_c_mesh_refinement_delta.sh
```

The command prints every level's released endpoint scores and coefficient error,
then prints the review archive path. Inspect `evaluations_complete` and individual
statuses before treating a task with a completed solve as numerically verified.

## Local verification

The focused tests cover exact polynomial transfer, source/worker boundaries,
incumbent preservation independent of continuation, failed-solve stopping,
cleanup and interrupted budgets, post-fit evaluation, and scheduler receipts.
A small linear native smoke uses an already training-fitted source, exercises
three resolutions and their continuation, and verifies exact terminal resume.
Hard alien fits are reserved for Delta.

Verification (2026-10-05):

- The full regression run passes: 4,028 passed, eight skipped (Torch unavailable).
  Two additional M11 failure-path tests added during that run are included in
  the final focused run below.
- 32 focused tests pass in both the checkout and portable source tree.
- The native linear smoke passes three distinct meshes (148, 400 and 1,444
  variables), all four tasks, six released endpoints and exact completed resume.
  In its continuation, maximum coefficient relative error decreases from 2.74%
  to 0.529% to 0.00379%. This is a small-control check, not the alien result.
- Production mesh planning alone verifies the four counts in the table, nesting,
  complete observation retention and unchanged forcing. No hard-case fit ran locally.
- Changed Python files pass Ruff, shell launchers pass `bash -n`, and whitespace
  checks pass. Repository-wide Ruff retains 37 unrelated pre-existing findings
  in untracked `analysis/claude` files.
