# Phase C fitting: plan and first qualification milestone

Agreed with the user on 2026-09-30. This task owns fitting; Astra owns
construction. Development proceeds one milestone at a time. The completed
28-cell development release is the input boundary. No test data, LLM calls,
benchmark changes or production-default promotion are part of milestone 1.

## Research objectives and five directions

Distinguish accurate output prediction, parameter/latent-state recovery, and
prediction under new interventions. A low observed NMSE cannot certify the latter
two. Separate numerical failure, weak/structural identifiability, inadequate
initialization information, and model-class mismatch before requesting revisions.

1. **Condition the joint optimization problem.** Center/scale parameters,
   collocation state variables, initial-value parameters and dynamical constraints.
   Preserve physical equations, domains, starts and observation weights in the
   scaling comparison. Test scientifically justified constraints separately.
2. **Approach dynamically consistent latent trajectories gradually.** Investigate
   multiple shooting, mesh/window adaptation and continuation. Final scoring must
   use a continuous free rollout with the deployed initialization rule.
3. **Exploit equation structure.** Investigate conditional linear subproblems,
   redundant latent scales and transfer of unchanged fitted blocks across revisions.
   Conditional least squares does not make the complete rollout objective linear.
   Earlier alternating/continuation trials had mixed outcomes; these are hypotheses.
4. **Allocate fitting effort using observed progress.** Retain useful checkpoints,
   distinguish improving/stalled/unavailable calculations, and avoid discarding
   incumbents or resetting consumed budgets after interruptions. Longer fitting
   is justified by progress, not by a generic timeout label.
5. **Return interpretable numerical uncertainty to construction.** Describe
   unexplained response features, weak parameter combinations and disagreement
   among fitted models. A weakly excited mechanism is not necessarily unnecessary.
   Experimental-design suggestions are future work outside the present paper.

Prioritize direction 1, while collecting evidence that will support directions
4 and 5. Existing collocation, sensitivities, multistart, local polling and
checkpoint recovery remain the baseline. Novelty must be established through
comparisons; established numerical techniques may already produce large gains.

The CSTR lead is documented in
[the September 28 findings](PHASE_C_FITTING_FINDINGS_2026-09-28.md): train/validation
NMSE 0.00014821/0.00006713 from known equations, despite substantially different
hidden initial values. That result bundled positivity, coordinate changes and
initializer extrapolation. It did not isolate conditioning or certify latent
recovery. The Phase C CSTR release now has common hidden preparation; this study
does **not** reuse the obsolete affine hidden-initial shift from that old release.

## Milestone 1: fixed-equation qualification

Protocol: `phase-c-fitting-qualification-1`.
Config: `configs/phase_c_fitting_v1.json`.

The first bounded study uses three predetermined development cells: named CSTR
easy, named CSTR hard, and coupled clean detention basins. These cover a nonlinear
thermal/concentration problem and a storage/threshold problem. Alien-device and
Dalla Man are deferred until this comparison is inspected. Dalla Man's accepted
continuous-ODE approximation of private bookkeeping is a separate model-class
question, not an exact-parameter qualification test.

Known equation skeletons are **evaluator assistance**. Original public training
and validation arrays are copied byte-verifiably from `phase-c-development-2`;
no trajectories are regenerated. CSTR reference parameters are verified against
the release's private specification hash. The portable input artifact retains
the public prompts, data identities, equation requests and diagnostic witnesses.
It contains no hidden trajectories or test payload. It must never become a
construction/proposer input.

The supplied CSTR template already uses a temperature-centered Arrhenius law and
its existing concentration/temperature guards. Every arm keeps this same template.
The study measures the additional effect of optimizer coordinates; it does not
compare raw Arrhenius parameterizations or automatically rewrite discovered laws.

| Case / fitted block | Arms | Numerical starts | Fits |
|---|---|---:|---:|
| CSTR easy: dynamics, measured initial auxiliaries | control, scaled | 3 | 6 |
| CSTR hard: dynamics, reference hidden initials fixed | control, scaled | 3 | 6 |
| CSTR hard: shared hidden initials, reference dynamics fixed | control, scaled, domain, scaled+domain | 3 | 12 |
| CSTR hard: dynamics and shared hidden initials jointly | control, scaled, domain, scaled+domain | 3 | 12 |
| Coupled basins: dynamics, public initial gauge | control, scaled | 3 | 6 |
| **Total** | | | **42** |

The known-block controls deliberately supply private reference values for their
fixed block; label this assistance when interpreting them. Generic starts for
the fitted block use fixed broad guesses with reproducible perturbations, not
reference-centered draws. Each seed's physical starting vector is identical
across its arms. Three starts measure numerical sensitivity, not three independent
scientific problems. Report every endpoint; do not select the best seed.

### What scaling changes

For each parameter and state use `physical = center + scale * coordinate`, with
strictly positive finite scales. Parameter centers are their numerical starts;
scales are `max(abs(start), 1)`. Shared initial-value parameters undergo the same
transform. State centers/scales use training channel means/standard deviations
(standard-deviation floor 1 in that channel's units), or candidate initial guesses
when no observed channel exists. The explicit CSTR unit proxies are reactor
temperature for T, feed concentration for C, and jacket-feed temperature for Tj.
These are recorded dimensional choices in this assisted experiment, not hidden
state estimates or automatic unit inference. Basin upstream depth falls back to
the public initial gauge. The policy is deliberately simple and may be suboptimal.

Collocation uses transformed decision variables, transformed parameter bounds,
and state-scale-normalized continuity defects. Observation residual weights,
physical node guesses, forcing interpolation and original equations are unchanged.
IPOPT uses its existing settings; constraint tolerance now refers to scaled
defects. Thus even an invertible transform can change numerical stopping behavior.
Checkpoint feasibility in these coordinates is not a physical accuracy certificate.
The intended pairing of physical node guesses was not fully achieved in the first
Delta run: the wall-clock warm-up produced different fallback patterns. See the
completed-run review below before interpreting this as an isolated scaling test.

Sensitivity refinement applies the chain rule to parameter gradients/Jacobians;
polling uses the same frozen parameter scales. Callbacks, checkpoints, returned
parameters and Jacobian reports stay in physical units. Native optimizer optimality
is labeled as measured in scaled coordinates. Actual ODE integration still uses
physical state units and unchanged solver tolerances. Time is not rescaled.
Initial-map basis rotation, full matrix preconditioning and adaptive latent-scale
inference are deferred.

### What the domain arm changes

Only the **shared initial CSTR concentration** becomes nonnegative, using its
explicit concentration interpretation in the supplied equation skeleton. No
reference numerical bounds are supplied. Dynamic coefficients retain their existing
domains in every arm. Jacket initial temperature is unchanged. Signed latent
variables are not automatically made positive.

This is an initial-domain experiment, not a certificate of nonnegative complete
state trajectories. No affine clipping or extrapolation change is bundled with it.
There are no redundant domain arms where initial concentration is already known.

### Budgets, scoring and provenance

All fits have 120 seconds of collocation initialization, 180 seconds of refinement,
240 refinement residual calls, five seconds of node warm-up, the same feasibility
screening policy, and local polling for guarded equations. Scaled and control arms
retain identical allowances. Startup/compilation and final scoring are recorded
overhead; a 45-minute scheduler allocation is not extra optimizer budget.

Before the array, the supplied full reference vector must pass both development
splits with NMSE <= 1e-6 and Radau/DOP853 disagreement <= 1e-4 training standard
deviations. It is a witness of attainable output error, not uniqueness.

Each fit selects parameters on training rollouts only. Validation uses the same
frozen parameters and causal initialization; no validation-specific initials.
Independent Radau/DOP853 replay has a separate 240-second total allowance.
Accuracy passes only when both split NMSEs <= 0.01 and the solvers agree within
1e-4 training standard deviations. Preserve optimizer convergence, execution
status, budget exhaustion and accuracy as separate fields. Record parameter
absolute errors without asserting identifiability, per-trajectory replay errors,
fit duration, actual residual calls and saved collocation checkpoints.

Plans, input content, Python source and numerical package versions are frozen.
Completed tasks are reused. An interrupted native fit retains its files and is
marked interrupted on reconciliation; it receives no automatic fresh budget.
If the backend completed before interruption, independent scoring can resume.
Floating-point results and wall-clock-limited endpoints need not be bitwise equal
across different machines. Resume guarantees artifact reuse and budget integrity,
not identical scheduler timing or portable native IPOPT state.

Exit criteria: account for all 42 attempts, verify reference attainability and
paired starting-point invariants, inspect within-case coordinate/domain effects
and scope-specific failures. Numerical failures are results, not missing rows.
Promotion to construction requires a subsequent explicit decision after these
results, including regressions and limitations, have been reviewed.

### Local qualification before submission

The three reference witnesses passed independent replay on the exact bundled
Phase C development arrays. CSTR easy/hard train/validation NMSE was
2.01e-17/1.88e-17; coupled basins was 2.42e-17/3.68e-18. Maximum Radau/DOP853
disagreement was 2.78e-6 training standard deviations. These are supplied-reference
checks, **not newly fitted performance**. The subsequent 42-fit Delta results are
reviewed below.

The focused tests exercise transformed bounds and Jacobians, a signed-latent
native collocation/refinement fit, physical checkpoint values, data/source drift,
paired starts, known-block isolation, resume budget preservation and uncertain
scheduler receipts. The signed-latent smoke fit reaches NMSE below 1e-6 on both
splits without fitting validation initials.

Implementation verification (2026-09-30): full `pytest -q` completed with
3,734 passed and eight optional-dependency skips. The standalone source/input
bundle passed its 14 focused tests and all three reference replay checks.
Changed Python files pass Ruff and the three shell scripts pass `bash -n`.
Repository-wide `ruff check .` still reports 37 pre-existing issues in unrelated
`analysis/claude` files; those files were not changed.

## Delta execution

Delta CPUs are sufficient. Use one CPU and 8 GiB per array task with at most two
concurrent fits; no GPU, model server, API key or ACES allocation is involved.
The 42 nominal optimizer allocations total 3.5 CPU-hours, plus reference replay,
independent endpoint replay and overhead. Queue time is not predicted.

Use the supplied `phase-c-fitting-m1.tar.gz` portable bundle. It includes source,
focused tests, configuration, sealed inputs, `SOURCE_COMMIT`, `SHA256SUMS`, and
the convenience script below. Upload the **single archive**, not its loose files.
An existing Delta project Python is reused. If CasADi is installed only in the
existing supplemental fitter dependency folder, the script detects it; no package
installation or environment mutation is performed.

```bash
bash <<'BASH'
set -euo pipefail
AF_REV=$(tar -xOf "$HOME/phase-c-fitting-m1.tar.gz" SOURCE_COMMIT)
AF_CODE="/projects/bibo/yxiao2/repos/phase-c-fitting-${AF_REV:0:7}"
mkdir -p "$AF_CODE"
tar -xzf "$HOME/phase-c-fitting-m1.tar.gz" -C "$AF_CODE"
(cd "$AF_CODE" && sha256sum -c SHA256SUMS >/dev/null)
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_c/fitting-m1
export AF_ACCOUNT=bibo-delta-cpu AF_CONCURRENCY=2
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_delta.sh"
BASH
```

Expected: 42 array tasks, with preparation, fitting and report job IDs written to
`/work/hdd/bibo/yxiao2/phase_c/fitting-m1/submission_manifest.json`.
Repeat submission with the same frozen source only to reuse receipts; a recorded
ambiguous scheduler reply stops for reconciliation instead of duplicating jobs.

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/fitting-m1
AF_IDS=$(jq -er '.jobs | [.prepare,.fit,.report] | join(",")' "$AF_ROOT/submission_manifest.json")
sacct -j "$AF_IDS" --format=JobID,JobName%26,State,ExitCode,Elapsed
jq '{status,expected,recorded,status_counts,groups}' "$AF_ROOT/summary.json"
```

The summary is created at preparation and refreshed after each fit, so unfinished
rows appear explicitly. For a failed preparation inspect its `.err` log and
`qualification/result.json`; the fit array should not proceed. For manual refresh
or a download archive, use `scripts/hpc/inspect_phase_c_fitting_delta.sh` from the
same bundle. It reports the existing campaign and packages its diagnostics; it
does not submit jobs or refit. Keep the original campaign for full call traces.

## Milestone 1 results: first Delta run, reviewed 2026-09-30

### Provenance and verification

- Source commit: `57d3271b2b2f7a596209368f16bfd08244d303df`.
- Plan SHA-256:
  `f7e626a8a414787039ea7c4da93ca4d0972ffef3f10661b8e40211c9ce4d9de7`.
- Supplied `review.tar.gz` SHA-256:
  `6a9c2173cb0b129bc569798414f0279b23ea18d130abd29bafd94e4375a5de61`.
- Delta prepare/fit/report jobs: `22583360`, `22583361`, `22583362`.
- Campaign: `/work/hdd/bibo/yxiao2/phase_c/fitting-m1`.
- Local review copy: `artifacts/phase-c-fitting-m1-review-6a9c2173cb`.

All 42 attempts completed, with no missing or interrupted rows. The 132 sealed
JSON records verify, the input artifact matches the original export, and the
summary reproduces from the archived records. Paired physical parameter starting
vectors are identical. All three reference witnesses pass on Delta. All fitted
endpoints pass solver agreement; maximum disagreement is 3.56e-5 training standard
deviations, below the 1e-4 limit. Thus inaccurate endpoints are not explained by
disagreement between these two replay solvers. This is not a guarantee of exact
integration for every possible parameter vector.

The study produced **25/42 accuracy passes**, requiring train and validation NMSE
at most 0.01. The aggregate is bookkeeping across deliberately different diagnostic
problems, not a benchmark-wide success rate. No test data, validation-specific
initial fitting or LLM calls were used. Review regenerated reports and inspected
saved diagnostics; it did not refit the supplied models.

### Results by fitted block

Each row reports all three numerical starts. Medians are over starts; the pass
count additionally requires both splits and independent solver agreement.

| Case / fitted block | Arm | Median train NMSE | Median validation NMSE | Passes |
|---|---|---:|---:|---:|
| CSTR easy / dynamics | control | 0.699945 | 0.365591 | 0/3 |
| CSTR easy / dynamics | scaled | 0.000563907 | 0.000345296 | 3/3 |
| CSTR hard / dynamics, true initials fixed | control | 0.808673 | 0.346161 | 0/3 |
| CSTR hard / dynamics, true initials fixed | scaled | 0.000671126 | 0.000272715 | 2/3 |
| CSTR hard / initials, true dynamics fixed | control | 2.00e-11 | 1.51e-11 | 3/3 |
| CSTR hard / initials, true dynamics fixed | scaled | 2.00e-11 | 1.51e-11 | 3/3 |
| CSTR hard / initials, true dynamics fixed | domain | 2.00e-11 | 1.51e-11 | 3/3 |
| CSTR hard / initials, true dynamics fixed | scaled+domain | 2.00e-11 | 1.51e-11 | 3/3 |
| CSTR hard / joint | control | 0.210252 | 0.0761192 | 1/3 |
| CSTR hard / joint | scaled | 0.962612 | 0.386155 | 0/3 |
| CSTR hard / joint | domain | 0.882372 | 0.319751 | 0/3 |
| CSTR hard / joint | scaled+domain | 0.210252 | 0.0761192 | 1/3 |
| Coupled basins / dynamics | control | 8.76e-10 | 1.49e-9 | 3/3 |
| Coupled basins / dynamics | scaled | 1.10e-9 | 1.86e-9 | 3/3 |

Scaling lowers both split errors in all six paired CSTR dynamics-only cases:
five passes versus zero. The basin problem was already solved accurately.
However, scaling alone regresses on joint CSTR fitting. For seed 1, validation
NMSE changes from 0.000230891 in control to 0.487214 with scaling, while
scaled+domain reaches 0.00184400. All four seed-2 joint arms end at exactly the
same parameter vector: collocation is worse than the common ordinary start, and
polling makes the same single exchange-coefficient update. These duplicate
endpoints are explained by fallback selection, not four independent confirmations.

Estimating only the two hidden initial values succeeds in every arm when the
dynamics are supplied. Estimating dynamics with supplied initials also can succeed.
The joint problem is the remaining obstacle. This supports investigating parameter
and initial-state coupling, nonlinear optimization basins and available excitation;
it does not distinguish these explanations or prove structural nonidentifiability.

Even a good joint fit need not recover the hidden preparation. Joint control seed 1
has validation NMSE 0.000230891 but fitted `(C0,Tj0) = (0.400284,332.506)`, versus
reference `(0.261932,347.566)`. The scaled+domain success estimates
`(0.307846,341.325)`. Additional unseen-intervention accuracy and latent recovery
remain separate questions. The initial-domain constraint alone gives no consistent
accuracy advantage in this run; it remains scientifically justified where the
public contract establishes a concentration.

### Numerical interpretation and experimental limitations

1. **The node-start comparison is incompletely controlled.** The five-second
   warm-up deadline starts before construction of the optimization problem and
   is shared across trajectories. Per-trajectory graph construction consumes
   this time between rollout attempts. Depending on runtime, later trajectories
   receive constant-hidden/observed-data guesses instead of rollout guesses.
   Three of six CSTR dynamics-only pairs have different source patterns; all
   three basin pairs also differ. The other three CSTR dynamics-only pairs still
   improve under scaling, and joint seed 1 still regresses with matching source
   patterns. This is encouraging evidence, but not a fully isolated numerical
   ablation. Matching recorded source labels also is not a saved node-array hash.
   Freeze and reuse physical node guesses across arms in the next comparison.
2. **Checkpoints, not optimizer termination, explain the useful CSTR outcomes.**
   Eighteen initializers converge (12 initials-only and six basin fits); 23 time
   out and one returns an invalid parameter vector. All seven passing CSTR
   dynamics/joint endpoints equal saved collocation parameter checkpoints exactly.
   Their actual free rollouts validate them despite unfinished initialization.
   Constraint residuals alone are not an endpoint selection rule.
3. **The experiment does not demonstrate faster sensitivity refinement.** All 42
   refinement stages use directional polling because the templates contain
   guarded/piecewise expressions. Its physical step scales are themselves part
   of the scaled-arm policy. These results cannot isolate a Hessian conditioning
   effect or establish the benefit of gradient scaling; the relevant numerical
   transform has separate implementation tests.
4. **Budget exhaustion is not equivalent to poor fitting.** All 42 endpoints
   exhaust the refinement allowance, including all 25 accuracy passes. In the
   12 initials-only fits, refinement takes another 180.0--181.9 seconds without
   changing the converged parameter vector. Polling also does not improve any
   of the seven passing CSTR dynamics/joint checkpoints. This motivates a
   training-only accuracy/progress stopping rule and reallocating effort; it
   does not justify selecting or stopping on validation error.

### Decision and proposed follow-up

The campaign and its diagnostic review are complete, but the intended isolation
of physical node starts needs a follow-up. Do not promote scaling as a universal
default or call joint fitting solved. Preserve this run and its shortcomings.

The recommended next bounded milestone combines the unfinished part of direction
1 with direction 4: freeze common physical node arrays and hashes; retain the
existing unscaled incumbent; compare strategies under an equal total budget; and
allocate further effort based on training rollout accuracy and progress instead
of always spending 180 seconds polling. Test this primarily on the difficult joint
case, with dynamics-only and basin controls to detect regressions. Any portfolio
must share its budget rather than granting every start a full extra allowance.
Known-block controls remain evaluator diagnostics, not a way to supply true
coefficients or hidden initial states to the deployed fitter.

Blockwise initialization and continuation are subsequent hypotheses if the joint
problem remains difficult. No follow-up implementation, production-default change
or new remote submission is included in this review.

Verification for this documentation update: report regeneration and all 132
artifact seals checked; original input identity and paired parameter starts
checked; all 14 focused fitting tests passed in 6.62 seconds. Repository-wide
Ruff still reports the 37
pre-existing findings in unrelated `analysis/claude` files. No implementation or
benchmark files were changed.

## References for later milestones

- Fides: https://doi.org/10.1371/journal.pcbi.1010322
- Multiple shooting/collocation: https://web.casadi.org/docs/
- Generalized profiling: https://www.jstatsoft.org/article/view/v075i02
- MAGI: https://doi.org/10.1073/pnas.2020397118
- Diffusion tempering: https://proceedings.mlr.press/v235/beck24a.html
- Observability/identifiability: https://doi.org/10.1007/s11538-025-01415-3

## Milestone 2: identifiable joint fitting controls

User-approved follow-up: establish coefficient **and latent-initial** recovery on
identifiable cases before interpreting joint CSTR failure. The implemented
[identifiable-control study](PHASE_C_IDENTIFIABLE_FITTING.md) freezes physical node
starts, shares each collocation pool across three matched refinement arms, tests
training-based stopping and conditional checkpoint rescue, and scores latent and
parameter recovery separately from observed error. It is 27 CPU endpoints with no
production-default or benchmark-release changes. It also clarifies the historical
alternating fitter: linear weights alternate with a joint trajectory/nonlinear
block; its initial conditions were fixed.

The [2026-10-01 Delta review](PHASE_C_IDENTIFIABLE_FITTING_RESULTS_2026-10-01.md)
closes this milestone: all 27 endpoints recover the supplied coefficients,
latent initial and trajectories. Training-based stopping reduces residual calls
from 74 to 60 across nine matched fits. Conditional-first rescue costs 331 calls
without increasing recovery. The next proposed bridge is guarded CSTR
sensitivity refinement from identical checkpoints; production defaults remain
unchanged. The review also records initializer timeout overhead and the limits
of these noiseless controls.

## Milestone 3: keep competing trajectory formulations open

The subsequent discussion broadens the proposed CSTR bridge into a
[matched fitting-strategy experiment](PHASE_C_FITTING_STRATEGIES.md): rollout-only,
adaptive direct collocation, adaptive multiple shooting and the existing hybrid.
Each receives the same 180-second total fitting ceiling. Four identifiable
synthetic controls (including a genuinely nonlinear shape parameter) precede
CSTR easy/hard stress tests with global identifiability explicitly unproven.
All original observations and input knots are retained. Native checkpoints may
violate dynamical constraints, but only complete training free rollouts select
parameters; independent development replay verifies endpoints. The planned
72 CPU fits run on Delta without changing production defaults or benchmark data.

The [Delta summary review](PHASE_C_FITTING_STRATEGIES_RESULTS_2026-10-01.md)
accounts for all 72 endpoints. Rollout-only passes output accuracy on all 18
case/start pairs, including all six CSTR fits; the hybrid passes 15, direct
collocation 12 and multiple shooting 11. Rollout-only recovers all 12 synthetic
controls and CSTR-easy coefficients closely, but hard CSTR retains coefficient
errors up to 35.8% despite accurate output. Rollout-first is the leading next
development candidate. The subsequent
[backend review](PHASE_C_FITTING_M3_OPTIMIZATION_ANATOMY.md) verifies the archive,
counts optimization dimensions and separates coefficient/initial recovery.

## Milestone 4: additional budget and fixed-mesh reuse

The [budget/reuse follow-up](PHASE_C_FITTING_BUDGET_REUSE.md) retains all four
strategies with 600-second allowances and adds a fixed-mesh reuse variant of
collocation. It compares three starts on the nonlinear-shape control and both
CSTR tiers (45 CPU fits). Repeated solves reuse the same graph with primal/dual
warm starts; unfinished coarse solves do not trigger larger meshes. Endpoint
reports compare dynamic coefficient accuracy, initial absolute errors and
independent prediction NMSE separately. Conditional affine-coefficient solves
remain a recorded next method, since nonlinear dynamic constraints prevent the
quadratic observation objective from making trajectory estimation a simple QP.
No production default changes or construction integration occur in this step.

The [2026-10-02 archive review](PHASE_C_FITTING_M4_RESULTS_2026-10-02.md)
reconciles all 45 endpoints. Rollout-only, the hybrid and original collocation
recover every CSTR coefficient within 1% in all six starts each. The new reuse
variant fails two hard-CSTR starts, and shooting retains low prediction errors
with inaccurate coefficients. The review records native restart, checkpoint
freshness and screening-budget issues for an isolated follow-up. It also shows
that useful nonconverged collocation checkpoints can seed successful refinement.

## Milestone 5: isolate formulation reuse from warm starts

The [reuse diagnostic](PHASE_C_FITTING_REUSE_DIAGNOSTIC.md) separates rebuilding,
cache-only cold solves, unconditional primal transfer and training-screened primal
transfer. It uses two uninterrupted solves on one fixed mesh, zero imported duals,
bounded rollout screens and retained ordinary-start fallbacks. Three matched
starts on the nonlinear control and hard CSTR produce 24 CPU tasks. Historical
arms and production defaults remain unchanged. The following harder-case milestone
will qualify basin and alien-device coefficient/initial identifiability before
extending the matched method comparison.

The [M5 archive review](PHASE_C_FITTING_M5_RESULTS_2026-10-02.md) verifies all 24
endpoints and closes this comparison. All pass prediction and coefficient recovery.
Cached and rebuilt cold solves produce identical parameters; caching removes a
second formulation build. Good primal starts cut hard-CSTR second-solve iterations
from 162/73/180 to 6/5/9. Screening selects the same latest checkpoint in every
case, so protection against harmful warm starts remains untested. The next step
is harder fixed-equation qualification, with screening stress cases and stopping
after already-adequate solves, while retaining all original fitting strategies.

## Milestone 6: larger latent dynamics with qualified parameter blocks

The approved [harder-case campaign](PHASE_C_FITTING_CHALLENGING.md) compares the
four original methods and the two primal-transfer policies on coupled basins and
alien-device hard. Three matched generic starts yield 36 CPU-only Delta tasks.
Alien-device fits 13 dynamic factors and five shared latent initials, conditional
on fixed internal couplings/nonlinear shapes that anchor its latent coordinates.
Both cases passed the local training-sensitivity preflight. Reference replay and
rank gates repeat on Delta before fitting. All original observations are retained;
no benchmark or production fitting defaults change. Report coefficient recovery,
initial recovery, actual rollouts and screening vetoes separately. Stopping after
an already adequate first solve remains a later scheduling experiment.

The [M6 archive review](PHASE_C_FITTING_M6_RESULTS_2026-10-03.md) accounts for all
36 endpoints. Basin passes all 18 fits. Alien rollout-only and hybrid each recover
predictions, dynamic coefficients and initials for two of three starts; the other
four strategies recover none. Hybrid successes use the original generic vectors
after unhelpful collocation. The third rollout start stagnates near NMSE 0.124;
shooting records almost no iterations before stage timeouts. Screening now
exercises vetoes, with mixed accuracy outcomes. Next diagnose fixed-mesh numerical
cost and training-based continuation/restart decisions on this same problem;
retain every method and do not promote a production default yet.

## Milestone 7: reduce mesh size and diagnose native cost/stagnation

The [M7 runbook](PHASE_C_FITTING_NUMERICAL_DIAGNOSTIC.md) implements the bounded
follow-up to M6: dense versus reduced collocation, exact versus limited-memory
shooting Hessians, and rollout continuation versus a training-triggered generic
restart. The same two qualified cases and three starts give 36 CPU fits under
900-second total ceilings. All observations and input interpolation are retained;
the conservative alien collocation mesh drops from 115,218 to 33,906 variables.
No adaptive enlargement follows an unfinished solve. Timing profiles survive
interruptions, and a worse restart cannot erase the verified incumbent. Earlier
methods and production defaults remain available unchanged.

The [M7 results review](PHASE_C_FITTING_M7_RESULTS_2026-10-04.md) verifies all 36
backends and 35 finalized evaluations; reduced-collocation start 2 has an uncaught
independent-replay timeout. Basin passes all 18 fits. Alien rollout arms still
recover two of three starts, with no recovery from the exercised restart.
Coarsening enables three native convergences but poor recovery. Exact shooting
Hessian evaluations cost 100--111 seconds; limited-memory increases iteration
counts without solving recovery. Dense-collocation checkpoint diagnostics consume
55--59% of recorded native-worker time. Next make checkpoints cheaper and replay
failures explicit, then revisit discretization/initialization with those costs
controlled. No production default is promoted.

## Milestone 8: checkpoint overhead and evaluation reliability

The [M8 runbook](PHASE_C_FITTING_CHECKPOINT_DIAGNOSTIC.md) implements a matched
legacy-versus-compact checkpoint comparison for three fixed-mesh native methods:
36 CPU fits on the same two cases and three starts. Routine compact callbacks
retain parameter/initial vectors and cheap feasibility/loss diagnostics; expensive
node diagnostics move to native exit. Independent replay failures become explicit
outcomes, with a journal that preserves partial evidence without granting another
budget. Partial restart decisions and completed-history counts survive reporting.
A separate M7 timeout closeout uses saved logs only. Historical results and fitter
defaults remain unchanged.

The [M8 review](PHASE_C_FITTING_M8_RESULTS_2026-10-04.md) accounts for all 36 fits:
34 complete replays and two explicit timeouts. Basin passes all 18; alien native
arms recover none. Compact dense checkpoint time falls from 363--391 to 15--40
seconds, with two newly completed native solves. Reduced collocation reaches the
same iteration totals faster but still produces inaccurate models. One expensive
checkpoint can consume the remaining screening budget. Next test bounded,
journaled screening and collocation initialized from already accurate
training-fitted rollout solutions, before adding benchmarks or more blanket
budget. These are proposed diagnostics; no production fitter is promoted.

## M9 — bounded screening and explicitly assisted collocation

The next implemented diagnostic is described in
[PHASE_C_FITTING_SCREENING_ASSISTANCE.md](PHASE_C_FITTING_SCREENING_ASSISTANCE.md).
It follows the two recommendations in the M8 review: bounded per-point RK45/Radau
screening with durable phase journals, and dense/reduced collocation initialized
from verified M7 training-fitted endpoints, first with coefficients/initials fixed
and then released. No hidden trajectory starts or production default changes.
The 48-entry Delta CPU roster separates 36 generic fits from 12 assisted entries
(two explicitly ineligible); upstream assisted cost is disclosed.

The [M9 review](PHASE_C_FITTING_M9_RESULTS_2026-10-05.md) accounts for all 48
entries: 32 complete evaluations, 14 without a selected vector, and two ineligible
sources. Basin passes all 24 selected endpoints, including six that retain the
supplied assisted fit. Alien's 20-second screen cap causes all 79 Radau screens
to time out, including the four eligible assisted rechecks; those collocation
experiments never start. RK45 yields eight complete alien endpoints but no
recovery. Next calibrate screening time and replay saved pools, then rerun only
the four blocked assisted tests. These are proposed diagnostics; no production
fitter or screening policy is promoted.

## M10 — calibrated screening, saved-pool replay and assisted reruns

The [M10 runbook](PHASE_C_FITTING_SCREENING_REPLAY.md) implements this follow-up
on alien-device only. Two method-specific timing pilots precede four assisted
reruns and 18 saved-pool replays. Complete historical training scores are reused;
uncached vectors receive calibrated allowances, and useful incumbents survive
timeouts. Point telemetry distinguishes setup from trajectory integration. The
released assisted stage reserves time to screen its final vector. Training-only
selection and independent post-selection evaluation remain separate. No generic
optimizer rerun, new benchmark data, provider call or production promotion.

The [M10 review](PHASE_C_FITTING_M10_RESULTS_2026-10-05.md) verifies successful
calibration and all four assisted fixed/released solves. Dense released endpoints
retain excellent training/coefficient accuracy in six iterations despite 115,218
variables. Reduced endpoints converge but shift one coefficient by 9.68% and a
latent initial by 0.219; mesh convergence and initialization robustness are now
separate targets. All four select their better supplied M7 incumbent. No generic
pool recovers an accurate model. Two independent replays time out and one cleanup
exception leaves a missing record; no full assisted rerun is needed. Next qualify
mesh refinement and perturbed training-fitted starts, with explicit assisted cost
and independent post-selection evaluation. No production policy is promoted.

## M11 — mesh convergence and explicit coarse-to-fine transfer

The [M11 runbook](PHASE_C_FITTING_MESH_REFINEMENT.md) implements the approved
mesh diagnostic first. The same two training-fitted alien starts feed four
independent resolutions and one sequential transfer each (ten CPU tasks).
Every released endpoint receives separate post-fit prediction and coefficient
assessment; a preserved supplied incumbent cannot mask mesh bias. Radau
polynomials, parameters and shared initials transfer only after a qualified solve.
All observations and input corners remain intact. This fixed ladder qualifies
mesh accuracy before an automatic adaptive policy or a matched generic-start
robustness experiment. Those remain the next targets, not established outcomes.

## M12 — generic-start recovery with training-verified early stopping

The [M12 runbook](PHASE_C_FITTING_GENERIC_RECOVERY.md) records M11's mesh-convergence
results and implements the approved next step: all three original generic starts
on alien hard, comparing rollout-only, medium-mesh plus rollout and progressive
meshes plus rollout under the same 1,200-second ceiling. Meshes retain input
corners and add bounded training-observation anchors. Complete independent
training rollouts can stop refinement early; validation and coefficient/initial
errors are evaluated only after training selection freezes. Nine CPU tasks run
on Delta. No production default or benchmark changes.

The [M12 results](PHASE_C_FITTING_M12_RESULTS_2026-10-06.md) account for all nine
tasks and complete independent replay. Rollout-only recovers two of three starts;
progressive meshes recover the same two by falling back to the original starts;
medium collocation recovers none. No run reaches a finer mesh. One converged
coarse solve lacks its final node checkpoint, and a medium checkpoint's first
sensitivity evaluation hits a local time cap. Lower current training NMSE does
not reliably identify the best start for further fitting. Next make endpoint
rejection explicit and compare bounded optimization trials from several retained
starts under one budget. This follow-up is recommended, not yet implemented.
