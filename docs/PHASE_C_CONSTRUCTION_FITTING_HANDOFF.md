# Orion handoff: fitting the returned construction models

## Scope and evidence

The user wants confirmation that construction fitting used Orion's current
algorithm, independent verification of the saved parameters, and concrete
fitting improvements. No new search or model change is authorized by this
verification campaign. Proposed fitting experiments below are follow-up work,
not changes to historical results.

Source archive SHA-256:
`1ec8f1deb8f25a698c6c3e4a948874398fdd89b446e81dbc5f0c3c7a7e2a6265`.
Source plan:
`7c017ab9d29700375335f0c8c6619f254b3c27002effc01639def7c4d8e25de1`.
Local preserved source: `output/construction-fit-analysis-20261021/inspection`.
Read-only equation/result audit: the parent directory's `REPORT.html` and
`analysis.json`. These are generated artifacts, not committed datasets.

All eight models are T2-easy: topology seed 1 / proposal seeds 1,2,3 and topology
seed 2 / proposal seeds 0,1,2,3,4. There are 16 training and four validation
trajectories, each with 301 samples and three targets. The two initializer
families are three analytic-map models and five trajectory-root models. All
eight remain in scope; no selection on error.

## Current source, specific execution route

The run used the current scientific tree
`5747e11830f6fa3c4672f74d0039d3b045a4a52fcd851f0fb4ec97f44b22fde0`:
scoped initializer fitting through `prefix_execution`/`prefix_initialization`,
analytic forward sensitivities, guarded relative scaling, bounded trust-region
least squares, DOP853/Radau verification, and measured checkpoint accounting.
It was one direct-rollout start with a fixed eight-minute validation prefix.

It did not route through the separate public fitter's certified affine profiling,
terminal-output profiling, restart portfolio, or collocation recovery. M47's
known-equation duration/noise controls are not the same problem: the construction
fits have many more coordinates and an unqualified model skeleton. Keep the
eight-minute construction rule fixed in any like-for-like comparison.

## What the original receipts show

| Topology / proposal | Completed search evaluations | Saved train NMSE | Original verification |
|---|---:|---:|---|
| 1 / 1 | 10 | 8.434942497 | 60-second timeout |
| 1 / 2 | 8 | 3.831008922 | 60-second timeout |
| 1 / 3 | 7 | 7.738445841 | 60-second timeout |
| 2 / 0 | 6 | 1.812508749 | 60-second timeout |
| 2 / 1 | 6 | 1.501872701 | 60-second timeout |
| 2 / 2 | 14 | 1.802079673 | 60-second timeout |
| 2 / 3 | 33 | 1.052356465 | DOP853/Radau agreed |
| 2 / 4 | 14 | 1.663064289 | 60-second timeout |

All searches stopped at their 600 numerical-active second budget. Seven
verification passes failed before the first full DOP853 sensitivity result;
their Radau passes never started. The saved search values are provisional until
independently checked; they are not convergence certificates.

For 2/3, training NMSE is 1.052356465 and validation suffix NMSE is 1.100672634.
Validation per-target NMSE is Gp 0.336630729, I 2.005066988, U 0.960320185.
Rank is 27/41. This candidate records 701.482 wall seconds, 698.135 numerical-active
seconds and 3.015 checkpoint-callback seconds. Across all eight candidates, the
corresponding totals are approximately 5,328, 5,318.5 and 7.55 seconds. Measured
checkpoint overhead does not explain the dominant cost in this campaign.

The new local supplement completed eight operations and verified seven exact
endpoints. Their training NMSE median is **1.802080**, unscaled MAD **0.749723**,
with 7/8 available. All seven agree with saved search errors; insulin NMSE remains
1.79–1.85. Candidate 2/0 timed out in state-only DOP853 after 11 complete training
trajectories; the twelfth started and Radau never started. Its first 11 completed
trajectories alone used 3,660,981 RHS evaluations. The 300-second allowance was
not renewed. This strengthens the solver-cost diagnosis but does not establish
that the saved endpoint would disagree with an independent integrator.

The local supplement used 1,102.801 CPU seconds, zero optimization/LLM calls,
and no new validation inference. The source snapshot, executed implementation,
per-row diagnostics and exact hashes are retained under
`output/saved-parameter-verification-20261021/`. ACES replay remains a separate
execution with its own runtime identity and costs.

## Concrete work items for fitting

1. **Separate output verification from sensitivities/rank.** The supplement checks
   the frozen endpoint with two state/output-only integrators. Preserve expensive
   derivative diagnostics as a separate status/budget so they do not erase an
   otherwise usable numerical prediction. Do not turn a timeout into a pass.
   Compare methods and per-trajectory cost on the same host before asserting a
   speedup. Completion and accurate fitting remain different outcomes.

2. **Profile stiffness and repeated segmented integration.** The completed local
   saved-endpoint results already show a strong method dependence: for topology
   seed 1 / proposals 1,2,3, DOP853 takes about 158.5/86.5/174.4 seconds and
   2,292,199/1,249,147/2,515,027 RHS evaluations, versus Radau
   10.2/6.0/7.4 seconds and 115,403/65,442/84,956 evaluations. The methods
   agree at the frozen tolerance. These are paired state-only checks on the same
   host, not a comparison with the original ACES fitting times. This is evidence
   to qualify an implicit sensitivity-search route and, separately, an independent
   stiff-system verification pair (or certified affine reference). It does not
   justify silently replacing the frozen verification protocol. The new receipts
   include per-method/per-trajectory wall/CPU and solver counters when available.
   The search needed only 6–33 completed evaluations to exhaust ten minutes.
   Inspect RHS calls, segments and scales; test an appropriate implicit-search
   route if stiffness is confirmed. An affine-system propagator may be useful
   for these linear models, but requires certified detection and exact treatment
   of the declared input interpolation and all initializers.

3. **Remove proven coordinate redundancy without changing the model class.** In
   candidate 2/3, 25 shared parameters collapse to 11 effective RHS coefficients;
   16 training roots give 27 effective coordinates, matching rank 27/41. Consumer
   gain × law amplitude products and overlapping linear terms are candidates
   for exact canonicalization or profiling. Preserve sign/domain constraints,
   shared-consumer coupling and a reversible map to original parameters. Do not
   merge effects merely because their drivers match. Qualify affine/terminal
   profiling for scoped roots rather than assuming the older router transfers.

4. **Compare search strategies only after controlling cost and coordinates.**
   Use these same eight frozen equations/initializers/data, keep original
   endpoints as controls, record all starts and budgets, and compare current
   direct rollout with justified scaling/reduction, collocation-informed starts
   or a small frozen restart set. Report training convergence and held-out prefix
   forecasts separately. No current evidence establishes the best attainable
   fit or that increasing the budget alone will solve it.

5. **Return actionable fit diagnostics to construction.** Distinguish a solver
   timeout, flat/redundant parameter direction, and a trajectory contradiction.
   Preserve target-wise residuals and input conditions. A bad endpoint is not
   proof that every parameterization of its skeleton is invalid.

## Equation evidence to retain

For candidate 2/3, the fitted insulin subsystem reduces approximately to

```text
I' = 3.313354*insulin_input - 5.599086*meal - 13.311635*I + 2.324277*I_mem
I_mem' = 0.673595*I - 4.906202*I_mem
```

The proposed "constant secretion" process actually depends on insulin input;
there is no additive constant in this fitted subsystem. A zero-meal,
zero-infusion training trajectory has constant observed insulin 25.58781346,
whereas this saved endpoint strongly decays; several validation rollouts become
negative. This is concrete endpoint evidence, not proof that every allowed
coefficient choice is infeasible: an alternative zero eigenvalue could preserve
a nonzero equilibrium. Better fitting and better scientific construction are
both still live questions.

## Supplement boundaries

The new campaign uses exact saved search parameters and training roots, with
unchanged `rtol=1e-9`, `atol=1e-11`, agreement tolerance `1e-6`. Its separately
declared budget is 300 numerical-active / 360 inclusive wall seconds per model.
No optimizer, sensitivity/rank calculation, validation-root inference, test data,
LLM, selection, or promotion is run. Source and new costs remain separate.
Seven missing validation scores remain missing. See
`docs/PHASE_C_CONSTRUCTION_SAVED_VERIFICATION.md` for reproducible execution.

## Orion review: fitting validity and construction feedback, 21 October 2026

### Independently checked evidence

Read-only verification passes all 90 source-snapshot hashes, original sealed
operation identities, policies and memberships of all eight saved best endpoints.
Recalculating seven training scores from stored predictions and observations,
using freshly computed pooled training standard deviations, differs from the
reported scores by at most 1.78e-15. The largest discrepancy from the original
search scores is 1.68e-10. The previously available validation suffix score for
2/3 also reproduces, but validation evidence is excluded from the advice below.
No new optimization or integration was performed in this review.

All eight searches remain unfinished. The saved best-point projected-gradient
infinity norms range from 0.00698 to 1.92559, with 0.04319 for 2/3, in the
optimizer's scaled coordinates. Every search terminated as `budget_limited`.
The model class and parameterization may both need improvement; differing from
the generating model does not by itself explain the large fitting errors.

Scoped initialization and held-out access are implemented correctly in the
inspected path: shared/map parameters and independent training roots have separate
coordinates, dependent initials are evaluated through the declared graph, and
validation inference freezes shared parameters and sees only the eight-minute
target prefix. Authorized future auxiliary schedules remain separate from future
targets. DOP853/Radau agreement checks integration of the same compiled model;
it does not independently certify the constructor-to-equation translation.

The frozen equations are affine in their states and forcing histories, although
their parameterization contains products and reciprocals and their rollout loss
is nonlinear in dynamics parameters. Linear functions are not automatically an
easy fitting problem. An implicit sensitivity solver, or a certified affine
propagator respecting piecewise-linear forcing, can address the present cost.
Eliminating exact gain redundancies must preserve shared-consumer constraints,
domains and a mapping back to original coefficients. It must not turn coupled
processes into independently fitted effects.

### Training witnesses that can already guide review

**Baseline glucose balance, all three topology-1 models.** At zero meal input,
the compiled glucose equation is exactly `Gp'=-U`. Their named basal-production
processes also depend on meal and therefore vanish. Public `train_000` has
constant Gp=172.58334024789 and U=1.91578598009 over 300 minutes; supplied EGP is
also 1.91578598009. Matching this continuous disposal curve would decrease glucose
by 574.735794, whereas the observed glucose is flat. This is a balance defect
before considering any particular nonlinear law. With discrete readings alone,
the integrated argument assumes the observed flat continuous behavior; it is not
a universal sampled-NMSE lower bound excluding arbitrary between-sample motion.

Feedback should ask for a compensating source and coherent units/balance, using
public evidence. A source outside the declared slot requires explicit topology
revision. The observations do not uniquely prescribe EGP, Gt, a private equation,
or a saturating response. Parameter tuning cannot change the identity `Gp'=-U`
at zero meal while also reproducing both continuous observed target curves.

**Direction and basal insulin, topology 2 / proposal 3.** In training trajectory
`train_003`, observed insulin peaks at 301.211603; prediction never exceeds its
initial 25.587813 and falls to -7.6299. In the zero-input baseline it decays toward
zero. The effective meal coefficient is -5.599086. Two inner secretion amplitudes
are approximately -1.395861 and have declarations `role=coefficient`, `domain=real`,
`bounds=null`. The fitter has not violated a nonnegative parameter domain: the
inner laws are allowed to reverse positive outer contributions. All three
topology-1 infusion amplitudes are also declared real and fit negative.

Feedback must display the effective function sign and its domain alongside the
declared process meaning. A positive topology operator is not a positivity or
monotonicity certificate for its inner law. Resolve any inconsistent function/domain
contract explicitly; do not silently tighten fitted bounds or label this as an
optimizer bounds bug. The basal insulin discrepancy also deserves a separate
homeostasis question; more flexible meal dependence alone need not fix it.

**Dose-dependent response shape, public training only.** Trajectories 000--005
have identical initial measured targets/auxiliaries, zero insulin infusion and
proportional meal schedules with amounts 0,30,60,90,120,150 g. For the positive
doses, observed insulin excursions and dominant peak times are:

| Meal amount (g) | Insulin excursion (pmol/L) | Peak time (min) |
| --- | --- | --- |
| 30 | 70.63398 | 111 |
| 60 | 163.16022 | 106 |
| 90 | 275.62379 | 101 |
| 120 | 406.47135 | 98 |
| 150 | 557.51581 | 95 |

Five times the dose produces 7.89 times the insulin excursion with an earlier
peak. This supports investigating dose/state-dependent response, not a blanket
instruction to use saturation. Glucose and utilization have different amplitude
patterns, so a target-specific packet is necessary. Free trajectory roots mean
that matching measured initial observations alone does not justify assuming
identical complete model states.

### A parameter-independent sampled check of the linear insulin family

For the five topology-2 models, expanding the process definitions verifies a
closed affine insulin subsystem with states `(I,I_mem)` and forcings only meal
and insulin infusion. No supplied auxiliary or Gp drives this subsystem. Allow
any independent `I_mem(0)` for each of the six dose trajectories, even where
the actual initializer is more restrictive. At fixed shared dynamics, sampled
insulin predictions have the form

    Yhat = a 1^T + h d^T + v m0^T,

where columns index trajectories, d is meal amount, and a,h,v are time vectors.
A common constant source would also be absorbed by a. If P projects onto the
orthogonal complement of `span{1,d}`, then `rank(Yhat P)<=1`. Consequently,

    ||Y-Yhat||_F^2 >= ||(Y-Yhat)P||_F^2
                    >= sum_{j>=2} singular_value_j(Y P)^2.

The observed projected singular values are approximately
610.9654,78.7826,30.8207,4.3549,0,0. The tail squared is 7175.58255, giving a
relaxed lower bound of 0.000319546 on insulin NMSE for these six trajectories,
or 0.0000399432 on the whole 16-trajectory, three-target objective. These are
floating-point calculations on supplied observations, not interval-arithmetic
or statistical certificates. The bound permits arbitrary time vectors and root
values, so actual ODE/domain constraints can only restrict the family further.

This shows a representational mismatch at exact accuracy even after allowing
trajectory-specific roots. Crucially, this lower bound is much smaller than the
current approximately 1.84 insulin NMSE. It does not explain the magnitude of
the present error or remove the need to improve fitting. Nonlinear laws, an
additional allowed driving channel, or a different memory representation are
alternative explanations; this test cannot identify one uniquely.

The reproducible, training-only audit and its small result file are at
`output/construction-fitting-review-20261021/audit_training.py` and
`training_witnesses.json`. It uses a restricted AST degree/dependency check and
linear algebra, with no reference equations, fitting, integration or LLM calls.

### Feedback contract and targeted revision tests

The first packet should contain:

1. Exact candidate, initializer graph, retained shared/root vector and training
   identity; output verification and search-progress statuses kept separate.
2. Per-target error and physical-unit baseline/excursion/peak/recovery comparisons,
   input schedules and uncertainty/censoring flags. Basal insulin can disappear
   yet score only 0.05248 on the baseline trajectory because its global training
   scale is 111.5073; normalized averages alone conceal meaningful failures.
3. Relevant executable laws, source sets, effective fitted signs, parameter
   domains and shared consumers. Names such as "basal production" are not
   accepted as evidence that a nonzero basal source exists.
4. A concrete revision hypothesis, competing explanations and predicted observable
   improvement, with a condition that would fail to support the hypothesis.
5. Explicit scope: an existing-source nonlinear law is an interaction edit;
   adding a source or replacing separate slots by a multivariable shared law
   requires topology authorization; new memory requires state/initialization work.

A suitable topology-2 insulin draft is:

> The saved training rollout is independently verified, but fitting stopped at
> its budget and no optimum is established. On train_003 the model loses insulin
> while the measured response rises, and the zero-input basal level also decays.
> The effective meal contribution is negative under the currently real-valued
> inner amplitudes. Propose a coherent correction of the response direction and
> basal behavior, preserving the delayed-action obligation and all observed
> targets. The dose table additionally suggests amplitude-dependent timing and
> gain; explain whether your change addresses those patterns or requests a new
> source. Predict the training windows that should improve after joint refitting.

We should not wait for global optimization proof before reviewing structural
witnesses, nor tell the proposer that every large residual proves a wrong law.
The controller may allocate a revision alongside unchanged-model continuation.

For nonlinear or multi-argument ranking, compare a small nested ladder: unchanged
law; flexible single-source law; and an explicitly authorized coupled-source law.
For example, compare `f(x)+g(y)` with a candidate `h(x,y)` only when the process
semantics justify joint action. The former has zero mixed derivative at the law
level, but coupled dynamics can still create interacting output responses.
Output correlation or a nonlinear target dose response alone does not prove a
particular bivariate mechanism. Do not merge unrelated source/sink mechanisms or
detach consumers of a shared process merely to improve fit.

A training-only local proposal screen can use normalized residual r, nuisance
Jacobian J (all current dynamics/map parameters and training roots), and new-law
sensitivity K. Assess `min ||r+J*delta+K*alpha||^2` under declared bounds and a
trust region. In the unconstrained local illustration, only `(Id-J J^+)K`
adds directions that current-parameter retuning cannot supply. This is a ranking
heuristic, particularly weak away from stationarity; it neither identifies a
unique causal term nor replaces bounded whole-model refitting and rollout checks.
Include state/input scales, signs, positivity and shared-consumer coupling.

Compare each child against unchanged continuation with matched fitting ceilings;
retain all proposal outcomes and the parent on failed or unverified changes.
Require the predicted response improvement without hiding damage to other
targets or the zero-input baseline. Freeze outcomes before separate validation;
never put historical validation/test metrics into proposer feedback.

### Next bounded implementation

First qualify output-only verification and a stiff-aware search route on all
eight unchanged equations/initializers, with exact inherited endpoints and
separate costs. Compare solver efficiency before enlarging budgets. Then qualify
safe coordinate reduction and applicable profiling; affine RHS dependence does
not imply affine integrated dependence on arbitrary dynamics coefficients.

In parallel, add a provenance adapter from these scoped-prefix receipts to the
M37/M38 descriptive feedback contract. The current exporter expects older staged
diagnostic snapshots and a legacy initialization plan; these saved receipts are
not directly interchangeable. Bind the authoritative constructor slots, sources,
signs and shared consumers, and preserve root scopes. Seven endpoints support
descriptive trajectory feedback; 2/0 must retain unavailable rollout evidence
while still allowing independently established equation/data witnesses.

Only after this qualification should the matched revision/refit experiment
expand functions or combine source sets. The present review updates the plan;
it does not change models, domains, benchmark data, production defaults or
historical results.
