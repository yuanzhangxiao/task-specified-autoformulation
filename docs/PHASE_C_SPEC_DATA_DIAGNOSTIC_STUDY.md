# Proposed controlled study: specification and data diagnosis

Design only, 13 October 2026. No fixtures, experiments, fitting, LLM calls,
remote operations, or benchmark/prompt changes are authorized by this document.
M33's remote qualification remains pending. This proposal follows the
[fallible-specification handoff](VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md) and
preserves the candidate-bound [M31 feedback contract](PHASE_C_FITTING_FEEDBACK.md).
It is an isolated synthetic development study, not a new basin benchmark result.

## Question, scope and falsifiable comparison

Can bounded joint diagnosis reduce false accusations of correct scientific
claims and sound measurements, while retaining useful detection of distinguishable
claim/data defects and abstaining on confounded or computationally unresolved
cases? Compare its paired outcomes with claim-only revision and observation-only
adaptation. An always-abstain procedure has zero accusations but fails the
predeclared detection, prediction and useful-action endpoints.

Start with exact analytic transfer and a finite, public equation catalog. This
deliberately separates diagnosis from unrestricted symbolic discovery. The first
implementation milestone must qualify known-sigma affine Gaussian fitting and
fixed-weight output profiling. Robust fitting, systematic-error hypotheses and
the full four-arm study are subsequent gated milestones; none is enabled here.
Continuous-rate nonlinear fitting and live proposers are later extensions.

## Restricted physical family and exact transfer

Use one dimensionless nonnegative basin state and one externally applied input:

\[
\dot x=g u-kx,\qquad x(0)=x_0,\quad 0\leq g\leq1.5,
\quad k\in K=\{0,0.4,0.6,0.8,1.2\}.
\]

The rate catalog is public and identical in every record; its true member is
private. Half the independent source blocks have k=0; the other half are balanced
across the four positive members. Each has a seed-derived gain in [0.55, 0.85]. The seed,
selected rate and generating gain are evaluator-only. These are design values,
not recovered parameters. Nonnegative inputs and initials ensure x remains
nonnegative; sensor errors may produce negative raw readings, which are retained.

For constant u=v on a segment of length d, transfer exactly by

\[
x(t+d)=e^{-kd}x(t)+g v\,a_k(d),\qquad
a_k(d)=\begin{cases}(1-e^{-kd})/k,&k>0,\\d,&k=0.\end{cases}
\]

Use a stable `expm1` implementation and a separately checked analytic reference.
At each fixed k, all predictions have the affine form
`mu = c_k(x0) + g * phi_k(u) + b * t`; b is present only in a drift hypothesis.
Gaussian weighted least squares is therefore a bounded convex subproblem in
(g,b); a fixed-scale Huber objective is also convex but needs its own qualified
solver. Enumerating every catalog member covers this finite family only.
Verified primal/dual bounds or an exact affine optimum can certify its restricted
objective optimum; an optimizer success flag or multistart agreement cannot.

The two reviewed revisable claims are `C_D: k in K excluding 0` and
`C_0: k=0`. All other balance, input, state and preparation assumptions are common.
Each source block appears with both claim packets: for positive k, C_D is correct
and C_0 is false; for k=0, their truth reverses. The permitted revision substitutes
the other reviewed claim;
it never changes the prediction target or silently edits observations.

## Inputs, preparation and sampling

Training comprises four open-loop trajectories with t=0,0.25,...,4 (68 readings).
The main study supplies independently calibrated, exact prepared x0, separate
from the potentially noisy target reading at t=0. No later target resets occur.

| Trajectory | Prepared x0 | Applied piecewise-constant u |
| --- | ---: | --- |
| Drawdown | 2.0 | 0.1 on [0,4] |
| Refill | 0.2 | 0.9 on [0,4] |
| Pulse and release | 0.3 | 0.9 on [0,1), 0 on [1,2), 0.6 on [2,4] |
| Alternating input | 1.0 | 0.1 on [0,0.5), 0.8 on [0.5,1.5), 0 on [1.5,3), 0.4 on [3,4] |

Input logs contain segment boundaries and levels, including right-continuous
switch semantics; state is continuous at switches. Analytic propagation restarts
at every switch. Do not pass these schedules through the existing basin
benchmark's piecewise-linear interpolation contract. A future implementation
must use a separate qualified adapter or exact affine design matrices.

Locked fresh-intervention evaluation uses x0=0.5 and 1.4, respectively, with
u=0.7 on [0,0.75), 0 on [0.75,2.25), 1.1 on [2.25,3.5), then 0 until 6;
and u=0 until 1.25, 0.4 until 4.25, then 0 until 6. Sample every 0.25 units.
Training and evaluation share physical coefficients and the instrument drift
within a source block; evaluation receives new independent noise/contamination.
Calibration is stable only over this declared domain, not arbitrarily far ahead.

## Eight factorial conditions

Write `y_i = x(t_i) + sigma*epsilon_i + z_i + b*t_i`, with independent standard
Gaussian epsilon. The two claim packets cross the following four conditions.
The baseline sigma is a known measurement-contract value, not a residual estimate.

| Condition | Generating observation process | What discovery receives |
| --- | --- | --- |
| Clean | sigma=0, z=0, b=0 | Raw readings and a justified deterministic error bound |
| Ordinary noise | sigma=0.03, z=0, b=0 | Raw readings and known sigma=0.03 |
| Sparse contamination | sigma=0.03; three training readings have additive errors of random sign and magnitude uniformly between 6 sigma and 10 sigma; b=0 | Same contract as ordinary noise; no mask or fault label |
| Systematic defect | sigma=0.03, z=0; b chosen from {-0.10,-0.06,0.06,0.10} | Same contract as ordinary noise; no true b or fault label |

Sparse training indices are drawn without replacement from the 64 noninitial
readings. Future-contamination count is `floor(0.05*n_noninitial)` with a new
mask. Signs, masks and amplitudes are private. Do not clip negative observations,
remove flagged samples, or replace raw targets with a clean companion trace.

For clean data, sigma=0 does not define a Gaussian objective or likelihood.
Use deterministic squared error and a separately declared numerical tolerance;
Gaussian likelihood and statistical coverage are not applicable. A proposed
absolute bound epsilon=1e-9 requires independently certified analytic-evaluation
and storage-rounding error before any bounded-error certificate is issued.
An ODE tolerance or agreement between two solvers is not such a bound.

The nonclean discovery contract permits the same reviewed alternatives on every
row: ordinary noise, bounded robust residual modeling, and a shared drift
b in [-0.12,0.12]. It never announces which alternative generated a case. Huber
influence reduction is not a sparse-error certificate or a recovered fault mask.
No free per-reading correction, noise inflation, unknown input or timing change
is allowed in this first study. The systematic defect is instrument drift, not
an extra physical source term; provenance must preserve that distinction.

## Hypotheses, distinguishability and evidence limits

Retain physical claim and observation hypotheses jointly; causes are not mutually
exclusive. In false-claim/contaminated and false-claim/drift cells, both defects
are present. A candidate can also be malformed or unsuccessfully fitted. Report
four separate statuses: candidate adequacy, claim compatibility, observation
explanation, and numerical/search completion, each with assumptions and evidence.

For exact direct observations, C_0 implies nondecreasing x for nonnegative u.
With known absolute errors epsilon_i, a decrease satisfying
`y_i-y_j > epsilon_i+epsilon_j`, t_i<t_j, contradicts that observation/claim
bundle. With an allowed drift, reject the whole bundle only if no allowed b
can satisfy all corresponding inequalities. For example, a single pair suffices
if `y_i-y_j > epsilon_i+epsilon_j + b_max*(t_j-t_i)`.
This is a narrow monotonicity certificate, independent of fitting failure.

Gaussian errors have unbounded support. Neither the same residual inequality nor
the best finite-family fit yields an absolute impossibility certificate there.
Any statistical challenge must name its null, selection procedure, nuisance
parameters, calibration population and uncertainty. The initial study measures
empirical error rates of a frozen decision rule; it promises no uniform p-value,
posterior truth probability or confidence coverage without separate qualification.
Robust and drift alternatives must also be included in the null being challenged.

Before launch, test affine design rank for every input/rate/observation branch;
assess separation between different rates after profiling all permitted nuisance
parameters over their domains. A local full-rank Jacobian does not establish
cross-family identifiability. A minimum separation computed only by multistart is
an empirical screen, not a global bound. Use exact finite enumeration plus a
qualified convex calculation where available; otherwise preserve uncertainty.

## Controls that prevent shortcut diagnoses

The first three existing M31 fixtures provide interface precedents, not new
independent study samples. Add isolated controls with new identities as follows.

| Control | Required distinction or action |
| --- | --- |
| Defective construction under C_D | Present an omitted-drainage candidate and an adequate conforming candidate on identical observations. Candidate failure cannot accuse the correct claim; a mismatched construction is recorded explicitly. |
| Wording-only and data-permutation shortcuts | The primary matrix balances true C_0 and true C_D. Compare rules seeing only claim wording, then deliberately shuffle observations across truth directions in a separate control. Require aggregate improvement over wording-only rules plus minimum performance in each direction; these controls cannot substitute for balanced primary scores. |
| Exact gain/calibration ambiguity | Set x0=0 and y=s*x, so y depends on g,s only through s*g. The witnesses (g,s)=(0.6,1.2) and (0.8,0.9) agree for every input. More pulses cannot resolve them; an independent known-level sensor calibration can. |
| Unexcited/obscured loss | Use x0=u=0 for exact indistinguishability, and separately a short near-equilibrium noisy trace. Abstain when evidence cannot separate drainage hypotheses; do not label an invisible mechanism absent. |
| Fitting/numerical failure | Reuse the reviewed nonlinear x'=g*u-k*x^2 family as an auxiliary control with an independently verified adequate witness and deliberately short allowance. Failure is computationally unresolved, never proof of a false claim. |
| Limited-regime approximation | Generate x'=g*u-k*x-q*x^2, q>0; compare the linear model inside a predeclared small-state regime and on a larger pulse. Report adequacy relative to a fixed tolerance/domain, not exact recovery. |
| Initial-observation uncertainty | Repeat a small subset with a known noisy initial measurement or a shared unknown preparation coefficient. Fit the permitted initial quantity jointly; never treat y(0) as exact or fit evaluation initials retrospectively. |
| Logical contradiction | Combine k=0 with k>=0.4 for the same scope and variable. A formal inconsistent-claims witness requires no residual or empirical accusation. |

For the approximation control, if 0<=x<=H, 0<=q<=q_max, k>=k_min>0,
the nonlinear and linear solutions under the same input/initial satisfy
`0 <= x_linear(t)-x(t) <= q_max*H^2*(1-exp(-k_min*t))/k_min`.
Propose g=0.7, k=0.8, q=0.1, x0 in [0,0.1], u in [0,0.1] and t in [0,4].
This invariant regime gives H=0.1 and error at most 0.00125, below a fixed 0.002
tolerance; a fresh u=1.5 pulse probes outside it. Independently qualify the state
bound and comparison bound. Shrinking the domain after seeing errors is not
success. The nonlinear and uncertain-initial controls are later
qualification stages and do not block the first affine Gaussian smoke milestone.

## Blind discovery and evaluator-only truth

An allowlisted discovery record contains an opaque task ID, the reviewed claim,
raw training arrays and hashes, available calibration/preparation information,
input semantics, the common hypothesis catalog and declared budget. It excludes
condition names, reference equations/coefficients, clean traces, corruption
masks, generating seeds, expected outcomes, witness parameters and certificates.
The claim's actual mathematical content is legitimate input; its truth label is
not. Give the fitter only its executable candidate and permitted training contract;
claim adjudication is a separate proposed layer, not a change to the M31 verdict.

Randomize task order and opaque IDs. Do not expose paired sibling IDs, filenames,
seed encodings or catalog positions that reveal truth. Discovery state and caches
must not communicate across paired tasks; content-addressed reuse may share only
identical allowed requests, never evaluator metadata or sibling decisions.
The evaluator stores immutable clean truth and intervention outcomes separately.
Audit serialization and filesystem access, including hashes whose human-readable
preimages might encode class labels. A label-shuffled or wording-only baseline
must fail; true-C_0 controls and data permutations test the suspected shortcuts.

The current `feedback_fixtures.py` deliberately gives `omitted_drainage` and
`false_no_drainage_claim` identical fitter inputs. Preserve that distinction:
their external claim contexts differ, while their numerical evidence alone
cannot decide which context is correct. Existing tests keep expected diagnoses,
adequate witnesses and certificates outside `bridge.Inputs` and feedback packets.
New study labels must never become additional fitter or proposer fields.

## Four matched-budget arms, after qualification

| Arm | Permitted adaptation |
| --- | --- |
| Fixed claims | Fit the original claim family with the declared baseline observation contract. |
| Claim revision | Compare the original and alternative claim; baseline observation contract fixed. |
| Observation adaptation | Original claim fixed; compare baseline, robust residual, bounded drift, and robust-plus-drift hypotheses. |
| Joint diagnosis | Compare both claim alternatives and the same observation hypotheses. |

Use known-sigma Gaussian negative log likelihood for its branch. A robust branch
uses a fixed Huber threshold, provisionally 1.345 standardized units, with its
properly normalized density if comparing likelihoods across branches. It must
not substitute unnormalized Huber loss into a Gaussian likelihood comparison.
Freeze branch/parameter/claim-revision penalties on development blocks, record
their sensitivity, and never choose the loss or noise scale to favor an outcome.
Clean records instead use a separately frozen deterministic-tolerance rule.

Propose a common ceiling of 40 catalog/convex-solve units and 60 CPU seconds per
arm/task, plus 10 diagnostic solve units and 30 CPU seconds; stop at either cap.
One unit includes a fixed inner iteration limit, whose value must be set from
preflight. All retries, nuisance fits, null alternatives and diagnostics count.
The richest catalog has five rates times four observation branches, so a single
complete enumeration uses at most 20 primary branch solves. Account actual
inner iterations too; one robust solve need not cost one Gaussian solve.
Simpler arms may finish early and do not spend their surplus on hidden assistance.
Measure achieved compute as well as equal ceilings; incompletion remains visible.

An oracle-known correction is a separate reference: disclose true b, or the
contamination mask, to that oracle only. It estimates recoverable performance,
never a discovery win. There are no LLM/provider calls in the deterministic pilot.
Later proposer budgets, caching and token accounting require a new reviewed plan.

## Replication, roles, endpoints and cost

Stage a 32-attempt engineering preflight (eight condition types x four arms)
before 24 development source blocks and 64 fresh locked evaluation blocks.
Each block crosses both claims with all four conditions and uses matched base
noise across nonclean cells and arms. Separate seed namespaces govern source,
noise, masks, drift and optimizer ordering; checkpoint their state privately.
The source block is the independent unit, not a trajectory, time point or arm.
The 24 development blocks contain 12 at k=0 and three at each positive rate;
the 64 locked blocks contain 32 at k=0 and eight at each positive rate.

Development blocks select penalties/decision thresholds only outside individual
candidate feedback. Freeze them and all code before opening evaluation blocks.
Within each block only training enters fitting or diagnosis; fresh-intervention
records are evaluator-only. The 64 blocks are a pilot assessment, not an automatic
calibration guarantee or a new production test split. Never use their errors to
revise the evaluated protocol; a revision requires fresh locked blocks.

Report paired, block-level differences with uncertainty and all attempted-task
denominators. Primary outcomes are false accusations of each correct claim;
false fault claims
on clean/ordinary-noise records; correct joint localization where identifiable;
abstention on the confounded controls; and unsupported certainty on failures.
Report sensitivity alongside false accusations, revision size, observation-change
size, retained alternatives, and whether the proposed next measurement separates
them. Report detection and false accusations separately for both truth directions.
Require improvement on the predeclared equally weighted aggregate diagnostic score
over the best wording-only baseline, plus a minimum detection rate and maximum
false-accusation rate in each direction. Set these thresholds on development and
freeze them before evaluation. Do not require strict improvement within a direction
where an always-drainage or always-no-drainage baseline already scores perfectly.
An aggregate improvement cannot hide failure of the per-direction gates.
Do not demand a unique fault label in the ambiguity control.

Score raw-observation predictive likelihood with the declared normalized density
on nonclean fresh records, alongside a common fixed-sigma Gaussian reference
score for comparing point predictions. Separately evaluate x predictions against
private clean physical truth, and y predictions against raw observations. Report
physical gain/drainage behavior, bias and drift accuracy only through the evaluator.
Do not score corrected data against themselves. Report predictive-error median
and unscaled MAD, and compliance mean/SD with equal-case aggregation; give counts,
unavailable attempts and block-level uncertainty for all diagnostic proportions.

The main matrix has (24+64)*8=704 datasets and 2,816 arm attempts; an oracle on
the two corrupted data conditions adds 352 attempts. Propose eight new control
types with four independent blocks each across four arms (128 extra attempts),
reported separately; this small control roster tests behavior, not calibrated
rare-event rates. At zero observed failures in 64 independent trials, even the
simple one-sided 95% binomial bound is about 4.6%; paired/cell pooling requires
its own dependence-aware analysis and cannot manufacture more independent trials.
For a fixed truth direction only 32 locked blocks are available; its corresponding
zero-failure bound is about 8.9%, not 4.6%.

Budget CPU work by `sum_over_attempts(core_count * (fit_seconds + diag_seconds
+ setup_seconds + evaluation_seconds))`, plus fixture/oracle qualification,
preflight and aggregation. With one core and the proposed caps, the 2,816 main
attempts alone reserve at most 2,816*90=253,440 CPU seconds (70.4 CPU hours) for
fit/diagnostic allowances, excluding overhead, controls and oracles. This is a
ceiling, not measured runtime; analytic solves may finish much sooner. GPU and
provider budgets are zero. Select concurrency after measuring memory/setup cost;
checkpoint complete branch solves and charge interrupted work without renewal.

## Launch gates and remaining limits

1. Review the claim meanings, input adapter, noise contract and all blind-access
   boundaries; preserve benchmark data/finalized prompts and pending M33 artifacts.
2. Qualify known-sigma affine Gaussian fitting first: exact/profiling agreement,
   numerical checks, deterministic resume, and correct zero-noise semantics.
3. Independently qualify reference transfer, drift identifiability, control
   witnesses, robust convex solves and initial-state handling before their stages.
4. Pass the 32-attempt preflight and leakage audit; set the per-solve iteration
   caps and measure costs. Preflight outcomes cannot count as evaluation evidence.
5. Freeze development-selected rules, seeds, code and denominators; review budget
   and obtain the separate authorization to implement/launch the controlled study.

This design does not establish novelty, field validity, calibrated uncertainty,
natural-language interpretation, or discovery outside the finite catalog. Its
strongest counterargument is that an enumerable family may make diagnosis too
easy; it is useful only if the controls expose the intended errors and the frozen
evaluation demonstrates an improvement over simple baselines. Moving to continuous
rates, general nonlinear dynamics, input/timing defects, unknown noise scales,
or live proposers must preserve these distinctions in separately reviewed stages.
