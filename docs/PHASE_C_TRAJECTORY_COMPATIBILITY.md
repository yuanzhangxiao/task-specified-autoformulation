# Training observations and model compatibility

Design review, 2026-10-13, based on source `e867372`. These are proposed features,
not implemented prompts, gates or budget changes. This supersedes the preliminary
evidence proposal in `PHASE_C_OWNERSHIP_FEEDBACK.md`.

Implementation update, 2026-10-14: the user selected only three narrow diagnostics
for initial offline qualification. They and a separate prompt/repair policy are
implemented in `PHASE_C_PROTOCOL_CLARITY.md`. The wider highlights and remaining
diagnostics below are still proposals; no new live compatibility gate is enabled.

## Decision

Provide factual training highlights from the first construction request. After
complete topology, evaluate necessary properties of that candidate and ask about
specific conflicts. Revisit deferred questions after functions, initialization
and fitting supply missing information. The runtime identifies what the model
cannot distinguish or reproduce; the proposer chooses the scientific dependence.
Rules operate on declared roles and mathematics, never benchmark/variable names.

Implement the evidence contract and highlights first. Add candidate compatibility
diagnostics as a separate, initially advisory milestone. Increasing repair budgets
is a further policy change, not a substitute for reliable feedback.

## What the model determines

Using the paper's notation, with optional time/covariates displayed explicitly:

\[
\mathcal M=(\Gamma,f,h,g;\psi),\qquad
\dot x=f_\Gamma(x,u,c,t;\psi),\quad
y=h_\Gamma(x,u,c,t;\psi),\quad
x(t_0)=g_\Gamma(\mathcal I_0;\psi).
\]

Here \(\mathcal I_0\) contains permitted initial information. Time, history
channels, covariates, resets and trajectory-dependent quantities are available
only when the task/model contract allows them. Expand algebraic and shared-process
dependencies to identify a target's complete ancestor subsystem. Do not substitute
observed histories for generated quantities in an open-loop model. One-step
prediction has a different conditioning contract.
Read that policy from the versioned task contract: current Phase C contexts have
empty `lagged_targets`, despite historical general documentation describing
one-step prediction. Causal-prefix comparisons use the actual interpolated
forcing; matching earlier stored samples alone may miss a ramp to the next knot.

Under uniqueness, identical complete initial states, forcing histories and
parameters determine identical trajectories on their common existence interval.
Local Lipschitz conditions are a standard sufficient condition for uniqueness;
topology alone does not certify them.
[Teschl, Theorem 2.2](https://www.mat.univie.ac.at/~gerald/ftp/book-ode/ode.pdf).

Distinguish three conclusions:

- **Observation:** a pattern in recorded training samples.
- **Conditional incompatibility:** the candidate cannot reproduce that pattern
  under stated assumptions, possibly for every function/parameter choice allowed
  by the restricted candidate class.
- **Fitted discrepancy:** this fitted realization does not reproduce the pattern.
  This alone does not exclude a different fit of the same topology.

Compatibility does not establish identifiability: several parameters or models
may explain the observations. Identifiability addresses uniqueness of inference,
not merely agreement with data.
[Villaverde et al. (2016)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005153).

## Initial prompt: facts without causal assignments

Start `training_observations` with short deterministic highlights:

1. Channel roles, units, initial observations, sampling intervals, recorded
   uncertainty, and the supplied-input interpolation/event contract.
2. Per-channel, per-trajectory sampled extrema and their locations. Distinguish
   isolated maxima, plateaus, boundary maxima and multiple local peaks.
3. Constant/changing input intervals and increases, decreases or reversals in
   observed channels, with sample locations and magnitudes.
4. Cross-trajectory contrasts: “At these samples, B has the same recorded value,
   while A differs.” State what was matched and what was not; distinguish
   instantaneous values from complete histories.

Do not select a special pair of peaks, infer a causal delay, prescribe an edge,
or turn correlation into a mechanism. Peak separation is a timing observation,
not a causal explanation. The proposer interprets the facts scientifically.

Maintain a complete machine-readable index of training trajectories. Bound the
prompt projection using a frozen deterministic rule, with omitted counts. “All
trajectories” must not silently mean the six displayed examples. Exact matches
and approximate matches remain separate; near-equality is never silently equality.

The current packet already includes extrema, bounded paths, turning points and
input changes. Defaults are six trajectories, seven path samples and four events.
Its tolerance is descriptive, not a measurement-noise bound. Its hash covers
target/input arrays but not auxiliary values or fixed covariates. A new version
needs those values and interpolation, initialization, history/reset and uncertainty
metadata before relying on them in comparisons. Missing metadata stays unavailable.
Equality of displayed samples is not equality of an entire input history. For
piecewise-linear inputs, for example, compare all defining knots on their common
time domain, including the union of knot times when grids differ.

## Properties and earliest useful stage

This table is our proposed application of the theory. “After topology” means a
complete dependency closure exists; a check can still be deferred if assumptions
are unavailable. Necessary-condition checks do not certify scientific adequacy.

| Property | Training information needed | Earliest useful stage | Question or limitation |
| --- | --- | --- | --- |
| Single-valued algebraic map | Repeated complete argument tuples and corresponding observations | Topology if arguments resolve; otherwise functions/fitting | Can this instantaneous map distinguish these observations? Generated/hidden arguments cannot be assumed equal. |
| Identical closed subsystem | Complete ancestor-state initialization, forcing histories, covariates, time origin, parameter/reset rules | Topology conditionally; initialization may be needed | What permitted information distinguishes trajectories that this subsystem treats identically? |
| Scalar autonomous dynamics | Directly observed scalar state, a rise/fall witness, constant external arguments throughout a reset/impulse-free interval | Topology conditionally | A closed scalar autonomous ODE with uniqueness cannot reverse direction. Coupling, varying forcing or a nonlinear readout changes the hypothesis. |
| Single-valued vector field | Complete states/arguments and trustworthy derivative estimates | Functions and usually fitting | Matching arguments require matching derivatives. Raw noisy finite differences cannot certify a contradiction. |
| Causal response | Matched past forcing and initial information, reset/observation policy | Topology plus initialization contract | Predictions cannot differ before permitted distinguishing information enters. Peak ordering alone is not this test. |
| State continuity | Input/event definitions, observation map and sampling | Variable/topology warning; functions for a specific claim | Finite forcing can change a derivative without jumping a state. Impulses/resets/feedthrough differ; steep sampled changes are not proven jumps. |
| Balance/conservation | Public conserved quantity, external exchanges, units/conversions and states | Interaction; earlier only with exact fixed assembly | Test the complete balance. Opposite signs alone do not ensure cancellation with independent gains/conversions. |
| Positivity and domains | Public admissible state/input domain, beyond sampled minima | Interaction and parameter constraints | Check relevant boundaries. Successful finite probes do not prove domain invariance. |
| Equilibria and recovery | Constant-forcing intervals, baseline and state/readout meaning | Interaction and fitting | Test equilibrium/restoring conditions on functions. A graph cycle does not establish stability. |
| Monotone/order-preserving response | Ordered states/inputs, certified derivative signs and equilibrium/readout assumptions | Interaction, sometimes fitted parameter region | Apply a response-ordering theorem only under its hypotheses. Assembly signs are not Jacobian signs. |

Monotone-systems theory is a promising later extension. A monotone system starting
at equilibrium has a monotone response to a monotone input, under the theorem's
state/input/output assumptions. This can invalidate some response shapes, but is
stronger than our positive/negative contribution labels.
[Angeli and Sontag (2012), Theorem 1](https://www.sontaglab.org/FTPDIR/12cdc_angeli_sontag_monotone_behaviors_final_version_28aug2012.pdf).

For example, the positive term \(+P(B)\) need not increase with B. Its derivative
depends on P. Defer such sign claims until the function class and domain support
them. Boundary invariance likewise needs a vector-field/domain certificate, not
merely positive measured samples.
[Angeli and Sontag (2003), Proposition III.2 and Appendix Theorem 4](https://www.sontaglab.org/FTPDIR/angeli-sontag-monotone-TAC03.pdf).

## Three early diagnostics

### Instantaneous maps

For \(A=h(B,C)\), the same complete \((B,C)\) and parameters must give the same
prediction. Same B with different A is only a preliminary observation if C is
unmatched. Nearly equal arguments do not exclude an unrestricted steep function.
If B is generated, matching measured B does not establish matching candidate B.

Feedback: “Your proposed algebraic relation receives the same complete supplied
arguments at these samples, but A differs. It cannot interpolate both exactly.
Is this an intended approximation under the task's error criterion, or which
permitted distinguishing dependence or alternative representation do you intend?”

An elementary bound makes exact indistinguishability quantitative. If two
observations must have one common prediction p, then

\[
\min_p\{(a_1-p)^2+(a_2-p)^2\}=\frac{(a_1-a_2)^2}{2}.
\]

This bounds these two equally weighted squared errors, not whole-dataset NMSE.
Use actual weights/normalization for an aggregate bound; do not double count
overlapping pairs. Nonzero unavoidable error need not exceed an acceptable
noisy-data error level.

### Matching dynamic subsystems

Let z contain all differential ancestors of the target and v all supplied forcing
used by those ancestors/readout. With common parameters and uniqueness,

\[
z^{(a)}(t_0)=z^{(b)}(t_0),\quad v^{(a)}(t)=v^{(b)}(t)
\quad\Longrightarrow\quad
\widehat A^{(a)}(t)=\widehat A^{(b)}(t),
\]

with equal covariates, time origin and readout rules. Restrict the comparison to
a common uninterrupted open-loop interval, or also match reset/impulse event
times and values, reset maps and any observations used to update state. Identical
reset rules alone do not suffice when their supplied data differ. Identical permitted
initial information gives identical initialization under a common deterministic
g. Latent initials whose equality cannot be established remain unresolved. An
unknown but shared initial parameter is equal across trajectories regardless of
its fitted value; it does not by itself require deferral.
Matching output projections is not matching the complete state.
[Sontag, Chapters 2 and 6](https://sontaglab.org/FTPDIR/sontag_mathematical_control_theory_springer98.pdf).

Feedback: “This subsystem receives identical permitted information in a and b,
but their target traces differ. Identify a permitted initialization/forcing
distinction that the comparison missed, explain an intended approximation under
the task's error criterion, or choose the scientific dependence to revise.”

This does not prescribe a variable, pathway or textbook law. Truly indistinguishable
trajectories can use the same error bound at aligned times. The intended ODE
class must justify uniqueness before a conditional statement becomes definitive.

### Scalar autonomous dynamics

For a directly observed scalar state \(\dot A=F(A)\), a nonconstant unique solution
cannot turn around: a turn requires reaching an equilibrium, and uniqueness makes
the solution through that equilibrium constant. Constant forcing can be absorbed
into F. This excludes
a rise followed by a fall only for that closed scalar hypothesis on an interval
without resets or impulses.
[Slastikov, Section 2.7, Theorem 6](https://people.maths.bris.ac.uk/~mazvs/handout1-1.pdf).

Coupled states, explicit time, varying forcing or a nonmonotone observation map
invalidate this shortcut. Near-flat/noisy reversals remain descriptive unless
uncertainty bounds support a stronger conclusion. Unknown uniqueness remains an
explicit assumption.

## Counterexamples every implementation must preserve

These elementary examples expose tempting but invalid shortcuts:

- Constant input does not imply constant state:
  \(\dot A=-A\), \(A(0)=1\), \(u=0\) gives \(A=e^{-t}\).
- Memory need not require self-dependence:
  \(\dot X=u\) stores its accumulated value after u stops.
- A scalar component can turn in a coupled system:
  \(\dot A=B,\ \dot B=-A\) permits oscillation.
- A monotone state can have a peaked algebraic output:
  \(\dot x=1,\ y=x(1-x)\), for \(0\leq x\leq1\).
- Equal A(0) and inputs do not suffice if a hidden initial state differs:
  \(\dot A=Z,\ \dot Z=0\) gives different A for different Z(0).
- Peak lag need not imply an explicit delay:
  \(\dot B=-B,\ \dot A=B-A\), \(B(0)=1,A(0)=0\), gives
  \(B=e^{-t},\ A=t e^{-t}\). A peaks later without a delay argument.

Thus “A changes under the same input” does not rule out \(\dot A=f(A,u)\).
Peak separation alone does not establish that a model can or cannot fit.

## Feedback and repair

At variables, show the summary plus existing declaration checks. At shared-process
exit, assess only what that partial graph supports. Missing whole-model pathways
remain provisional until ordinary topology is complete. Run compatibility after
complete topology and relevant repairs; defer formula/initializer-dependent tests.

Use a compact record beside the single existing model view:

```text
property: identical closed subsystem
scope: target A and complete differential/algebraic ancestors
evidence: training trajectory IDs, samples/intervals, observed differences
assumptions verified: ...
assumptions unavailable: ...
conclusion: supported conflict / conditional conflict / no conflict found / deferred
question: which permitted scientific dependence or representation is intended?
```

“Supported conflict” needs an established model hypothesis and a declared error
criterion. Label failure of exact interpolation separately from inadequate fit;
the proposer cannot change the task's noise/error tolerances to dismiss a witness.
“No conflict found” concerns only performed checks.
Keep structural validity, public-requirement coverage and compatibility separate.
Show all issues in a short checklist and expand one or two actionable questions;
reference the model/evidence instead of duplicating them. The proposer owns edits.

The current focused policy allows five global repairs. Each local response or
completion sequence gets three attempts, plus two recovery attempts for variables.
Accepted partial replies can start another sequence within the total budget;
these are not whole-stage call caps. Focused variable-stage exhaustion stops the
episode before global repair. Global repair reserves 131,072 budget units within
a 128-call / 524,288-token cap. Admission also depends on context capacity and
conservative request reservation; more calls alone may not be usable.

Current stagnation detection compares the full draft/error signature, so a
description-only change can reset it. A future policy should track substantive
structural changes, resolved questions and changed assumptions separately from
wording. Withdrawing an optional unsupported assertion can resolve that assertion's
inconsistency, but not an independently required mechanism or supported conflict.
A substantive rejected alternative may still justify a clarifying question.

After offline diagnostic qualification, a pilot can allow up to ten global repairs
and retain checkpoints after calls 5 and 10. Version the token reservation and
hard cap explicitly, preserving the initial construction allowance while adding
the extra repair reservation. Successful completion still requires existing
topology eligibility and explicit stage completion. Freeze a semantic-repetition
threshold before launch; stagnation/exhaustion/deferred assumptions remain distinct
outcomes rather than new ways to declare success.
These checkpoints describe within-episode repair yield, not a randomized
comparison against a separate five-call policy. Never reset consumed budgets.
Budget exhaustion is not proof that no compatible model exists.

## Milestones and verification

1. **Evidence/highlights:** freeze all relevant training values and conditioning
   metadata, build a full index and short reproducible observations. Preserve
   acceptance and repair budgets. Unknown metadata stays explicitly unavailable.
2. **Topology diagnostics:** implement the three early checks with assumption
   records and counterexample fixtures; audit saved drafts before live feedback.
   No reference equations, benchmark-specific heuristics or held-out outcomes.
3. **Bounded repair pilot:** retain all seeds; report diagnostic coverage/validity,
   public-requirement retention, changes by repair index and cost. Warning removal
   or structural completion alone is not scientific success.
4. **Interaction/fitting handoff:** assembled balance/domain checks, then Orion's
   numerical reliability and residual diagnostics. Separate fitting failure from
   model-class incompatibility. Defer broader monotonicity/identifiability work.

Fixtures must include exact/approximate matches, different latent initials,
covariates/history, explicit time, coupled states, nonlinear readouts, missing
interpolation/noise, constant nonzero inputs, plateaus and missing channels. Each
check needs a valid conflict, a near-miss counterexample and an abstention case.
Changing any relevant source/policy must invalidate the new evidence hash. Resume
preserves calls/budgets; reject held-out splits before reading their arrays.

Current implementation references:

- `src/autoformalism/search/training_evidence.py`: training packet.
- `src/autoformalism/search/construction_ledger.py`: final topology assessment.
- `src/autoformalism/search/construction_focus.py`, `construction_schedules.py`:
  focused prompts and stage/repair control.
- `src/autoformalism/search/public_graph_obligations.py`: public graph predicates.
- `src/autoformalism/search/causal_initialization.py`: shared initial parameters
  and common maps from permitted initial information.
- `tests/test_training_evidence.py`, `test_construction_focus.py`,
  `test_construction_coverage.py`, `test_construction_ownership_feedback.py`:
  existing train-only, prompt, bookkeeping and resume tests.

This review changes documentation only. Production prompts, checks, budgets and
historical campaigns remain at their existing versions.

Verification: independent theory and code reviews; 60 relevant existing pytest
tests passed; analytic counterexample/error-bound smoke checks passed; focused
Ruff and whitespace checks passed. Repository-wide `ruff check .` reports 37
pre-existing findings in unrelated untracked `analysis/claude` scripts. No live
model calls, optimization, solver rollouts or test-data access were performed.
