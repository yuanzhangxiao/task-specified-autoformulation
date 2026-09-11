# Bounded collocation-and-sensitivity feedback pilot

This development-only experiment is the first numerical feedback loop over
complete staged candidates. It is deliberately narrow: it reuses the two
anonymous-system candidates that remained unresolved after the bounded fitter
rescue, performs two revision-and-fit rounds, and changes neither the production
fitter default nor the benchmark prompts.

## Frozen scope

The source is the completed six-candidate staged fitter-rescue campaign. Source
task indices 3 and 4 are the anonymous hard-cell seeds 0 and 1 whose attribution
was `unresolved_after_bounded_rescue`. Their candidate JSON, rescue diagnostics,
public train/validation files, prompt, and manifests are copied and hashed before
any new call. The pilot has two tasks and two rounds, hence four fits at most.

No candidate is regenerated from scratch. Test data and private references remain
closed. The scientific judge is not called, and no automatic winner is declared.
Validation is not an optimizer input: parameters are selected from training only,
then both training and validation are replayed causally for descriptive scoring.

## Numerical route

Each revised candidate is compiled through the restricted expression grammar.
Eligible smooth, single-target candidates then use:

1. train-only CasADi integral collocation with two-stage Radau IIA constraints to
   obtain a global-parameter start while retaining initial values resolved
   causally at each trajectory boundary;
2. a SciPy trust-region least-squares refinement whose Jacobian is computed by
   integrating the exact forward-sensitivity equations derived with CasADi AD;
3. fresh causal train and validation rollouts from the declared initial values.

The collocation node states are discarded and never enter the reported score.
Unsupported nonsmooth/domain-restricted expression graphs fail closed. The
current adapter supports one output (`v01`), global parameters,
parameter-independent fixed or restricted analytic state initials, and open
rollouts. Direct-observation initializers such as `v01(t0) = observed v01(t0)`
are resolved independently for each trajectory by the same runtime path as final
causal simulation, with certified zero initial parameter sensitivity. Fitted
initial-state ranges remain outside this adapter's contract. These constraints
make this a transfer pilot, not a general production-fitter replacement.

## Feedback priority and routing

The runtime first checks contract validity. For the frozen sources the diagnosed
failure is numerical instability, and the deterministic selector identifies the
smallest generated strongly connected component containing a superlinear
dependence. In the reviewed candidates this selects `f` and `v01`.

Round 1 is always a **function revision**. The proposer must replace the complete
expressions of exactly those selected components while preserving their
nonparameter source sets. The runtime rejects missing/extra symbols, undeclared
symbols, parameter-role conflicts, exact functional duplicates, and any topology
change before fitting.

Round 2 reuses the same selected component boundary:

- after a numerically stable round 1, it performs a second function refinement;
- after persistent instability, it performs one topology backtrack, allowing only
  those selected equations to add or remove dependencies among already available
  variables; the variable inventory and target mapping remain fixed.

This policy tests the proposed priority—function before topology—without claiming
that it is universally optimal. The selector is a conservative structural proxy
for likely impact, not an estimate of validation improvement. A future campaign
can compare it with counterfactual repair ranking once enough paired revisions
exist to estimate impact without test leakage.

A numerical-adapter contract failure is not scientific evidence against either
the functions or topology. It terminates that task's fitting attempt with the
typed `fitter_contract` class and is never routed into a proposer revision.

Every rejected LLM response receives the exact deterministic contract error on
the next bounded retry. Every accepted revision, fit, call, and progress record is
checkpointed, content-bound, and deterministically reusable.

## Version-2 role and locality repair

The first live run never reached fitting: both candidates exhausted all three
round-one attempts because a reused parameter name was returned with a role that
did not match its parent declaration. Version 2 removes that transport-level
failure without relaxing numerical or scientific contracts:

- a reused parameter identity always retains its parent role; a conflicting
  proposed role is logged as a deterministic normalization rather than rejected;
- a new parameter used once as a direct scalar multiplier of one scientific term
  receives the runtime role `nonnegative_coefficient`, because the expression's
  explicit outer `+` or `-` owns direction;
- a new bare top-level additive parameter receives the runtime role `offset`;
- a new internal parameter that cannot be certified by those rules must use a
  specific `shape`, `positive_shape`, `rate`, `time_constant`, or `scale` role.

The provider sets `role` to `null` for reused parameters, direct term gains, and
additive constants. Thus it does not redundantly decide optimizer signs for
ordinary linear coefficients. The nullable field remains explicit only because
the strict structured-output transport requires a fixed object shape.

The certification is intentionally syntactic and narrow. It does not infer a
scientific sign for a nonlinear source law, alter any expression, or reinterpret
an ambiguous nested parameter. A rejection names the parameter, occurrence,
allowed roles, and reason. The full rejected response and structured diagnostic
are durably written after every attempt, including the final exhausted attempt.

Version 2 also narrows the initial repair boundary. It selects the first directly
superlinear generated component and its coupled target-producing component. It
does not expand the whole strongly connected component merely because the
candidate graph is dense. For the reviewed sources this is `f` and `v01`.
Topology backtracking remains available in round 2 after persistent instability.

Before new provider calls, `scripts/replay_staged_multiround_roles.py` can replay
all stored version-1 round-one responses through the version-2 role policy. The
replay performs no LLM call, fitting, judging, test access, or private-reference
access; it reports any later contract failure that was previously masked by the
role mismatch.

## Interpretation

A successful round establishes that a localized revision plus the stronger fitter
can produce a causally scoreable candidate. It does not establish scientific
correctness or superiority to a baseline. A failed round may still combine poor
functional form, an unsuitable topology, or remaining numerical limitations.
The recorded route, selected components, initializer status, sensitivity optimizer
status, causal rollout failures, and train/validation NMSE keep those explanations
separate.

The summary reports completion, route counts, stability by round, collocation and
forward-sensitivity success rates, stable validation NMSE, LLM resources, and all
per-round records. There is intentionally no scalar winner.

## Revision-4 failure-harvesting protocol

Revision 4 extends the same two frozen public candidates to four bounded rounds
inside one allocation.  It is an engineering campaign for revision transport,
static function-domain evidence, numerical diagnostics, and feedback routing; it
does not change the benchmark prompts or define a model winner.

Revision replies are no longer all-or-nothing.  Every named component is checked
independently against the frozen parent topology.  A valid component is retained
provisionally, the parent candidate remains externally unchanged, and the next
request contains only components that are still invalid.  The complete candidate
is committed only after every originally selected component passes.  Exhausting
one bounded revision action records the complete responses and named diagnostics,
retains the last valid parent, and continues the next campaign round instead of
terminating the task.

Existing parent parameter names may be used without another declaration.  Their
roles are inherited exactly.  Only genuinely new parameter names are declared;
direct outer gains and additive offsets receive runtime-derived roles, while an
ambiguous nonlinear internal parameter still requires a qualitative role.

Before each fit, a narrow static audit reports state-dependent denominators that
are not certified away from zero on the unrestricted generated-state domain.  It
accepts total-domain constructions such as `1 + x**2` and `1 + abs(x)`, and it
records a symbolic possible-zero condition for expressions such as `1 + x`.
This certificate does not claim that a trajectory reaches the singular set.  A
finding keeps the repair at function level and prevents an expensive fit until
the affected function is replaced.  Once the static function audit passes,
persistent rollout instability may route to a topology backtrack; later rounds
alternate back to function repair after a topology change.

Numerical success uses the verified fitter-validity contract.  Native optimizer
termination is reported separately and is insufficient: at least one finite
training residual evaluation, a fresh returned-point training check, and a fresh
production training replay are required before the fit is accepted.  Bounded
observed rollout-failure evidence is preserved for feedback without asserting an
unproven scientific cause.

Latent initial values remain the fixed or causal proposer-originated values in
this revision.  Learning global latent initial values from training data, and
defining a causal validation/test initializer without outcome leakage, are a
separate numerical milestone so their effect is not confounded with revision
transport and diagnostic changes.
