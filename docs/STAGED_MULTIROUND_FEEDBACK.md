# Two-round collocation-and-sensitivity feedback pilot

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
   obtain a global-parameter start while retaining the candidate's fixed latent
   initial values;
2. a SciPy trust-region least-squares refinement whose Jacobian is computed by
   integrating the exact forward-sensitivity equations derived with CasADi AD;
3. fresh causal train and validation rollouts from the declared initial values.

The collocation node states are discarded and never enter the reported score.
Unsupported nonsmooth/domain-restricted expression graphs fail closed. The
current adapter supports one output (`v01`), global parameters, fixed numeric
state initials, and open rollouts. These constraints make this a transfer pilot,
not a general production-fitter replacement.

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
