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
