# Qualified admission correction and initialization reminder

This opt-in milestone closes the concrete admission gap identified in the
28 October repair-contract results. It does not introduce a function toolbox,
change benchmark prompts/data, prescribe physiological mechanisms, or launch
fitting. Historical policies and results remain reproducible.

## Defined-domain zero-input proof

`defined-common-zero-forcing-affine-1` extends the existing certified common-zero
restriction with a bounded domain case split. For a literal quotient whose
numerator is provably zero and whose restricted denominator depends only on
shared global parameters, the checker distinguishes:

- Nonzero denominator: the quotient is zero and the existing exact affine
  trajectory identity applies.
- Zero denominator: the literal expression is undefined; there is no declared
  removable-singularity continuation that supplies a valid solution.

The proof never modifies candidate equations, bounds or parameter values. It
records original expressions, restricted denominators, inherited obligations,
all substituted input schedules and the exact initial/readout premises. Nested
parameter aliases retain their obligations. State/time-dependent denominators,
unsupported expressions, unavailable schedule/error evidence and genuine state
nonlinearity remain outside this correction. It is not a universal compatibility
test or a bound on approximate fitted NMSE.

All conditions remain visible in feedback beside the target-specific witness.
A zero-input reminder or a conditional sign concern is not itself a rejection
criterion. In particular, `Gp'=-U` is not rejected merely because it lacks a
positive term; applicable observations and qualified assumptions are required.

## General initialization reminder

The new profile includes this guidance in `editing_scope.initialization_review`
when proposing or revising a candidate, beside the permitted initializer fields:

> When choosing or revising initialization, consider the recorded starting
> conditions, including trajectories with zero or unchanged external input.
> Zero external input does not imply zero internal states, zero rates,
> equilibrium, or equal hidden initial states across trajectories. Use only the
> permitted initialization information and fixed window policy; subsequent
> behavior must also be supported by the proposed dynamics. This reminder alone
> is not evidence of a model-data conflict.

This is benchmark-independent guidance. It adds no requirement to use decay,
basal production, a particular hidden state, or equal initial conditions. Public
task text and the fixed observation-window policy are unchanged. Historical
profiles do not receive the new reminder.

## Repair bookkeeping feedback

`typed-branch-action-feedback-3` shows mutually exclusive action fragments when
a reply both requests a component removal and drops that pending removal. It
distinguishes keeping the existing component from deleting it and explicitly
repairing its consumers. A missing pending action is not silently manufactured,
and protected or unknown-scope components receive no unauthorized removal option.

For two conflicting initializer declarations, feedback identifies the owning
`new_states` entry and the explicit duplicate action to drop. For qualified
conflicts it displays the actual typed initial scope. It does not infer a
scientific choice from prose or silently translate a claimed zero initial value
into a structured constant.

## Saved results and fitting boundary

All 24 saved calls and 48 candidate outcomes replay exactly under their original
flags. Prospective reassessment of each actual saved patch changes exactly three
decisions: 120B initial option 0, seeds 0/1/2, now receive
`QUALIFIED_MODEL_DATA_CONFLICT`. The other 45 decisions are unchanged.

The remaining 14 records represent 14 distinct complete-model/domain identities:
eight 20B and six 120B models. They are construction candidates for fitting
qualification, not scientifically certified models. Their full problems and
audits are retained under
`output/construction-admission-guard-delivery-20261028/replay/fitting-candidates/`.
Remaining consistency checks are explicitly deferred; the conditional balance
concerns in the preceding equation audit remain relevant. No fit, numerical
selection or model promotion was performed.

Prospective prompts for the historical selected branches are counterfactual
feedback previews. The corrected policy may select different live branches;
old replies are never presented as responses to these new prompts. In particular,
the three newly blocked candidates have no saved repair response.

## ACES retest

The fresh `admission-guard` profile uses the same frozen `P00-r1` parent and
seeds 0, 1 and 2 on both model sizes. Each episode asks for five initial
alternatives and permits at most four shallow follow-ups: at most fifteen calls
per model arm, with the existing output/context/token limits. There is no
automatic follow-up or fitting. This is the targeted admission confirmation,
not the later eight-benchmark portfolio qualification.

Upload the delivered `phase-c-search-admission-guard-<commit>.tar.gz` to ACES
group scratch. Set `AF_ADMISSION_CODE` to the exact extracted directory shown in
the delivered `COMMANDS.txt`; run each command as its own terminal line:

```bash
AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
env AF_CONSTRUCTION_SEARCH_ROOT="$AF_GROUP/phase-c-search-admission-guard-20b-v7" bash "$AF_ADMISSION_CODE/run_search.sh" run 20b full wave1 admission-guard
env AF_CONSTRUCTION_SEARCH_ROOT="$AF_GROUP/phase-c-search-admission-guard-120b-v7" bash "$AF_ADMISSION_CODE/run_search.sh" run 120b full wave1 admission-guard
```

Replace `run` with `status` or `inspect` for the corresponding arm. `inspect`
creates its `inspection.tar.gz`. Optional one-episode `smoke` is included within
the full roster and budget. Use the fresh v7 roots; do not resume an older root
under the new contract. Reconcile ambiguous scheduler receipts before retrying.

Prompt review and replay:
`output/construction-admission-guard-review-20261028/PROMPTS.html` and
`REPLAY.html`. All generated artifacts remain outside Git.

## Offline verification

The seven affected test modules pass all 457 tests. The exact historical replay
reproduces 24 calls and 48 outcomes, while the new-policy reassessment changes
only the three documented decisions. Input files remain unchanged. All 27
prospective requests fit the 24,000-token input allowance; the largest contains
22,281 tokens. Counts were calibrated against all 24 recorded server counts.
The saved responses do not measure success under the revised prompts; that is
the purpose of the bounded live retest. Verification receipts, full-suite and
repository lint logs are retained in the generated delivery directory.
