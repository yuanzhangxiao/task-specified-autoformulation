# Repair contract and dependency feedback confirmation

Returned 28 October: both arms completed; all three episodes per arm found
eligibility. Offline replay reproduces all 48 candidate decisions, while the
independent equation audit identifies a missed exact-reproduction conflict in
three eligible 120B meal-saturation candidates. See
[the results, limits and next milestone](PHASE_C_REPAIR_CONTRACT_RESULTS_20261028.md).

This opt-in milestone addresses the interface and feedback defects in the
28 October returned 20B action-feedback and 120B focused runs. It preserves the
scientific acceptance gates, parent models, benchmark data and fitting boundary.
See `PHASE_C_CONSTRUCTION_FEEDBACK_RESULTS_20261028.md` for the motivating evidence.

## Initial options, decisions and bounded recovery

With `repair_contract=True`, the initial option schema requires each patch's
decision to be `revise`, matching runtime expectations. Instructions explicitly
state that the existing scope already permits supported changes to states,
equations, dependencies and latent initialization. Five options remain required;
the schema is not narrowed to function-only repair.

If a provider nevertheless returns an invalid decision in an otherwise usable
option envelope, the runtime preserves the raw patch and metadata, reports a
decision-contract error, and permits bounded clarification. It does not change
the decision, apply the actions, or treat the proposal as scientifically checked.
Selected follow-ups remain explicit same-branch delta/replacement/withdrawal
replies. Genuinely unsupported changes can retain the documented no-actions
wider-revision outcome in a follow-up. No path grants more than the existing
five calls per episode: one batch and at most four branch follow-ups.

Exactly identical action entries can be consolidated during prospective branch
composition, with normalization provenance and original response evidence kept.
Conflicting duplicates remain errors. Original list limits apply before
consolidation; duplicate records cannot evade those limits. Historical composition
and initial-option schemas remain unchanged when the new flag is absent.

## Specific movement and initializer feedback

The new feedback version identifies both directions of a declaration-location
mistake: a new component placed among existing edits, or an existing component
incorrectly declared as new. It shows the exact pending action to drop and the
correct destination for the proposer's intended definition. It does not silently
move a component or choose its scientific role. Parameter/component collisions
remain explicit choices, not assumptions based on symbol spelling.

Initializer feedback uses the actual local alias declarations to explain a
mismatch. An example demonstrates the permitted representation without choosing
a new scientific map, freeing a protected initial identity, or adding an input.

## Qualified conflict and dependency view

For a validated trial with a qualified model-data conflict, feedback traces
whether newly added states/processes have a declared route to the affected
target. Algebraic definitions and latent-initializer dependencies must be
considered; an initialization-only connection must not be reported as absent.
Unsupported or unavailable evidence stays explicit rather than being treated as
a disconnected model.

A path is potential influence, not proof of a nonzero effect, an adequate
mechanism or a successful repair. Absence of a path explains why a disconnected
addition cannot affect that target in the represented model. The view is
advisory explanation beside the existing mathematical witness: it introduces no
new hard gate, no prescribed dependence, and no automatic model modification.
Linearity alone is not an error, and a nonlinear expression does not earn a pass.

## Offline verification and live confirmation

The saved 20B and 120B replies are replayed under their original flags to protect
historical behavior. Prospective reassessment and prompt examples are separately
labeled, preserving source files and raw responses. Cases whose routing would
request a new clarification have no invented provider response. Authored
demonstrations, if used, are not live results or scientific approvals.

Verification on 28 October:

- 451 targeted tests pass, including historical policy behavior, bounded resume,
  decision clarification, exact duplicate provenance, declaration movement,
  protected initializers, unresolved graphs and initialization-only connections.
- Exact historical replay preserves 75 candidate records from 39 saved calls.
  New-contract substitution preserves every eligible record in the five episodes
  with all needed historical replies. One identical-state declaration now clears
  composition but reaches the existing mathematical conflict and stays rejected.
- The remaining 120B episode retains all five invalid-decision options and needs
  a new clarification response. No response or scientific success is invented.
- All 25 prospective requests use the current schema and system text; CPU token
  counts match all 24 corresponding saved server counts. Maximum new input is
  20,948 tokens against the 24,000-token admission limit. Future model responses
  can change that size; live admission remains authoritative.
- Changed files pass Ruff, shell syntax and whitespace checks. Repository-wide
  `ruff check .` still reports 37 unrelated issues under `analysis/claude/`.

Implementation is divided between `construction_search_policy.py` (option
schema/routing), `construction_branch_patch.py` (exact normalization provenance),
`construction_branch_feedback.py` and `construction_repair_influence.py`
(conditional instructions and bounded dependency explanations), plus the campaign
and ACES launchers. Corresponding tests cover both successful and rejected paths.
The generated prompt review is
`output/construction-repair-contract-review-20261028/PROMPTS.html`.

The new `repair-contract` profile uses P00-r1 with seeds 0, 1 and 2 independently
on 20B and 120B. Per arm: three episodes, five initial alternatives per episode,
at most fifteen generation calls, 8,192 output tokens and 32,768 context tokens.
The 20B arm requests one H100 and 120B requests two. Either arm can proceed
without the other; no automatic follow-up or fitting is launched.

Fresh roots:

```
/scratch/group/p.nairr260351.000/u.yx126462/phase-c-search-repair-contract-20b-v6
/scratch/group/p.nairr260351.000/u.yx126462/phase-c-search-repair-contract-120b-v6
```

Upload and extract the delivered source bundle in ACES group scratch. Use the
exact extracted directory as `AF_REPAIR_CODE`, and set the result root explicitly per
command to avoid inheriting another campaign's environment:

```bash
AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
AF_CONSTRUCTION_SEARCH_ROOT="$AF_GROUP/phase-c-search-repair-contract-20b-v6" bash "$AF_REPAIR_CODE/run_search.sh" run 20b full wave1 repair-contract
AF_CONSTRUCTION_SEARCH_ROOT="$AF_GROUP/phase-c-search-repair-contract-120b-v6" bash "$AF_REPAIR_CODE/run_search.sh" run 120b full wave1 repair-contract
```

Replace `run` with `status` to check a submission or `inspect` to create its
`inspection.tar.gz`. `full` is only the three-episode roster. Optional `smoke`
work is included in that same budget. Reconcile ambiguous scheduler receipts
before resubmitting; checkpoints and charges remain attached to the sealed plan.

This is a targeted end-to-end confirmation, not a causal comparison of feedback:
fresh initial model responses may differ even with the same prompts and seeds.
Report initial and follow-up outcomes separately, distinguish interface progress
from scientific progress, and count duplicate eligible models explicitly.
