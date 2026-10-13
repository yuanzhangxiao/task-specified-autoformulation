# Memory assignments and scientific questions in construction

Policy: `minimal-public-coverage-2`. Study: `coverage_decision_confirmation`.
This revision is implemented for prompt review; no live experiment was launched.
It stops after topology and preserves the previous scientific checks and budgets.

## Evidence motivating this revision

Public-coverage v1 completed 40 episodes, with 38 structurally eligible topologies.
All variable stages completed; 38 shared-process stages completed locally.
Its 176 cached calls accounted for 1,892,819 observed tokens. Structural completion
does not establish scientific consistency. Inspection found incorrect flow uses
and disconnected physical effects among accepted models.

Both incomplete episodes added an optional memory assignment during global repair:
T2-easy seed 2 assigned the meal mechanism to its insulin-memory state in event 3;
T2-hard seed 4 did so in event 5. Neither request demanded that assignment. Both
listed all requirements in `memory_binding_options`, including ones for which
`memory_assignment_required` was false. This may have encouraged confusion with
general mechanism coverage, but the saved traces do not establish causation.

The runtime checks every declared memory assignment, including optional choices.
An assignment claims that the chosen differential states carry the mechanism's
history, with the applicable driver-to-memory and memory-to-target pathways.
The direct meal-to-target path was already present in both examples. Their new
assignments introduced an additional, unsupported claim. Conditional withdrawal
was therefore appropriate: `remove_bindings` removes the claim, not the public
mechanism, equations, processes, or states. The runtime did not choose removal.

The same logic applies to every public requirement and benchmark. No glucose,
insulin, meal, basin, or reactor names are embedded in this policy.

## Prompt changes

- Explain that `mechanism_bindings` assigns memory states, not general mechanism
  coverage. Required memory choices and existing assignments remain visible.
  Unassigned optional choices no longer appear as a menu to fill out; intentional
  optional memory realizations remain allowed and described explicitly.
- Use a generic example: a direct input-to-target process does not need an
  assignment to an unrelated state merely because that state exists.
- For a failed optional assignment, ask whether history was intended or the
  entry was only intended to mark the mechanism represented. Offer explicit
  keep/reassign and conditional withdrawal options. Do not infer the answer.
- Show this decision even when an unrelated incomplete equation makes the
  corresponding missing paths provisional. Keep provisional evidence labeled;
  it does not create a new rejection. Deduplicate the choice across missing
  driver and target paths for the same assignment.
- Make the associated memory-path repair instructions conditional on retaining
  that optional assignment. The same choice covers an invalid assigned state
  type; independent public input-to-target requirements remain unconditional.
- Ask what a positive process value means before choosing its signed uses.
  Distinguish a nonnegative inflow/outflow from an explicitly signed net rate.
- After variables, include `feedback.scientific_review` in every request. It
  references the single existing model view, lists the actual process and local
  equation names when present, and asks about direction, receiving equations,
  and consistency of generated rates/effects with related balances.
- Related-quantity questions list public targets but do not assume they are
  connected. Equivalent inline or shared representations are allowed. The
  proposer is explicitly told not to create artificial connections or sharing.

These questions apply to both local and shared processes across benchmarks.
They extend scientific prompting beyond the old conditional transfer-format
feedback, which only appeared for certain malformed transfer declarations.
An influence entry with a scientifically wrong sign can still pass the unchanged
structural checks; the questions themselves are not a natural-language verifier.

The proposer uses existing `scientific_meaning`, `scientific_role`, and topology
edit fields. There is no new response questionnaire, score, mandatory review call,
automatic sign correction, forced memory binding, or inferred scientific edge.
If a declaration is already consistent, no edit is required. Fresh declarations
in a final successful reply do not receive an extra review call under this policy.

## Scope and provenance

System instruction, editing transaction rules, response schemas, model view,
public science/training evidence, requirement checklist, acceptance predicates,
fitter, and train/validation/test boundaries remain unchanged. Variable-stage
wording changes only to distinguish assignments from mechanism coverage.
Historical `minimal-public-coverage-1` prompts remain available unchanged.

The new study defaults to the same eight Full cases and seeds 0–4, including
both basin cases: 40 constructions. It uses the existing five global repairs,
local recovery limits, request/token caps and early stops. The Jetstream wrapper
uses a fresh `decisions-seeds5-v1` root. No automatic follow-up is configured.
Historical jobs must resume from their original pinned checkout or bundle.

## Offline review

From the repository, after extracting the saved coverage inspection:

```bash
PYTHONPATH=src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/jetstream-coverage-inspection-20261013 \
  --output output/decision-prompts-review-20261013 \
  --policy minimal-public-coverage-2
```

The output includes `PROMPTS.html` and exact before/after requests for each saved
predecessor. Red/green differences are marked. Historical responses are evidence
of the old wording, not responses to the new wording or a simulated new chain.
The source archive is never overwritten. No LLM, fitting, or test-data access is
needed for this review.

Review the short stage instructions, the repeated memory-assignment rule, and
the specific optional-binding and scientific-review sections. The user reviews
these prompts before another live test; launch registration is not submission.

## Review audit

The offline review re-presented all 176 saved requests from 40 episodes (49
variable, 54 shared-process, 43 remaining-topology, and 30 repair requests).
Public briefs, model views, requirement checklists, response schemas, request
settings, and diagnostic evidence were preserved. Five repair requests for the
two optional meal-memory claims receive a coordinated decision, including one
with provisional paths. Scientific questions appear in all 127 requests after
the variable stage, covering all eight cases. User-message UTF-8 bytes increase
by 6.3%; this is not a token estimate. All 177 review pages passed link checks.

One independent prompt audit identified an unconditional path-repair instruction
competing with optional withdrawal. That instruction is now explicitly conditional
on retaining the assignment; independent public pathways remain mandatory.
The focused tests cover missing and valid required/optional assignments, invalid
assigned state types, provisional paths, withdrawal preserving public effects,
generic renaming, unchanged evidence and predicates, live/preview agreement, and
deterministic resume. The combined prompt/integration suite passed 129 tests in
a fixed source snapshot. No live experiment or scientific improvement is claimed.
