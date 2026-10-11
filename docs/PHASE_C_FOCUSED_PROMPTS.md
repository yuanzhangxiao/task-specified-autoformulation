# Focused construction requests — review before testing

Policy: `minimal-focused-construction-1`. Study: `focused_prompt_confirmation`.
This is an opt-in implementation and an offline review, not a new live result.
Historical plans must continue with their pinned code. Benchmark prompts, data,
fitter, scientific acceptance rules and test-data separation are unchanged.

## Proposer view

The full draft remains in sealed runtime events. The LLM receives:

- The short system message, current-stage instructions, and the unchanged public
  brief including training evidence.
- One `model_view`: variable types/meanings, symbolic process definitions and
  assembled balances, with joint arguments, signs and fixed conversions. Symbols
  are topology placeholders; function forms and fitted parameters remain later
  work. Unknown conversions are explicitly not claims of unity.
- One `requirements_checklist`, separating declaration/path observations from
  deferred/unavailable scientific checks. `memory_assignment_required` refers
  only to an explicit state assignment, not whether the public mechanism matters.
- `receiver_choices`, a small `previous_edit` receipt, and specific `feedback`.
- Editing rules and the existing stage-specific JSON response schema.

`bookkeeping_policy`, `interaction_review_handoff`, duplicate draft declarations,
and nested copies of diagnostics are absent from the LLM packet. The runtime still
retains the policy, original before/after declarations, calls, receipts, and final
interaction handoff. No full rejected model is repeated. Naming errors retain the
conflicting identifiers, drivers, receivers and conditional repair choices.

The model view is accepted state. The previous-edit receipt says what changed or
why a transaction was rejected. A rejected transaction still commits no partial
edits. Generated algebraic observations and external supplied inputs remain
distinct: observations may be explicitly modeled when permitted; external inputs,
time and covariates cannot become generated receivers.

## Feedback and continuation

1. Missing memory assignments show structured syntax examples for each
   type-eligible state. No example is a scientific recommendation. The proposer
   chooses the appropriate state(s), including new declarations if needed.
2. Invalid supplied receivers show a conditional process edit retaining its
   drivers and valid uses. Moving the intended effect to a local term is an
   alternative. Recently removed process definitions appear in the next request;
   the runtime neither restores nor reinterprets them automatically.
3. Missing RHS entries no longer hide all other dependency observations.
   Provisional driver/memory/target paths and reviewed feedback/forbidden-path
   observations use the declared edges directly. They remain advisory; no fake
   equation is inserted, no binding is inferred and no final check is weakened.
4. A variable-stage delivery/completion repair sequence receives the original
   local attempts plus two recovery calls (five consecutive attempts with the
   current three-attempt setting). Valid incremental declaration turns can still
   continue under the overall request/token cap. Successful completion resumes
   shared processes, then remaining topology. Exhausting a repair sequence stays
   explicitly incomplete rather than jumping to whole-model reconstruction.
5. The new study reserves at most five global repair calls, under the existing
   total request/token ceilings and repair-token budget. Success stops early.
   Two consecutive no-change replies with identical remaining diagnostics stop
   with an explicit no-progress reason. This is a different allocation of the
   existing overall budget; it is not a matched-budget result against earlier runs.

These changes do not infer function-level duplication, impose textbook names or
certify scientific correctness. Multiple drivers in one term remain one joint
function. The variable and shared-stage reports retain their existing first,
post-local and post-global checkpoints.

## Offline review command

From the project checkout, with a saved inspection unpacked outside the new output
directory (no LLM, optimizer or scheduler calls):

```bash
PYTHONPATH=src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/minimal-deferred-v2-inspection-d04b08f9 \
  --output output/focused-prompts-review-20261010
```

Use a fresh output directory. `PROMPTS.html` lists every stage and request, with
direct feedback examples, the new model view, complete request/schema/settings,
and marked differences. `PROMPTS.json` and `SUMMARY.json` support inspection.

The renderer verifies sealed historical event/call chains and projects each
actual displayed predecessor using the same function as the live runner. It does
not chain old replies through hypothetical new feedback. Byte counts are not
token counts. Existing focused-policy calls are shown as recorded. A fresh live
confirmation is needed to measure actual prompt tokens and proposer behavior.

No live test is submitted by the renderer. Review the prompts before using the
new opt-in campaign study.
