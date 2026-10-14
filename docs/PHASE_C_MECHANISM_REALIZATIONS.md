# Mechanism realizations and optional memory assignments

The opt-in `minimal-mechanism-realizations-1` policy replaces the memory-only
bookkeeping field with one explicit proposer-selected record per public mechanism.
Historical policies, saved requests, and public benchmark evidence are unchanged.
This milestone covers variables through topology; no live confirmation was run
while preparing the prompt review.

## Contract

```json
{
  "mechanism_realizations": [
    {
      "requirement_id": "public_requirement_id",
      "components": ["chosen_quantity", "chosen_shared_process"],
      "memory_states": []
    }
  ]
}
```

Components may be supplied quantities, generated algebraic variables,
differential states, or shared processes. Several components can cooperate. For
a local contribution, cite the generated variable whose equation contains it;
this contract adds no per-term identifiers. The runtime derives component types,
so the proposer does not repeat a potentially contradictory realization-type tag.
A record containing only supplied quantities has unresolved generated type.

At variable-stage completion every public mechanism needs a record identifying
its chosen quantities. Later stages extend or revise those records as processes
and pathways are introduced. Partial drafts may reference missing components,
but their declaration checklist cannot pass until those references resolve.
Each edit replaces the complete record. `remove_realizations` removes a record by
requirement ID; it does not make the public requirement optional.

`memory_states` is a subset of components and contains only declared differential
states. It is empty for an instantaneous algebraic realization unless history is
explicitly claimed. Existing public requirements that explicitly demand memory
still require selected differential mediators and the applicable input-to-memory
and memory-to-target paths. Withdrawing an optional history claim clears this
list while retaining or explicitly revising the mechanism's components.

## What is checked

The checklist distinguishes three levels:

1. **Declaration:** each required record exists; component names resolve; memory
   selections have the required type and refer to components in that record.
2. **Structure:** the existing applicable public and selected-memory path checks
   hold, and cited components contribute to a public target. Fixed conversion
   covariates can support a realization without being newly treated as dynamic
   pathways. Unavailable graph or endpoint evidence stays unavailable/unresolved.
3. **Behavior:** unverified here. A declaration or path does not establish a
   correct function, sign interpretation, conservation, units, stability, initial
   conditions, or agreement with trajectories.

The runtime does not choose a scientific association or manufacture a latent
state. A mechanism can be entirely algebraic within a dynamical model. The
existing compiler still requires the overall model to have a differential state.
General component references identify whole variable definitions, not a verified
attribution to individual local terms.

## Prompt and runtime changes

The dictionary now says mechanisms can involve **variables**, processes, and
pathways. The variable prompt requests every mechanism's components and makes
memory conditional. Shared-process, topology, and repair instructions maintain
the same records. The equation view includes them; actionable feedback points
to missing records, stale references, wrong memory types, and failed paths.
Old memory-only fields are rejected with migration guidance rather than silently
reinterpreted. Wrong-category deletions identify the retained owner.

Transactions, receipts, stage reports, final topology handoffs, and deterministic
resume preserve the full records. The handoff carries the compiled topology hash
and separate declaration/structural/behavior status. It is available for later
interaction work; the current function builder, fitter, initializer and pruning
code do not consume this new general map directly.

## Review and confirmation scope

Generate the marked review from retained evidence, without LLM calls:

```bash
cd /Users/yuanzhangxiao/Projects/autoformalism
PYTHONPATH=src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/jetstream-ownership-inspection-20261013 \
  --output output/mechanism-realizations-review-20261014-final \
  --policy minimal-mechanism-realizations-1
```

The output directory must be new or empty. The review contains all 160 historical
request positions across 40 episodes, re-presented under the new contract. Only
already-explicit memory selections are projected for this offline preview; other
general realization records remain missing. Historical replies are not answers
to the new requests. The public brief is unchanged.

`mechanism_realization_confirmation` is the separately versioned next study:
eight Full cells, seeds 0–4, with the preceding policy's 133-request and
655,360-accounting-unit episode caps. It reserves ten global repairs and 262,144
accounting units, stopping early on success or three repeated semantic failures.
These accounting units are not model-token context capacity. Initial allowances
are unchanged. New training highlights and conditional topology/data diagnostics
remain offline; this change enables none of them as live gates.

## Verification

- 157 focused regression tests passed, including full construction/report/preview
  and cached resume, missing records, algebraic realizations, optional memory
  withdrawal, transactional rejection, and zero-call checkpoints.
- The full suite passed 5,096 tests and skipped eight Torch-dependent tests.
  Twelve additional tests detected source changes made during the first run;
  all twelve passed when rerun against the settled source.
- `ruff check src/autoformalism tests scripts` passed. Whole-repository
  `ruff check .` reports 37 existing findings in unrelated `analysis/claude` files.
- The generated review audited all 160 request positions: public briefs unchanged,
  new schema present, no stale memory-only instructions, and valid local HTML links.
  Historical rejected old fields appear only as explicitly obsolete-field feedback.
- These are software and prompt-delivery checks, not a live scientific evaluation.
