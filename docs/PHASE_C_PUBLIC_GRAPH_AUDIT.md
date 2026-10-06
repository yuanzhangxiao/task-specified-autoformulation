# Public dependency audit before functions

The historical version-4 implementation below is retained for provenance.
Version 5 replaces its permissive ancestor-cycle predicate with target-local
feedback and explicit proposer readout coordinates; see
`PHASE_C_CONSTRUCTION_COMPARISON.md`. The audit now reports whether its exact
reviewed contract matches the source plan, and keeps overlap clarifications
separate from graph errors. Use the original commit to reproduce old audit values.

Construction comparison version 4 added a reviewed public graph contract, shown
from the first request and repeated during repair. This is a prospective
operationalization of the public task, not a claim that these predicates were
already stated explicitly in historical prompts. Finalized benchmark prompts,
data, historical campaigns, fitting and independent evaluation are unchanged.

## Scope across all eight cases

| Case | Existing structural obligations | Added predicate | Still needs scientific/function inspection |
| --- | --- | --- | --- |
| T1 easy | Generate targets; meal-to-glucose path | None | Balance roles, physical signs, duplicate contributions, initialization |
| T1 hard | Generate targets; meal-to-glucose path | None | Same; a textbook compartment skeleton is not required |
| T2 easy | Meal path; insulin through declared dynamic memory to disposal; supplied Uii composition | None | Actual delayed disposal, physical signs, source/sink identity and duplication |
| T2 hard | Meal path; insulin through declared dynamic memory to glucose | None | Disposal/source identification, physical signs and latent meanings |
| CSTR | Generate temperature; `controlled_balance` has no executable driver endpoints | Dynamic feedback affecting temperature | Distinct feed/reaction/jacket roles, heat-flow signs, scaling and restoring behavior |
| Alien device | Input through declared dynamic memory to output | None | Response behavior and initialization; input memory does not necessarily mean fading memory |
| Coupled basin | Local inflow path and upstream dynamic-memory path | Dynamic feedback affecting downstream depth | Which storage the feedback represents, upstream depletion, transfer cancellation, area conversion and threshold law |
| Independent basin | Local inflow path | Dynamic feedback affecting depth; no upstream-inflow path to depth | Threshold law, physical crest versus warning threshold, nonnegative water balance |

The source catalog being available does not require every supplied channel to
be used. In particular, the new contract does not force CSTR to read C or Tj
directly, force a particular basin latent name, or demand a named shared process.
No exact Arrhenius, overflow or textbook glucose equation is supplied.
No self-dependence rule is added to the Dalla or alien memory requirements.

## General deterministic predicates

The runtime uses two domain-independent predicates on the assembled dependency
graph, including ordinary equations and named process definitions/uses:

- `dynamic_feedback(target)`: there exists a differential state s on a nonempty
  directed cycle, with a path from s to the target (or s is the target itself).
  This accepts direct feedback, coupled states, and a dynamic energy/volume
  coordinate with an algebraic temperature/depth readout. A zero-length path
  alone is not feedback. A disconnected cycle does not pass.
- `forbidden_path(source, target)`: no declared directed path exists from source
  to target, including paths through generated variables and named processes.

Each check reports its identity, public quote, reviewed interpretation, status,
and a concrete path/cycle witness when present. Public quote matching validates
provenance only; the runtime does not determine the science by reading prose.
The reviewed eight-case catalog is separate from the general checker.

Feedback is only a necessary structural capability under this reviewed balance
interpretation. A positive loop can pass, and a loop through the wrong physical
state can pass. It proves neither dissipation nor correct scientific assignment.
Likewise, a declared path need not remain effective after functions/parameters
are chosen. These remain later checks; no functions are rewritten or invented.

Checks wait for whole-graph compilation and then run before interaction functions.
Compilation failure produces `unavailable_graph`, with a null pass value. Known
failures enter the existing bounded overall repair with the same public context,
current assembled draft and contract. There is no extra LLM, prose judge, budget,
confirmation phase or automatic scientific role assignment.

`controlled_balance` remains listed among unresolved public predicates. Adding a
feedback-capability check does not make feed/reaction/jacket separation verified.
The report separately lists deferred scientific checks for every case.

## Saved live confirmation audit

Source plan: `cc917b501a637115f884a8bf4598417872af391525cb62c5dc8c0d90dc55e277`.
All 24 final drafts were available. Call records and transaction chains were
verified; initial and final drafts were checked in their actual saved forms.
There was no hypothetical new response chain or promotion of historical models.

| Quantity | Count |
| --- | ---: |
| Historically structurally eligible final drafts | 17 |
| Eligible under current bookkeeping without added predicates | 17 |
| Eligible under prospective added predicates | 16 |
| Added final predicate checks: pass / fail / unavailable | 8 / 1 / 3 |

The one newly flagged final draft is `cell04_seed0_full_joint_fixed`: CSTR's
temperature derivative depends on feed temperature, concentration and jacket
temperature, but no generated dynamic state feeds back into its evolution.
The predicate permits an energy-state alternative; it does not demand a literal
T source or a particular heat law.

Passing the new checks does not resolve the previously observed T2 sign/duplication
or basin crest/transfer questions. The audit is not a new mechanism-compliance
score and cannot establish any model-capacity or strategy-ranking claim. No LLM
calls, optimizer calls, solver rollouts or trajectory/test-data reads occurred.

## Reproduction and future use

Using the current pinned source and an extracted historical inspection package:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AF_REPO_ROOT/src"
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/audit_construction_graph.py" \
  --source "$AF_SAVED_CONSTRUCTION_ROOT" \
  --output "$AF_AUDIT_OUTPUT_JSON"
```

Output must be outside the historical source directory. The sealed JSON records
public contracts, source checkpoint hashes, every initial/final check, witnesses,
deferred checks and missing drafts. Identical reruns are deterministic; changed
source/tool results require a different output path. An incomplete campaign is
not treated as zero failures. Missing/tampered call evidence fails closed.

Version 4 freezes the public contracts and their source-brief hashes into each
new campaign. Resuming verifies the reviewed contract as well as source identity.
Older plans must use their pinned code. New defaults are
`phase-c-construction-comparison-v4` and `phase-c-construction-live-v3`; this audit
does not submit either. The existing live wrapper still stops after variables,
topology and bounded repair, before functions or fitting. Any future comparison
must give all three strategies the same added contract and fresh calls.
