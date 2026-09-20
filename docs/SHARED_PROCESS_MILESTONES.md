# Shared-process modeling: milestones and first audit

This is a development protocol. Keep both `full` and `brief_only`; the final
method choice remains open. The numerical fitter remains frozen. Incorrect or
contradictory specification experiments and alternating optimization are deferred.

The optional explicit process-review follow-up is preceded by the new
[detention-basin qualification](DETENTION_BENCHMARK.md). This adds a realistic
positive control and an independent-basins negative control outside the existing
domains, with isolated development data and a frozen-fitter attainability audit.
It does not yet change the construction flow below, enable the review phase,
or select a new model/reasoning setting. Review these CPU results first.

## Where shared processes enter

A shared process is **introduced during variable construction**, using the existing
algebraic-variable concept. It has a scientific meaning and may be used in several
equations. It is not a new latent state and has no initial condition.

| Stage | Proposer's scientific decision | Runtime responsibility |
| --- | --- | --- |
| Variables | Identify states and reusable processes; give each a name, role and dynamic/algebraic type. Reuse an existing process where appropriate. | Maintain one inventory and detect duplicate/conflicting definitions. Do not impose a shared process where none is justified. |
| Topology | State a process's drivers and the equations it affects, including contribution signs and any scientific conversion factors. | Keep process identity across consumers. Derive the dependency graph through that identity, including its upstream state dependencies. |
| Interactions | Define the analytic process law once. Define necessary conversions at its consumers. | Compile one law and one parameter registry, rather than independently generating a new law/gain for every consumer. |
| Assembled review | Examine the complete coupled equations and declared shared relationships. | Check closure, signs, parameter identity and explicit balance claims. Keep broader scientific interpretation separate. |
| Revision | Add, change, split or remove a process with all affected definitions in one coherent patch. | Commit atomically only after whole-model checks; preserve learned parameters only where their meaning is unchanged. |

Example of a same-unit transfer, where `q` is a named algebraic process:

```text
q = k * A / (K + A)
A' = input - q
B' = q - loss
```

The runtime must not turn these consumers into `-a*q` and `+b*q` with independent
unrequested gains. If states are concentrations in different volumes, an explicitly
declared amount flux could instead enter as `-q/V_A` and `+q/V_B`. Equality of the
concentration derivatives is not the conservation requirement.

A shared response need not be a transfer:

```text
r = tanh(x - baseline)
y' = a*r - decay_y*y
z' = b*r - decay_z*z
```

Here independent `a` and `b` may be scientifically appropriate. The contract must
preserve them. A process can be signed; positivity is not inferred merely because
it is called a process. Internal differences and nonlinear sign changes remain
part of its law. Opposite equation signs alone do not establish conservation.

The first construction pilot distinguishes transfer, conversion and response
through existing scientific-role fields and guidance. It introduces no redundant
process type or machine-certified balance declaration. The detailed saved-model
audit did not justify imposing a new transfer schema; richer checked relationships
remain a possible later extension, guided by evidence.

## Milestone 1 — inventory and removal of revision quotas

Implemented in this commit:

- `search/review_revision_v6.py` is an opt-in whole-model revision adapter. It
  removes the six-equation, two-new-variable and hidden twelve-total-variable
  transaction checks. Associated removal and initializer list quotas are removed
  so a coherent whole-model patch does not fail in a nested legacy schema.
- Actual model counts remain reported. Unused new parameter metadata is cleaned
  up, while expression safety, parameter meaning, initializer completeness,
  dependency closure, and output causality checks remain active. Public-mechanism
  and ablation checks remain the campaign's separate deterministic gate.
- Historical v2–v5 schemas and campaigns retain their original behavior. The v6
  adapter is tested but deliberately **not dispatched by an existing campaign**.
  Milestone 2 will wire it into a new, explicitly versioned construction campaign.
  No existing pinned job or saved result is silently changed.
- The count changes apply to revision patches. They do not silently remove
  initial-construction agenda limits. Initial construction's resource policy is
  part of Milestone 2. Restricted expression size/depth, provider context and
  request/fit budgets remain finite; no extra numerical budget follows from a
  larger model.
- The underlying candidate serializer still has its pre-existing capacities
  (64 states, 256 algebraic processes, 256 parameters). The new prompt/payload
  expose those capacities explicitly. They are distinct from the removed small
  revision quotas; this milestone does not change the candidate storage schema.
- `scripts/audit_shared_processes.py` inventories sealed saved public candidates.
  It reads equations, context and provenance, not trajectory tables, and invokes
  no optimizer or LLM. There is no automatic rewrite or scientific selection.

The inventory produces four distinct kinds of evidence:

1. **Named processes already shared:** one algebraic definition directly used in
   at least two other definitions, including observation mappings.
2. **Exact repeated terms:** the same parsed unsigned outer term occurs in at
   least two definitions. Outer signs are recorded separately; internal differences
   remain intact. These are conservative syntax matches, not full algebraic proofs.
3. **Exact repeated nonlinear calls:** the same parsed call occurs in different
   definitions, potentially beneath different gains.
4. **Similar terms for review:** consistent renaming of parameter symbols makes
   two terms match, with state/input names and constants preserved. Parameters may
   have different roles or fitted values. This is a review prompt only; tying those
   parameters would change the model and requires a scientific decision and refit.

Witness groups can overlap, include already named processes, or be irrelevant to
a physical balance. Do not add their counts to estimate unique mechanisms or
report them as a mechanism-compliance percentage. The report leaves
`missing_shared_mechanisms` unset. Equivalent factorizations, commuted products
and undeclared physical unit relations may be missed. A human equation review is
needed to decide which opportunities matter.

Unavailable or invalid models stay explicit; they never become zero-opportunity
models. The input snapshot, audit implementation and each row are hashed. Reruns
reuse checked rows; a different input/implementation requires a new output root.
The audit does not change or advance the historical search.

Run this audit on the retained global round-15 models, **both prompt variants**.
Expected coverage is six cells × two seeds × two variants = 24 rows, with any
unavailable models retained in that denominator. Inspect at least one witness
per detected group before designing the shared-process prompt pilot.

The private reference inventory is separately documented in
`SHARED_PROCESS_REFERENCE_INVENTORY.md`. It must never be loaded into a proposer,
critic, residual packet, or public construction audit. It is not a list of hidden
requirements that discovered reduced models must reproduce.

Milestone exit: inspect coverage and representative witnesses, confirm that the
proposed shared-process contract fits the task-sufficient models, then proceed to
construction. This audit needs no new fitting experiment.

Local verification (2026-09-19): 123 focused revision/audit/transaction tests
passed. The full suite reported 2,398 passed, four skipped and two source-freeze
failures because package files changed during that run. Both affected complete
test files were rerun against stable source: all 34 tests passed. The audit CLI
and identical resume passed; the historical revision smoke also passed its real
small fitting/fallback checks. Changed-file lint passed. Repository-wide
`ruff check .` reports 37 pre-existing findings in the unrelated, untracked
`analysis/claude` scripts; those files were left unchanged.

## Milestone 2 — shared-process construction and coherent revision

Implemented as the opt-in `shared-process-pilot-1` comparison; see
[SHARED_PROCESS_PILOT.md](SHARED_PROCESS_PILOT.md) for the frozen matrix, commands,
reporting and limitations. The existing variable/topology/interaction prompts
receive optional stage-specific guidance. Whole-model v6 revision is wired into
both arms. Both also use explicit broader construction capacities, with unchanged
LLM and fitting budgets. Historical campaigns retain their own policy.

The detailed round-15 audit found 23 available models, 16 exact groups in eight
models and 67 similar groups in eighteen. Exact groups already share parameter
names; factoring them is not a parameter reduction. The sole directly reused named
process was an output expression consumed by its mapping and one state equation.
The pilot therefore reports governing-equation reuse separately and does not turn
syntax witnesses into transfer certification.

The matched pilot covers three families, two seeds, both initial-information
variants and guidance on/off: 24 fresh lineages and one revision round, at most
48 fits. Review round 0 before explicitly submitting round 1. Tests compare inline
and named law values and parameter derivatives; an offline prescribed-provider
smoke exercises the real frozen fitter and atomic revision. Scientific benefit
requires the live development results. No pruning, new judge or test access follows
automatically.

## Milestone 3 — process-aware pruning

Start from the retained fitted model. Propose removal of redundant terms or whole
processes; a transfer must be changed consistently at all its consumers. Remove
orphan parameters/initializers. Re-run deterministic requirements and a bounded
training refit with the frozen fitter. Use a predeclared validation/complexity
selection rule and retain the original if the pruned model is not preferable.
Pruning cannot delete a target, violate a declared balance, or use test results.

Measure parameter/state/process reductions, NMSE changes, runtime and deterministic
mechanism evidence with pruning on/off. This is model selection, not evidence that
every small fitted coefficient is scientifically unnecessary.

## Milestone 4 — calibrated judge as critic, with separate routing

Reuse the existing calibrated GPT-OSS-120B scientific judge and its assessment
rubric. Do not silently add fitting scores or routing instructions to its
calibrated input. Its output is advisory: deterministic executable requirements
remain blockers, and scientific persuasiveness cannot override them.

Start with the established scientific review before fitting. If pruning or
revision changes the equations, review the changed model again using the same
cached judge contract. Avoid repeated calls for an unchanged structure.

A separate adapter joins the judge findings, deterministic evidence and qualified
training residuals for the proposer. The proposer can revise any coupled part of
the model. If LLM advice about what to investigate next is added, log it as a
new, uncalibrated routing extension; it is not a new scientific certification.
Keep deterministic budget and acceptance controls, and preserve the incumbent.

## Milestone 5 — focused component comparisons

After the preceding pieces work, run shared-process on/off and pruning on/off
comparisons. Separately cross critic on/off with deterministic scientific verifier
on/off. The four cells are both, verifier only, critic only, and neither;
"no critic" and "verifier only" are the same cell, not extra ablations.

Turning off the scientific verifier does not turn off parsing, causality, domain
or execution safety. Evaluate every arm afterward with the same independent
deterministic mechanism assessment, regardless of which checks guided selection.
Retain no-latent, no-revision/refit-only and no-spec controls with their explicit
interpretations. No-spec is a prediction-only task contrast. Do not multiply all
factors into one large campaign before the individual comparisons are understood.

Keep both initial-prompt variants during development. Controller changes such as
a small beam are a later independent pilot; a lineage is a search path rooted
in a task/seed/arm, not necessarily a seed once branching exists.

## Milestone 6 — fresh fixed-protocol campaign and final freeze

Run fresh multi-round lineages from one pinned implementation once these choices
are settled. Keep the interrupted historical campaigns as developmental evidence;
do not combine their token costs with fresh-run efficiency estimates. Report
round curves, input/output tokens, model complexity, mechanism evidence and output
NMSE with failure denominators. Freeze the final method, controls and selection
policy before opening held-out test data.
