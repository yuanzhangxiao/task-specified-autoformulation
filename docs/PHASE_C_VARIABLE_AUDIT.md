# Phase C: variable-stage and handoff audit, 2026-09-30

The first Phase C pilot exposes both proposer errors and runtime interface
defects. It does **not** establish that the proposer became worse than in Phase B.
Do not tune a new search controller or change the model before repairing the
information and ownership problems below.

## Evidence and scope

The inspected `inspection.tar.gz` has SHA-256
`ac61e030a4655af4392e2a28e411b1553039fe70c0d2fecb7cff379f9f44a255`.
Its plan seal is
`315ca6edb7c6dac776f00fb450e8a9f0cc584e7a9c0cbc1fbea5e0c47fd92780`;
the pilot source is `21598e270dfb0cf464e179c6a4491d304c7a4d09`.
There are eight cases, two seeds and two prompt variants: 32 constructions.

This audit inspects saved public prompts, requests, responses, inventories,
runtime events and equations. It makes no provider calls, numerical fits,
rollouts, model edits or promotions. Public data embedded in the plan are not
evaluated. Test data and private reference equations are not accessed.
Provider explanations are evidence for human review, not executable rules or
automatic scientific scores. The human interpretations below are distinct from
the mechanical counts produced by the audit script.

## Why this is not a matched Phase B comparison

| Factor | Phase B final component campaign | Phase C construction pilot |
| --- | --- | --- |
| Search | 15 visits, incumbent retention | One fresh construction with local retries/fallback |
| Critic/pruning | Enabled in corresponding arms | Neither enabled |
| Cases | Nine-case suite | Eight cases, including two basin controls |
| Public data | Historical releases | Corrected input pulses and CSTR/alien preparation/data |
| Proposer | Pinned GPT-OSS-20B, low reasoning | Same revision and principal inference/budget settings |
| Fitter | `collocation-multi-target-v1` | Same frozen profile; new fitting improvements not adopted |

There is also a **Phase C adapter regression**. Phase B's
`search/fresh_shared.py::fresh` uses
`review_multi_construction.construction_brief` to display the enforced target
contract before variable selection. `fresh_shared.propose` also reuses up to
three public-contract repairs of executable, unfitted drafts. Phase C's
`research/construction_baseline.py::propose` calls `shared_construction.construct`
directly with the raw brief. It omits both wrappers, while retaining the final
admission check. Reusing the inner stages did not preserve the complete Phase B
construction behavior. This audit identifies that omission; it does not silently
alter the frozen experiment or imply that a repaired draft would fit well.

The pilot produced 27 constructed candidates and 25 complete finite fits;
five constructions failed, one fit failed, and one candidate was outside fitter
capability. All 340 provider calls returned responses. These data do not support
an outage explanation. They also do not separate numerical optimization error
from model inadequacy. A numerical comparison needs matched data, budgets and
stages; its absence prevents attributing a fraction of the NMSE gap to any cause.

## Variable definition: mechanical performance

| Quantity | Result |
| --- | ---: |
| Completed variable inventories | 32/32 |
| Called variable agenda items | 44 |
| Completed on first reply | 36/44 (81.8%) |
| Physical variable calls | 54 |
| Schema-valid replies | 54/54 |
| Runtime-accepted replies | 44 |
| Rejected/retried replies | 10 |
| Replies retaining partial progress | 5 |
| Empty replies rejected despite existing differential candidates | 5, across 3 constructions |

An agenda item is one mechanism/target-completion request, not an entire model.
There are eight constructions with variable retries. Partial acceptance is not
an extra call or a separate success. These counts are not scientific accuracy.

| Case (four constructions each) | First-reply agenda completion | Inventory review |
| --- | ---: | --- |
| Dalla T1-easy | 4/4 | All generate `Gp`; one adds gut memory, one elects to model supplied auxiliaries. Minimal inventories alone do not violate the narrow meal-path specification. |
| Dalla T1-hard | 4/4 | All use a `Gp` state. The task does not demand the full textbook state list. |
| Dalla T2-easy | 6/8 | All eventually include an insulin-memory candidate. Three define `U` dynamically, conflicting with the undisplayed final role policy. |
| Dalla T2-hard | 6/8 | Three declare plausible insulin-memory states. One returns meal absorption `G_abs` and instantaneous insulin-dependent `U`; the runtime assigns `G_abs` to insulin memory. |
| CSTR | 4/4 | All generate `T` dynamically; `C`, `Tj` and other declared channels can remain supplied. This inventory does not establish the correct heat balance. |
| Alien device | 3/4 | All eventually include latent `m`; `v01` is dynamic in three and algebraic in one. Public target role is unspecified. |
| Coupled basin | 5/8 | All contain `h_up` and `h_down`. Three runs receive unnecessary empty-reply rejections; one subsequently adds `mem_up`. |
| Independent basin | 4/4 | All contain `h_down`; one also adds a downstream volume state. Possible coordinate redundancy needs equation inspection. |

### The U role is an inferred policy, not an explicit public instruction

The public T2-easy prompt describes `U` as total disposal rate and says the
mechanisms' dimensionality and internal representation must be inferred. It
does **not** explicitly say that `U` must be algebraic/instantaneous. The legacy
machine-readable target contract nevertheless requires
`expected_representation: instantaneous_process`. The staged brief deliberately
does not import the older mass/rate keyword inference; the final gate still does.

Here, a *typed representation* means only an explicit mathematical role:
`differential` gives `dU/dt = f(...)`, whereas `algebraic` gives `U = h(...)`.
A quantity with rate units can itself be modeled dynamically; its units alone
do not prove that an algebraic representation is mandatory.

The three conflicts are `cell02_seed0_full`, `cell02_seed0_brief_only` and
`cell02_seed1_brief_only`. Their executable retained drafts reach the final gate,
where the target representation fails despite graph requirements passing.
Some also have earlier shared-route failures; the role conflict is not their
only unsuccessful attempt. None of their variable requests displays the Phase B
target-contract block.

Explicit public requirements must be disclosed and checked during construction.
An additional runtime interpretation must be justified and declared as policy,
or relaxed/advisory in a separately versioned experiment. Simply calling it
public, or silently inserting it into a finalized benchmark prompt, is not a fix.

### The runtime chooses a scientific assignment by list order

`staged_topology_runner._record_memory_candidates` takes the first accepted,
returned differential variable that is not a public target/current driver and
binds it to the current memory requirement. `VariableReply` has no explicit
mechanism-to-variable assignment field.

For `cell03_seed1_brief_only`, `variables_1`, attempt 0, request prefix
`f0d310e60ec5`, the reply describes `G_abs` as meal absorption and `U` in terms of
current insulin. The runtime nevertheless binds
`delayed_insulin_action -> G_abs`. Later topology retries require `I` to reach
this meal-absorption state: six `equation_G_abs` calls across the process and
fallback routes fail. Both sides contributed: the proposer did not explicitly
provide the requested insulin memory, and the runtime accepted a mismatched
scientific assignment. It should ask the proposer to declare/revise the binding.

### Reusing an existing state is not adequately represented

In `cell06_seed0_full`, `variables_1`, attempt 0 (`ebce4372dd29`), the displayed
inventory already contains differential `h_up` and `h_down`. The prompt permits
an empty list when existing inventory serves the agenda. The reply is empty,
but `_agenda_gaps` rejects it because no memory binding has been recorded.

There are five such replies across that run and both seed-1 coupled-basin runs.
In seed-1 Brief-only, two empty replies are rejected; the third returns existing
depths plus new `mem_up`. The runtime binds `h_up` anyway, leaving `mem_up` in
the inventory. Requiring a new state would be the wrong repair. Allow an explicit
proposer assignment to an existing state; the runtime can check existence and
definition kind without interpreting the prose.

## Who loses track of committed process uses?

The runtime preserves signed declarations. The later **proposer replies** conflict
with them, but the interface contributes: it displays required uses for the
current equation while listing other process names as allowed sources, and its
generic error offers no focused way to reconsider the earlier consumer choice.
Exact repetitions of a declared use are already tolerated and inserted once.

There are **22 distinct rejected calls**: 19 propose a use at an undeclared
consumer; three conflict with an existing use's sign/source group. Duplicate
copies in progress/result files are counted once. These counts are neither 22
failed models nor 22 proven false rejections.

Concrete examples:

1. `cell02_seed0_brief_only`, `equation_U`, attempt 0 (`64add9f71bd8`):
   `DelayedInsulinAction(I)` was declared to drive `Uid`, whose dynamic equation
   already contains that process. The new `U` reply uses the instantaneous process
   directly, bypassing `Uid`. A delayed-sounding name is not dynamic memory.
2. `cell02_seed1_brief_only`, `equation_Gp`, attempt 0 (`d4cc7593cb59`):
   `MealGlucoseInput` was declared positive in `Gp`; the reply changes its sign
   to unrestricted even though the required positive use is displayed.
3. `cell07_seed0_brief_only`, `equation_h_down`: the proposer subtracts
   `outlet_flow` directly, although it previously assigned that process only to
   the algebraic `outlet_down` equation. Using `outlet_down` may express its
   intention, but scientific equivalence requires checking its full gain and
   conversion. The runtime must not invent that equivalence.

The repair should show the full current model, fixed uses and exact conflict,
then offer an explicit proposer-owned coordinated revision where needed.
Unambiguous mechanical normalization is appropriate; changing scientific
consumers or signs by interpreting explanations is not.

## Equation review remains necessary

The variable stage only establishes potential capability. For example,
`cell00_seed1_full` yields a fitted `dGp/dt = k * meal_rate_g_per_min` model that
passes the tested meal-response requirement; that does not recover the textbook
glucose skeleton. `cell04_seed0_brief_only` achieves validation NMSE about
0.00973 but fails the controlled-balance boundary check. Conversely, a structurally
reasonable variable list can yield a poor fit. Preserve equations, independent
predicate results, fit availability and NMSE as separate evidence.

## Next bounded milestone

1. Reconcile public requirements and machine target-role policy, then expose
   every enforced rule consistently before variable selection. Preserve old runs.
2. Add explicit proposer-owned memory bindings, including reuse of existing states.
   Validate references/kinds mechanically. Do not infer assignments from prose.
3. Test these interfaces first with saved/synthetic replies. Then run a small
   variable-only confirmation on T2 and coupled basins, preserving first replies
   and all repairs. No fitting is needed to evaluate this boundary.
4. After variable handoff is sound, isolate process/topology/function stages with
   reviewed predecessors. Keep Orion's fixed-model fitting study separate.

This audit implements evidence collection only. It does not yet implement these
construction changes or run the confirmation experiment.

## Reproduce locally or on a CPU host

Using the repository Python environment, with the original inspection extracted
at `SNAPSHOT` and a separate report directory:

```bash
python scripts/audit_construction_stages.py --source "$SNAPSHOT" --output "$AUDIT"
```

`audit.json` contains per-call evidence and mechanical counts. `VARIABLES.html`
shows all 32 inventories and the 54 variable prompts/replies in stage/attempt
order, plus the 22 shared-use conflicts. Complete provider responses and schemas
are escaped for display. Request hashes, plan/proposal identity and saved call
accounting are checked. This is evidence indexing, not counterfactual replay;
missing or contradictory evidence raises an error rather than becoming a zero.
The input directory is never modified. Reports contain provider responses and
must remain outside Git.

Validation: seven focused audit tests pass; the complete `pytest -n 4 -q` run
passes 3,741 tests with eight environment-dependent skips. The CLI reproduces
the counts above on the supplied snapshot and verifies all 340 request records.
Tracked Python files pass Ruff. `ruff check .` also encounters 37 existing
violations in unrelated, untracked `analysis/claude` files; those are untouched.
