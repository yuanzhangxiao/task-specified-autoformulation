# Phase C construction schedules

Version 7 adds exact consumer accounting, covariate-aware overlap clarification,
and concrete cycle/binding repair feedback. The current protocol is
`phase-c-construction-comparison-7`. The next confirmation is **six fresh
constructions**: coupled and independent basins, each under separate, joint-fixed
and joint-adaptive schedules, Full/seed 0. This targets the four affected basin
drafts and retains the two adaptive basin constructions as matched comparisons.
No other benchmark is rerun. Model settings and the three-request overall repair
allowance are unchanged. No functions, fitting, critic or test access is launched.

## Version 7: consumers, overlaps and feedback

Every declared signed use must appear exactly once in its generated equation.
Uses aimed at runtime-generated process definitions are diagnosed explicitly;
they are never silently discarded or counted as shared consumers. A named law
can depend on another named law through `depends_on`, but extra signed uses cannot
be inserted into its automatic definition. The proposer must choose the intended
representation. Forward consumer references remain permitted in provisional
transactions; assembly/whole-draft assessment waits for their equations.
Reports separate declared consumer counts from assembled uses (including their
sign and conversion metadata). Only verified assembly is labeled local/shared;
incomplete or inconsistent accounting is `unverified`. A local law remains legal.

Overlap clarification retains the existing exact-source comparison and also
detects equal, nonempty **non-covariate driver sets** when the only differences
are public fixed covariates. Inputs, observations, states and time are never
discarded as covariates. The prompt shows both source sets and process conversions.
This asks whether the contributions are distinct; it does not infer a duplicate
physical effect or delete one automatically. A distinction confirmation is bound
to the exact evidence, including the covariate comparison.

Algebraic-loop feedback includes a concrete cycle and its assembled equations.
Memory-path feedback names the actual `mechanism_bindings` entry, target ancestors
and appropriate edit options; it does not assign a state or substitute
`feedback_bindings`. Optional claims may be explicitly withdrawn; mandatory
assignments still require a valid replacement. Accepted-edit feedback now includes
variable changes. A repeated process-definition equation is normalized only if
its complete term declaration exactly matches the runtime-generated definition;
conflicts and different definitions still require a proposer edit. Raw replies
and the normalization log are retained.

The live-v5 offline audit covers 99 actual-predecessor replies: all 92 historically
accepted transactions remain accepted, and one exact repeated definition is newly
accepted. Of 24 saved final drafts, 22 were historically eligible and 20 remain
eligible under these checks. The two intended changes are independent/joint-fixed
(self-consumer) and coupled/joint-fixed (covariate-only overlap). This does not
recover or promote any model and cannot establish live scientific improvement.

## Basin confirmation on ACES

Use a fresh pinned source bundle with `inputs/topology-source/plan.json`. Upload
the archive to the group scratch directory, verify its `SHA256SUMS`, then run:

```bash
bash scripts/hpc/start_phase_c_construction_basin.sh run
```

The wrapper defaults to
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-construction-basins-v1`,
wave `basins-1`. It uses the existing one-H100 service, a three-hour worker window,
quota write probes, logged/cached calls and submission receipts. Repeating `run`
does not submit a duplicate wave. If a new allocation is needed for unfinished
work, use the same source/root and a new `AF_COMPARISON_WAVE`; completed tasks
remain cached. It does not continue automatically.

After jobs finish, from that same source directory:

```bash
bash scripts/hpc/start_phase_c_construction_basin.sh inspect
```

Read `SUMMARY.md`, `TOPOLOGY.html` and the linked per-task request/response traces.
Download `phase-c-construction-basins-v1/inspection.tar.gz` for inspection.
Structural completion remains separate from scientific adequacy. Basin threshold
laws, physical conversion choices, and successful fitting remain untested here.
The general 24-task and 96-task options remain available with new defaults
`phase-c-construction-live-v6` and `phase-c-construction-comparison-v7`.
Historical campaigns still require their original pinned source.

Historical implementation changes follow for provenance.

Version 6 clarifies the binding and named-process handoff. The protocol is
`phase-c-construction-comparison-6`; fresh defaults are
`phase-c-construction-comparison-v6` and `phase-c-construction-live-v5`.
The confirmation stays at eight cases x three schedules, Full/seed 0 (24 tasks),
with the same model, generation settings and three-request overall repair budget.
It stops before interaction functions, initialization or fitting. This is a
bookkeeping confirmation, not a strategy ranking or a scientific success rate.

## Binding and consumer handoff

Every request now includes `binding_context`: concrete field examples, exact
public requirement IDs, type-eligible state choices (without a preselection),
applicable feedback targets, and exact removal edits for existing bindings.
`mechanism_bindings` assigns differential mediators to public requirements;
`feedback_bindings` assigns physical coordinates to an algebraic readout only
when a reviewed target-feedback predicate requires that assignment. These fields
are not interchangeable. Required mediator assignments remain proposer decisions.
An annotation for an exact public target outside the applicable feedback set is
removed and logged, including a stale saved entry. It creates no mechanism
assignment and does not waive any actual pathway or target-feedback requirement.
Unknown target names remain errors. Empty edit lists preserve prior entries;
removal feedback explicitly names the target/requirement key to delete.

A singleton equation source `P` with an explicit positive/negative outer sign
means use of the existing named law P. The runtime records an unambiguous new
consumer and removes the ordinary reference so assembly includes it exactly once.
A new conversion remains `null` (unknown); a matching existing use retains its
known conversion. Process drivers, scientific kind and existing consumer signs
are unchanged. This applies only to equations explicitly supplied in the reply
and declared generated consumers. Multiple references, composite sources,
unrestricted signs, conflicts with an existing sign, or a use simultaneously
withdrawn by a process replacement require an explicit proposer decision. The
feedback shows the conflicting term and current consumer declaration, and offers
coordinated edits rather than simply ordering the physical contribution removed.
The underlying process validator still checks distinct consumers and transfer
signs. No conservation claim is inferred from assembly syntax.

All edits remain atomic. Normalizations, original replies and before/after drafts
are retained. Subsequent accepted-edit feedback includes equation/process/binding
changes; the full rebuilt model remains in every request and the final report.
Reports count normalizations by code, separately from structural completion.
No benchmark-specific sign, threshold or water-balance rule is added.

## Saved-response regression check

`python scripts/audit_construction_handoff.py --source OLD_ROOT --output AUDIT.json`
is a read-only audit of version-5/6 recorded responses. It verifies sealed events
and calls, then applies each reply to its actual historical predecessor. It never
chains hypothetical corrected drafts, regenerates responses, promotes a model or
opens trajectory data. The output must be outside the historical campaign.

On the inspected live-v4 package, all 133 attempts were audited: all 119 historically
accepted transactions remain accepted. There are six explicit new-consumer
normalizations, eleven matching identity repetitions, nine inapplicable incoming
readout bindings and eleven stale binding removals. Counts may overlap and refer
to saved replies, not recovered models. In particular, missing alien-device memory
assignments still require a fresh proposer answer. This audit cannot establish that
the revised prompts will improve live scientific construction.

Historical implementation changes follow for provenance.

Version 5 narrows local-feedback witnesses and clarifies ambiguous contributions.
The protocol is `phase-c-construction-comparison-5`; fresh defaults are
`phase-c-construction-comparison-v5` and `phase-c-construction-live-v4`.
The next live confirmation retains eight cases x three policies, Full/seed 0,
24 fresh constructions, and the same three-request overall-repair allowance.
It stops before functions, fitting or a scientific critic. Old campaigns require
their pinned source; no finalized benchmark prompt/data is changed.

## Narrow topology feedback milestone

The public basin task explicitly requires downstream discharge/local water balance.
The version-4 predicate was too weak: an upstream feedback cycle with a path to
h_down could pass despite missing downstream feedback. Version 5 uses
`target_feedback`: a differential target must itself lie on a nonempty cycle.
For an algebraic target, the proposer declares `feedback_bindings` with its actual
storage/energy coordinates. Each coordinate must be differential, have feedback,
and reach the target through algebraic readouts only. This permits alternative
coordinates and coupled feedback, without guessing scientific roles from names.
The CSTR thermal balance uses the same generic predicate. These are necessary
topological conditions, not evidence of restoring signs or a correct release law.

Memory bindings and feedback coordinates have separate namespaces. Every request
lists all allowed public memory-requirement IDs, including optional ones. If a
memory entry names an exact automatic graph-check ID that is not also a genuine
public requirement, the runtime drops it and logs the original entry. A stale such
entry is explicitly removed during a later transaction. It never maps that entry
to another requirement or chooses memory states. Unknown unrelated IDs still
receive an error with an exact removal action. Raw responses remain immutable.

Per-LHS views display inserted process uses before ordinary RHS declaration.
`ordinary_rhs_declared=false` means additional terms have not been specified; it
no longer visually implies an empty equation. The proposer can acknowledge a
complete process-supplied RHS with `terms=[]`.

An ordinary term with the same drivers as a named process used in that equation
raises a **clarification**, not a scientific duplication verdict. During the same
bounded overall repair the proposer either removes the repetition by explicit
edits, removes/restructures the process and its consumers, or confirms distinct
scientific effects. Confirmation uses a runtime-issued hash of the exact term,
process and variable declarations plus a scientific distinction. Changed declarations
invalidate it. The runtime does not interpret the explanation. Unanswered questions
remain unresolved and prevent a topology-complete label; they are reported separately
from structural errors. Explicit references to the process itself still assemble
once and do not trigger the expanded-driver question. This detector does not find
all possible physical duplications or general symbolic equivalences.

Same-sign transfer feedback offers three choices: correct the signs for a real
internal transfer, reclassify as influence while preserving intended signs, or
replace the named process with ordinary contributions. Runtime never flips signs
or changes scientific type automatically. Accepted edits are labeled separately
from rejected replies in subsequent repair context.

The offline audit of the 24 saved version-4 drafts preserves all historical
records: 23 were historically eligible; the new checks leave 15 ready as written.
One coupled-basin draft lacks target-local feedback; two algebraic basin readouts
need explicit coordinate bindings that the old schema could not provide. Eight
ordinary/process overlaps in five other drafts require proposer clarification.
The remaining historical failure is the invalid binding namespace. These are
requirements for fresh responses, not eight newly disproved scientific models or
a ranking of construction schedules. No LLM calls, fits or test data were used.

Historical implementation changes follow for provenance.

## Feedback corrections after the live confirmation

Version 3 retains the scientific acceptance rules and budgets, with one narrow
delivery normalization: an exact string `"null"` in a process-use conversion is
converted to JSON `null`. It is not normalized if `null` names an available fixed
covariate. Arbitrary expressions, fitted coefficients, other spellings and
scientific assignments are not repaired by this rule. Original provider records
remain unchanged; each transaction logs the normalization, including when another
error subsequently rejects the transaction. Cached replay reproduces that log.

Protected public inputs, covariates and time receive feedback in the actual edit
vocabulary: omit their generated declarations/equations and keep intended RHS
references. They are already available. Supplied observations remain optionally
modelable. A conflicting replace/remove transaction names the entries and explains
the two valid alternatives; it never chooses a scientific intent or partially
commits the reply.

Public and memory pathway checks now require a successfully compiled whole graph.
Compilation failure reports `graph_check_status=unavailable`; it does not create
missing-path verdicts from an empty or partial graph. Known declaration/type/ID
errors are still checked. Once compilation succeeds, the existing pathway checks
run before functions, including indirect feedback paths when explicitly required.
Unresolved public predicates are exposed alongside the status in the report and
repair context. For example, the current CSTR `controlled_balance` requirement
has no declared drivers: its scientific prose does not supply an executable
temperature-feedback predicate. A structural pass does not certify that balance.
This change does not alter benchmark contracts or add a universal decay condition.

A named law with one distinct declared consumer is **local**. A **shared** law has
at least two distinct consumers. Reports expose consumer names/counts and this
classification separately from validity; a repeated consumer does not count twice.
The prompt prefers ordinary terms for local effects and never asks for invented
consumers. Explicit local laws remain valid representations. Internal pairwise
transfers still require two opposite-signed consumers; shared influences need not
have opposite signs. Naming and assembly syntax do not certify conservation.

An offline audit of all 114 replies from the first 24-task live confirmation,
against each reply's actual historical predecessor, retained all 100 accepted
replies and newly admitted four quoted-null replies. Ten remained rejected.
Five final drafts now report unavailable graphs; final eligibility is unchanged.
These are delivery/feedback findings, not recovered models or fresh live evidence.
The audit used no LLM calls, optimization or test data.

Version 3 refuses version-1/version-2 plans. Preserve their original source and
reports. New defaults are `phase-c-construction-comparison-v3` and
`phase-c-construction-live-v2`. No fresh campaign is automatically submitted.

## Earlier contract corrections after the first inspection

Version 1 is retained as diagnostic evidence, not a valid ranking of schedules.
All 19 finished separate constructions exhausted relationship planning before
reaching their scheduled equation stage. The scope check rejected any nonempty
variable list, including exact repetition of the committed inventory. Version 2
checks changes in names/types instead; unchanged declarations and rephrased roles
are allowed. Fixed equation scheduling similarly tolerates exact repeated earlier
equations. Genuine changes outside the selected scope still require repair.

Failed delivery of optional relationships now records the failure and proceeds
to ordinary equation construction with the accepted draft. It does not erase
accepted processes or waive final checks. The initial checkpoint records the
outcome of every attempted stage, including this continuation. This restores the
principle that absent/invalid optional process suggestions must not by themselves
prevent construction of ordinary equations.

Memory bindings recognize every existing public requirement. Mandatory delayed
memory still needs a distinct differential mediator on the required path.
An optional binding can name a generated differential target itself; its type
and available graph endpoints are checked. Missing public endpoints remain
unresolved, and genuinely unknown requirement IDs remain errors. Optional memory
is not silently converted into a mandatory benchmark mechanism.

The prompt distinguishes an accumulator, `dX/dt=u`, from fading memory such as
`dX/dt=a*u-b*X`. It asks the proposer to consider restoring dynamics, possibly
through coupled states, when the public task calls for return toward baseline.
There is no universal self-dependence gate or automatically inserted decay term.
Both forms can be scientifically appropriate. Merely including X also would not
prove stable relaxation.

Every request shows a complete empty edit template, including all lists and the
completion flag. Output-limit failures receive delivery feedback describing the
trailing-whitespace count and reminding the proposer to finish the JSON object.
Incomplete responses are still rejected, never repaired by inventing missing
fields. This addresses the observed delivery pattern; a live serving probe is
still needed to establish whether the new prompt prevents whitespace loops.

The prompt and targeted diagnostics distinguish generated variables from later
function parameters, and known covariate conversions from unknown fitted gains.
They do not infer scientific intent from names or prose. Actual empty RHSs remain
invalid; an explicit constant term has an empty source list, while an equation
with only shared uses can have an empty ordinary-term list.

New comparison workers and reports write one small linked trace per task. They
retain all original call/event JSON and do not duplicate them into `trace.json`.
A quota/space error writing this derived view cannot mask a saved proposal or
the original checkpoint error; report rows expose an unavailable trace. This does
not create disk space or make failed primary checkpoint writes safe to ignore.

Version 2 refused version-1 plans and used
`phase-c-construction-comparison-v2`. Confirm the fixes before a full new
comparison. The earlier 32-task topology continuation imported already inspected
inventories and used a different response/repair protocol; its completion rate
is not an equal-budget baseline for these fresh constructions. Retain the
established method as a reference for a later matched scientific comparison;
none of the three alternatives is assumed to outperform it.

## Matched experiment

### Bounded live confirmation

`prepare --study live_confirmation` freezes a separate 24-task plan: all eight
cases, seed 0, Full only, and all three construction policies. Both basin cases
are included. Within this protocol, the live confirmation and full comparison
share public inputs, reviewed graph contracts, model revision, serving image,
generation settings and construction/repair budgets. Only the roster and worker
window differ. No old inventories or models are imported.

The worker visits every case within its first eight tasks, rotating policies
within each case. It stops after the frozen 24 tasks or the three-hour window,
whichever comes first. Pending tasks retain their caches for explicit resume;
there is no automatic extension to the 96-task comparison. A 3.5-hour 1xH100
allocation accommodates server startup/draining, followed by a CPU report job.

The report labels this study as a delivery/bookkeeping confirmation, not a
strategy ranking. Inspect `all_tasks_terminal`, each separate construction's
`equation_stage_reached`, initial/final errors, and `delivery` counters. The latter
count output-limit responses and the subset with at least 97% trailing whitespace
(a diagnostic threshold matching the observed failure pattern). Zero observed
delivery failures with pending work does not establish completion. Inspect actual
skeletons and responses before considering interaction generation or larger runs.

Submission and worker startup test a real 128 MiB write in the output directory,
flush it, and remove only that temporary probe. This catches an already exhausted
quota; it neither reserves space nor guarantees later writes. A failed probe
prevents submission or new provider calls. Historical results are never deleted.

From a pinned code bundle on ACES:

```bash
bash scripts/hpc/start_phase_c_construction_live.sh run
# After the jobs finish:
bash scripts/hpc/start_phase_c_construction_live.sh inspect
```

The default output is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-construction-live-v4`.
The wrapper fixes the study to `live_confirmation` and uses wave `live-1`;
repeating the command returns existing submission receipts. If the allocation
finishes with pending tasks, use the same code/root with an explicit new wave:

```bash
AF_COMPARISON_WAVE=live-2 bash scripts/hpc/start_phase_c_construction_live.sh run
```

### Full comparison

The experiment contains 96 fresh constructions: three policies × eight cases ×
two seeds × Full/Brief-only. Cases are Dalla Man T1-easy/hard and T2-easy/hard,
CSTR, alien device, and coupled/independent detention basins. Both basin cases
are retained. Policies rotate within each matched case/seed/prompt block so a
draining allocation does not systematically omit the same policy.

| Policy | Construction schedule |
| --- | --- |
| `separate` | Declare generated variables (several may be returned together); plan relationships; define ordinary equation skeletons one LHS at a time. |
| `joint_fixed` | Plan relationships and preliminary variables; define one LHS skeleton at a time, allowing accompanying variable declarations. |
| `joint_adaptive` | Plan relationships and preliminary variables; let the proposer choose the order and grouping of variable, equation and process edits. |

Fixed schedules visit public targets first, then their unresolved dependencies,
then other declared generated variables. Adaptive construction receives the same
pending-work record but selects its own work units. The adaptive policy changes
order and grouping together; this comparison cannot isolate their individual
effects. It evaluates these construction policies for the pinned proposer, not
a universally optimal ordering.

Inputs come from the sealed `phase-c-topology-v1/plan.json`: public briefs,
corrected public target contracts, validation contexts, descriptive training
packets and existing model/serving settings. No old variable inventories,
topologies, results, validation scores or evaluator rules enter the new plan.
All arms start from an empty generated-variable draft. Full receives the saved
training packet; Brief-only does not. The source plan is needed only to prepare
the new campaign, not to resume it.

## Common source and editing contract

The public catalog controls information availability. Non-target public sources
are available for RHS use without a `supplied`/`unused` decision. Variables in a
proposer reply specify generated differential or algebraic quantities only.
Public targets must be generated. If a supplied observation is elected as a
generated quantity, its equation governs RHS use; the existing compiler aliases
it to prevent substitution of its measured future trajectory. No new observed
channel can be invented. Initial readings and diagnostic covariates retain the
meaning stated in the public task; the runtime does not infer scientific roles
from names or explanations.

Every response uses the same structured edit format:

- `variables`: replace named generated declarations, including their type/role.
- `equations`: replace each named LHS's **complete ordinary RHS**, not append.
- `processes`: replace named shared-law declarations, drivers and signed uses.
- `mechanism_bindings`: replace explicit memory assignments for named requirements.
- `remove_variables`, `remove_equations`, `remove_processes`, `remove_bindings`:
  remove only explicitly named existing entries.
- `stage_complete`: finish the current stage. During equation construction it
  means the entire topology draft is ready, not just the selected equation.

Omitted entries survive. Unambiguous batches are applied atomically. Each shared
process has one runtime-generated definition; its current uses are inserted at
all consumers. Ordinary equation replies omit these already-included uses.
An explicit process edit propagates to every consumer in the next displayed
skeleton. The runtime does not infer additional scientific edits from prose.
There is no six-equation/two-variable response restriction. Existing public
whole-model limits and serving/output budgets still apply equally to all arms.

Each prompt contains the unchanged public task, public source catalog, required
target types, memory obligations, current declarations, assembled contributions,
already-included process uses and pending work. Failed replies and diagnostics
are labeled separately from the accepted draft. No hidden conversation history
is required for the proposer to recover the current model.

Unknown RHS names and not-yet-defined equations remain pending. Immediate checks
cover response syntax/schema, conflicting edit operations, protected public data
roles, reserved names, conversion-expression safety and stage editing scope.
Pathway completeness and algebraic cycles are checked on the finished draft.
Differential feedback is permitted. A process can remain provisionally incomplete
while other equations are constructed; invalid scientific prose is not a gate.

## Initial draft and bounded repair

The initial construction ends when its schedule finishes, the proposer requests
completion, local response repair exhausts its allowance, or the initial budget
is exhausted. Its exact draft, cost and structural assessment are saved BEFORE
overall repair. Local response/schema retries are separately recorded and are
not counted as an untouched first response.

Final structural checks reuse the ordinary compiler and public predicates:
generated targets, source/equation closure, unsupported algebraic cycles,
declared process consumers and transfer signs, explicit public fixed signs,
composition paths and explicitly bound driver–memory–target paths. Requirements
without an explicit graph predicate are reported as unresolved, not certified.
Missing self-relaxation, apparently duplicated mechanisms, physical units and
scientific explanations are not silently converted into new hard predicates.

All policies then receive the same repair allowance: up to three physical replies
and 131,072 charged tokens. Repair can coordinate variable, equation, process and
binding edits across the whole draft. Every subsequent request shows the rebuilt
model. Success requires structural eligibility and completion; otherwise the
outcome remains `topology_incomplete`. An empty process list is valid. No automatic
second construction without processes adds an unequal hidden budget.

The inherited total ceiling is 128 requests / 524,288 charged tokens, partitioned
into at most 125 initial requests / 393,216 tokens and the repair allowance above.
Initial and repair work share the same cache and accounting. Unspent initial
budget does not buy extra repair requests. As in the existing client, an unknown
usage event keeps its conservative reservation; observed tokens can be incomplete.
Inference remains pinned GPT-OSS-20B, low reasoning, temperature 0.2, maximum 8192
output tokens and 32768 served context. Full context is checked before calls;
it is never silently truncated to make a request fit.

## Evidence and interpretation

- `SUMMARY.md` and `summary.json`: before/after structural completion, pending
  tasks, actual calls, observed/unknown costs and median [unscaled MAD] by policy
  and prompt variant. Cost medians cover finished tasks only and say so.
- `TOPOLOGY.html`: before/after equation skeletons, declarations, failed checks,
  policy identity and links to every exact prompt/response trace.
- `results/<task>/construction/before_repair.json`: immutable initial draft.
- `results/<task>/construction/events/*.json`: every batch, its exact predecessor,
  resulting draft, selected LHS, acceptance/error and pending references.
- `results/<task>/TRACE.html`: exact requests, schemas, raw responses and decisions.

In version 2, the trace is an index linking those original JSON records rather
than embedding duplicate copies. The legacy expanded renderer remains available
for other protocols.

Inspect the actual equations and compare public scientific adequacy independently
of acceptance. In particular, inspect required memory, target composition,
source/sink interpretation, shared balance structure, unnecessary variables and
duplicate mechanisms. No NMSE or independent fitted mechanism score exists at
this stage. Raw events support analysis of batch sizes, ordering, repeated edits
and unresolved-reference lifetime; these are not scientific correctness scores.

Checkpoint resume reconstructs the same state from cached calls. Each transaction
is sealed with its response digest and predecessor/successor drafts. Missing or
changed call evidence fails closed. An interrupted unknown HTTP outcome is charged
and not silently resent under the same request key. Terminal outcomes are not
rerun when another allocation resumes pending tasks.

## ACES launch and inspection

Upload the commit-pinned bundle to
`/scratch/group/p.nairr260351.000/u.yx126462`, extract it, and verify `SHA256SUMS`.
From its extracted directory run:

```bash
bash scripts/hpc/start_phase_c_construction_comparison.sh run
```

Defaults:

- Source: `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-topology-v1`.
- Output: `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-construction-comparison-v6`.
- One existing 1×H100 server, followed by a CPU report job; no fitting jobs.

Override these with `AF_COMPARISON_SOURCE` and `AF_COMPARISON_ROOT` if necessary.
Existing wave receipts prevent duplicate submission. The six-hour worker window
does not guarantee all 96 constructions finish in one allocation.

After completion:

```bash
bash scripts/hpc/start_phase_c_construction_comparison.sh inspect
```

Download the printed `inspection.tar.gz`. It includes the plan, reports, all
provider responses, transaction records, runtime logs and scheduler receipts.
If the report still has pending tasks after the allocation has ended, resume the
same root and pinned source with a new wave name:

```bash
AF_COMPARISON_WAVE=comparison-2 bash scripts/hpc/start_phase_c_construction_comparison.sh run
```

There is no automatic follow-up or test-data access. This launcher uses ACES;
switching model, provider or serving platform would require a separate matched
experiment rather than mixing backends inside this comparison.

## Recovering a quota-failed report

An `EDQUOT` while writing `TRACE.html` is a storage failure, not a scientific
rejection. The original renderer duplicates requests, responses and transaction
snapshots into both HTML and JSON views. A failure in that renderer can also
interrupt the worker after its proposal was already saved. A failed write of
`runtime/finished-<job>.txt` alone does not establish construction completion.

`scripts/recover_phase_c_storage.py` is a standalone operational tool. Run it with
the original pinned source directory and the existing campaign, after its jobs
have stopped. It verifies that source against the frozen plan, then validates
saved calls, transaction chains and proposal accounting. It never changes the
source, plan, prompts, model decisions, caches or budgets, and makes no LLM calls.
It refuses an existing active construction lock.

`inspect` prints saved completion/partial/unstarted counts, checkpoint errors and
the space occupied by reproducible views. `recover` rechecks the records, deletes
only `results/<task>/TRACE.html` and `trace.json` for valid tasks, and regenerates
the original report with one small HTML index per task linking to authoritative
call/event JSON files. It does not recreate `trace.json`, reducing both bytes and
file count. It leaves derived evidence for invalid tasks intact and stops for
inspection instead of discarding or repairing their checkpoints. The scoped
renderer substitution affects presentation only; original source verification,
namespaces, report calculations and accounting remain enforced.

Upload just the recovery script to group scratch; no new source checkout is
needed. If quota prevents even that small upload, removing only generated HTML
views frees some space while keeping `trace.json` and all original records:

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-construction-comparison-v1
find "$AF_ROOT/results" -mindepth 2 -maxdepth 2 -type f -name TRACE.html -delete
```

Then, after uploading `recover_phase_c_storage.py`:

```bash
(
  set -euo pipefail
  module load GCCcore/13.2.0 Python/3.11.5
  export PYTHONDONTWRITEBYTECODE=1
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python \
    "$AF_GROUP/recover_phase_c_storage.py" recover \
    --repo "$AF_GROUP/phase-c-construction-111ab14" \
    --root "$AF_GROUP/phase-c-construction-comparison-v1" \
    --archive "$AF_GROUP/phase-c-construction-comparison-v1/inspection-compact.tar.gz"
)
```

The optional archive retains original requests/responses, transactions, temporary
checkpoints, logs, receipts and compact reports. It excludes lock files; an
interrupted archive write removes only its own temporary archive and preserves
any previous finished archive. The views and archive still require some space.
This reclaims duplicated reports, not unrelated runtime caches or allocation
storage. Use the recovered counts and worker logs to decide what needs resuming;
the original `inspect`/worker renderer would recreate the large views. This tool
does not submit jobs or extend an interrupted construction budget.

Local public-plan smoke (no LLM calls):

```bash
PYTHONPATH=src python scripts/phase_c_construction_comparison.py prepare \
  --source /path/to/topology-campaign --root /path/to/new-comparison
PYTHONPATH=src python scripts/phase_c_construction_comparison.py verify \
  --root /path/to/new-comparison
PYTHONPATH=src python scripts/phase_c_construction_comparison.py report \
  --root /path/to/new-comparison
```
