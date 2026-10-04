# Phase C construction schedules

`phase-c-construction-comparison-1` compares three ways to construct variables
and equation topology. It stops before interaction functions, initialization,
fitting, numerical mechanism assessments and the scientific critic. Historical
campaigns and finalized benchmark prompts/data are unchanged.

## Matched experiment

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
- Output: `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-construction-comparison-v1`.
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
the original report with small indexes linking to the authoritative call/event
JSON files. It leaves derived evidence for invalid tasks intact and stops for
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
