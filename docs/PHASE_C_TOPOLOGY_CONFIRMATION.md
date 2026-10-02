# Phase C topology from inspected inventories

`phase-c-topology-confirmation-1` continues the variable-stage inspection through
optional process declarations and equation topology. It stops before interaction
functions, initialization, fitting, scientific-critic calls or model selection.
It does not modify benchmark prompts/data or historical campaigns.

## Frozen inputs and scope

The roster remains eight development cases, two seeds and Full/Brief-only:
32 constructions, including coupled and independent detention basins.

- Import the 12 T2-easy/CSTR/alien-device inventories from
  `phase-c-variable-usage-v1`.
- Import the other 20 inventories from `phase-c-variables-v2`.
- Require a completed, sealed inventory for each selected task. Never replace
  a missing/failed selected inventory with a different source automatically.
- Verify the common original public plan, identical model settings and public
  contexts, source proposal identity, and cached variable-call accounting.
- Recheck variable and binding schemas without assigning scientific meanings or
  changing the saved decisions. Preserve source plan/proposal/call hashes.

The new plan contains public briefs, target contracts, contexts, descriptive
training summaries and the saved inventories/bindings. It contains no raw
trajectories, validation scores, private reference equations or independent
mechanism assessments. Full receives the original descriptive training summary;
Brief-only does not. The same GPT-OSS-20B settings and per-episode limits apply.
New topology costs are separate from historical variable costs; the two variable
sources had different prompts and review budgets. This is a staged diagnostic,
not an equal-budget ablation of the source variable policies.

## Existing construction behavior

The adapter reuses the existing signed-process proposer, per-equation topology
schemas, bounded local repair, public-polarity policy, deterministic lowering,
provider cache, token preflight and optional-process fallback.

1. Make one optional process call. Empty or invalid suggestions do not prevent
   ordinary topology construction. Each admitted process has a single defining
   interaction; the runtime inserts its declared signed uses at consumers.
2. Construct the remaining equation skeletons. The proposer chooses joint source
   sets, outer weight signs and scientific explanations.
3. Validate source names, generated left-hand sides, algebraic cycles, process
   uses, public source/composition paths and selected-memory obligations. Repairs
   retain the full public context, current inventory, bindings, selected variable
   meaning, accepted equation sketch and complete process declarations.
4. If admitted processes lead to a failed topology, reuse the original inventory
   in the existing ordinary fallback under the same new-call budget. Empty/invalid
   process suggestions do not earn an extra retry route.
5. A legitimate `inventory_revision` is reported as
   `inventory_revision_requested`. It does not silently change a saved variable,
   start another inventory review, or trigger fallback. The next decision can be
   made from the explicit requested addition/activation/type change.

`explicit-topology-inventory-context-1` adds the complete selected usage and memory
bindings to process and topology prompts. Supplied channels have no generated
LHS and may be RHS sources. Unused channels have neither. Natural-language roles
remain advisory; their apparent inconsistency is not a new semantic rejection
rule. Later initialization must still reconcile public preparation and observed
initial outputs. Neither an algebraic-output inventory nor a graph path settles
that question.

Historical default prompts are unchanged when this opt-in context is absent.
The new context participates in both the topology evidence contract and the
signed-process checkpoint identity. All calls, including failed routes, share
one cached client and remain charged on resume. Source changes require a new
plan; no cache deletion or implicit budget reset is supported.

## Inspection outputs

- `TOPOLOGY.html`: additive skeletons, exact term roles, selected sources,
  shared-process declarations, graph witnesses and links to each `TRACE.html`.
- `summary.json`: every outcome, failed/unfinished routes, inventory requests,
  declaration-versus-RHS usage, first replies, repairs, fallback and separate costs.
- `SUMMARY.md`: compact progress table.
- `results/<task>/TRACE.html`: complete requests, schemas, raw replies and runtime
  decisions, including repairs.

The readable skeleton uses unknown `phi` laws, nonnegative outer magnitudes `a`,
and unrestricted weights `b`. Shared process names denote one later law reused
at its consumers. Outer signs do not imply global sign or monotonicity of the
unknown intrinsic function. These are equation skeletons, not executable ODEs.

Graph inspection is recomputed from declared source sets. It reports actual
declared RHS occurrences and paths from public drivers through the specifically
bound memory states to targets, allowing intermediate nodes. Incomplete topology
and requirements without explicit graph endpoints stay unresolved. A path may
disappear through cancellation or an omitted dependency in later functions;
these observations are not the independent fitted mechanism-compliance score.
No NMSE or fitted scientific compliance can be reported at this stage.

Review should cover CSTR feed/reaction/jacket separation, T2 disposal composition
and delayed paths, basin storage/transfer/independence, and alien-device state and
initialization feasibility. These are human inspection questions from the public
tasks, not benchmark-specific runtime additions. Joint topology/inventory repair
and initialization remain outside this bounded continuation.

## ACES commands

Upload the commit-pinned source bundle to group scratch, extract it in its own
directory and verify `SHA256SUMS`. From that directory:

```bash
bash scripts/hpc/start_phase_c_topology.sh run
```

Defaults are:

- Earlier inventories: `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-variables-v2`.
- Targeted inventories: `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-variable-usage-v1`.
- Output: `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-topology-v1`.

Override with `AF_TOPOLOGY_BASE_SOURCE`, `AF_TOPOLOGY_USAGE_SOURCE` and
`AF_TOPOLOGY_ROOT` if necessary. The script submits one H100 proposer and one
dependent CPU report job. It does not submit fitting jobs. Preparation runs once;
later invocations verify the frozen plan without needing the original source
directories. Existing submission receipts prevent duplicate jobs.

After the jobs finish:

```bash
bash scripts/hpc/start_phase_c_topology.sh inspect
```

Download the printed `inspection.tar.gz`. It includes plan, reports, provider
traces, stage checkpoints, logs and submission receipts. If an allocation drains
with pending work, inspect it first, then explicitly submit a new
`AF_TOPOLOGY_WAVE=topology-2` using the same output root and source bundle. Terminal
failures and revision requests remain visible; they are not retried by relaunch.

For local public-only preparation without a scheduler:

```bash
PYTHONPATH=src python scripts/phase_c_topology.py prepare \
  --base-source /path/to/earlier-variable-results \
  --usage-source /path/to/targeted-variable-results --root /path/to/new-root
PYTHONPATH=src python scripts/phase_c_topology.py verify --root /path/to/new-root
PYTHONPATH=src python scripts/phase_c_topology.py report --root /path/to/new-root
```
