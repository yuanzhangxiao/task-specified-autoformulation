# Phase C: foundation, scope and handoff

This is the short entry point for the next research phase. It does not replace
AGENTS.md, change a frozen protocol, or announce a new production algorithm.
Read this before the chronological campaign history in PROJECT_CONTEXT.md and
PIPELINE_DESIGN.md. Implementation still follows their relevant contracts.

## Objective and ownership

Build reliable scientific model construction before expanding search. Separate
reference/data defects, insufficient specification, proposer capability,
interface rejection, numerical fitting limits and selection effects. Good
prediction need not imply a unique textbook realization; textbook recovery is
an appropriate claim only under adequate specification and identifiability.

- Orion owns benchmark correction, input semantics and reference qualification.
  Do not edit his benchmark/data/loader files concurrently or mix corrected
  releases into historical campaign directories.
- Codex owns Phase B artifact/scheduler reconciliation, operations tooling,
  stage-contract diagnostics and integration of separately reviewed changes.
- Scientific interpretation and experiment selection stay with the researchers
  and supervisor. Bounded workers extract evidence or implement isolated fixes
  under [work orders](BOUNDED_RESEARCH_WORK_ORDERS.md).

The first maintenance milestone delivered observation and a handoff. The
subsequent explicit recovery commands reuse the original construction path.
No automatic cleanup, test evaluation or resubmission is enabled.

## Phase B remains a separate historical experiment

### Latest recovery checkpoint: 2026-09-28

The snapshot captured at `2026-09-28T20:55:53Z` has audit seal
`0c0606878336236360c822110b341d49d2e688d9886245e4210d9748e93b91f2`.
Its summary and scheduler reconciliation reproduce locally. Prior round-result
hashes and completed/skipped endpoints are unchanged. The queue observation is
empty, not a live guarantee. The bundle's focused recovery manifest still names
wave 1; the complete audit/scheduler roster identifies wave 2 correctly.

There are 1,884 complete round rows (+122 since the prior snapshot), 14
construction failures, 6 fit failures, 24 interruptions and 472 missing rows.
The 72 critic-disabled lineages have 71 complete endpoints and one skipped
without a finite parent. The 88 critic-enabled lineages end at zero-based round
8 (32 lineages) or 9 (56), each awaiting its next critic operation. Thus 120 of
192 endpoint rows remain missing, including secondary pruning comparisons.
Test data remain unopened. Interim comparisons can use the universally recorded
round 8, before final pruning, but must retain failures and distinguish equal
visit counts from equal compute costs or completed final ablations.

Wave 2 completed 123 proposer, critic and fit operations each. The single critic
ran 5:01:16 without the former lock failure. Two proposers performed all 123
operations; two others started after the CPU services ended and each spent
5:58:00 with zero operations. Reduce the next proposal allocation to two workers;
more GPUs alone will not remove the current critic bottleneck.

Three fit allocations (`2166855_13`, `_14`, `_15`) failed after two seconds,
before worker-start receipts. Supplied stderr identifies an import failure:
`ModuleNotFoundError: No module named 'autoformalism.config'` in the original
checkout. Thirteen other fit workers completed. Pruning allocation `2166856_0`
failed after two seconds with exit 120:0 and no worker receipt. Follow-up evidence
places all four failures on `ac042`; the 13 successful fit allocations used other
nodes. The login-side `config.py` SHA-256 matches the original commit exactly
(`7f4bf1c5dba7a0687bddd57e26a6888fce150ba20db072bb607cff58937cf572`).
Both pruning logs are empty. This supports avoiding `ac042` for the next wave,
not claiming a proven filesystem root cause or replacing source/dependencies.
Use a distinct wave-3 submission with two proposers and one critic; preserve
all consumed attempts. These failures do not establish an API problem.

The additional interrupted round belongs to `cell07_seed0_full_c1v1s1`, now
with nine interruptions and no finite incumbent; its diagnostic hash is unchanged.
`cell07_seed0_brief_only_c0v1s1` remains the skipped no-incumbent endpoint.
Keep any corrected-runtime recovery separate from these historical failures.

Actual quota headroom is 2,267 personal-scratch files and 45,682 allocation-wide
group-scratch files. Group file usage increased by 43,019, but this observation
does not attribute that increase to a particular workload. Review generated
cache paths and live dependencies before any cleanup; no deletion is authorized
by directory names or this checkpoint alone.

### Earlier checkpoint and diagnosed concurrency failure

The reviewed portable audit is identified by
`93dc4d898f83f3bff23bff40cb75a3ecdaad0e37d25b9a308e114edad2ffd711`.
Its ACES source is
`/scratch/group/p.nairr260351.000/u.yx126462/final-components-v1`.
It records 160 lineages and 2,400 visits: 1,619 complete, 14 construction failures,
5 fit failures, 21 interrupted workers and 741 missing visits. All 192 endpoints
are missing. The first unresolved stages are critic for 88 lineages (round 6
or 7, zero-based) and pruning for 72 lineages (after round 14).

The supplied scheduler snapshot records these array tasks, excluding steps:

| Submission | Stage | Completed | Failed |
|---|---|---:|---:|
| 2161347 | propose | 8 | 0 |
| 2161348 | critic | 4 | 0 |
| 2161349 | fit | 32 | 0 |
| 2161350 | prune | 8 | 0 |
| 2162100 | recovery critic | 1 | 3 |
| 2162104 | recovery fit | 32 | 0 |

Recovery critic tasks `2162100_0`, `_1`, `_2` exited `1:0` after 3:00, 1:25 and
1:33. The subsequent stderr inspection established the immediate cause in all
three: `component_critic.review_one` could not acquire `public._lock(cache)`.
The cache deduplicates reviews across lineages, but the lineage claims do not
serialize this shared cache. Its nonblocking lock raises the misleading message
`public fit directory is in use`. These failures occurred at cache acquisition;
they do not establish an API failure. Earlier calls, if any, remain accounted for.
The refreshed `JobIDRaw` snapshot reconciles all 88 recorded worker identities.
All recorded allocations are terminal; check the live queue before new work.

Resume the historical runtime with one critic worker across the campaign; see
the bounded recovery command in the operations runbook. Do not delete lock files,
reset attempts or patch the frozen source hash to permit a different runtime.
Future parallel critics need a targeted contention/requeue fix and a concurrency
test, separately from this historical continuation.

Successful worker allocations do not mean their dependent work was ever ready. The worker
loop deliberately exits after its allocation window, leaving time for bounded
operations. Earlier pruning workers may have finished before final fits, but
the original snapshot lacks timestamps and operation counts to establish that.

Use the [operations runbook](RESEARCH_OPERATIONS.md) to refresh observations,
including `JobIDRaw`, live queue state and worker finish receipts. Preserve
consumed attempt budgets and partial critic-call accounting. Verify the original
runtime, authorization and remaining state before planning a recovery wave.
The observational audit is not a substitute for those runtime checks.

Keep historical corrected and uncorrected reference versions separate. Finishing
Phase B describes the original experiment; it cannot validate the corrected
CSTR reference. A corrected comparison requires a new, matched experiment.

## Active component map

These are navigation anchors, not an exhaustive dependency closure or permission
to remove anything else. Several current entry points import historical modules.

| Concern | Start here |
|---|---|
| Frozen campaign and workers | `scripts/component_campaign.py`; `src/autoformalism/rebuttal/component_campaign.py`, `component_workers.py` |
| Campaign semantics and matrix | [FINAL_COMPONENT_CAMPAIGN.md](FINAL_COMPONENT_CAMPAIGN.md) |
| Staged proposal | `src/autoformalism/search/staged_proposer.py`; `schemas/staged_topology.py`; `schemas/staged_functions.py` |
| Shared declaration and law delivery | `search/shared_process_contract.py`; `search/identified_function_stage.py`; `search/process_assembly_contract.py` |
| Fitting and initialization | `src/autoformalism/fitting/public_fitting.py`, `fitter.py`, `conditional_collocation.py` |
| Pruning | `src/autoformalism/pruning/process_aware.py`; [PROCESS_AWARE_PRUNING.md](PROCESS_AWARE_PRUNING.md) |
| Critic | `src/autoformalism/rebuttal/component_critic.py`; [GENERAL_CALIBRATED_CRITIC.md](GENERAL_CALIBRATED_CRITIC.md) |
| Assessment | [DETERMINISTIC_MECHANISM_ASSESSMENT.md](DETERMINISTIC_MECHANISM_ASSESSMENT.md); fitted public-mechanism tests |
| Read-only operations | `scripts/audit_component_campaign.py`; `scripts/inventory_research_storage.py` |

When changing a component, inspect its actual imports, callers and relevant
tests. Keep historical campaign behavior pinned instead of silently routing it
through a new default. Do not move `rebuttal/` wholesale: file paths and source
hashes can be part of saved execution identity.

## Next research milestones and exit criteria

1. **Close the evidence gap (first week).** Reconcile Phase B workers and
   unfinished visits; retain historical failures. Incorporate Orion's qualified
   reference/input release as a distinct version. Run known-admissible-structure
   fitting diagnostics under the same observations and initialization rules.
   Exit: reference attainability and optimizer limitations can be distinguished
   from proposer mistakes; historic/corrected results have explicit labels.
2. **Isolate construction stages (second week).** Fix a small roster covering
   single/multiple outputs, memory, shared transfer, ordinary interactions and
   causal initialization. Compare actual predecessors with independently reviewed
   predecessors. Measure first-attempt scientific adequacy, mechanical acceptance,
   false rejection, repair calls and fitting utility separately. Select examples
   by predeclared strata, not by held-out scores. Exit: each recurring failure has
   a reproducer and a scientific/interface/numerical attribution or remains
   explicitly unresolved.
3. **Simplify and test specification value (third week).** Reuse a common model
   renderer and atomic edit path. Test only confirmed fixes; compare staged and
   coherent whole-model proposals on the same contracts and budgets. Qualify
   detention and one mechanical-system family with Orion. Use weak, mechanistic
   and stronger structural specifications without implying unique recovery from
   insufficient information. Exit: matched pilot results and equation inspection
   show which change helps, rather than only a successful demonstration.
4. **Freeze and evaluate (fourth week).** Fix algorithm, data versions, resource
   budgets, comparison roster and metrics; run focused ablations rather than a
   Cartesian sweep. Report prediction with median/MAD, compliance with mean/sample
   SD and unresolved evidence, completion and actual compute. Keep assisted
   diagnostics separate. Exit: a reproducible archive and defensible claims.

These are planning windows, not a guarantee about queues or scientific success.
Beam/MCTS, learned routing and more extensive searches follow a demonstrated
need; they are not prerequisites for the first stage audit.

## Fresh task handoff

Start a new task using this document and the most recent operations report;
copying the full historical conversation is unnecessary. Use the same repository.
A `codex/phase-c` branch can be created from the agreed clean baseline once
Orion's concurrent changes are committed; do not switch the shared checkout
under an active writer. Creating a branch or task does not establish a new data
release or transfer a running cluster job.

Suggested initial message:

> Work on Phase C in this repository. Read AGENTS.md,
> docs/PHASE_C_START_HERE.md and the linked active contracts. Orion owns benchmark
> corrections. First inspect the latest foundation/scheduler report and his
> qualification handoff. Continue one bounded milestone at a time. Preserve Phase B
> code/data/budgets and existing edits. Prefer reusable tested core components;
> do not add another campaign adapter unless existing entry points cannot express
> the experiment. Commit and push completed changes and provide site-specific
> commands. Keep test outcomes out of model and method selection.

Before archiving this task, retain the exact audit/report locations, source
commits, open work orders and confirmed decisions. Archiving conversation history
does not archive cloud storage or cancel jobs. Retain useful historical readers
until a verified archive and restore path supersede their working copies.
