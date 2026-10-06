# Phase C: foundation, scope and handoff

This is the short entry point for the next research phase. It does not replace
AGENTS.md, change a frozen protocol, or announce a new production algorithm.
Read this before the chronological campaign history in PROJECT_CONTEXT.md and
PIPELINE_DESIGN.md. Implementation still follows their relevant contracts.

The first executable step is the [construction baseline](PHASE_C_CONSTRUCTION_BASELINE.md):
32 constructions on eight corrected public cases, including coupled/independent
noiseless basin controls, independently assessed for NMSE
and mechanism compliance, with exact prompt/response and equation reports. It
starts stage isolation; reviewed-predecessor experiments remain subsequent work.
The [first variable-stage audit](PHASE_C_VARIABLE_AUDIT.md) inspects all 32 saved
constructions. It identifies an undisplayed target-role policy, omission of the
Phase B construction wrappers, and mechanism-assignment/reuse problems. Its
offline reports expose exact prompts and decisions; no construction fixes or
new experiment are silently applied to the frozen pilot.
The next [variable-only confirmation](PHASE_C_VARIABLE_CONFIRMATION.md) restores
the public-contract wrapper, corrects the reviewed mass/rate role inferences,
and adds explicit proposer-owned memory bindings. It uses the same 32-task
roster, preserves full repair context, and submits no fitting jobs.
In parallel, the [fitting qualification plan](PHASE_C_FITTING_PLAN.md) starts a
42-fit, CPU-only Delta study of coordinate scaling and shared initial values on
fixed CSTR/basin equations. It changes no production defaults. The same
construction runbook supplies a separate CPU-only Phase B
endpoint closeout without changing the historical runtime.

Fitting has since completed the [M4 budget comparison](PHASE_C_FITTING_M4_RESULTS_2026-10-02.md)
and [M5 reuse review](PHASE_C_FITTING_M5_RESULTS_2026-10-02.md). All 24 M5 fits pass
prediction and coefficient recovery. Formulation reuse preserves the cold-solve
results; good primal starts reduce repeated-solve iterations. M5 did not exercise
screening vetoes. The [M6 harder-case results](PHASE_C_FITTING_M6_RESULTS_2026-10-03.md)
now account for all 36 CPU fits: basin passes all 18; alien rollout-only and hybrid
each recover two of three starts, while collocation/shooting and both reuse arms
recover none. The hybrid falls back to original parameter guesses. Screening
exercises vetoes but has mixed outcomes. Next isolate native numerical cost and
stagnation/restart behavior on this qualified case. The implemented
[M7 diagnostic](PHASE_C_FITTING_NUMERICAL_DIAGNOSTIC.md) adds an observation-preserving
reduced collocation mesh, a fixed-mesh Hessian comparison, and a bounded
continuation/restart comparison. The [M7 review](PHASE_C_FITTING_M7_RESULTS_2026-10-04.md)
verifies 36 backends and 35 finalized evaluations; one independent replay times
out without saving a terminal record. Basin passes all 18 fits; alien rollout
recovery remains two of three starts. Mesh reduction enables native convergence
without accurate recovery; exact shooting Hessians and detailed checkpoint
callbacks are measured bottlenecks. The next proposed fitting milestone reduces
checkpoint overhead and handles replay failures explicitly. Existing methods
remain available; there is no production fitter promotion.

The implemented [M8 fitting diagnostic](PHASE_C_FITTING_CHECKPOINT_DIAGNOSTIC.md)
pairs legacy and compact checkpoint policies across 36 CPU fits. It also journals
independent replay, reports timeouts without partial-data accuracy, and preserves
interrupted restart metadata. A separate log-based M7 closeout performs no new
fitting/replay. The [M8 results](PHASE_C_FITTING_M8_RESULTS_2026-10-04.md) account
for all 36 endpoints, including two explicit replay timeouts. Compact checkpoints
substantially reduce overhead, but alien recovery remains unsuccessful in these
native arms; basin still passes all 18. The implemented
[M9 diagnostic](PHASE_C_FITTING_SCREENING_ASSISTANCE.md) adds bounded RK45/Radau
screening and fixed-then-free collocation from explicitly training-fitted starts.
Its [M9 results](PHASE_C_FITTING_M9_RESULTS_2026-10-05.md) account for all 48
entries. Basin still passes, but the 20-second cap times out every alien Radau
screen, including all four eligible assisted rechecks before collocation starts.
The implemented [M10 diagnostic](PHASE_C_FITTING_SCREENING_REPLAY.md) calibrates
screening on saved models, replays 18 saved pools, and retries the four blocked
assisted tests with final-screen headroom and trajectory timing logs. Existing
rollout success remains unchanged; no production fitting default is promoted.

The [M10 results](PHASE_C_FITTING_M10_RESULTS_2026-10-05.md) show that calibrated
screens unblock all four assisted fits. Dense collocation stays accurate from
good fitted starts, while reduced grids introduce coefficient/initial bias. The
four selected models remain the supplied M7 incumbents, and generic saved pools
still show no recovery. Next investigate mesh convergence and initialization
robustness separately; a cleanup exception and two replay timeouts also need
explicit closeout. This remains a diagnostic, not a production promotion.

The [M11 mesh diagnostic](PHASE_C_FITTING_MESH_REFINEMENT.md) is now implemented:
four independent resolutions and coarse-to-fine polynomial transfer from each of
two previously training-fitted alien starts. Ten Delta CPU tasks evaluate every
released endpoint separately from the retained incumbent. It qualifies mesh
accuracy before generic-start robustness; it is not automatic adaptive fitting
or a production default. Historical data/results are preserved.

## Objective and ownership

Build reliable scientific model construction before expanding search. Separate
reference/data defects, insufficient specification, proposer capability,
interface rejection, numerical fitting limits and selection effects. Good
prediction need not imply a unique textbook realization; textbook recovery is
an appropriate claim only under adequate specification and identifiability.

- Orion owns benchmark correction, input semantics, reference qualification and
  the Phase C fitting milestone; Astra owns Phase C construction.
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

### Latest recovery checkpoint: wave 4, reviewed 2026-09-30

The snapshot captured at `2026-09-30T09:59:02Z` has audit seal
`34431cc024e2425f22d4a33e43c7485e41a97ae03bb689381bbacd4ad35b9c92`;
the supplied archive SHA-256 is
`8ead03126b6b24c3223077cf2ee7b863abc5c4eeea9f6ff140f640c3c58ea4a4`.
Both archived audit summaries reproduce locally and agree with the freshly
generated campaign report. All 2,075 previously recorded round-result hashes
and all previously recorded endpoints are unchanged. No artifact issues were
reported. Test data remain unopened and evaluation is not frozen.

**Search is finished:** all 160 lineages have records through zero-based round
14, accounting for all 2,400 planned visits. There are 2,350 complete visits,
14 construction failures, 6 fit failures and 30 interruptions, with no missing
visits. Since wave 3, 321 complete visits and four interruptions were added.
The four interruptions belong to `cell07_seed0_full_c1v1s1`, retain its known
diagnostic hash and have no fit-start receipt. Both this lineage and
`cell07_seed0_brief_only_c0v1s1` have 15 interrupted visits and no finite incumbent;
the other 158 lineages have finite retained models.

**Endpoint processing is unfinished:** 111 endpoint rows are complete, 79 are
missing, one is model-unavailable and one is skipped without a finite parent.
The next actions are pruning for 66 lineages and final critic review for one.
Of those 66 pruning lineages, 65 have finite incumbents and can subsequently
require final critic reviews. Thus one currently ready final review is not the
entire remaining critic workload. Search proposer and fit workers are no longer
needed; paired pruning performs its own bounded refits. A CPU closeout should
use parallel pruning and a single critic, preserving the frozen runtime,
consumed budgets and the shared critic-cache serialization workaround.

Wave 4 (`serial-critic-recovery-4`, jobs 2174079--2174082) has 20 successful
allocations with exact scheduler matches and worker finish receipts. They
record 237 proposal, 346 critic, 325 fit and 22 pruning operations. One proposer
started after proposal work was exhausted and performed no operations. Among
87 critic-enabled lineages with finite incumbents in both snapshots, 25 improved
validation NMSE and 62 were unchanged; one further lineage remains unavailable.

At round 14, before pruning, case medians over both planned seeds followed by
the macro median and unscaled MAD over all nine cases are:

| Critic | Verifier | Full NMSE [MAD] | Brief-only NMSE [MAD] |
|---|---|---:|---:|
| on | on | 0.248603 [0.178266] | 0.384436 [0.306497] |
| off | on | 0.336642 [0.155912] | 0.474422 [0.230833] |
| on | off | 0.386238 [0.138658] | 0.929019 [0.547110] |
| off | off | 0.459450 [0.297354] | 0.618690 [0.296344] |

These are the primary shared-process-enabled arms. Failures receive positive
infinity before either median, rather than being omitted. Full with both
components and Brief-only without the critic each have 17/18 finite repetitions;
the other arms have 18/18. A median over two seeds is their midpoint and does
not itself suppress one extreme seed. The nine-case median limits the influence
of extreme cases but is not a confidence interval or evidence of equation
recovery. These comparisons have equal search visits, not equal compute, and
use the historical CSTR/alien reference versions. They cannot establish
performance on the corrected Phase C release or replace fitted mechanism tests.

Recorded critic usage rose by 979 HTTP attempts and 8,641,209 tokens, to 4,072
attempts and 33,269,858 observed tokens. The same three usage events remain
unmeasured. Group scratch has 467,122/500,000 file entries (32,878 headroom);
personal scratch remains 247,733/250,000. The queue was empty at capture. The
snapshot does not contain the cache archival receipts, so quota alone does not
establish how many entries the archive operation retired.

Phase C stage-isolation development can proceed independently of this historical
CPU closeout, using the separately qualified development release. Do not import
corrected data or fitting changes into the original campaign. Keep the proposed
CSTR fitting improvements separately qualified before changing defaults.

### Previous recovery checkpoint: 2026-09-29

The snapshot captured at `2026-09-29T18:10:24Z` has audit seal
`2358f5f652c1bb0fa00f2f333664edd2407c151592a14d652975b621c7a27cef`.
Both summaries reproduce locally. All 1,928 earlier round-result hashes and
previously recorded endpoints are unchanged. The focused manifest now correctly
identifies `serial-critic-recovery-3`; its 20 allocations all completed with
matching worker finish receipts. There were 147 proposer, critic and fit
operations each, and zero pruning operations. Both proposer workers did useful
work (66 and 81 operations). The wave-2 startup failures did not recur; this
does not establish their underlying cause.

The campaign has 2,029 complete round rows, 14 construction failures, 6 fit
failures, 26 interruptions and 325 missing rows. The 88 critic-enabled lineages
end at zero-based round 10 (61) or 11 (27), awaiting their next critic operation.
The 72 critic-disabled lineages remain terminal: 71 complete endpoints and one
skipped without a finite parent. Thus 120 endpoint rows still await completion.
The two new interrupted rounds belong to the same `cell07_seed0_full_c1v1s1`
lineage and retain its historical diagnostic hash; neither has a fit-start
receipt. Keep these model-level failures distinct from allocation failures.

Of 87 critic-enabled lineages with finite incumbents in both snapshots, 23
improved validation NMSE and 64 were unchanged. The remaining lineage still has
no finite incumbent. All lineages have a recorded round 10, allowing an interim
comparison at initial construction plus ten revisions. This is before final
pruning, on the original reference suite, with equal visits rather than equal
compute. It does not establish equation recovery or fitted mechanism compliance.
Test data remain unopened.

The critic completed 147 operations in 5:02:53. At that rate, 325 remaining search
operations represent about 11 critic processing hours; up to 88 final reviews
would add roughly three hours if similarly expensive. Queues, allocation gaps,
fitting and pruning are additional; this is not a completion guarantee.

Storage is now the immediate operational concern. Personal scratch remains at
247,733/250,000 files; allocation-wide group scratch is at 476,355/500,000, leaving
23,645 file slots. The observed group increase of 22,037 is almost that entire
headroom and cannot be attributed solely to this campaign. Inventory the exact
`component-runtime-cache` tree and reconcile its numeric job directories with
scheduler records before another wave. This is a metadata review, not deletion
authorization. Preserve source archives, model weights, fit checkpoints and
critic/proposer response caches. The queue was empty at capture, not necessarily
now. No new wave was submitted by this review.

The subsequent cache inventory at `2026-09-29T21:04:51Z` is complete: 18
job directories, 42,040 regular files and 5,867 directories, no symlinks or
special entries. Exact scheduler raw IDs identify all 18 as successfully
completed `component-propose` allocations. The pinned launcher's per-job
compiler-cache bindings account for all five observed namespaces; scientific
call records, fits, runtime logs and model weights live elsewhere. Group file
usage is now 476,926/500,000. The prepared wave-4 command archives each of these
explicit directories, verifies every archived file and rechecks live scheduler
state before removing the loose copies. About 47,900 entries could be retired,
leaving roughly 71,000 group file slots after small archive/receipt overhead;
the actual quota must be checked after execution. No remote deletion is implied
by this inventory. The archival helper is outside the frozen runtime.
See the operations runbook for the bounded archive-and-resume command.

### Previous checkpoint: 2026-09-28

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

Benchmark preparation now has a separate development release described in
[PHASE_C_DATASET_READINESS.md](PHASE_C_DATASET_READINESS.md): CSTR and
alien-device preparation fixes, detention-basin controls and Dalla Man T1/T2.
The 28-cell development release retains exact-interface replay gates for the
first three families. Dalla retains the existing multi-meal inputs and reference
dynamics under an accepted reduced-ODE approximation scope; its numerical gates
still apply. Final held-out publication remains a separate milestone. Fitting
ideas remain deferred in
[PHASE_C_FITTING_FINDINGS_2026-09-28.md](PHASE_C_FITTING_FINDINGS_2026-09-28.md).

These are navigation anchors, not an exhaustive dependency closure or permission
to remove anything else. Several current entry points import historical modules.

| Concern | Start here |
|---|---|
| Phase C development datasets | [PHASE_C_DATASET_READINESS.md](PHASE_C_DATASET_READINESS.md); `scripts/prepare_phase_c_benchmarks.py` |
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
