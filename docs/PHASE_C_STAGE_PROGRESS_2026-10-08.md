# Minimal-prompt stage inspection and bounded refinement

This review uses the saved 4869909 prompt comparison, plan
`6fcd7396560be6266a7d265d4ce0a71484e0a9a700349dccaa22c7ec28def11f`, and
the current-bookkeeping-2 confirmation from bec075c, plan
`2001c278340952e1e911acaa1b4789773635e06977f06291b89d026f5f074033`.
Only public specifications, saved requests/replies and topology are inspected.
No functions, fitting, test access or new LLM calls support these findings.

## What the comparison can establish

The practical objective is the best working version of each prompt system, not
an artificially identical call schedule. A future system comparison should freeze
each version's native schedule and repair policy, use the same public case/seed
roster and model, and report actual calls/tokens, completion, and independently
reviewed topology. Resource ceilings must be disclosed; unused budget need not be
spent to match the other arm. This comparison evaluates systems, not wording alone.
The earlier eight-case current control used the common staged schedule. The latest
current confirmation covers only the two basins. They do not yet constitute an
all-eight-case comparison of the latest systems. Neither a single seed nor a
post-hoc best construction is a justified strategy ranking.

## Recorded stage boundaries

The frozen minimal schedule was variables -> shared processes -> ordinary topology
-> at most three global repairs. Within a stage, invalid edits/delivery received
bounded retries; accepted partial replies could continue. Most structural and
scientific-representation checks were deferred until the assembled topology.
In particular, the provisional process schema accepted an invalid transfer before
the final process-use check. There was **no separate immediate scientific repair
pass after each stage**. Repeated declarations are not evidence of scientific
correction, and a schema-valid reply is not a correctness verdict.

| Recorded fact, minimal arm (8 cases) | First reply | At stage exit | After global repair |
| --- | ---: | ---: | ---: |
| Every public target declared | 7/8 | 8/8 | 8/8 |
| Variable-stage completion flag true | 0/8 | 8/8 | not applicable |
| Required memory bindings supplied (4 applicable cases) | 0/4 | 0/4 | 4/4 |
| Shared-stage transaction accepted | 8/8 | 8/8 (one call each) | not applicable |
| Pairwise transfer format valid for all declared transfers | 7/8 | 7/8 | 8/8 |
| First ordinary-topology transaction accepted | 7/8 | 7/8 | not applicable |
| Whole topology structurally eligible | 3/8 | 3/8 | 6/8 |

Here first/exit refers to that row's named stage. The whole-topology first and exit
values happen to agree: seven cases made one topology reply; coupled basin made
three rejected replies. Eligibility includes unresolved overlap clarifications.
Binding absence is an annotation issue, not evidence that the state itself was
missing. T2-hard supplied its binding in ordinary topology; the other three
applicable cases supplied it during global repair.
Transfer-format counts include three cases that declared no named processes;
they are not counts of correctly identified shared physical laws.

## Per-case scientific interpretation

These are human inspections of declared meanings and topology, not automated
science grades or claims of exact reference-equation recovery. D= differential,
A=algebraic. Variable types alone cannot establish necessary/sufficient mechanisms.

| Case | Variables: first -> stage exit -> global exit | Shared-process decision | Ordinary topology -> local retries -> global exit |
| --- | --- | --- | --- |
| T1-easy | `Gp:D, A:D`; unchanged. Plausible target plus gut/meal state. | Gut-to-plasma transfer has opposite uses; ingestion was also named as a local, single-consumer process. | Both ordinary lists empty. Structural pass without repair, but plasma only receives absorption; no disposal/restoring contribution. |
| T1-hard | `Gp:D, Ggut:D`; unchanged locally. Global repair adds `k_abs:A, k_clear:A`, then constant equations for them; neither is used as a driver. | Plausible gut-to-plasma transfer, plus local ingestion process. | First reply supplies no ordinary entries. Global repair supplies them and removes duplicate gut effects, but leaves the unnecessary coefficient variables. |
| T2-easy | First has `Gp,I,A,M:D` but omits target `U`. Next adds `U:A`; final keeps it. Types form a plausible meal/insulin-memory representation. | No shared process; legal for this chosen representation. Whether meal depletion and appearance should be tied remains a scientific question. | Initial topology contains meal and insulin memory, total `U` includes `Uii`, and `Gp` uses supplied `EGP,E`. Global repair adds only the required insulin-memory binding. Functional adequacy is untested. |
| T2-hard | `Gp,I,I_mem,G_abs:D`; unchanged at stage/global exit. | `meal_absorption(meal_rate)` feeds both `G_abs` and `Gp` positively; three other named processes are local. `G_abs` does not mediate plasma appearance despite its stated absorption role. | Ordinary contributions repeat named processes. Global repairs remove one repeat but leave overlaps. Prose proposes deleting processes; removal lists do not enact that. Final incomplete. |
| CSTR | `T:D; Qf,Qr,Qj:A`; unchanged. Sensible temperature plus instantaneous heat quantities. | Empty; reasonable because each heat quantity feeds only `T`. | Each heat law includes `T`. The temperature RHS is one joint function of all three heats, not three separately signed contributions. Structural pass; physical additive balance still unestablished. |
| Alien device | `v01,m:D; v02,v03:A`; unchanged. Input-memory state exists. Electing to generate supplied telemetry is permitted, but the asserted gain/decay proxy meanings are not established. | `mem_accum(u01)` initially labeled transfer with four positive consumers; invalid transfer format. It represents direct input uses, not the value of memory `m`. | Ordinary effects overlap process uses. Global repair changes transfer to influence, adds binding and memory decay, then describes deleting the process but leaves its removal list empty. Final incomplete. |
| Coupled basin | `h_up,h_down:D; Q_up_to_down,Q_down_out:A`; unchanged. Plausible depth and discharge inventory. | One opposite-signed internal transfer and one local outlet. The transfer includes instantaneous inflow among its drivers; the physical law is not yet established. | Three replies try an ordinary definition of `Q_up_to_down` with drivers different from the existing process. All are rejected. Global repair keeps the process definitions and adds just ordinary inflows plus the upstream binding; structural pass. |
| Independent basin | First `h_down:D`; later adds `q_out,S_down:A`. Algebraic storage is compatible with depth as the dynamic coordinate. | Empty; no spurious upstream coupling. | One joint depth term uses inflow, storage and outlet. Outlet depends on `warning_depth`, not the physical crest. Structural pass without repair does not make that interpretation correct. |

## Specific questions resolved from the saved calls

- Minimal editing rules defined `stage_complete`, but its variable instruction did
  not explicitly say when to return true. Every first reply returned false. Five
  repeated no-change replies were rejected. We cannot know the proposer's private
  intent, but the records support a completion-contract ambiguity, not eight bad
  inventories. The current control explicitly requested true when the inventory
  was ready. The response template used false in both arms.
- Retained declarations, generated equations, potential overlaps and edit effects
  were already displayed in the minimal run. The newer current policy's stronger
  separation of editable versus read-only representations was not included.
  T2-hard event 5 described deleting processes but sent no process removals; it
  removed only `G_abs`, which event 6 restored after its remaining references were
  reported. Alien event 7 likewise left `remove_processes=[]`. Runtime did not
  infer deletions from those explanations.
- Coupled basin events 4-6 treated previously declared algebraic discharge names
  as needing ordinary equations. `Q_up_to_down` had process drivers
  `[inflow_up,h_up]`; the proposed second definition changed the drivers. This was
  an actual competing definition, not a harmless repeat. The original rejection
  did not identify the competing process by name. Global repair finally omitted
  the second definition. A clearer editing location and named conflict receipt
  address the observed ambiguity without selecting the scientific drivers.

## Implemented refinement (opt-in, separate from historical policies)

`minimal-clarity-1` keeps the short system and stage-specific text. It adds explicit
stage completion, the parameter/variable distinction, read-only generated
definitions and balances, concrete process removal keys, and before/after edit
receipts. Its duplicate-definition comparator remains the historical strict one;
the current policy's description-only normalization is not silently inherited.
Variable completion checks only missing public targets, retaining valid partial
declarations. It does not grade scientific variable completeness. Final graph
criteria, provisional process acceptance and atomic transactions are unchanged.

`current-repair-fidelity-1` extends current-bookkeeping-2 only. It distinguishes
keeping a differential coordinate from explicitly choosing an algebraic readout,
asks the proposer to preserve every intended effect when replacing a joint term,
and records exact before/after types, balances and removed contributions/uses.
These are receipts, not a gate that assigns types, signs, conversions or meanings.
A final accepted repair may still be scientifically flawed; no mandatory extra
self-review call is introduced.

The next diagnostic is **11 tasks**, not a new factorial comparison:

- Eight fresh minimal constructions: all eight public cases, Full, seed 0.
- Three current-prompt repair episodes, starting from their actual saved
  pre-global drafts in the bec075c confirmation: independent/separate/integrated,
  independent/joint-guided/integrated, coupled/joint-adaptive/integrated.
  Each has at most three new repair calls and no repeated construction.

Starting checkpoints, public context, task IDs, policies and source fingerprints
are sealed. Repair cases are selected for diagnosed failures, not scores, and
are not counted as fresh-current controls. Every call is cached/logged, interruption
resumes against the same starting draft, and no fitting or test jobs are submitted.
The pinned launcher prepares from the embedded public source plus
`phase-c-current-bookkeeping-v2` on ACES. Use a fresh output root; old campaigns
continue to require their original pinned bundles.

```bash
bash scripts/hpc/start_phase_c_prompt_refinement.sh run
# After the jobs finish:
bash scripts/hpc/start_phase_c_prompt_refinement.sh inspect
```

Read-only stage evidence can also be rebuilt locally, outside the source archive:

```bash
PYTHONPATH=src python scripts/audit_construction_stage_progress.py \
  --source /path/to/saved/inspection --output /path/to/new/stage-review
```

This writes `STAGES.html` and `stages.json`, including the first actual reply,
retained first/exit snapshots, rejection diagnostics, process consumer facts,
pre-global checkpoint and final result. No scientific verdict is inferred from
natural-language explanations. The review must continue to inspect both effects
and state/readout assignments before moving to interaction.
