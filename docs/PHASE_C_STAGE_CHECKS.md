# Minimal shared-process stage checks and topology identifiers

The opt-in `minimal-stage-checks-1` policy extends `minimal-variable-checklist-1`.
It preserves the short stage prompts and adds concrete editing examples and local
shared-process completion checks. Earlier studies require their original pinned
code; their results, public specifications and fitting settings are unchanged.

## Evidence and scope

The eight-case variable-checklist confirmation declared every public target in
the first retained reply. Seven first inventories passed declaration checks;
all eight passed at variable-stage exit. Coupled basin supplied its missing
memory/readout bindings after targeted feedback. This establishes declaration
consistency, not scientifically sufficient state selection or mechanism recovery.
There were no explicit mandatory target-type restrictions in these eight inputs.
Four final topologies passed the existing structural checks.

The saved T2-easy and coupled-basin replies used derivative labels as topology
identifiers. The new instruction illustrates the distinction directly:

> For a differential variable named `Gp`, return `{"name":"Gp","terms":[...]}`,
> not `name="dGp_dt"`. Its declared type already makes this the topology of
> `d(Gp)/dt`. If the declared identifier is `G_p`, preserve `G_p` exactly.

Feedback names possible derivative-label mismatches and shows the explicit
`remove_equations` edit needed to remove an old alias. The runtime does not rename
variables or infer a scientific identity. For CSTR's missing ordinary topology,
the prompt explains that `{"name":"T","terms":[]}` explicitly declares no
additional ordinary contributions while keeping all previously declared process
uses. Omitting an entry means no edit. Restating shared uses is unnecessary.

A supplied `Uii(t)` can be a process driver but cannot receive a generated term:
it has no candidate equation. If the proposer instead intends to generate that
observed quantity, it must explicitly declare its type and provide its topology.
This is a modeling decision, not a runtime correction. External inputs and fixed
covariates cannot be converted into generated quantities. The meaning of `Uii`
and whether the intended disposal mechanism was identified remain scientific
questions; a name or prose explanation does not certify them.

## Shared-stage boundary

At each shared-process turn, the runtime checks:

- each receiving variable occurs once per process;
- a pairwise transfer has exactly two opposite signed uses;
- a known supplied channel is not used as a generated receiver.

These are existing final rules, applied earlier with specific repair options.
Proposers retain control of signs, process kind, receiver selection, variable
types and removals. Accepted partial transactions remain visible, but inconsistent
declarations cannot complete the shared-process stage. Up to three requests use
the existing local allowance; exhaustion is recorded and ordinary topology may
continue with the partial draft. Variable checks are repeated after edits, and
all final structural checks still apply. There is no extra successful self-review.

Unknown future references remain deferred until the complete topology, as do
algebraic cycles, required pathways and consumer assembly. Existing single-receiver
influences remain legal named local contributions; the report does not count them
as shared laws. Empty shared-process decisions remain legal. These checks cannot
decide that sharing was scientifically required, that a sign is scientifically
correct, or that a mechanism is recovered.

## Live confirmation and inspection

`stage_check_confirmation` uses eight fresh minimal constructions, including both
basins, Full context, seed 0. The proposer, settings, total request/token limits,
three global repair requests and variables -> shared processes -> ordinary
topology schedule are unchanged. It stops before interaction functions, fitting
and test data. This is a diagnostic confirmation, not a matched strategy ranking.

The portable bundle includes the sealed public source plan, checksums and a
`RUN_ACES.sh` wrapper. Upload it to ACES group scratch. The repository equivalents
are:

```bash
bash scripts/hpc/start_phase_c_stage_checks.sh run
# After jobs finish:
bash scripts/hpc/start_phase_c_stage_checks.sh inspect
```

Default root:
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-stage-checks-v1`.
One H100 worker uses a three-hour work window in a 3.5-hour allocation, followed
by a CPU report. Calls and transactions are cached and checkpointed under a frozen
identity; repeated submission uses its saved receipt.

- `VARIABLES.md/json`: first variable reply, stage exit and final declaration checks.
- `SHARED_PROCESSES.md/json`: first retained shared-process reply, last snapshot
  after immediate repair, pre-global topology and post-global topology. It shows
  stage completion separately from local consistency, shared/local counts, each
  declaration, exact blockers and per-turn request hashes.
- `SUMMARY.md/json`, `TOPOLOGY.html` and cached calls: structural outcomes,
  assembled contributions, full prompt/response evidence and costs.

Pending assessments are missing, not zero. A locally consistent empty decision is
not evidence of successful sharing. Read the actual retained process definitions
alongside these checks. `inspect` refreshes reports without model calls and creates
`inspection.tar.gz` for review.

## Verification

Read-only checks on the previous eight shared-stage snapshots flag T2-hard's
supplied `Uii` receiver and CSTR's duplicate `T` receivers. The other six snapshots
pass locally, including coupled basin's initial transfer. This does not create
new repaired results or establish that empty shared-process decisions were right.

The 156 focused tests cover valid and invalid declarations, future references,
explicitly modeling an observed auxiliary, derivative-label suggestions without
mutation, bounded local repair, retained partial edits, interrupted cached resume,
stage reporting and the eight-case campaign/submission path. The runtime-agenda
and incremental-construction smoke tests pass. The new/modified Python files pass
Ruff; the full checkout retains unrelated lint findings in `analysis/claude`.
