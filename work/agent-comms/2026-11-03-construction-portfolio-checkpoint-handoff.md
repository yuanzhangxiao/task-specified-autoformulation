# Compact checkpoint continuation: implementation handoff

3 November 2026. Implements the milestone recommended in
`2026-11-02-construction-portfolio-residual-results.md`.
The operational runbook is `docs/PHASE_C_CONSTRUCTION_PORTFOLIO_CHECKPOINT.md`.

## What changes

The opt-in `SearchCheckpointPolicy` applies to root-profile and residual-only
Powell searches. Historical calls without the keyword preserve their previous
behavior. Routine counter/progress writes occur at most once every 30 seconds.
Every strictly improved incumbent, accepted iteration, initial-sweep completion
and eligible final checkpoint still forces persistence. A failed checkpoint
halts further numerical work. Checkpoint serialization and hashing failures are
also counted, not just failures inside the file-writing callback.

Compact progress retains the complete best endpoint, parameters, root-solve
evidence, counters, timing and bounded failure examples. Repeated static bodies
are replaced with explicit content hashes, and progress histories retain their
latest eight entries with the total count. Setup and qualification receipts
retain their evidence; full terminal results retain the complete search result.
The compact snapshot explicitly states that it is not the complete scientific
record. No objective, candidate comparison, bound, conditional KKT gate, domain
check or integration tolerance changes.

The operation journal reserves the whole stage before work starts. Counter-only
progress can lag after a hard interruption, but that never renews an allowance.
Previously persisted incumbents remain available to independent verification.
Full Powell state is not checkpointed; a separately authorized continuation is
a warm start from saved parameters, not an exact internal-optimizer resume.

## New campaign

Protocol: `construction-portfolio-checkpoint-1`.
Source: `/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-residual-v1`.
New root: `/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-checkpoint-v1`.

| Model | Verified starting training NMSE | Search |
| --- | ---: | --- |
| 10 | 1.7129412519089338 | Residual-only, 24 coordinates |
| 12 | 0.43648487564637667 | 9 shared coordinates, 32 profiled initial roots |
| 14 | 1.6887254012285298 | Residual-only, 24 coordinates |
| 15 | 0.4388608844130449 | 8 shared coordinates, 32 profiled initial roots |

All four immediate source records, the source plan, training payload and
numerical settings are pinned. Preparation reconstructs every prior terminal
exactly from its operation receipts before producing a nonnested projection.
Models 10/14 retain rtol/atol 1e-11/1e-13; 12/15 retain 1e-9/1e-11 and their exact
affine propagation. The per-record settings are inherited from authenticated
setup receipts. Source results and their old termination labels are untouched.

Each search retains 1,800 numerical-active seconds, a 1,920-second overall wall
safety ceiling and 2,000 objective attempts. Stage reservations total at most
4,620 seconds per task. Slurm requests one CPU and 8 GB per worker, at most two
concurrently, with a four-hour worker ceiling and 30-minute report job. There
are no GPU/LLM calls, equation revisions, benchmark changes, validation/test
access or new causal-prefix forecasts.

Reports now separate measured overall-wall and numerical-active limits from
evaluation caps, checkpoint failures and interruptions. A hard-killed worker's
stale ordinary progress cannot be reported as still running: the journal's
interrupted outcome takes precedence when no more specific timeout exists.
Operational completion still does not establish optimization convergence.
The old residual campaign retains its historical reporting semantics so its
sealed terminals continue to reconstruct exactly.

## Controlled persistence tests

The slow sink advances a simulated clock by 0.7 seconds per write. Both policies
see the same deterministic sequence of 109 objective evaluations and retain the
same best point when the overall allowance is sufficient:

| Search | Original writes | Compact writes | Original callback seconds | Compact callback seconds |
| --- | ---: | ---: | ---: | ---: |
| Root profiling | 223 | 9 | 156.1 | 6.3 |
| Residual-only | 226 | 6 | 158.2 | 4.2 |

With the same tighter five-second numerical / 125-second overall allowance, the
old paths hit the overall limit after 88/86 objective evaluations; the compact
paths complete all 109. These are controlled persistence tests, not a measured
Delta speedup or a forecast of how many new best points the real searches find.
Actual Delta write counts and callback time remain experimental outcomes.

Evidence is in ignored `output/compact-search-checkpoint-smoke-20261103.json`.
Compact-JSON byte counts describe callback payload encoding; they do not claim
to measure filesystem traffic or the bytes of the final journal seal.

## Verification and limits

Tests cover identical objective sequence/selection, forced-best durability,
periodic and bounded progress, serialization/write failure, keyboard and hard
interruptions, no repeated spent search stage, source authentication, independent
retention and report reconstruction. Bounded three-attempt checks exercise the
actual four source endpoints, with fresh qualification and independent original
replay; long fitting is reserved for Delta.

All four bounded checks passed. Original-replay prediction discrepancies were
1.95e-10, 1.26e-9, 2.26e-10 and 1.03e-9 for models 10/12/14/15. Each stopped at
its deliberately small three-attempt smoke cap. Their verified NMSEs reproduce
the saved endpoints; these checks do not claim new fitting gains. Evidence is
in `output/construction-portfolio-checkpoint-preflight-20261103/smoke.json`.

All 304 targeted tests pass (24.27 seconds), changed Python files pass Ruff,
and the Bash syntax and CLI-help checks pass. Repository-wide Ruff retains the
same 37 pre-existing findings under unrelated `analysis/claude` files.

Local preparation and receipt-only reporting are also checked. The launcher
recovers the submitted checkout from `launcher_execution.json.value.repo`, so
status and inspection work from any directory. No remote session or submission
is performed by this implementation turn.

Remaining limitations: frequent genuine improvements still force frequent
writes; an already blocking storage call is not preempted; hard interruption can
lose recent counter-only updates; no full optimization or parameter-confidence
conclusion follows from the bounded checks. Historical and new costs stay
separate, and search-view counters must not be added as disjoint physical calls.
