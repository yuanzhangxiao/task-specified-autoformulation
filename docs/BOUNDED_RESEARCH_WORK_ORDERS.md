# Bounded research work orders

Use one supervisor for scientific decisions, integration and final review.
Delegate repeatable extraction, contract mapping, fixture construction and
small scoped fixes to a lower-cost worker. Reuse deterministic scripts once a
task repeats. Do not spend expert context supervising open-ended rediscovery.

For this milestone the worker mapped existing artifact contracts read-only;
the supervisor implemented and verified the audit. Orion owns benchmark
corrections. Workers do not modify those files. This document describes the
workflow; it is not an automatic agent launcher or authorization to start jobs.

Before dispatch, specify:

```yaml
id: <stable work-order id>
objective: <one observable result>
model: <explicit supported worker model>
inputs: [<exact paths and scientific artifact hashes>]
read_scope: [<necessary source, instructions and tests>]
write_scope: [<specific files or new output directory; empty for read-only work>]
fixed_decisions: [<semantics, metric, denominator, task roster>]
deliverables: [<typed report, diff, fixtures or reproducer>]
acceptance: [<independent expected outcomes and at least one failing case>]
verification: [<exact commands>]
limits:
  correction_cycles: 1
  recursive_delegation: false
  live_model_calls: 0
  optimizer_calls: 0
  scheduler_mutations: false
escalate_if: [<scientific ambiguity, provenance mismatch, out-of-scope change>]
```

Raise these limits only when required by the particular authorized work order.
No changing denominators, inferring missing runs, silently correcting reference
data, or weakening acceptance rules. Test access requires an explicit isolated
evaluation task. Shared-checkout implementation has one writer per file; begin
with read-only parallel work, not overlapping refactors.

Return one compact completion record:

```json
{
  "ticket": "...",
  "status": "complete|needs_review|blocked",
  "artifacts": [],
  "files_changed": [],
  "checks": [{"command": "...", "result": "..."}],
  "counts": {},
  "assumptions": [],
  "unresolved": [],
  "usage": {"measured": false}
}
```

The supervisor checks evidence and tests, not confidence. Read every diff that
affects scientific semantics; use deterministic checks and bounded spot checks
for extraction. After one unsuccessful correction cycle, inspect the reproducer
or redesign the contract instead of extending a conversation indefinitely.
Model choice alone does not establish lower total cost: record actual usage
when available and avoid inventing savings when it is not measured.

Next suitable bounded assignments, after the campaign census is available:

| Work order | Worker output | Supervisor acceptance |
|---|---|---|
| Stage examples | Small list of task/round/route/source hashes for a fixed stratum | Each resolves to the displayed predecessor; no score-based cherry-picking |
| Regression fixtures | Synthetic minimal inputs reproducing a confirmed contract bug | Both failure and intended success; no physical rule silently added |
| Metric reconciliation | Fixed-roster coverage and per-case values | Independent aggregation; failed/missing/unresolved kept distinct |
| Benchmark manifest audit | File/input/protocol hashes and qualification status | Orion resolves scientific discrepancies; worker never edits data |

Keep high-level scientific diagnosis, specification design, admissible-model
decisions, experimental selection and paper claims with the supervisor and
researchers. New mechanisms or search algorithms are not housekeeping tasks.
