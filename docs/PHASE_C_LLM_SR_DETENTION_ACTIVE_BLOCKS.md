# Detention fitting with reliability-guided active parameters

Date: 2026-10-26. An opt-in, training-only numerical pilot.

## Evidence motivating this milestone

The corrected-coordinate audit completed all 24 fixed-point checks. Its report
was reconstructed exactly from authenticated operation receipts (report SHA256
`3b9018bfc952c6434f3f438fae348ce5785fefe8feed06d517d44dbc15936814`).
Twenty-two coordinates qualified for search:

| Model 15 start | Previously blocked | Corrected audit still blocks |
|---|---|---|
| `perturbed_0` | `eps_s`, `Q_max` | `eps_s` |
| `perturbed_1` | `Q_max` | None |
| `generic_1` | `beta`, `eps_s` | `eps_s` |

At `perturbed_0`, the `eps_s` central derivative estimates agree closely, but
the smallest forward/backward probes fail the local consistency gate. At
`generic_1`, the gradient uncertainty estimate exceeds the search allowance
by about five times. Neither observation establishes an actual equation kink,
an unidentifiable parameter, or an incorrect equation family. The two zero
`Q_max` derivatives qualify search only; they do not establish parameter
information. The audit performed no optimization and changed no fitted values.

The audit used 2,525 summed operation wall-seconds and 2,345 CPU-seconds,
including operation overhead. It recorded 308 completed rollout evaluations
and no failed evaluations. These historical costs are retained separately.

## Algorithm and interpretation

At each independently verified incumbent, assess all eight coordinates of the
corrected, family-preserving Model 15 parameterization. Let `A` contain those
that meet the existing search-precision checks and `F` the others. Search solves

```text
minimize training_rollout_loss(x_A, incumbent_F)
subject to the original feasible parameter family.
```

Held coordinates retain their current values. The optimizer does not compute
their derivative columns or claim their values are correct. The same full
coordinate transformation is used for the reduced search, and the resulting
original parameters are independently verified using the original equations.

Each of three starts receives at most three 600-second search segments. The
first active sets contain seven, eight, and seven coordinates, respectively.
After each improved, verified checkpoint, all eight coordinates are rechecked
at that exact point. A newly qualified parameter can be released; a direction
that becomes unreliable can be held again. Solver termination, including a
plateau, or the segment time cap triggers the checkpoint and reassessment.
Thus evidence determines the active set; the time cap only bounds work.

If the same point makes no verified progress with the same evidence, stop and
record unused search allowance. Repeating the identical deterministic segment
is not a new initialization. If a candidate fails verification or is worse,
retain the incumbent. A useful checkpoint observed before an interruption can
still receive verification, but the interrupted operation's allowance is not
renewed. If no coordinate qualifies, report the unresolved fitting problem.

The final endpoint has full-coordinate search evidence, where available. This
pilot does not claim full stationarity, coefficient uniqueness, calibrated
confidence, global optimality, or scientific validity. It tests whether temporary
restriction enables fitting progress. Strong coupling can require simultaneous
changes, so a blocked coordinate must be revisited, and different starts remain
an important future option.

## Frozen experiment

The portable input authenticates the complete corrected-coordinate audit and
its preceding seven-task continuation history, including unsuccessful arms.
Equations, public domains, initial observations, training data and saved starting
points remain fixed. There is no cross-start seeding or held-out data access.

| Tasks | Work | Numerical allowances |
|---|---|---|
| 0–2 | Three adaptive continuations | Up to 3 × 600 s search per start; 120 s per original-model verification and per coordinate recheck |
| 3–4 | `eps_s` precision controls at `perturbed_0` and `generic_1` | 240 s each; no optimization |

The two precision controls reduce the tight integration tolerances while keeping
the optimizer finite-difference step, local probe ladder and acceptance
thresholds unchanged. Their receipts have distinct policy identities. They do
not silently override eligibility in the fitting tasks. Comparing common
completed local steps can help distinguish integration noise from a persistent
one-sided discrepancy; an unresolved diagnostic remains unresolved.

Every operation is checkpointed, identity-bound and charged once. Evidence reuse
requires the same point, chart, data, numerical policy and precision. Historical
and new costs, unused allowances, active/held transitions, verification outcomes
and incomplete records remain visible. Comparisons with earlier original-coordinate
fits are historical references, not a fresh matched-budget benchmark.

## Delta execution

Use the delivered source/input bundle in a fresh extracted directory:

```bash
export AF_LLMSR_DETENTION_ACTIVE_BLOCKS_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-active-blocks-v1
bash scripts/hpc/start_llm_sr_detention_active_blocks_delta.sh run
```

The launcher freezes inputs before submitting five array tasks with two concurrent
tasks. Each uses one CPU and 8 GB, with a two-hour scheduler ceiling covering
search, diagnostics, authentication and output overhead. The report has a
30-minute ceiling. No GPU, API key or LLM is required. Scheduler ceilings are
not fitting allowances; precision-only tasks finish after their bounded work.

From the same extracted directory:

```bash
bash scripts/hpc/start_llm_sr_detention_active_blocks_delta.sh status
bash scripts/hpc/start_llm_sr_detention_active_blocks_delta.sh inspect
```

Inspection writes `llm-sr-detention-active-blocks-v1-inspection.tar.gz` beside
the output root. Status/report/inspection never run a fit or renew an allowance.
Ambiguous scheduler replies block duplicate submission until reconciled.
Production fitting defaults and the separate construction campaigns are unchanged.
