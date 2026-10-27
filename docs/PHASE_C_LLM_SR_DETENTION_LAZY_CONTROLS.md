# Detention residual and derivative controls

Date: 2026-10-27. Opt-in numerical development; training data only.

## Returned active-block evidence

The returned report reconstructed exactly from saved operation receipts, with
SHA256 `d41a220e00f9eda110c04378f31860cb1d3d34a05b3dac7a324b5d359de13c12`.
Four of five tasks have terminal records. The generic start saved an independently
verified improved endpoint before its last three derivative checks were interrupted.

| Model 15 start | Initial training NMSE | Active blocks | Historical original coordinates |
|---|---:|---:|---:|
| perturbed_0 | 3.026179 | 2.296089 | 0.0000952805 |
| perturbed_1 | 2.991611 | 2.991552 | 0.212965 |
| generic_1 | 0.799478 | 0.217279 | 0.799478; search blocked |

These historical arms are not a newly matched comparison. The generic start
improved with `eps_s` held, improved further with `C` also held, and then improved
after both directions were released. The second perturbed start held no coordinates:
its poor result cannot be attributed to holding alone. Its two 600-second attempts
accepted no optimizer steps. Rejected trial points consumed about 110–132 seconds
each because the residual callback also built a full numerical Jacobian. A budget
stop without progress is not a stationarity certificate.

Both higher-precision `eps_s` controls passed unchanged search thresholds at the
original points. This supports a bounded precision retry before holding a direction;
it does not establish smoothness at every step size or parameter identifiability.

## What changes when a parameter is held

Write the full vector as `(x_A, x_H)`, with `A` the active coordinates and `H` held
at the incumbent. The subproblem minimizes `||r(x_A, incumbent_H)||² / n`.
Its gradient is `2 J_A.T r / n`; its local least-squares matrix is `J_A.T J_A`.
The held components of the full gradient need not be zero.

Active-step scaling, projected gradients, feasible steps, local model predictions,
and active stationarity checks use `A` only. Held derivative columns may be
unavailable and must not be replaced by fabricated zero derivatives. The full
parameter vector, original domains, state/integration validity and independent
rollout verification remain mandatory. Held values remain exact in optimizer
coordinates throughout a segment. In a reduced coordinate system, a held parameter
combination does not necessarily hold each original coefficient separately.

No active-only stopping result certifies full-model stationarity, full parameter
rank, or identifiability. Those require a separate assessment of all relevant
directions. A complete coordinate audit may record an explicitly failed held
direction: that failure does not prohibit optimizing the qualified active block.
Missing records remain missing evidence, rather than successful or failed checks.

## Lazy derivatives

Residual trial evaluations compute only predictions and residuals. The optimizer
requests a Jacobian separately at points where it needs derivatives. Useful finite
trial checkpoints remain available even if a later derivative calculation fails,
but independent original-equation verification is required before retention.

Initial scaling and accepted-point derivative checks still run. Residual-only
telemetry has no gradient estimate; it cannot supply a stationarity or confidence
claim. Numerical invalid trials, bounds, deadlines, accepted-step histories and
cache identities retain their explicit semantics. Unsupported profiled inner
problems are rejected by this opt-in path. Historical default search behavior
and previous result roots remain unchanged.

This opt-in evaluator requires the Jacobian call to return exactly the same
deterministic/cached residual as the residual-only call. It rejects inconsistent
baselines instead of pairing a residual with derivatives from another numerical
solve. Supporting approximate agreement from separate augmented solves needs its
own error contract. The matched eager control uses the same corrected
accepted-point bookkeeping as the lazy arm; it is not a replay of historical
trial-point stationarity checks.

## Controlled pilot

The same three original starts each receive six arms:

1. Original coordinates, all parameters, eager derivatives, one 1800-second segment.
2. Original coordinates, all parameters, lazy derivatives, one 1800-second segment.
3. Reduced coordinates, all parameters, eager derivatives, one 1800-second segment.
4. Reduced coordinates, all parameters, lazy derivatives, one 1800-second segment.
5. Reduced coordinates, all parameters, lazy derivatives, three 600-second segments.
6. Reduced coordinates, adaptive active blocks, lazy derivatives, three 600-second segments.

Pairs 1–2 and 3–4 isolate derivative scheduling; 2–4 compare coordinates; 4–5
compare segmentation; 5–6 compare adaptive holding. Starts, equations, data,
domains and total search ceilings remain fixed. Actual search time, unused budget,
derivative costs and verification costs are reported separately. A blocked arm is
reported as blocked, not silently given a different start or numerical policy.

Six shared qualification tasks precede fitting: one for each start and coordinate
system. Each first uses the baseline precision. When needed, one separately charged
full-coordinate tighter pass is available, avoiding mixed-policy derivative evidence.
Compared arms use the same selected initial qualification and numerical policy.
Changed endpoints receive their own checks; evidence is never transferred between
different parameter points. Diagnostic checks have separate time limits and do not
consume or renew optimization allowances.

Three additional tasks assess the interrupted generic endpoint's coordinates 5–7.
These are explicitly new diagnostic allowances; old consumed receipts remain in the
source history. No fitting is repeated to finish those checks.

## Execution and recovery

The 27-task roster comprises six qualification tasks, eighteen fitting tasks and
three diagnostic supplements. Delta uses one CPU and 8 GB per task, at most two
array elements concurrently. Qualification has a two-hour scheduler ceiling;
fitting/diagnostic workers have four hours. The report has thirty minutes. These
ceilings include startup and audit overhead and do not enlarge numerical allowances.
Workers log startup before Python imports. Invocation timings distinguish startup,
numerical work and orchestration when available.

Use the delivered source/input bundle in a new extracted code directory. Set:

```bash
export AF_LLMSR_DETENTION_LAZY_CONTROLS_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-lazy-controls-v1
bash scripts/hpc/start_llm_sr_detention_lazy_controls_delta.sh run
```

Qualification is submitted first; fitting depends on its completion. The runtime
also verifies each task's own qualification before allowing search. Ambiguous
scheduler replies block duplicate submission until reconciled. Every operation
has an immutable identity and nonrenewable reservation. Completed work is reused;
interrupted allowances are never silently reset. Source history is authenticated
once per invocation rather than repeatedly inside the optimization loop.

From the same extracted checkout:

```bash
bash scripts/hpc/start_llm_sr_detention_lazy_controls_delta.sh status
bash scripts/hpc/start_llm_sr_detention_lazy_controls_delta.sh inspect
```

Inspection writes `llm-sr-detention-lazy-controls-v1-inspection.tar.gz` beside the
result root. It does not fit, submit jobs or renew allowances. No GPU, LLM call,
validation/test access, scientific model promotion, or ground-truth coefficient
selection is involved. New result collection is required before claiming speedup
on these detention models or improved fitting reliability.
