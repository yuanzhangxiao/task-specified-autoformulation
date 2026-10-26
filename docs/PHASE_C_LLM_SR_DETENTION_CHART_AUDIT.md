# Detention reduced-coordinate consistency and saved-point audit

Date: 2026-10-26. Opt-in numerical qualification using training data only.

## Returned continuation results

The seven-task continuation completed 133 operation records. Its report
reconstructs exactly from the saved receipts under the delivered source.
All retained endpoints passed independent Radau/DOP853 verification.

| Model/start | Starting training NMSE | Retained training NMSE | Work performed |
|---|---:|---:|---|
| 14, perturbed 0 | 4.736310 | 8.472941e-6 | Seven-coordinate search, then full release |
| 15, perturbed 0 | 3.026179 | 9.528051e-5 | Original-coordinate continuation |
| 15, perturbed 1 | 2.991611 | 0.2129653 | Original-coordinate continuation |
| 15, generic 1 | 0.799478 | 0.799478 | Original-coordinate search blocked |

All three reduced-coordinate searches were blocked before optimization. No pair
had both optimizers start, so this experiment does not rank their optimization
performance. All seven endpoint diagnoses retained evidence of feasible descent;
none certified full local stationarity or coefficient uniqueness.

Model 14's seven-coordinate search alone reached 8.476098e-6. The full derivative
recheck then qualified, permitting release of `h_max`; another 600 seconds changed
NMSE by only about 0.037%. The final `h_max` check was again unresolved for strong
conclusions. This is evidence that a temporary restriction can enable useful
progress, not that a parameter is permanently irrelevant.

The continuation used 14,802 summed operation wall-seconds, of which about 59%
was outside optimization. Four 1,800-second search allowances went unused.
Historical costs, failed checks, and unused allowances must remain visible.

## The numerical consistency defect

The original reduced chart returned the exact original parameter vector at the
exact starting coordinate. Every neighboring coordinate instead reconstructed
parameters through rounded fractional coordinates. Varying only `Q_max` could
therefore also perturb `k_loss`, `kappa`, and `h_max` by tiny amounts. At a locally
inactive `Q_max`, the two neighboring residual vectors were identical, while the
center differed slightly. One-sided finite differences amplified that offset
as their steps shrank and could incorrectly suggest a nonsmooth response.

A four-rollout diagnostic at the saved perturbed-0 point reproduced the offset
locally. Holding the other original coefficients exactly unchanged removed it.
Its magnitude differed from the cluster's saved traces; the architectural source
of the discrepancy was reproducible. Other blocked directions still require
their own checks. Passing a zero-effect test does not prove all derivatives safe.

The correction is a separate `reduced_model15_consistent` mode. Historical
`reduced_model15` behavior and identities remain available for reconstruction.
The new chart uses an exact rational, monotone transformation of each bounded
fraction so that the representable starting coordinate maps to its exact
starting physical combination, with the same rule at the center and neighbors.
The correlated feasible family and original bounds are preserved. It does not
fix an arbitrary groundwater gain or replace the feasible combinations by an
independent box.

For an interior floating-point coordinate anchor p and its exact physical
fraction q, the transformation is

```text
K = q (1-p) / [p (1-q)]
w(t) = K t / [1-t + K t]
```

It maps 0 to 0, 1 to 1, and p to q, with an exact inverse and no new kink at p.
Endpoint and collapsed-interval cases are handled separately. The conditional
order remains b, then M, then d. This construction removes the center-only
reconstruction exception; it does not remove piecewise boundaries already
present in the feasible domain or the original equation.

The original reduced probe settings, acceptance thresholds, and coordinate
scales are retained to isolate the consistency correction. The new identity
records its transformation. Near exceptional endpoint anchors, these inherited
scales need not equal exact physical-unit scaling after transformation; numerical
checks still govern eligibility. Smaller probes alone are not a remedy for
integration or reconstruction noise.

Local checks at all three saved anchors preserved every unrelated coefficient
bitwise in 48 one-coordinate neighbor checks. A bounded replay of the previously
blocked perturbed-0 `Q_max` direction then produced identical center and neighbor
residuals. Its 33-second corrected audit met the search-uncertainty threshold
without changing any acceptance tolerance. Its zero derivative remained
inconclusive for stronger parameter-information claims. This checks one
direction, not the full chart; the cluster audit below is still required.

## The next experiment

Freeze all eight corrected reduced coordinates at each of the three saved
reduced-arm starting vectors: 24 independent coordinate checks. Each receives
at most 120 seconds of numerical work. There is no optimization, coefficient
selection, equation revision, new data generation, validation access, or test
access. Initial observations and the original scientific domains stay fixed.

The portable input preserves the complete authenticated continuation history.
New reports compare corrected qualification with the historical checks and
retain their separate costs. This is a historical diagnostic comparison, not
a fresh paired timing benchmark. Each coordinate is checkpointed and an
interrupted reservation cannot silently acquire a new allowance.

Aggregate qualification requires evidence for all eight coordinates at the
same exact endpoint, chart, data, scales, policy, and integration precision.
Reports enumerate every row that is not search-qualified, including inconclusive
rows whose error exceeds the search allowance. A list of explicitly failed
indices alone is insufficient. Missing records remain missing, not passed.

After this audit, qualified starts may enter a separately frozen optimization
comparison from their original paired checkpoints. Unqualified directions need
targeted investigation or an explicitly bounded temporary-subset experiment;
there is no automatic relaxation of derivative tolerances or model promotion.

## Generalization: reliability-guided optimization blocks

The Model 14 result motivates an adaptive controller, not yet a production
algorithm. Let x collect continuous model coefficients and any permitted fitted
initialization variables, and let L(x) be the fixed training rollout objective.
At a verified incumbent, divide coordinates into an active set A and a temporarily
held set F using numerical derivative evidence. One bounded step solves

```text
minimize L(x_A, incumbent_F) over the original bounds on x_A.
```

Retain a checkpoint only after original-model rollout verification. Then recheck
the full coordinate set at that exact retained point. Release newly qualified
coordinates, and record whether the active set or numerical precision changed.
Rechecking can be triggered by useful progress, a plateau, or a fixed maximum
number of accepted steps; held coordinates must not remain forgotten indefinitely.
Search, derivative checks, and verification all count toward separate declared
budgets and a total cost ledger.

The policy must distinguish:

- **Numerically unreliable derivatives:** candidate reasons for temporary
  restriction, local probe adjustment, or greater integration precision.
- **Weakly identified parameters:** uncertainty in what the data determine;
  holding a value does not make it identified or eliminate that uncertainty.
- **Exact redundant parameters:** remove only with a family-preserving algebraic
  reduction and correct feasible domains.
- **Reliable feasible descent:** use it for progress even if another direction
  remains unresolved, without asserting full-model convergence.
- **Poor fits with no trustworthy active direction:** report an unresolved
  fitting problem; do not attribute the error to the equation family.

Block improvement is not proof that the held parameters are wrong, right, or
irrelevant. Simultaneous changes may be required to leave a poor solution.
Always retain the current incumbent values for held coordinates, periodic full
rechecks, and the possibility of a genuinely different start. A full-space
stationarity claim requires full-space evidence; a good prediction can instead
be reported as accurate but incompletely optimized or weakly constrained.

## Delta execution

Use the delivered portable source/input bundle and a new root:

```bash
export AF_LLMSR_DETENTION_CHART_AUDIT_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-chart-audit-v1
bash scripts/hpc/start_llm_sr_detention_chart_audit_delta.sh run
```

The array uses one CPU and 8 GB per task, with two concurrent tasks and no GPU.
The scheduler wall allowance includes authentication and receipt overhead as
well as the 120-second numerical check. From the same extracted directory:

```bash
bash scripts/hpc/start_llm_sr_detention_chart_audit_delta.sh status
bash scripts/hpc/start_llm_sr_detention_chart_audit_delta.sh inspect
```

Inspection writes an archive beside the output root. No fitting or remote
submission occurs while reporting or inspecting. Production defaults are unchanged.
