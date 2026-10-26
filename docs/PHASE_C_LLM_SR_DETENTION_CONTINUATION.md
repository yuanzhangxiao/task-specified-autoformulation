# LLM-SR detention: continuation and exact parameter reduction

Date: 2026-10-26. This is an opt-in, training-only numerical experiment.

## Returned results and consistency follow-up

All seven tasks completed. Model 14's temporary subset/release reached training
NMSE 8.47e-6, and Model 15's original-coordinate perturbed-0 continuation reached
9.53e-5. All three reduced searches were blocked before optimization. A center
versus neighbor reconstruction inconsistency was identified; the correction,
24-direction saved-point audit, and proposed adaptive block strategy are in
[the consistency follow-up](PHASE_C_LLM_SR_DETENTION_CHART_AUDIT.md). The sections
below preserve the original continuation protocol.

## Evidence motivating the experiment

The six-task diagnostic recovery completed all 139 operation records. Its report
reproduces from the saved receipts under the delivered source bundle. All six
retained endpoints agree between independently integrated Radau and DOP853
rollouts. There were no diagnostic timeouts.

The two previously accurate endpoints remain at training NMSE 8.48295e-6
(Model 14) and 6.81784e-5 (Model 15); they received diagnosis only. Three formerly
blocked Model 15 starts entered optimization. Their training errors changed
from 75.8537 to 3.02618, 3.13107 to 2.99161, and 83.9158 to 0.799478. All three
used their 600-second search allowances. Their final improving base-point
checkpoints survived unfinished derivative calculations and passed independent
verification. Model 14's perturbed start remained at 4.73631 because its
`h_max` derivative did not qualify; its other seven coordinate checks passed.

Every retained point has a locally checked, feasible coordinate with a gradient
larger than its empirical numerical uncertainty and the stopping threshold.
These results do not establish the best attainable error of either equation
family. Approximately 80% of this diagnostic follow-up's summed operation time
was spent on qualification and diagnosis, rather than optimization.

## Frozen comparison

The next roster has seven tasks:

| Parent recovery task | Model/start | New work |
|---|---|---|
| 3 | 15 / perturbed 0 | Original coordinates and reduced coordinates |
| 4 | 15 / perturbed 1 | Original coordinates and reduced coordinates |
| 5 | 15 / generic 1 | Original coordinates and reduced coordinates |
| 2 | 14 / perturbed 0 | Seven-coordinate search, followed by conditional release |

Each arm starts from its own parent's independently verified retained endpoint.
The paired Model 15 arms receive identical 1,800-second search allowances and
the same data, equation family, working bounds, and initial observations. No
arm inherits another start's coefficients. Qualification and verification costs
are recorded separately; equal search allowances do not mean equal total cost
or equal numbers of optimizer evaluations. A failed qualification remains a
reported outcome, not an omitted comparison.

Authenticated original-coordinate derivative checks are reused at the identical
saved endpoint when their parameter meaning and policy are unchanged. The
generic-1 original-coordinate arm currently has an unresolved `eps_s` check;
if it remains blocked, its unused search allowance is reported explicitly.
Reduced-coordinate checks are performed anew because their derivatives differ.

Model 14 holds `h_max` at that parent's value during its initial seven-coordinate
search. This is a temporary working subset, not pruning or a claim that `h_max`
is scientifically irrelevant. After verification, all eight original
coordinates are checked again. A separate, bounded 600-second full-coordinate
search is allowed only if the new full check qualifies. Failure to qualify
retains the verified checkpoint without a full-model convergence claim.

The two accurate diagnosis-only endpoints are retained in historical evidence;
this experiment does not spend another search budget on them. All earlier
outcomes, including failures and their costs, remain part of the portable input.

## Preserving the reduced model family

For Model 15, let `g=k_gw`, `l=k_loss`, `k=kappa`, `H=h_max`, and `T=dt`.
The four original coefficients enter the vector field through three combinations:

```text
b = g - l
d = g * k
M = H + T * g
```

Together with `C`, `beta`, `eps_s`, `Q_max`, and `T`, these give eight effective
coordinates. The final guards preserve this identity. Nine original
coefficients therefore cannot be uniquely identified even with ideal numerical
precision. Additional uncertainty can occur when training does not excite a
branch.

The feasible combinations are correlated. An independent box around `b,d,M`
would change the equation family. The reduced arm instead uses a bounded
conditional parameterization:

1. Choose `b` within the exact attainable interval for `g-l`.
2. Intersect the original `g` interval with the interval implied by `b` and the
   original loss bounds.
3. Choose `M` within the interval attainable from that `g` interval, the original
   `H` bounds, and the positive `T`. Restrict the feasible `g` interval further
   using the selected `M` and the original `H` bounds.
4. Choose `d` within the products of that remaining `g` interval and the
   nonnegative `kappa` interval.
5. Lift each selected combination back to an original bounded parameter vector
   and evaluate the unchanged original equations.

Fractional coordinates encode `b` and the two conditional intervals for `d` and
`M`.
They do not make the physical combinations independent. Zero products, signed
gains, interval boundaries, and collapsed intervals require explicit handling.
Numerical scaling and derivative checks apply to the actual optimizer
coordinates. Final retention always requires independent integration of the
lifted original model. A reduced-coordinate result does not identify a unique
original coefficient vector.

The reduced chart starts its shrinking derivative-probe ladder at at most
`1e-5` of the local coordinate scale, while its optimizer stencil uses `2e-6`.
This avoids spending the entire check allowance on excessive truncation error;
the acceptance tolerances are unchanged. Original-coordinate checks retain
their previous policy. On a representative saved endpoint, the reduced
capacity-direction check passed in 99 seconds; its stronger gradient-precision
condition remained unqualified and is reported separately.

## More informative diagnostic decisions

New reports distinguish three outcomes:

- **Qualified local stationarity:** every required feasible direction has
  adequate numerical precision and the projected gradient is sufficiently small.
- **Reliable descent remains:** at least one locally passed, feasible direction
  has gradient magnitude greater than its empirical error envelope plus the
  stopping tolerance. Full rank information is not required for this conclusion.
- **Numerically unresolved:** available evidence supports neither conclusion.

At an active bound, a gradient pointing outside the allowed domain is not a
feasible descent direction. The checks use the original parameter bounds and
the exact retained endpoint. A statement about the temporary seven-coordinate
subproblem never becomes a statement about all eight parameters.

Raw historical diagnoses remain unchanged. The new assessment is separate,
authenticated derived metadata. Reports preserve the numerical rank interval
and separately show its intersection with any proven algebraic rank ceiling.
A contradiction is reported rather than rounded into a confidence claim.
Finite-difference error envelopes are empirical numerical evidence, not
statistical confidence intervals or rigorous bounds on the global optimum.

## Execution and interpretation

Use a new campaign root, one CPU and 8 GB per array task, at most two concurrent
tasks. No GPU, LLM, validation, or test data are needed. The portable input
contains the authenticated recovery history, so the old Delta directories are
not needed. Every operation has a nonrenewable reservation and a durable result;
resuming reuses completed records and preserves interrupted work and charges.
Reporting and inspection perform no fitting.

These are fixed LLM-SR numerical controls with one observed depth state. They
test numerical optimization, not latent-state recovery or mechanism compliance.
The supplied coefficients are references, not ground truth. Production fitting
defaults and construction policies are unchanged.

After extracting the delivered bundle on Delta:

```bash
export AF_LLMSR_DETENTION_CONTINUATION_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-continuation-v1
bash scripts/hpc/start_llm_sr_detention_continuation_delta.sh run
```

Use the same bundle and root for status and packaging:

```bash
bash scripts/hpc/start_llm_sr_detention_continuation_delta.sh status
bash scripts/hpc/start_llm_sr_detention_continuation_delta.sh inspect
```

The inspection archive is written beside the campaign root. Download from a
workstation terminal after completion. Exact source revision and validation
results accompany the delivery bundle.
