# Candidate 15 direct-meal coefficient control — numerical handoff to Orion

The matched Delta fits completed. The nested child lowers full-training NMSE from
0.4238668851085676 to 0.2610107571537155, while all three prespecified shifted
90 g meals still produce insulin peaks 91 minutes early. This qualifies the
separate construction revision under the predeclared routing rule. No proposer
call or revised-model fit has run as part of this returned campaign.

## Provenance and operational audit

- Frozen implementation: `9d7f5e3195b89bea14c4983e2bee91cddcc44dcf`.
- Delta fitting array: `23575634` (parent 0, child 1), report `23575635`.
- Remote campaign: `/work/hdd/bibo/yxiao2/phase_c/insulin-meal-control-v1`.
- Returned archive SHA-256: `a522283dd246668cc37a3aa70bb83664e72d484ea812ba25b1b2d4edfda8f3ac`.
- Plan SHA-256: `c2ac4c6e173950ae587f1bd36e4123ba5684ba3b7ed849f1b970ee174929e8a1`.
- Public training SHA-256: `169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795`.
- Local results: `output/insulin-meal-control-analysis-20261105/returned`.
- Visual report: `output/insulin-meal-control-analysis-20261105/REPORT.html`.
- Actual fitted-child request: `output/insulin-meal-control-analysis-20261105/PROMPTS.html`.

All 32 sealed JSON artifacts and the operation/terminal chains reconciled.
The returned stderr files are empty. Both searches terminated successfully,
with finite feasible evaluated terminal points and no failed objective calls
or exhausted allowance. Parent used 170 calls and 206.3 numerical-active seconds.
Child used 1,418 calls and 1,697.8 numerical-active seconds. Both received the same
1,800-second / 2,000-call allowance. Parent retained the original warm endpoint
because its searched improvement of about 1.29e-11 was below the 4.24e-9 guard.

The parent and child inner bounded solves were full rank (34 and 35 columns),
with conditional KKT residuals around 1e-13. Original-equation Radau replay of the
fixed retained parameters independently returned losses 0.4238668851085678 and
0.2610107571549462. The largest discrepancy from saved loss is 1.23e-12.
This does not certify outer stationarity, global optimality, or identifiability.

## Results and interpretation

The only change to the model family is a nonnegative shared direct-ingestion
coefficient in the insulin equation, fixed at one in the parent. The child
fits it to 3.1141617364718406. Both fits also optimize the other shared dynamics
and profile training initial states, so the final improvement is not an isolated
fixed-parameter gain intervention.

| Quantity | Parent | Child |
| --- | ---: | ---: |
| Full-training NMSE objective | 0.423867 | 0.261011 |
| Trajectory NMSE median [unscaled MAD], n=16 | 0.303295 [0.258858] | 0.185688 [0.166996] |
| Glucose NMSE, pooled | 0.193199 | 0.176912 |
| Insulin NMSE, pooled | 0.734590 | 0.404829 |
| Utilization NMSE, pooled | 0.343811 | 0.201291 |
| Glucose trajectory NMSE median [MAD] | 0.157542 [0.141345] | 0.136450 [0.117924] |
| Insulin trajectory NMSE median [MAD] | 0.564256 [0.433491] | 0.263457 [0.258264] |
| Utilization trajectory NMSE median [MAD] | 0.186797 [0.147011] | 0.114051 [0.091912] |

The objective drops 38.4%. Fourteen of sixteen trajectory losses improve.
No-meal glucose median NMSE rises slightly from 0.013969 to 0.014256, while
no-meal insulin and utilization medians improve. The report contains all five
no-meal controls, the five doses, all single-meal schedules, and channel errors.

| 90 g schedule | Observed peak minute | Parent peak minute | Child peak minute | Observed height | Parent height | Child height |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_003, meal 0–10 | 101 | 10 | 10 | 301.212 | 114.333 | 296.308 |
| train_006, meal 60–70 | 161 | 70 | 70 | 301.212 | 115.828 | 294.728 |
| train_007, meal 120–130 | 221 | 130 | 130 | 301.212 | 116.894 | 293.931 |

All three witnesses have zero external insulin and unique sampled peaks.
The median absolute timing error remains 91 [0] minutes. The three-schedule
median absolute height error improves from 185.383 [1.065] to 6.484 [0.797].
The smaller/larger start-zero doses also still peak at minute 10, whereas the
observations peak at 95–111 minutes. Improved height does not recover the shape.

## Equation audit at the retained child

Writing r for supplied meal rate, v for supplied external insulin, X for
`X_insulin_action`, and Q for the latent generated `Uii` alias, the assembled
fitted equations are approximately

```text
Gp' = r - U + 0.00205851347 A
 A' = 24.6376953 r - 0.00205851347 A
 I' = 3.11416174 r + 0.000975926039 Gp - 0.00717575139 I + 2.90478483 v
 X' = 0.000513988269 I - 0.0213261928 X
 Q' = 0
 U  = Q + X
```

The displayed insulin clearance is the assembled `-I - insulin_decay` term.
Its fitted decay-process coefficient is negative, but the complete coefficient
of I is negative. Do not infer growth from that isolated parameter name.

A separate diagnostic starts at observed insulin at meal completion (27.556),
uses interpolated observed glucose with the child's fitted coefficients, and
integrates to the observed insulin peak. It predicts 32.993 versus observed
301.212 on each shifted schedule. Thus inaccurate glucose prediction alone
does not explain the remaining mismatch at these fitted coefficients. This is
conditional evidence, not a proof that every fit of this equation family fails.
The current I timescale is about 139.4 minutes and A timescale about 485.8 minutes.
Neither is established as a physiological estimate.

## Division of the next work

Codex has prepared the existing one-call 120B construction request from the
verified child. It can replace I's RHS or `insulin_production`, with at most one
connected extra state and one connected process, causal initialization, and up
to eight shared parameters. The response undergoes the existing deterministic
compilation, public-path, connectivity and qualified training-consistency checks.
No new scientific rejection based solely on the measured peak error is added.

Orion should retain the child endpoint as the numerical reference and review
any remaining optimization limitations without delaying the already-qualified
construction experiment. Once an eligible revision exists, qualify a backend
for its actual equations and initializer scope, then fit it under the matched
training contract. A nonlinear revision must not be forced through this affine
profiler. Use `compare_refit_replay` in `scripts/propose_meal_insulin_revision.py`
with the independently verified original-equation residual to compare all three
channels, peak timing/height, dose behavior and no-meal controls with this child.

The authoritative child coefficients are in
`results/child/result.json` → `value.retained_endpoint.parameters`.
Its model family is defined by the original input problem plus the pinned
`child_problem` transformation, with model hash
`d07ecb5745cb440dc6369640623becd289bf874c4ae21911d78c3203b8b04c4b`.
The public data and original source endpoint are in `inputs.json`.

All reported fits reconstruct the 16 training trajectories using initial states
estimated from complete training trajectories. No validation/test forecasting
claim follows. Preserve the fixed eight-unit observation-prefix contract for a
later held-out evaluation. No test data was opened in this review.
