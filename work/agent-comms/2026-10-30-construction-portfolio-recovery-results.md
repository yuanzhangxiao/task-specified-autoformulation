# Returned construction portfolio recovery: forecasts restored, fitting still unresolved

30 October 2026. Orion reviewed the returned CPU campaign. All 23 terminal
records and summary rows reconstruct exactly from authenticated operation
receipts. All 15 affine models now have independently verified fresh-prefix
forecasts for all 16 training trajectories. None of the eight nonlinear models
is yet qualified for shared-parameter search. The recovery did not refit shared
parameters or change equations, data, domains, or the prefix policy.

## Evidence and scope

- Code: `d4555321a745f95c3c0a52fa7bf0d4eb7e9ee84e`.
- Delta root: `/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-recovery-v1`.
- Run array `23364086`; report `23364087`.
- Archive SHA256:
  `565d78d68bdb3e870437c1e46092ae743069c1e87429d78a1ac2140ffd9ab61e`.
- Recovery plan:
  `8f213fad2056c99cab1d00a971a24cc9fcd6e23453a9622d13e03b71089bd0bc`.
- Source plan:
  `8498d0d08631ed81543ef1f4a96eb3471b1017f295a06ccb72da7bbd02961725`.
- Generated local evidence: `output/construction-portfolio-recovery-review-20261030/`.
  `audit-replay.json` records exact reconstruction; `review-details.json` contains
  compact scores, failure diagnostics, coordinate receipts and timing totals.

These are 23 construction alternatives for one T2-easy parent, not 23 benchmark
cells. Shared coefficients were learned from the full training trajectories.
Fresh initial states use only the eight-unit prefix (nine samples), with scores
on the subsequent 292 samples. These are training diagnostics, not independent
validation or test estimates. No remote jobs, LLM calls or test access occurred
during this review. Small fixed-point affine checks performed no optimization.

## Affine recovery worked

All 240 prefix inferences and forecasts completed, up from 122/240 available
affine forecasts and five complete affine models. Independent forecast agreement
is within approximately `2.04e-9` in normalized coordinates. All 240 root problems
used bounded least squares successfully. Every retained shared endpoint matches
the preceding campaign exactly.

| Candidate | Full-trajectory reconstruction NMSE | Fresh-prefix suffix NMSE | Suffix Gp / I / U NMSE |
| --- | ---: | ---: | --- |
| 12 | 0.564509 | 3.144752 | 7.592 / 0.830 / 1.013 |
| 15 | 0.564608 | 3.122805 | 7.533 / 0.825 / 1.011 |
| 21 | 1.552123 | 2.824981 | 4.873 / 0.662 / 2.940 |
| 20 | 2.360880 | 2.429496 | 1.462 / 1.950 / 3.876 |
| 16 | 2.291426 | 2.512135 | 1.537 / 1.920 / 4.080 |

Reconstruction uses the previously jointly fitted trajectory initials and all
training times. The suffix column uses newly inferred prefix-only initials, so
the two columns differ in both information and scoring interval. Candidate 20
has the lowest complete suffix score among these affine fits; it is still poor.
There is no accurate demonstration model in these results.

### Why 12 and 15 lose their apparent advantage

Their absorption-state decay times are about 305 and 309 time units, much longer
than the eight-unit prefix. At the saved coefficients, the initial absorption
state affects the suffix much more strongly than the prefix. For candidate 12,
training trajectory 000:

- Full-trajectory-fitted absorption initial: approximately 112.60.
- Fresh prefix-optimal initial: approximately 3.83.
- Prefix NMSE improves from approximately 0.2864 to 0.2309.
- On the same suffix, NMSE deteriorates from approximately 0.4618 to 5.2638.

Candidate 15 shows the same tradeoff: prefix `0.2836 -> 0.2292`, suffix
`0.4539 -> 5.2372`. Both models place this root at its lower bound for ten of
sixteen fresh prefixes. Fixed-point audits of representative 12/15/19 rows
confirm affine residual identities near `1e-15` and projected prefix gradients
below `1e-16`; there is no observed bounded-LS or normalization mistake.

Thus the current model/coefficients prefer different roots under the prefix and
whole-trajectory objectives. The short window and slow state explain sensitivity
to this distinction, but do not prove the window is intrinsically insufficient.
Model mismatch and unconverged shared coefficients remain possible causes. Do
not lengthen the frozen prefix or seed it from future observations to hide the
gap. Future fitting must report both reconstruction and causal-prefix forecast
behavior, and predeclare any change to its training objective.

All root designs have numerical full rank under the ordinary rank test, which
does not establish useful initial-state information. Candidate 20 illustrates
this: condition number near `6.95e12`, with one filtered root's normalized prefix
sensitivity around `2.1e-14` on a checked row. That weak column also has tiny
suffix influence. This is not evidence that every root is well identified, nor
necessarily a dangerous correlated-root ambiguity.

## Nonlinear starts: stronger evidence, but a diagnostic scheduling defect

### Five starts fail under both implicit solvers

| Candidate | Saved start behavior on train_000 |
| --- | --- |
| 01 | Approaches `1 + Gp = 0` near time 3.738826. |
| 02 | Approaches `1.5 + Gp = 0` near time 13.585520. |
| 04 | Approaches `1.5 + I = 0` near time 5.577275. |
| 13 | Approaches `1.5 + Gp = 0` near time 3.119548. |
| 17 | Polynomial feedback produces enormous attempted states near time 0.912172. |

Both Radau and BDF terminate with a required-step-size failure. The four literal
denominator singularities occur in the proposed equations at these parameter
values. Last attempted Radau denominators are approximately `1.06e-7`, `4.25e-10`,
`1.83e-10`, and `4.14e-11`. Candidate 17 has no state denominator and reaches
attempted insulin magnitude about `1.68e49`. These diagnostics describe attempted
RHS evaluations, not accepted solver states or mathematical proofs of finite-time
blow-up. They support rejecting these starts as usable rollouts. They do not
prove no feasible parameters exist in the model family.

### Three models replay correctly, but no derivative direction is tested

| Candidate | Radau numerical seconds | BDF numerical seconds | Maximum normalized discrepancy |
| --- | ---: | ---: | ---: |
| 03 | 67.50 | 16.10 | 2.88e-8 |
| 10 | 60.05 | 14.02 | 4.17e-9 |
| 14 | 58.68 | 13.94 | 4.41e-9 |

All pass the unchanged `1e-7` agreement threshold. Their supplied starts still
have NMSEs around `1.28e5`; these are not optimized fitting results.

The derivative recovery has a budgeting/cache design error. Its reusable baseline
pair is stored only after the whole state-plus-sensitivity calculation succeeds.
Each coordinate's 60-second allowance must first build that same pair:

- 03 and 10 repeat 57 unsuccessful state-only baseline calculations; no Jacobian
  starts. The already completed independent replay is not reused in this cache.
- 14 completes its state baseline once, then repeats 29 unsuccessful full
  augmented/Jacobian calculations.
- All 86 coordinate operations check zero directions and run zero finite-
  difference stencils. Their outcome is unavailable evidence, not a measured
  derivative disagreement.

Coordinate operations spend about 5,192.5 summed wall seconds (1.44 hours),
94.8% of the nonlinear operation time, without obtaining a derivative verdict.
This limitation is in our diagnostic execution, not evidence against the models.
Caching alone is insufficient: a central two-scale check needs at least four
residual rollouts, already about 235--270 seconds with Radau or 56--64 seconds
with BDF before overhead at these starts.

There is also a reporting mismatch: nonlinear replay receipts use generic
`calls_started/completed`, while the cost aggregator reads verification-specific
keys. Generic summary verification counts therefore show zero. The authenticated
receipts establish 16 replay starts, six completions and ten failures. Wall/CPU
timings remain recorded; the counter normalization needs correction.

## Optimization status and next milestone

All 15 affine fits retain `optimization_unresolved`. Correcting historical budget
bookkeeping removes spurious reasons, but does not certify convergence. Candidate
15's nominal projected gradient is `1.49e-9`, while the precision-aware upper
bound remains `7.82e-4`; its retained point also lacks sufficient matching progress
history. Function-form blame from an allegedly solved fitting problem remains
premature.

Recommended next bounded milestone:

1. Journal state and sensitivity prerequisites separately. Reuse authenticated
   completed values and stop dependent coordinate work after a failed prerequisite.
   Normalize replay/backend call counters without rewriting historical receipts.
2. Test BDF and directional or block sensitivities on representative 03/10/14
   calculations, preserving reduced-coordinate lift derivatives, independent
   solver checks and local derivative thresholds. Reserve enough time for each
   actual primitive; do not repeat the current 60-second coordinate jobs.
3. For 12/15, remove remaining additive redundancies only where exact equations,
   feasible domains and parameter lifts can be certified, then run a small
   precision-qualified incumbent continuation. Track fresh-prefix forecasts
   alongside reconstruction at retained checkpoints. Do not silently replace
   the declared fitting objective or claim the prefix gap is solved by refitting.

The five invalid starts need a separately bounded feasibility-start recovery
before optimization, preserving the declared model and domains. A failed start
alone is not a reason to request a different function form from the proposer.
No production equations, fit objective or benchmark policy changed in this review.

## Cost and verification

Recorded operation costs: 7,928.26 summed wall seconds (2.20 hours), 7,344.51
CPU seconds (2.04 hours), including 445.42 seconds of synchronous callback wall
time. These totals exclude queue, source authentication, reporting and resumed
adapter construction; summed task time is not campaign elapsed time.

Exact receipt reconstruction passed for all 23 records; all source-retained
endpoints match the preceding archive. All 89 focused recovery, launcher,
affine-prefix and qualification tests passed. Repository-wide Ruff reports the
same 37 pre-existing unrelated analysis findings. No full-suite pass is claimed.
