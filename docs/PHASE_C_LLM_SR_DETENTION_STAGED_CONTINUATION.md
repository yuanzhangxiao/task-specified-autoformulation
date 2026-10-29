# Continued detention polishing from the verified staged endpoints

Date: 2026-10-28. Opt-in, two-task CPU experiment using training data only.

## Returned evidence

The returned staged-polishing report reconstructs exactly from its operation
receipts. Saved residuals reproduce both losses, and all 17 saved gradient
coordinates agree with recomputation to floating-point precision.

| Start and coordinates | Previous training NMSE | Retained training NMSE | Accepted steps |
|---|---:|---:|---:|
| `perturbed_0`, original | 4.2763648676e-5 | 3.5854978934e-6 | 11 |
| `generic_1`, reduced | 2.1334319661e-5 | 3.1999016188e-6 | 12 |

Independent Radau/DOP853 verification differs by 1.05e-15 and 5.75e-15 in loss.
Both retained endpoints are their own best and last accepted checkpoints. Each
search exhausted its 1,800-second allowance during a later Jacobian calculation;
the useful preceding endpoint survived. The final accepted steps still reduced
loss by approximately 12.2% and 1.2%, respectively.

Reliable feasible descent remains at both endpoints. The original-coordinate
`kappa` gradient is -0.00662553 with empirical error 4.57e-8; the reduced-coordinate
`b_fraction` gradient is 0.0294959 with error 3.43e-6. These chart-dependent values
should not be compared as measures of difficulty across the two starts.
Both full search-quality audits completed, and both strict audits were correctly
skipped. Completion is not convergence or coefficient confidence.

Observed endpoint audit wall time was 436.66 and 433.45 seconds, about 7.25 minutes
each. Combined time was 44.9% lower than in the preceding polishing experiment,
but differing endpoints and machine conditions prevent attributing the whole
difference to staging. Strict allowances that were not reserved are not measured
time savings. Long startup waits were also visible; their cause is not established
by the numerical archive. Both tasks nevertheless completed.

Returned archive SHA256:
`3035bf557fdc9794cc511622b40d4094a6e96086d27fc9da7c5602dee1444771`.
Source report content SHA256:
`2e367f471fe781f2dd3461e48f6e9e51af27590c74944aff5638e69486e6bb00`.
Source report file SHA256:
`e84c3863bfcaca65c1fba8cc2bc9c6411fa259865361fca2c0aad6159f67a9ac`.

## Approved next experiment

Give each of these two endpoints one additional uninterrupted 1,800-second
search using the existing numerical algorithm and staged endpoint checks.
Keep equations, training data, initializers, parameter domains, coordinate
charts, solver settings, scaling and acceptance thresholds fixed. There is no
new random start, cross-start substitution or held-out selection.

The new `llm-sr-detention-staged-polishing-input-2` package authenticates these
exact returned endpoints, their independent verification and complete matching
residual/Jacobian qualification. Export reconstructs the original report before
extracting the package. Its content SHA256 is
`5606f66b19db3ba14fcd5d93a0f55b70c4c4720d1e51e6546d01548822964737`.
The old input profile and pins remain supported for historical report replay.
No historical receipt or spent allowance is changed.

Execution uses the compact package without reading prior campaign directories.
The first residual and active Jacobian columns are reused only at their exact
authenticated point and integration settings. Subsequent points require fresh
evaluation. This is checkpoint-based continuation with newly initialized optimizer
bookkeeping, not restoration of an interrupted trust-region internal state.

The existing sequence is unchanged:

1. One nonrenewable 1,800-second search per start, checkpointing finite improving
   base rollouts before their Jacobians are computed.
2. Independent original-equation verification with a 120-second allowance.
   Failed or worse candidates leave that start's verified incumbent intact.
3. Search-quality checks of every chart coordinate, at most 360 seconds each.
4. Strict checks, with separately recorded 360-second coordinate allowances,
   only when complete qualified evidence permits the existing escalation rule.

The maximum possible new numerical allowances remain 8,400 and 7,680 seconds.
Resuming an interrupted operation consumes its existing reservation; it does
not restart the clock. Repeating a completed launch/run reuses recorded jobs or
results. Only this explicitly approved new root receives new search allowances.
Historical scopes stay separately labeled, and nested timing records are not
summed as independent costs.

Afterward, compare incremental loss reduction, accepted-step progress and
measured compute cost against this immediate predecessor. If reliable descent
remains but improvement becomes small, reconsider allocation rather than
automatically starting another segment. No guaranteed convergence, unique
coefficients, scientific mechanism validity or held-out accuracy is claimed.

## Delta commands

Extract the supplied versioned bundle into a new persistent project directory
and check `SHA256SUMS`. The bundle includes the compact input. Run:

```bash
export AF_LLMSR_DETENTION_STAGED_CONTINUATION_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-staged-continuation-v1
bash scripts/hpc/start_llm_sr_detention_staged_continuation_delta.sh run
```

The dedicated wrapper overrides inherited staged-polishing root/input variables
and preserves the old campaign. Two tasks may run concurrently, each requesting
one CPU and 8 GB; no GPU or LLM is used. A four-hour scheduler ceiling covers
the optional diagnostic branch and I/O overhead without enlarging the thirty-minute
search allowance. The report job has a thirty-minute ceiling.

Keep this code directory for queued jobs and later inspection. From it:

```bash
bash scripts/hpc/start_llm_sr_detention_staged_continuation_delta.sh status
bash scripts/hpc/start_llm_sr_detention_staged_continuation_delta.sh inspect
```

Inspection creates
`/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-staged-continuation-v1-inspection.tar.gz`.
Confirmed scheduler receipts prevent blind duplicate submissions. After an
ambiguous scheduler response, inspect the saved intent and Slurm state before
attempting recovery. A disconnected terminal does not imply accepted jobs stopped.

## Verification

Verification covers both input profiles, receipt reconstruction, exact endpoint
and raw-array reuse, rejection of altered or mismatched inputs, preservation of
historical costs, nonrenewable interrupted/completed execution, and wrapper
routing under stale environment variables. Saved-endpoint smoke checks perform
no integration or real local refitting. Detailed test results and checksums
accompany the delivery bundle.
