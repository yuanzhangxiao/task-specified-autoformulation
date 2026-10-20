# M45: reliable verification of saved prefix initializations

Implemented 19 October 2026. This is the execution-reliability follow-up to
[M44](PHASE_C_FITTING_PREFIX_WINDOW_SELECTION.md), before multiple-root and
construction-export fitting qualification. It changes no historical kernel,
benchmark, finalized prompt, production default, or prefix-duration selection rule.

## Why this milestone

M44's six training fits were verified. Four noisy duration sweeps completed;
two clean sweeps had three numerical verification timeouts. Synchronous checkpoint
writes, including filesystem synchronization, ran inside five-second numerical
deadlines. The large wall/CPU discrepancy showed waiting, but did not prove whether
storage or scheduling caused each delay. Failures must remain failures in M44.

Each unsuccessful inference saved its best root candidate. M45 verifies those
exact candidates without repeating optimization, fitting shared parameters, or
using a longer prefix. One candidate is approximately 0.650510285: its search also
stopped early. M45 deliberately keeps that value instead of silently improving it.
Whether its forecasts meet the original tolerance is decided by the unchanged rule.

## Exactly what runs

The source is a terminal M44 campaign. Preparation validates all source seals,
operation receipts, saved arrays, scores and selection rules, then preserves a
byte-identical portable snapshot in a separate directory. Selection for recovery
uses missing verification only, never evaluator-only root errors or desired scores.
A missing saved candidate is an explicit unsupported case, not permission to refit.

For the returned Delta archive the new work is:

| Source task | Additional verification |
|---|---|
| `clean_r0_s0` | Prefix candidate at duration 0.2, `val_0`, start 1.1; its blocked forecast; failed forecast at duration 2, `val_3`, start 0.5 |
| `clean_r0_s1` | Prefix candidate at duration 1, `val_3`, start 1.1; its blocked forecast |

This is five bounded operations, each using DOP853 and Radau: at most ten new
integrator calls. The six fitted shared parameter sets, all training roots and all
successful source endpoints remain fixed. The four noisy tasks receive no numerical
rerun. No optimizer, LLM, test data or new ground-truth evaluator is called.

Inference receives only the original observed prefix and frozen training fit.
Forecasting receives the inferred roots and target-free schedule. The report scores
predictions against the same validation arrays, selects starts by prefix loss, and
requires the original complete duration/row/start grid. All durations are compared
on the common late horizon `t > 2`; the operational NMSE and start-spread limits
remain 0.01. Validation is tuning data, not an independent performance estimate.

Historical information diagnostics remain attached to their original endpoint.
If the selected inference receipt changes, the old diagnostic is preserved separately
and its applicability to the new receipt is marked unavailable. No information value
is silently transferred to a different estimate. Information is advisory, not a gate
for duration choice.

## Timing and checkpoint contract

`prefix_verification.py` implements an isolated verification path:

- The numerical-active allowance remains five seconds per two-integrator operation.
  It includes setup and all non-callback elapsed wall time; descheduling remains
  charged. Only directly measured synchronous checkpoint time is excluded.
- A separate 60-second overall wall deadline includes those callbacks. The runtime
  checks before and after callbacks and numerical calls, and passes each integration
  the minimum of its remaining active allowance and the overall deadline.
- These are cooperative deadlines. A blocked filesystem call cannot be preempted
  by this Python timer. When it returns after expiry, no further integration runs
  and the result cannot qualify. Slurm separately caps the task allocation.
- CPU time, overall wall time, numerical-call time and callback time are recorded.
  Timing subcategories are nested, not extra costs to add to the total.
- `attempted` counts a pre-integration checkpoint reservation; `started` counts an
  actual integrator invocation; `completed` counts a returned invocation, which
  can still fail numerical agreement; `failed` counts an invocation that raises.

The source uses the older pre-checkpoint meaning of `started`. Its counters remain
unchanged. The combined bookkeeping sum is explicitly named `mixed_start_counter`,
not reported as an exact physical-integration count. Supplemental counters have
separate attempted/started/completed/failed meanings.

Every attempt has durable identity, progress and terminal receipts. Completed work
is reused exactly. An interrupted attempt remains charged and is not automatically
retried; abrupt termination leaves unknown totals and observed lower bounds.
Measured partial timing snapshots never become exact totals. New policy/code/runtime
requires a new campaign identity. The original source directory is never written.

Reported operation cost includes setup, numerical work and progress callbacks.
Source auditing/copying, final receipt sealing, report writing and scheduler waits
are outside that cost scope. The snapshot retains historical failed attempts and
costs, so the supplement never resets the original budget or relabels M44 as a pass.

## Delta commands

Upload `phase-c-fitting-m45.tar.gz` into the directory in `AF_BUNDLE`. The existing
M44 source stays in place; no additional experiment data download is needed.

```bash
bash <<'BASH'
set -euo pipefail
AF_BUNDLE=/work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m45.tar.gz
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/prefix-verification-m45.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_CODE"
cd "$AF_CODE"
export AF_PREFIX_WINDOW_SOURCE=/work/hdd/bibo/yxiao2/phase_c/prefix-window-selection-v1
export AF_PREFIX_WINDOW_SUPPLEMENT_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-window-supplement-v1
export AF_ACCOUNT=bibo-delta-cpu
export AF_CONCURRENCY=2
bash scripts/hpc/submit_phase_c_prefix_window_supplement_delta.sh
BASH
```

The launcher prints source/output paths and progress before reading the source.
It uses the existing Delta Python/dependencies. The returned archive produces two
CPU tasks, one CPU and 16 GB each, concurrency two, with a 20-minute scheduler
limit. The limit covers verification, filesystem overhead and bookkeeping; it is
not a fitting target. Preparation runs the focused tests. There are no GPU jobs.
Uncertain scheduler replies stop for inspection rather than submitting duplicates.

After the report job completes:

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-window-supplement-v1
jq '{status, expected, recorded, status_counts, source_status_counts,
     recovered_forecast_gaps, selections, source_cost,
     supplemental_cost, combined_cost}' "$AF_ROOT/summary.json"
AF_REVIEW="$AF_ROOT/review-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_ROOT" \
  plan.json inputs.json portable_audit.json summary.json results source_snapshot
printf '%s\n' "$AF_REVIEW"
```

`status: complete` means all supplemental terminal records exist. Check
`status_counts`, recovered gaps and selections for numerical qualification.
An unsuccessful supplement remains explicit and never drops a failed start from
the denominator. Preserve the source snapshot when transferring the review archive.

## Exit criteria and next step

The local qualification must reproduce M44's numerical tolerances at fixed saved
roots, demonstrate slow-callback isolation and total-deadline failure, retain
interrupted ledgers, and preserve source/terminal artifact bytes across resume.
The remote gate is both clean sweeps qualified with the four noisy sweeps reused,
separate old/new costs and no new fitting. A local pass does not establish remote
filesystem behavior or universal prefix sufficiency.

After that gate, move to multiple independent latent roots and one actual
construction-exported model, with a frozen duration policy on new trajectories.
Uncertain initial observed readings, shared-parameter uncertainty, nonlinear
initializer robustness and immediate post-prefix forecasts remain separate work.

## Local replay of the returned M44 archive

The complete replacement Delta archive was used as the source. All 2,198 source
JSON artifacts included in the snapshot remain byte-identical. Both supplemental
tasks complete and all three forecast gaps recover using exactly the saved root
candidates. The clean regime selects duration 0.1; the noisy regime retains 0.2
without new numerical work. Both clean 0.1 mean NMSEs remain approximately
6.80e-21 because those forecasts were already verified in the source.

The supplement performs five operations, ten attempted/started/completed
integrator calls, zero failed integrations and zero search calls. Summed new
operation CPU is 1.051 seconds and wall is 1.089 seconds; measured callbacks take
0.018 wall seconds. These local timings do not predict Delta storage latency or
establish a speedup relative to the original remote run.

One selected inference receipt changes at clean start 1, duration 1, `val_3`.
Its old conditional-information diagnostic is retained as historical evidence,
with current applicability unavailable. This does not affect the 0.1 selection.
Both terminal resumes preserve all source and supplement JSON bytes and timestamps.
Artifacts are uncommitted under `output/phase-c-fitting-m45-smoke/`.

## Verification

The 250 focused initialization, duration-selection, timing, campaign and launcher
regressions pass. They include source preservation, independent integrator
agreement, slow checkpoint isolation, numerical/overall expiry, invalid shared or
prefix identities, retained-root substitution, partial-cost handling, interruption,
portable resume and scheduler uncertainty. Shell syntax, Python 3.11 compatibility,
standalone CLI help and `git diff --check` pass. Full-repository pytest is also run
before delivery. Ruff passes outside the unchanged 37 existing findings in
untracked `analysis/claude/` files; those are not part of this change.
