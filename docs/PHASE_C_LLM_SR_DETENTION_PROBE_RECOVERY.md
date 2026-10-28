# Detention diagnostic recovery and incumbent continuation

Date: 2026-10-27. Opt-in numerical development on training data only.

The returned six-task run is complete. Its two continuations reached verified
training NMSEs 6.26811e-5 and 6.74663e-5 without establishing stationarity.
The [endpoint polishing follow-up](PHASE_C_LLM_SR_DETENTION_POLISHING.md) records
that result and the next bounded two-task experiment.

## Returned evidence

The 27-task lazy-controls report reconstructed exactly from its saved receipts.
All tasks terminated, but only six of the eighteen fitting arms optimized.
Twelve arms were blocked by initial derivative qualification. Terminal task
completion is not optimizer convergence or a successful numerical comparison.

| Start | Original eager | Original lazy | Reduced adaptive |
|---|---:|---:|---:|
| perturbed_0 | 0.4608770312 | 0.0013033658 | blocked |
| perturbed_1 | blocked | blocked | 2.9854279249; stopped after one segment |
| generic_1 | 0.4138438467 | 0.4138438467 | 0.0932183407 |

These are independently verified training NMSEs. No held-out data were opened.
The eager/lazy perturbed_0 arms share their first five accepted vectors. Lazy
search skips derivative calculations at rejected residual trials and makes more
progress within its wall allowance. Differences in CPU work and unmeasured waits
prevent interpreting the observed improvement as a machine-independent speedup.

The adaptive generic run initially holds `eps_s`, then releases it. Its retained
NMSE progresses 0.799478 -> 0.526890 -> 0.202998 -> 0.093218. The final `b_fraction`
derivative check fails. Fixed reduced-coordinate controls never entered search,
so this does not isolate the effect of freezing from coordinates and segmentation.

Initial tight checks had 240 seconds per coordinate, while equally tight final
checks received only 120 seconds. Original generic endpoints completed the first
probe scale but routinely exhausted that allowance during the second. Every
coordinate rebuilt its own cache and repeated the same base rollout. Some
otherwise useful passed pairs were lost when optional later refinement timed out.
A separate check spent about 1492 seconds elapsed but only 21 seconds CPU; its
cooperative deadline could not interrupt the blocking operation. The records do
not identify the cause of that wait.

The source archive SHA256 is
`01e54a546ab01ddee329c5f133611708ce426da6a501795e0216f3c2334a6a7d`.
Its report file SHA256 is
`6717d370ff13ba9e3876dfafc4ce766cf899434fe5d8ce77c32c659e9f03bb03`.

## Recovery contract

Historical files, numerical outcomes, consumed allowances, equations, parameter
domains and public data remain unchanged. This is a new campaign with new,
explicitly bounded allowances. It is not a rerun of all eighteen controls.

Completed finite residual samples are reused only for the exact model, training
payload, parameter point, coordinate chart, numerical settings and implementation
identity. Coordinate audits share the base rollout. A timeout preserves completed
samples; unfinished samples never become zero residuals or passing evidence.
Historical raw residuals/reference columns require matching identities and array
digests. Scalar summaries alone cannot reconstruct missing raw probe samples.

Search qualification and convergence assessment remain distinct. A search check
can finish once derivative agreement and the existing search-error requirement
are satisfied. It need not pursue the stricter gradient-precision target required
near stationarity. No explicit derivative disagreement is relabeled as passed.
If the same coordinate previously had a resolved disagreement, its recovery
must cover at least the previously attempted probe scales and satisfy the strict
check; an early passing pair cannot conceal that earlier disagreement. Failed
columns are not imported as trusted reference derivatives.
Defaults outside this opt-in experiment retain their prior behavior.

New per-coordinate diagnostic allowances are 360 seconds. Continuations have
at most three 600-second lazy-search segments, each followed by independent
original-equation verification and retained-point qualification. Valid earlier
coordinates are reused at identical points only. A changed endpoint needs its
own assessment. Temporarily held coordinates remain fixed; active-only checks
do not certify full-model convergence or parameter confidence.

## Fixed six-task roster

| Index | Source | Work |
|---|---|---|
| 0 | lazy task 7: original perturbed_0, NMSE 0.0013033658 | Qualified incumbent continuation |
| 1 | lazy task 23: reduced generic_1, NMSE 0.0932183407 | Qualified active-block continuation |
| 2 | identical endpoints of lazy tasks 18 and 19 | One shared diagnostic recovery; no fitting |
| 3 | lazy task 17 endpoint | Diagnostic recovery; no fitting |
| 4 | selected tight initial audit of lazy task 1 | Recover missing reduced perturbed_0 evidence; no fitting |
| 5 | historical generic endpoint, NMSE 0.2172791844 | Complete missing `b_fraction` evidence; no fitting |

The two original-coordinate generic endpoints and their saved raw reference
columns are identical, so their diagnosis is performed once and referenced twice.
No successful start supplies coefficients to a different start. Independent
rollout verification governs retention; coefficient truth is not used.

## Reporting and operations

Reports distinguish campaign completion, whether search occurred, diagnostic
availability, explicit check failure, timeout, and retained error. Missing timing
instrumentation is unknown rather than zero. Search, diagnostics, startup and
record-writing costs have separate scopes; nested costs are not added twice.
There is no automatic confidence or scientific-model promotion.

The portable input contains only the required model, training payload and selected
saved evidence with provenance. Export authenticates the older history once;
workers do not repeatedly deserialize the entire nested historical bundle.
Submission logs authentication progress and each confirmed scheduler ID.

Delta uses one CPU and 8 GB per array task, at most two tasks concurrently, and
no GPU or LLM calls. A six-hour scheduler ceiling accommodates diagnostics,
continuation and filesystem overhead; it does not enlarge numerical allowances.
The report has a thirty-minute ceiling. Deadlines remain cooperative, so a blocked
filesystem or solver call can overrun a numerical deadline; the overrun is recorded.

After extracting the delivered versioned bundle:

```bash
export AF_LLMSR_DETENTION_PROBE_RECOVERY_ROOT=/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-probe-recovery-v1
bash scripts/hpc/start_llm_sr_detention_probe_recovery_delta.sh run
```

Keep that extracted checkout available for queued jobs. Use the same checkout for:

```bash
bash scripts/hpc/start_llm_sr_detention_probe_recovery_delta.sh status
bash scripts/hpc/start_llm_sr_detention_probe_recovery_delta.sh inspect
```

Do not resubmit blindly after an ambiguous scheduler response. Inspect the sealed
submission intents and accepted jobs first. A completed operation is reused;
interrupted numerical allowances are never silently renewed. Any additional
allowance requires an explicit new experiment identity.

## Local verification

The targeted regression run passed 223 tests. A final 54-test run covered the
finished implementation and retried two source-identity checks after concurrent
source edits stopped; both retries passed. The broader repository run was
interrupted after 541 passes and three optional Torch skips; full-suite completion
is not established. Repository-wide Ruff still reports 37 unrelated errors under
`analysis/claude`; changed Python files pass Ruff.

All six real saved cases authenticate, and their permitted raw arrays import
without integration or fitting. The compact sealed input is 1,882,171 bytes.
These checks establish readiness to run the recovery experiment, not successful
optimization or restored derivative qualification on Delta.
