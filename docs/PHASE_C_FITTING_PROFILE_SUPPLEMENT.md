# M40: finish practical profiles and measure diagnostic time

Implementation milestone, 16 October 2026. This completes the bounded follow-up
to [M39](PHASE_C_FITTING_PRACTICAL_UNCERTAINTY.md), before any nonlinear expansion.
It performs no parameter refitting, new acquisitions, LLM calls or test access.
It does not change construction, benchmark contracts or production fitting.

## Why this step

M39 returned 150 independently checked fits and 141 complete profile diagnostics.
Nine profiles reached their 30-second wall allowance after using only
0.144–0.431 CPU seconds. There is no evidence that these were difficult numerical
optimizations. Waiting on shared storage is plausible, but the original timings
cannot localize the delay.

M40 retains every original fit and profile. It selects the nine incomplete
profiles from a verified source snapshot and gives each a separately recorded
90-second diagnostic allowance. This is a new diagnostic attempt at exactly the
same fitted endpoint, not a reset of the old attempt or a rerun of fitting.
Original costs remain in cumulative totals. The original 141 completed profiles
remain selected without extra work.

## Numerical and persistence changes

An opt-in execution policy separates the nuisance-reoptimized profile from the
frozen-nuisance slice. The old implementation already cached exact target values,
but unnecessarily solved bounded least squares at new frozen-only root points.
The new path evaluates those frozen points algebraically. It retains the same
objective, domains, thresholds, root-finding tolerances and numerical checks.
Profile-node and frozen-node caches remain distinct: evaluating a frozen point
never certifies a nuisance-reoptimized point.

Timing separates numerical operations, progress callbacks, JSON serialization,
durable journal writes and remaining overhead. Started work is persisted before
it runs; a crash cannot silently earn another allowance. Repeated commands
verify immutable completed results or expose interrupted work. Terminal artifact
write time is explicitly distinguished from the bounded diagnostic interval.
New-only and original-plus-extension cost totals are both reported.

The bounded diagnostic timer includes its durable start and progress writes but
excludes final result serialization/write. Source snapshot import, integrity
verification and process startup are also outside that timer; the reported
diagnostic seconds are not total CLI runtime or Slurm elapsed time.

This does not establish a cluster speedup: the saved M39 run and the extension
occur under different hosts and filesystem load. Numerical equivalence and
reduced bounded solve counts were checked locally. The returned ACES fallback
qualifies its timing breakdown and completion rate, as recorded below; it does
not qualify Delta performance. Broad nonlinear profiles and coverage calibration
remain later work.

## Delta commands

Upload the supplied `phase-c-fitting-m40.tar.gz` code bundle to your Delta home
directory. The existing M39 output stays at
`/work/hdd/bibo/yxiao2/phase_c/practical-uncertainty-fitting-v1`.

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/practical-uncertainty-m40.XXXXXX)
tar -xzf "$HOME/phase-c-fitting-m40.tar.gz" -C "$AF_CODE"
cd "$AF_CODE"
sha256sum -c SHA256SUMS --quiet
bash scripts/hpc/submit_phase_c_practical_uncertainty_supplement_delta.sh
BASH
```

The submission uses one CPU per diagnostic, two concurrent tasks by default,
with prepare and report jobs. No GPU or API key is needed. It writes a new root:
`/work/hdd/bibo/yxiao2/phase_c/practical-uncertainty-supplement-v1`.

After completion:

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/practical-uncertainty-supplement-v1
AF_CODE=$(jq -er '.identity.repo' "$AF_ROOT/submission_manifest.json")
bash "$AF_CODE/scripts/hpc/inspect_phase_c_practical_uncertainty_supplement_delta.sh" "$AF_ROOT"
```

The inspector prints the report and the review archive to download. A repeated
submission command reuses confirmed receipts. If Slurm's reply is unconfirmed,
inspect the saved intent and scheduler before recovery; do not delete receipts
or resubmit blindly.

## ACES fallback after the Delta storage outage

On 16 October the Delta submission process waited for over two hours in
`cl_sync_io_wait`, used about four CPU seconds, created no supplement root and
had no matching Slurm accounting records. Delta's notice reported OST recovery.
This is consistent with blocked filesystem access before submission, not a slow
profile calculation. Interrupt that launcher before starting a replacement;
delivery of the interrupt can be delayed while filesystem I/O is blocked.

The ACES fallback imports the already downloaded M39 review archive. It preserves
the exact plan, inputs and result JSON/JSONL bytes, including original costs.
It prepares a new M40 plan on ACES, pinning that host's code and numerical runtime;
it does not copy a locally prepared M40 plan or Delta scheduler receipts. Results
and timing must be labeled ACES: these measurements cannot establish a Delta
speedup. The nine-task roster, two-task concurrency, CPU resources and diagnostic
policy are unchanged. No original fit is rerun.

Upload `phase-c-fitting-m40-aces.tar.gz` to `/scratch/user/u.yx126462`. It contains
committed code and the historical source snapshot in separate directories, with file checksums.
No access to Delta is required:

```bash
bash <<'BASH'
set -euo pipefail
AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
mkdir -p "$AF_GROUP/repos" "$AF_GROUP/phase_c"
AF_PACKAGE=$(mktemp -d "$AF_GROUP/repos/practical-uncertainty-m40.XXXXXX")
printf 'Extracting into %s\n' "$AF_PACKAGE"
tar -xzf /scratch/user/u.yx126462/phase-c-fitting-m40-aces.tar.gz -C "$AF_PACKAGE"
cd "$AF_PACKAGE"
printf 'Checking transferred code and historical result bytes\n'
sha256sum -c SHA256SUMS --quiet
export AF_PROFILE_SOURCE="$AF_PACKAGE/m39-source"
export AF_PROFILE_ROOT="$AF_GROUP/phase_c/practical-uncertainty-supplement-aces-v1"
bash code/scripts/hpc/submit_phase_c_practical_uncertainty_supplement_aces.sh
BASH
```

The ACES wrappers load the existing GCC/Python modules in both submission and
workers, validate numerical imports, and retain the same prepare-stage tests.
The launcher now prints checksum/import/preparation/submission milestones to
distinguish these stages. Integrity checks remain mandatory; progress messages
do not prove an I/O operation is advancing.

After the jobs finish:

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase_c/practical-uncertainty-supplement-aces-v1
AF_CODE=$(jq -er '.identity.repo' "$AF_ROOT/submission_manifest.json")
bash "$AF_CODE/scripts/hpc/inspect_phase_c_practical_uncertainty_supplement_aces.sh" "$AF_ROOT"
```

The imported source and code paths must remain available for queued workers and
resume. Use the recorded code directory for recovery, rather than extracting a
second copy and changing an existing submission identity. Never delete receipts
to retry an unconfirmed submission.

Local fallback validation: all 82 relevant numerical, supplement and launcher
tests pass, including 17 launch/receipt tests. The portable source audit verifies
all 150 retained fits and selects exactly nine partial profiles from 3,842 exact
historical files. Changed-file Ruff and shell syntax pass. Whole-checkout Ruff
reports 37 pre-existing findings confined to `analysis/claude`. These were local
preflight checks; the subsequent user-run results are recorded below.

## Returned ACES results, 16 October 2026

The supplied summary reports **9/9 complete extensions and 150/150 complete
selected profiles**, with complete extension accounting and zero new failed
diagnostic calls. No fitting, live LLM calls or test access occurred. Comparing
the returned rows against the earlier downloaded M39 source confirms all 150
backend records and all 150 original profiles are unchanged; all nine extensions
retain their original fitted parameters. The three historical failed diagnostic
calls remain in the cumulative accounting.

All 150 analytic objective checks, 150 conditional-information checks and 150
frozen-information checks pass. All 75 direct/profiled objective pairs agree,
with maximum absolute difference `2.48e-12`. Independently recomputing all 180
interior profile widths from `2*sqrt(2*c/I)`, where
`I = 187*delta^2/sigma^2`, gives maximum relative discrepancy `1.10e-9`.
These calculations validate this affine control; they do not certify nonlinear
nuisance optimizations in future models.

For fixed noise `sigma=0.01`, medians over the five direct-fit noise seeds are:

| Input separation | Clean sensor-reading RMSE | Absolute gain error (evaluator only) | Profile width at objective rise 2 |
| --- | ---: | ---: | ---: |
| 0.5 | 0.002610 | 0.001422 | 0.005850 |
| 0.1 | 0.002610 | 0.007109 | 0.029251 |
| 0.02 | 0.002610 | 0.035544 | 0.146254 |
| 0.005 | 0.002610 | 0.142175 | 0.585018 |
| 0 | 0.002227 | 0.339768 | 2.950000 (entire gain domain) |

The nonzero rows have full joint rank. Gain error and proper support width grow
100-fold between the first and fourth rows while sensor-reading error stays
essentially fixed. The gain and sensor drift compensate for one another.
At zero separation only their sum is identified, and the entire permitted gain
range remains supported. Frozen-nuisance slices conceal this compensation.
Range widths are objective-rise diagnostics, not confidence percentages.
Evaluator errors use synthetic references after fitting, not held-out test data.

The nine extensions add **1.698 CPU seconds and 5.485 summed diagnostic wall
seconds**, with individual wall times from 0.427 to 0.857 seconds. Within the
5.420-second instrumented interval, numerical operations use 0.441 seconds,
progress callbacks use 4.714 seconds (87.0%), and orchestration uses 0.265 seconds.
The 4.535-second write/fsync total is nested within callbacks and must not be
added again. Import, integrity checks, process startup, scheduler wait and final
artifact writes are outside these diagnostic totals. Historical detailed timing
remains unknown; cross-host speedup and the cause of each old timeout are not
established. Combined old-plus-new diagnostic CPU cost is 50.497 seconds versus
58.261 seconds for the retained fits (86.7% extra in this tiny control).

This closes the M40 diagnostic-completion qualification on the supplied evidence.
The new raw journal/plan archive was not supplied, so its byte-level replay audit
is not claimed. Preserve the original cluster artifacts.

Local review: `output/phase-c-practical-profile-m40-aces-review/` (uncommitted).
Supplied summary SHA-256:
`2d051754eb1947521deb0b4c78af603c66d531ca515ee4eb8492cb731dce22ce`.

Next proposed step, not implemented: introduce one nonlinear nuisance rate in
a small known-structure dynamical control before general benchmark profiling.
Retain the qualified rollout fitter, joint initial/drift treatment, separating
and weak input regimes, and fixed budgets. Check each constrained profile solve
against an independent one-dimensional nuisance search where available; report
unresolved searches as unavailable, not as narrow parameter support. This tests
the distinction between weak data information and failed nuisance optimization
needed by fitting-to-interaction feedback. Coverage calibration remains separate.

## Exit criteria

- All 150 source fits and their selection remain unchanged.
- Each of the nine selected diagnostics is complete or explicitly unavailable;
  partial ranges do not count as completed uncertainty assessments.
- Support calculations agree with the unchanged numerical contract and analytic
  references on the certified affine control.
- Original, additional and cumulative diagnostic costs remain visible, including
  interruptions and unknown usage. Timers do not double-count nested categories.
- No claim of calibrated confidence, nonlinear global optimality or automatic
  model-versus-data fault attribution is introduced.

## Local verification, 16 October 2026

All **227 relevant tests** pass across M39 numerical/fixture/campaign/launcher
coverage and the new extension tests. Targeted Ruff, shell syntax and whitespace
checks pass. Whole-checkout Ruff reports 47 findings in unrelated analysis and
concurrently developed construction files; these were not changed for M40.

The full pytest attempt was bounded at 300 seconds, with progress recording
1,174 passes, five skips and one early adapter-test failure. That adapter test
(`test_fixed_endpoint_is_never_sent_to_fitter`) passed its isolated rerun. The
whole repository suite is therefore not claimed to pass.

An immutable local code export completed all nine actual saved profiles and
reported 150/150 combined complete profiles. Source bytes remained unchanged;
terminal resume preserved result/journal bytes and modification times, and
repeated reports were identical. Two final malformed-journal/counter-consistency
guards were subsequently covered by the regression suite and a final real-data
stage replay: all nine still completed with complete accounting. That final
replay used 1.088 CPU seconds and 1.168 summed diagnostic wall seconds; import,
integrity checks and report work are outside these totals. No fitting or new
truth evaluation occurred.

Artifacts are retained under
`output/phase-c-practical-profile-m40-local/` and
`output/phase-c-practical-profile-m40-final-stage-smoke/`. They are uncommitted.
These local filesystem results do not establish a Delta speedup. The cluster
completion rate and breakdown of shared-storage time remain pending.
