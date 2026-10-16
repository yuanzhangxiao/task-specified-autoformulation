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
occur under different filesystem load. Numerical equivalence and reduced bounded
solve counts can be checked locally; Delta will qualify the new timing breakdown
and completion rate. Broad nonlinear profiles and statistical coverage calibration
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
