# M10: calibrated screening and the four blocked assisted runs

This is the approved follow-up to the
[M9 review](PHASE_C_FITTING_M9_RESULTS_2026-10-05.md). M9's 20-second screening
ceiling rejected every alien-device Radau evaluation before a complete training
score was available. All four eligible assisted tests stopped at this recheck;
their collocation stages were never tested. Basin already passed and is not rerun.

## Frozen experiment

The input export verifies M9 qualification, worker identities, saved vectors and
cached training scores. It copies the original alien-device training/validation
arrays and three requests. It does not generate new data, change equations, add
reference starts or read test data. Historical M9 records remain unchanged.

1. **Calibrate complete training evaluations.** Two CPU jobs independently test
   RK45 and Radau on the three ordinary starts and the two eligible M7
   training-fitted endpoints. Each point has 120 seconds including process
   startup, setup, all trajectories and output. Both good endpoints must recheck
   at training NMSE at most `1e-8`. A method's allowance becomes
   `max(60, ceil(2 * slowest_completed_point_seconds + 10))`, at most 300 seconds.
   A missing required recheck or an allowance above that ceiling blocks the
   affected tasks with an explicit status. Ordinary-start failures are reported;
   they do not invalidate the good-start calibration.
2. **Retry four assisted tests.** Dense and reduced collocation each use the two
   eligible training-fitted endpoints. The source is reverified, node trajectories
   are generated from that fitted model, coefficients and initials are first fixed
   and then released. Each test has the original 900-second overall ceiling,
   at most 180 seconds for node generation and 250 for the fixed stage. Released
   optimization now reserves the calibrated point allowance plus five seconds
   for its final training screen. The source fit remains an eligible incumbent.
3. **Rescore 18 saved generic pools.** Evaluate every saved parameter vector in
   each original pool, including retained native checkpoints that were not
   screened. Keep the original pool's RK45/Radau method; do not rerun its optimizer.
   Reuse verified complete M9 scores (35 among 149 pool entries); rescore the
   remaining 114 under a 900-second pool ceiling. Identical vectors within a pool
   are deduplicated. A cached incumbent survives subsequent failures/timeouts.
4. **Evaluate after selection.** Freeze the training-selected vector, then run the
   existing independent Radau/DOP853 train/validation replay (300-second ceiling)
   and coefficient/initial recovery diagnostics. Evaluation never changes the
   selected model. Report the released assisted endpoint separately from the
   retained supplied source; retaining an accurate source is not new recovery.

These are additional diagnostic costs, not an equal-budget improvement claim.
Reports retain source fitting cost, calibration cost and new task cost separately.
The 22-task run array lists assisted tests first; the scheduler determines actual
execution order. No production fitting default is promoted.

## Progress and interruption behavior

Each point writes `progress.json` with startup/setup time, the current trajectory,
completed trajectory count, elapsed integration times and solver work counters.
Timeout logs identify whether setup or a particular rollout consumed the allowance.
No partial-trajectory average is accepted as a training score. Symbolic setup and
integration share the process deadline. Hard termination remains available.

Every task and calibration has a frozen start record. Terminal reruns reuse their
sealed results. An interrupted computation is closed explicitly without silently
granting a new budget; saved telemetry remains available. A completed backend can
resume its separately journaled evaluation. Submission intents and receipts prevent
duplicate jobs after uncertain scheduler replies. Missing jobs remain missing in
the report rather than being counted as fitting failures.

## Delta commands

Upload the supplied `phase-c-fitting-m10.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. The bundle contains the sealed saved-pool export;
there is no need to move the M9 campaign or install another environment.

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m10"
tar -xzf "$AF_BASE/phase-c-fitting-m10.tar.gz" -C "$AF_BASE/code/fitting-m10"
export AF_CAMPAIGN="$AF_BASE/fitting-screening-replay-v1"
bash "$AF_BASE/code/fitting-m10/scripts/hpc/submit_phase_c_screening_replay_delta.sh"
BASH
```

The submission uses the existing fitting environment and CasADi dependencies,
one CPU and 16 GB per task, two calibration jobs and at most two concurrent run
jobs. Wall limits are 15 minutes for preparation/calibration, 25 for each run and
five for reporting. No GPU, API or LLM is used. The output root is
`/work/hdd/bibo/yxiao2/phase_c/fitting-screening-replay-v1`.

Inspect and package the results:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m10/scripts/hpc/inspect_phase_c_screening_replay_delta.sh
```

The command prints calibration decisions, every task outcome, and the review
archive path. Calibration can still find the time allowance insufficient; that
is an explicit diagnostic result, not evidence that the fitted model is poor.

## Local verification

Focused tests cover provenance, training-only payloads, timing calibration,
checkpoint retention, interruption/resume and scheduler idempotence. The small
linear end-to-end smoke uses an exported, previously training-fitted M9 source:

```bash
PYTHONPATH=src python scripts/smoke_screening_replay.py \
  --inputs /path/to/linear-replay-inputs.json --root /path/to/new-smoke-root
```

Hard alien fits are reserved for Delta. No local hard-case refitting is required.

Verification completed on 2026-10-05 for implementation `fcb505e`:

- Focused M9/M10 tests: 18 passed, both in the checkout and portable release.
- Small linear end-to-end smoke: all eight entries passed prediction and
  coefficient gates; both assisted tests completed fixed and released phases,
  screened their released endpoints, and resumed exactly without new computation.
- Frozen-source full pytest run: 3,899 passed, 35 skipped; 62 initially lacked
  local fixture/environment paths in the snapshot. All 62 passed after making
  those existing paths available, for 3,961 passing tests overall.
- Changed Python files pass Ruff; shell scripts pass `bash -n` and the diff
  passes whitespace checks. Repository-wide `ruff check .` still reports 37
  pre-existing findings in unrelated `analysis/claude` files, left unchanged.
- The sealed production export contains 18 pools, 149 distinct vectors within
  those pools (35 cached scores), four assisted tasks and five calibration points.
  All three ordinary vectors equal the original M9 start payloads exactly; all
  four assisted payloads reproduce their source identities before budget changes.

The portable archive contains no hard-case rerun results. Submit it on Delta to
obtain the new calibration and fitting evidence.
