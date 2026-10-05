# Phase C M8: compact checkpoint results

Review of `review-20261005-015729.tar.gz` (UTC archive name), on 2026-10-04
Pacific/Honolulu. M8 tests the [checkpoint policy](PHASE_C_FITTING_CHECKPOINT_DIAGNOSTIC.md)
on the same development cases, correct equation templates and generic starts as
M6/M7. It is an engineering diagnostic, not a new benchmark or discovery run.

**Checkpoint overhead falls substantially and reporting is complete, but alien
prediction/coefficient/initial recovery remains unsuccessful.** Basin passes all
18 fits. All 36 endpoints are accounted for: 34 complete independent evaluations
and two explicit `no_complete_replay` timeouts. Neither timeout receives a partial
training/validation score. No fitter default, dataset or historical result is
changed by this review.

## Provenance and verification

- Archive SHA256:
  `43582b602bc03afe1c3235808e7bf928d598a015f37b727de6c5e6589307edb3`.
- Ignored extraction: `artifacts/fitting-m8-review-20261005`.
- Experiment commit: `cf90b395337cd77b28c9b0bf7e7cdb91f0c0f327`.
- Plan SHA256:
  `a306b4f09e3032ddb91622cf43c432568e029c213129fc873dca2ce69f5c6092`.
- Inputs SHA256:
  `97aacd4d137ad3631b11798795ad857d4febcae5780565d351277806119adfc6`;
  unchanged from M6/M7.
- Source SHA256:
  `3149236477cc44fa427b762d55bd4fb796e06f384c802a145d2cb454ff8fd214`,
  matching the preserved M8 release bundle. Subsequent checkout changes concern
  construction/operations; fitting code is unchanged from the experiment commit.
- Delta jobs: prepare `22664049`, fit array `22664050`, report `22664051`.
  One CPU per fit, concurrency two, no GPU or LLM calls.
- All 149 sealed JSON records verify. The full summary reproduces exactly in a
  separate report directory. All 36 worker payload/backend identities reconcile,
  all 18 legacy/compact pairs differ only in checkpoint mode, and all 36 terminal
  replay journals agree with their sealed replay results.
- All reported coefficient errors recompute from the retained vectors; all
  training/validation scores agree with the saved independent replays. Maximum
  normalized Radau/DOP853 disagreement among complete endpoints is 8.227e-6,
  below the 1e-4 gate.
- The reference replay and conditional local sensitivity gate pass. Alien's
  free block has numerical rank 18 and singular ratio 0.006420 near the reference.
  This does not prove global uniqueness or easy optimization from distant starts.
- No new fitting, hard-case integration, LLM call or test-data access was performed
  for this review. Raw artifacts and the local audit output remain untracked.

## What improved computationally

The table shows cumulative checkpoint seconds and native iteration counts for
alien starts 0, 1 and 2. A timeout's iteration count is the last progress record;
a completed solve uses IPOPT's total, which can exceed the latest retained
checkpoint iteration. Checkpoint time includes final detailed extraction when it
was successfully recorded.

| Method | Legacy checkpoint seconds | Compact checkpoint seconds | Legacy iterations | Compact iterations |
|---|---|---|---|---|
| Dense collocation | 391 / 364 / 363 | 40 / 35 / 15 | 84 / 85 / 97 | 274 / 203 / 139 |
| Reduced collocation | 116 / 100 / 217 | 18 / 11 / 37 | 147 / 146 / 313 | 147 / 146 / 313 |
| Limited-memory shooting | 74 / 71 / 70 | 51 / 48 / 53 | 14 / 15 / 11 | 15 / 16 / 14 |

Dense checkpoint work falls by 90--96% in these paired runs, despite more native
iterations. All legacy dense workers time out without a saved successful native
result. Compact starts 1 and 2 record `Solve_Succeeded`; start 0 still times out.
This is a useful improvement in numerical progress, not evidence of accurate
model recovery.

Reduced collocation is the cleaner timing comparison: native iteration totals are
identical across policies. Total native-worker seconds fall from
237.6 / 265.0 / 478.4 to 145.9 / 166.3 / 341.0 (29--39% shorter).
Whole-fit times for starts 0 and 1 fall from 687 to 557 seconds and 567 to 442
seconds. Start 2 still consumes its full 900-second fitting allowance while
screening. A faster native solve therefore need not shorten the whole fit.

Shooting checkpoint savings are smaller (23--32%), and all six native workers
still time out. The expensive integrated dynamics/Jacobians remain relevant;
compact records alone do not remove those costs.

These are observed timings from three paired starts, not a hardware-controlled
speed guarantee. Compact callbacks run every iteration and can retain a different
pool. Wall-limited iterations and screening opportunities can differ. The
comparison includes that policy effect, not just faster serialization.

## Prediction and recovery

Basin passes prediction and coefficient recovery in all 18 fits. The two policies
retain identical parameter vectors within every basin pair. Worst validation
NMSE is 4.914e-9; worst maximum relative coefficient error is 0.0002610 (0.0261%).

Alien passes **0/18** prediction, coefficient and shared-initial recovery checks.
Each method/policy has three planned starts; unavailable evaluations remain in
the denominator. Below are validation NMSEs for each start, legacy to compact.

| Method | Start 0 | Start 1 | Start 2 |
|---|---|---|---|
| Dense collocation | 0.437585 → 0.996544 | 0.832282 → 0.636806 | 0.907542 → 0.862609 |
| Reduced collocation | 0.304437 → 0.304323 | 0.877640 → 0.877972 | unavailable → unavailable |
| Limited-memory shooting | 0.996544 → 0.996544 | 0.542132 → 0.512830 | 0.604866 → 0.604866 |

The alien prediction gate requires both training and validation NMSE <=0.01 and
solver agreement; coefficient recovery requires every free dynamic coefficient
within 1%, and shared-initial recovery requires absolute error <=0.01. The complete
alien endpoints are far outside these gates. Individual improvements do not
justify claiming successful recovery or a uniformly better fitter.

M8 did not rerun rollout-only fitting. Its M6/M7 recovery of two out of three
starts remains valid; these 18 failures are native collocation/shooting arms.
M8 does not establish that rollout fitting became worse.

Reduced collocation again ends in poor discretized solutions, with latest
retained nodal losses approximately 0.225, 0.608 and 0.628. Faster convergence to
those regions does not produce a good model. The compact dense start-1 endpoint
has nodal loss 0.429335 and independent training NMSE 0.429279: poor observation
fit, not just a discrepancy between an excellent nodal fit and rollout.

Some solves finish without a newly retained final snapshot: compact dense start 2
reports 139 native iterations but retains checkpoint 113 as latest. The snapshot
code can skip nonfinite or out-of-bounds values; without the final vector saved,
this archive does not identify the exact rejected condition. `Solve_Succeeded`
describes the solver exit; it must not be transferred to a different selected
vector. Full final node reports are present only where detailed extraction
succeeded.

## A screening bottleneck is now visible

Compact dense start 0 progresses from 84 to 274 recorded native iterations, but
its endpoint is the original generic start: training NMSE 1.225138, validation
NMSE 0.996544. No new checkpoint has a saved complete training screen. The old
incumbent is correctly preserved rather than replaced by an unverified point.

The first queued checkpoint has nodal NMSE 0.233887 and equality defect 5.77e-8,
but `decay_x3` is about 41,416 and its shared initial value about 2.50 million.
The native stage uses about 668 seconds, followed by the whole-fit 900-second
kill. The screening code grants each point the entire remaining screening time.
Together the ordering, saved progress and lack of a completed new screen indicate
that the first expensive rollout consumes the remaining opportunity. Extreme
rates are consistent with costly/stiff explicit integration; this review does not
reintegrate that vector or claim a formal stiffness diagnosis.

Reduced start 2 retains exactly the same parameter vector under both policies.
Its `decay_x5` is about 1,263 and `input_x0` about 190,741. Its saved complete
training screen is 0.565127 and takes roughly 264--267 seconds. This is already an
inaccurate model. Later evaluation timeouts do not conceal evidence of a good fit.
A reduced start-0 final node report also has a maximum off-node residual indicator
around 19,165. Small enforced nodal defects do not certify accurate dynamics
between nodes, particularly at extreme fitted timescales.

The current pool protects the least-defective point, the best nearly-feasible
nodal point, and six recent vectors. It does not protect every historically useful
rollout point. Combined with sequential screening and no small per-point cap,
that can use the remaining budget poorly. This is distinct from the solved
checkpoint-serialization overhead.

Accounting caveat: a killed worker can leave `screen_budget.json` stale, and the
ordinary and post-NLP screening oracles share `screens-0` with call numbering
restarted. Some numbered records can overwrite the earlier ordinary screen.
The retained incumbent remains recorded; the complete attempted screening count
cannot be reconstructed from filenames. Partial backends correctly leave total
residual calls unknown. Future diagnostics should journal started/finished points
and give screening phases distinct directories.

## Replay-timeout handling worked

Both reduced-collocation start-2 endpoints now terminate as
`no_complete_replay`, with `replay_wall_budget_exhausted` recorded explicitly.
Each saves eight complete training trajectories, then times out in DOP853 on
`train_008`; all 20 expected train/validation rows remain represented. Aggregate
training and validation NMSE are null. The completed rows and partial per-solver
scores remain inspectable, without being converted into an accuracy pass.

This is improved accounting over M7's missing result. It is not a numerical
recovery improvement. Existing budgets, fitted vectors and historical results
are unchanged; no extra replay allowance was granted.

## Recommended next milestone

Keep compact checkpointing available for fixed-mesh diagnostics, while retaining
the other methods. Do not launch more hard cases or simply increase every budget
on the strength of this run. Two bounded diagnostics are more informative:

1. **Make checkpoint screening robust.** Bound each point's wall time, preserve
   the verified incumbent, journal each attempt, and avoid overwriting phase logs.
   Compare a declared stiff-capable screening solver with the current explicit
   one under the same total budget. Any solver change is a new protocol; a timed-out
   point remains unavailable rather than receiving a fabricated penalty score.
2. **Test collocation near an attainable solution.** Use the already accurate,
   training-fitted rollout endpoints from M6/M7 as explicitly assisted starts.
   Generate latent-node guesses by integrating those fitted equations, not by
   importing private hidden trajectories. First fix the fitted coefficients and
   initials and optimize only node states; then release the 18 model unknowns.
   Compare dense/reduced meshes and independent training rollouts. If this works,
   poor-start optimization becomes the priority; if it fails, inspect the
   transcription/mesh/conditioning before investing in broader search.

The second diagnostic must disclose its start provenance and upstream fitting
cost. It is not a new generic-start recovery rate or a reason to omit failed
starts. Selection uses training evidence only; validation and reference errors
are evaluated afterward. A failure or success from a good start alone does not
prove the cause of every generic-start failure.

After these checks, log/scale parameterization, bounded multi-start policies,
continuation and alternating trajectory/coefficient fitting remain candidates.
Any new bounds must follow public constraints or an explicit numerical policy,
not hidden reference magnitudes. The current evidence supports better numerical
engineering, but not a claim that additional compute alone will solve recovery.

## Review checks and scope

The 36 focused checkpoint/numerical/challenging-case regression tests pass.
The frozen six-arm smoke's completed results replay exactly without new fitting.
Repository Ruff still reports 37 unrelated findings in `analysis/claude/`.
The full suite at the M8 implementation commit passed 3,924 tests with eight
skips; it is not rerun for this documentation-only review.

Changed files: this review, the fitting plan, and the Phase C entry document.
No implementation or production default is changed. Follow-up work is proposed,
not implemented by this review.
