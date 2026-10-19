# M42: boundary-search repair for nonlinear uncertainty diagnostics

Implementation milestone, 18 October 2026. This follows the
[M41 result review](PHASE_C_FITTING_NONLINEAR_PRACTICAL_UNCERTAINTY.md).
It changes a diagnostic reference search, not the fitted models, production
fitting defaults, benchmark data, specification or construction stages.

## The observed defect and correction

M41 returned 36 qualified fitted endpoints but only 34 complete gain profiles.
The two incomplete profiles belong to `d0p5_sigma0p01_s1`, one for each fitting
method. At fixed gain `2.3116666667`, the primary optimizer finds a rate near
`1.4953773`, whereas the dense reference scan incorrectly retains the upper rate
bound `1.5`. The grid decreases toward that bound; the actual minimum is inside
the last grid cell. Refining sampled interior minima alone misses it.

The opt-in `reference_boundary_cells=true` policy now refines both the first and
last rate-grid cells, in addition to the existing sampled interior basins.
Exact boundary candidates are retained because bounded scalar minimization need
not evaluate its endpoints. Failed refinements stay unresolved. Every evaluation
uses the existing call/deadline ledger; edge searches receive no free budget.
Default policy serialization and M41 behavior remain unchanged. The enabled flag
is included in the new diagnostic identity, so old checkpoints cannot silently
resume with a different search policy.

Regression tests cover lower/upper near-boundary minima, minima exactly at the
bounds, the observed M41 discrepancy, failed refinement, exhausted call allowance
and deterministic checkpoint recovery. Separate algorithms still share the
analytic response and bounded affine solver. Agreement is numerical evidence,
not a proof that all minima of a general nonlinear objective have been found.

## Separate supplement and accounting

`phase-c-nonlinear-practical-supplement-1` validates all original fits and freezes
a byte-preserving, portable snapshot of their artifacts. Selection depends only
on incomplete diagnostic status after endpoint qualification; evaluator-only
coefficient errors do not choose cases. For the returned M41 archive this selects
exactly two tasks.

Each selected task performs a complete new profile diagnosis, using the original
fitted parameters, input data, noise model, gain grid and support thresholds.
The only default numerical-policy change is boundary-cell reference refinement.
The supplement does not call parameter fitting, the truth evaluator, an LLM or
test data. A full diagnosis rechecks the endpoint and all gain nodes under the
corrected reference policy rather than replacing a few saved node numbers.

The allowance remains **60 seconds / 15,000 evaluated calls per diagnostic**,
with **10 seconds / 1,000 calls per operation**. This is a new, separately charged
allowance, not a reset of the original work. Checkpointed operations resume under
the same new identity; interrupted work keeps conservative reservations and is
not silently retried. Terminal reruns reuse saved results exactly. A changed
source, runtime, policy, original endpoint or original diagnostic is rejected.

The report keeps historical fitting, historical diagnostics, supplemental
diagnostics and historical evaluator costs separately. Combined diagnostic cost
adds the old and new work exactly once. Unknown usage remains unknown. Failed,
partial or adverse new diagnoses remain visible rather than falling back to a
more favorable historical result. Summary completion and numerical diagnostic
qualification are separate fields.

This milestone does not certify continuous support widths or calibrated
confidence intervals. Adaptive refinement of support-threshold crossings and
reducing persistence overhead remain later steps.

## Delta commands

Use a fresh code directory from the committed `phase-c-fitting-m42.tar.gz` bundle.
The source must be the completed original M41 campaign; no earlier M39/M40 roots
are needed. Upload the bundle to the path assigned to `AF_BUNDLE` below.

```bash
AF_BUNDLE=/work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m42.tar.gz
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/nonlinear-practical-m42.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_CODE"
cd "$AF_CODE"
export AF_NONLINEAR_SOURCE=/work/hdd/bibo/yxiao2/phase_c/nonlinear-practical-uncertainty-v1
export AF_NONLINEAR_SUPPLEMENT_ROOT=/work/hdd/bibo/yxiao2/phase_c/nonlinear-practical-supplement-v1
export AF_CONCURRENCY=2
bash scripts/hpc/submit_phase_c_nonlinear_practical_supplement_delta.sh
```

The wrapper prints code/source/output paths before accessing the source storage,
checks the bundle and numerical imports, and reports snapshot/submission progress.
It uses the existing fitting Python environment. The preparation job verifies the
snapshot and runs focused regression tests. Each of the two array tasks requests
one CPU, 16 GB and a ten-minute scheduler limit, with concurrency two. The limit
includes imports and artifact I/O; it is not the optimizer's target duration.
The report uses an `afterany` dependency so failed array tasks stay visible.
Accepted scheduler receipts are reused; uncertain submissions are never blindly
repeated. No GPU is requested.

After completion, from the same code directory:

```bash
bash scripts/hpc/inspect_phase_c_nonlinear_practical_supplement_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/nonlinear-practical-supplement-v1
```

This checks recorded job IDs, regenerates the evidence report and prints a
timestamped review archive to download. If all diagnostics complete, expect
`expected=2`, `recorded=2`, `status_counts={"complete":2}` and
`combined_profile_status_counts={"complete":36}`. These are anticipated exit
criteria, not a claim about the unrun Delta supplement.

## Local verification

All 105 M41/M42 fixture, numerical-profile, campaign and launcher tests pass.
Scoped Ruff, shell syntax and whitespace checks pass. Whole-checkout Ruff still
reports the 37 pre-existing findings in unrelated `analysis/claude` files.
A full repository pytest attempt reached 14% with no reported failures before
the 180-second verification limit; it was stopped and is not a full-suite pass.

Both real saved partial cases completed a local supplemental diagnosis: 40/40
gain nodes each, no unresolved nodes, and 11,384/11,389 recorded calls. Both
remain below the unchanged call allowance. The combined local report has 36
complete profiles with unchanged original fitted parameters. The original source
file hashes are unchanged; terminal reruns preserve all result bytes and
modification times; repeated reports agree exactly.

Local diagnostic totals are 5.162 CPU / 5.204 summed wall seconds, including
1.565 seconds in persistence callbacks. These are a smoke qualification on a
different host, not a Delta speedup result. Artifacts remain uncommitted under
`output/phase-c-fitting-m42-smoke/`. The separately requested Delta supplement
has not been run by the agent.
