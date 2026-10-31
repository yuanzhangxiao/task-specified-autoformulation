# Construction portfolio numerical recovery

30 October 2026. This CPU-only milestone implements the numerical recovery
recommended in [the returned-results audit](../work/agent-comms/2026-10-30-construction-portfolio-fitting-results.md).
It preserves the 23 equations, their domains, the public training data, the
eight-unit prefix, and every historical receipt. No shared parameters are
refitted and no proposal, judge, validation or test call is made.

## What changes

The previous campaign used certified affine propagation for shared fitting,
then reverted to generic explicit integration for prefix-root inference and
forecasting. Several saved endpoints are extremely stiff. The new opt-in path
keeps the affine propagation certificate through both stages and compares its
outputs against an independent original-equation Radau solve.

For fixed shared coefficients, affine dynamics alone do not imply an affine
prefix optimization problem. The runtime separately certifies that the state
generator is independent of the free roots, the initial-state map is affine in
those roots, and all composite readouts are jointly affine in states and roots.
Only then does it solve bounded linear least squares for the roots. Otherwise
it uses bounded nonlinear root optimization with affine propagation. The
independent verification still applies. This is a numerical representation of
the same model, not a change to its equations or initialization contract.

Inference receives only the permitted prefix. Shared parameters and training
normalizers stay frozen; dependent initials are evaluated through the original
graph. Forecasting receives a target-free schedule and verifies the inference's
fit, boundary, policy and forcing identities. Composite readouts and declared
root domains remain authoritative. A failed verification cannot supply a usable
forecast. Shared parameters were originally learned from all training samples,
so these fresh-prefix training forecasts remain diagnostics, not validation.

Search receipts now state whether their allowance was exhausted. Historical
missing budget fields are interpreted as normal termination only when a complete,
error-free receipt records a recognized normal stopping reason; otherwise they
remain unknown. Neither a small optimizer step nor correction of bookkeeping
certifies convergence. Gradient precision, exact-endpoint history, parameter
ambiguity, and scientific validity remain separate checks.

Structured integration failures are retained through an opt-in bounded serializer:
trajectory, interval, last attempted state/time and related diagnostics survive
without being interpreted as proof of a pole or structural model defect. Old
invalidation defaults retain their historical receipt shape.

## Recovery roster and allowances

The new campaign authenticates every old terminal against its immutable operation
receipts, then freezes a compact source snapshot. Workers need only this new
snapshot, not continued access to the old result tree. The source plan is pinned
to the reviewed `construction-portfolio-fitting-v2` identity. Resume rejects a
different source, runtime, policy, task, or checkpoint.

All 23 models remain visible:

- **15 affine models:** keep the source-retained verified shared fit; reassess
  local adequacy, infer fresh roots for all 16 training prefixes, and independently
  verify all available forecasts. No jointly fitted trajectory roots seed the
  prefix solves. Diagnosis receives 180 numerical seconds; each prefix receives
  45 seconds for inference and 60 for verification; each forecast receives 90.
- **Eight nonlinear models:** replay the unchanged supplied start independently
  with Radau and BDF, 180 numerical seconds per method. Both attempts run even
  if the first fails. Tolerances tighten to `rtol=1e-11`, `atol=1e-13`; the
  agreement threshold remains `1e-7` in normalized residual coordinates.
  Only mutually agreeing completed replays allow derivative qualification.
- **Coordinate qualification:** each coordinate in the existing certified reduced
  chart owns a separate 60-second numerical allowance. Fixed-Radau state-only
  and sensitivity predictions must agree before pairing their Jacobian. Local
  finite-difference checks preserve their existing step and precision policies.
  Full coverage, identical chart/start/precision/baseline hashes, and agreement
  with the original-equation replay are required for the aggregate verdict.
  A passed qualification means eligibility for a future search, not a fitted
  model or confidence in its parameters.

Completed coordinate receipts survive interruptions. An interrupted coordinate's
allowance remains spent and its missing qualification stays explicit; resuming
can perform only the unattempted coordinates. A separately authorized future
continuation would be needed to retry a spent coordinate. A process-local cache
avoids repeatedly computing the same paired baseline Jacobian; rebuilding it on
resume is charged to the next coordinate's new operation.

Slurm requests one CPU, 8 GB, two hours per model, at most two simultaneous
workers, and 30 minutes for the report. Inclusive reservations (including bounded
checkpoint overhead) are 4,920 seconds for an affine task and at most 6,060 seconds
for a nonlinear task. The admission ceiling is 6,600 seconds. These are ceilings,
not elapsed-time estimates. Queue time, source authentication, reporting and
adapter rebuilding on partial resume are excluded from operation cost totals.

## Verification before submission

Small synthetic tests cover stiff affine dynamics, composite outputs, shared and
dependent initials, nonlinear maps/readouts that must not use linear least
squares, active root bounds, prefix isolation, independent-solver disagreement,
interrupted reservations, exact report replay, source tampering and copied Slurm
worker paths. The original campaign's saved report still reconstructs unchanged.

The focused regression suite passed 419 tests; after the final status-output
correction, all 52 launcher tests passed. Whole-repository `ruff check .`
reported 37 existing findings in unrelated analysis files; all changed Python
files pass scoped Ruff checks. This is not a claim that the entire pytest suite
was run. A read-only preparation smoke authenticated all 23 saved terminals,
froze the expected 15/8 routing split, and verified the 6,060-second maximum
reservation without modifying the source campaign.

Four bounded saved-endpoint checks succeeded without another shared fit:

| Candidate / training row | Prefix and forecast wall time | Maximum normalized verification discrepancy |
| --- | ---: | ---: |
| 12 / 000 | 0.84 s | 4.92e-10 |
| 15 / 000 | 0.88 s | 5.06e-10 |
| 00 / 011 | 0.46 s | 1.22e-9 |
| 06 / 007 | 0.40 s | 4.92e-10 |

The generated details are in
`output/construction-portfolio-recovery-delivery/representative-endpoints.json`.
These checks establish recovery of the sampled numerical failures; they do not
establish forecast accuracy across the population. The first-row suffix NMSEs
for 12 and 15 are approximately 5.26 and 5.24 despite their much lower joint
training reconstruction errors. Full prefix evaluation therefore matters.

## Delta runbook

Upload the delivered code bundle from the Mac. The original results already on
Delta are the source; the code bundle does not need to duplicate those results.
Substitute the supplied commit-tagged filename and persistent directory:

```bash
scp /absolute/path/to/phase-c-construction-portfolio-recovery-COMMIT.tar.gz \
  yxiao2@login.delta.ncsa.illinois.edu:/projects/bibo/yxiao2/
```

On Delta:

```bash
set -euo pipefail
AF_REPO=/projects/bibo/yxiao2/repos/construction-portfolio-recovery-COMMIT
mkdir -p "$AF_REPO"
tar -xzf /projects/bibo/yxiao2/phase-c-construction-portfolio-recovery-COMMIT.tar.gz \
  -C "$AF_REPO"
cd "$AF_REPO"
sha256sum -c SHA256SUMS --quiet
export AF_ACCOUNT=bibo-delta-cpu
export AF_FIT_RUNTIME=native
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CONSTRUCTION_PORTFOLIO_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-fitting-v2
export AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-recovery-v1
bash scripts/hpc/start_construction_portfolio_recovery_delta.sh run
```

Keep that code directory for all workers and subsequent commands. Preparation
authenticates old evidence but runs no broad fitting suite on the login node.
Do not overwrite the old campaign or its code. Confirmed submission IDs are
reused; ambiguous scheduler replies block duplicate submission.

From the same directory, with the same recovery-root export:

```bash
bash scripts/hpc/start_construction_portfolio_recovery_delta.sh status
# After the workers finish:
bash scripts/hpc/start_construction_portfolio_recovery_delta.sh inspect
```

Then download from the Mac:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-recovery-v1/inspection.tar.gz \
  "$HOME/Downloads/construction-portfolio-recovery-v1-inspection.tar.gz"
```

The archive contains the compact source provenance, new operations, explicit
recovery statuses, and bounded worker-log tails. A terminal task is not necessarily
a successful qualification or forecast; inspect `recovery_status`, per-trajectory
availability, and `qualification` separately.

## Remaining boundaries

Timers are cooperative and cannot forcibly interrupt a single blocked filesystem
or linear-algebra call. Root-affinity certification is deliberately sufficient,
not complete. Guarded matrix exponentials do not eliminate all float64 error in
coefficient assembly or state accumulation. Independent solver agreement is
evidence of numerical consistency, not scientific correctness or coefficient
identifiability. Additional additive parameter reductions and shared fitting
continuations are deferred until this recovery provides trustworthy diagnostics.
