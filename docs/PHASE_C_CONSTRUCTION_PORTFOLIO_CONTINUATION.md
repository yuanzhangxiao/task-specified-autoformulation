# Construction portfolio prerequisite repair and incumbent continuation

30 October 2026. Follow-up to the [recovery audit](../work/agent-comms/2026-10-30-construction-portfolio-recovery-results.md).
The five tasks cover nonlinear models 03, 10, 14 and affine models 12, 15.
All source results remain immutable. Equations, parameter domains, training
objective, normalization and eight-unit observation prefix remain unchanged.
There are no LLM calls, GPUs, validation/test access or automatic model revisions.

## Shared numerical prerequisites

The old coordinate checks repeatedly spent their allowance rebuilding a common
baseline and Jacobian. They therefore never reached an actual derivative probe.
The new campaign has separate authenticated operations for:

1. Importing both successful original-equation Radau/BDF replays, with matching
   equations, data, supplied parameters, solver settings and residual hashes.
2. Computing a full BDF sensitivity Jacobian once. Its paired residual must agree
   with both saved original-equation residuals at the existing `1e-7` threshold.
3. Testing each optimizer coordinate using local state-only BDF perturbations.
   Every probe uses the same fixed precision (`rtol=1e-11`, `atol=1e-13`).

A prerequisite failure blocks all dependent probes. No per-coordinate retry of
the common solve is allowed. Complete matching coordinate coverage is required
before search eligibility can be reported; no nonlinear search runs in this
milestone. Passing local derivative checks is not parameter confidence or proof
of global convergence. The existing local step policy is not relaxed.

New numerical allowances are 600 seconds for the shared Jacobian, 240 seconds
per coordinate and at most 90 seconds per state-only probe, clipped to the parent
coordinate's remaining time. Checkpoint overhead has separate bounded room.
Baseline import performs no new rollout. Imported computation is reported as
inherited evidence, not charged again as new solver calls.

Primitive probes have their own receipts inside the coordinate's journal. Their
cost is included in the parent operation, never counted twice. Deleted, changed
or orphaned receipts are rejected. A resume reuses completed prerequisites and
coordinates; an interrupted reservation stays spent. It cannot renew a budget
or silently retry an interrupted probe. New unattempted coordinates may proceed.

## Certified additive reductions and continuation

Models 12 and 15 already have product/unused-coordinate reductions. The new
pass finds exact proportional coefficient consumers across **all** state
equations, composite readouts and initialization expressions. A sum may replace
several coefficients only when a feasible canonical lift preserves its entire
original domain. This implementation requires an unrestricted real pivot;
unsupported constraints and expressions retain their original coordinates.
It does not infer redundancy from small sensitivities or remove terms based on
their fitted magnitudes.

The saved models have these reductions:

| Model | Original coordinates | Previous reduced coordinates | New coordinates |
| --- | ---: | ---: | ---: |
| 12 | 52 | 44 | 41 |
| 15 | 50 | 42 | 40 |

The removed additive freedom includes interchangeable insulin-action drive and
decay coefficients in both models and duplicate glucose-production coefficients
in model 12. Fixing a redundant coefficient in a canonical lift does not remove
the corresponding physical effect: its contribution is retained in the summed
coefficient. Original equations and domains remain the verification authority.

Before continuing, the runtime checks the saved incumbent against its canonical
lift, verifies that lift using the original-equation Radau solver, and qualifies
all local derivative columns. A failed or incomplete check prevents search.
Then a single bounded warm search receives 1,800 numerical seconds and at most
500 calls. It starts at the same predictions as the saved model, in fewer
coordinates. There are no extra restarts or changes to the loss.

Only an independently verified strict improvement in the original full-training
objective replaces the saved incumbent. A failed/worse search retains that
incumbent. Any retained improvement receives new causal prefix inference and
forecasting for all 16 training trajectories. Prefix scores do not select the
endpoint. These remain training diagnostics, not independent generalization
estimates. The source model's optimization assessment remains visible; a short
continuation is not automatically a convergence certificate.

## Verification and limitations

Bounded checks at the actual saved endpoints found zero discrepancy between the
original saved affine predictions and the reduced canonical predictions. Their
independent original-Radau discrepancies were `9.41e-10` (12) and `9.20e-10`
(15). The three/two newly combined coordinate probes passed the unchanged local
derivative rule. Other columns still require the complete campaign qualification.
Training NMSE remained `0.5645090613` and `0.5646083251`; these checks made no fit.

For nonlinear model 14, a bounded local BDF sensitivity check completed in
13.32 seconds, producing a `14448 x 29` Jacobian and a normalized residual
discrepancy of `4.39e-9` against the saved state-only BDF replay. This motivates
the new prerequisite organization; the full 86-coordinate campaign remains
necessary. Local timings are not Delta runtime guarantees.

Tests cover exact/prohibited reductions, domain-preserving lifts, original-solver
disagreement, complete derivative coverage, stale receipts, callback failure,
child deadlines, prerequisite failure, no-repeat resume, receipt deletion,
incumbent retention, causal prefixes and safe/idempotent Slurm submission.
Historical campaigns and their receipt formats are preserved.

The affected regression suite passed **436 tests**. Scoped Ruff checks and shell
syntax checks passed. Whole-repository `ruff check .` still reports 37 existing
findings in unrelated `analysis/claude` files; those files were not changed.
The full repository pytest suite was not run. Read-only production preparation
authenticated all 23 historical terminals and selected exactly the five declared
tasks, with a maximum reservation of 9,600 seconds.

## Delta commands

The launcher authenticates every terminal in `construction-portfolio-recovery-v1`
and freezes the five-model snapshot. It requests one CPU and 8 GB per task,
four hours maximum, two concurrent tasks, and 30 minutes for reporting. Inclusive
reservations are 9,300/9,600 seconds for the nonlinear tasks and 8,400 for each
affine task. These are ceilings, not runtime estimates. Preparation and reporting
are outside numerical operation totals. No fitting tests run on the login node.

Upload the delivered commit-tagged code bundle from the Mac. On Delta, extract
it into a persistent repository directory, then run:

```bash
cd /projects/bibo/yxiao2/repos/construction-portfolio-continuation-COMMIT
sha256sum -c SHA256SUMS --quiet
export AF_ACCOUNT=bibo-delta-cpu
export AF_FIT_RUNTIME=native
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-recovery-v1
export AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-continuation-v1
bash scripts/hpc/start_construction_portfolio_continuation_delta.sh run
```

Keep this directory while jobs are active. Subsequent commands, from that same
directory with the same root export:

```bash
bash scripts/hpc/start_construction_portfolio_continuation_delta.sh status
# After completion:
bash scripts/hpc/start_construction_portfolio_continuation_delta.sh inspect
```

Download from the Mac:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-continuation-v1/inspection.tar.gz \
  "$HOME/Downloads/construction-portfolio-continuation-inspection.tar.gz"
```

Confirmed scheduler receipts are reused. An ambiguous submission stops for
reconciliation; it does not guess whether Slurm accepted the job. Use a new root
only for an explicitly new experiment, not as an automatic retry of spent work.
