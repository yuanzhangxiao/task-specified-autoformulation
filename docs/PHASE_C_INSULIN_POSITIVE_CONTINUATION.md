# Insulin fitting continuation and concentration control

6 November 2026. This is the next bounded fitting milestone for the admitted
candidate-15 revision. It does not change Astra's equations or reopen the old
campaign. The separate campaign for the other 22 models is unaffected.

## Why further fitting and a separate domain control

All three returned searches exhausted 1,800 numerical seconds while still
improving. Their negative insulin predictions may improve with more fitting,
but optimizer convergence would not guarantee nonnegativity. Least squares
can prefer a negative tail to reduce errors elsewhere. A stable linear system
also need not preserve positive concentrations: the fitted negative X-to-I
coupling can keep driving I down when I reaches zero.

The public prompt describes I as an insulin concentration in pmol/L, not a
signed deviation. This milestone makes that physical interpretation explicit
in a **separate constrained arm**. It does not pretend that the original scalar
parameter bounds already enforced concentration positivity. The new coupling
keeps its declared real domain. No predictions are clipped, no coupling sign
is prescribed, and no ground-truth parameters or trajectories are used.

## Six matched tasks

Each of the three saved endpoints has two new tasks: unconstrained continuation
and concentration-constrained continuation. The labels nested-zero,
signed-negative and signed-positive refer to the historical starting conditions;
the saved fitted coupling is negative in all three.

Both arms start from the same saved outer coefficients and receive another
1,800 numerical search seconds / 2,000 objective attempts. Inner variables are
re-solved under the arm's own constraints before search. The parameter chart is
recentered at the saved endpoint, using the existing frozen net-coefficient
scaling rule. Additional allowances cover qualification, original-equation
replays, a 120-second local audit and checkpoint persistence. These costs are
reported separately; equal wall budgets do not imply equal objective counts.

The original saved fit remains an immutable reference. For the positive arm it
is not eligible for retention if it fails the domain check. The earlier nested
direct-meal fit, NMSE 0.261011, is independently checked as an admissible fallback.
The search starts from its own qualified conditional solution, so an infeasible
lower-error source cannot suppress saving feasible improvements.

## Conditional convex solve

At fixed outer matrix coefficients theta, the predicted sampled trajectories
are affine in beta, the three shared forcing coefficients and 32 training
initial roots. The unconstrained arm uses the existing bounded least squares.
The positive arm solves

\[
  \min_{\ell\leq\beta\leq u}\frac12\|A(\theta)\beta+b(\theta)\|_2^2,
  \qquad C_I(\theta)\beta+d_I(\theta)\geq0.
\]

A and b are the same normalized residual design on the original training
observations. C_I and d_I are insulin **predictions**, not residuals, at all
observation nodes and midpoints. The shared coefficients appear once across
all trajectories. This is a convex quadratic program, not bounded LS with
clipped outputs.

The implementation normalizes columns, uses QR to compress the objective, and
adds violated concentration rows to a small working QP. Acceptance checks all
9,616 concentration rows, original coefficient bounds, nonnegative multipliers,
stationarity and complementarity. Solver success alone is insufficient; a
numerically qualified KKT point may be accepted despite a solver status warning.
The tolerances and solve overhead are recorded. These checks qualify only the
conditional problem, not the outer search or coefficient identifiability.

Both arms independently replay the original equations with Radau at eight
subintervals per original sample interval: 38,416 insulin predictions across
the 16 training trajectories. The positive arm must pass this finer check for
retention. Interpolated observations in this diagnostic are bookkeeping for
recovering predictions from residuals; they never enter the fitting objective.
The physical tolerance is 1e-8 times the original training channel scale.

This is a sampled training-domain constraint with a finer verification gate.
It is **not** a proof of positivity for every continuous time or unseen input.
Only I is constrained in this pilot; minima of Gp and U are also reported.
No general scientific-validity certificate is issued.

## Stopping and feedback

Reports distinguish a time/call limit, an optimizer stop paired with an actual
finite evaluation, and a separate local descent audit. The audit perturbs each
free outer coordinate in feasible directions at 1% and 0.1% of its frozen
scale, re-solving the conditional problem each time. These are objective probes,
not derivative checks. Useful points are checkpointed and must pass independent
original and domain verification before retention. Incomplete probes and failed
checks remain explicit; even a complete audit without descent is not a global
or joint convergence proof.

The existing full-training reconstruction estimates 32 trajectory roots.
Prefix-conditioned forecasting, validation/test performance, mechanism validity
and whether another equation revision is necessary remain separate questions.
No LLM is called and no automatic proposer revision is triggered.

## Bounded local qualification

At the three saved outer points, conditional positivity enforcement gives the
following starting fits, before any new outer search:

| Historical label | Saved NMSE | Positive conditional NMSE |
| --- | ---: | ---: |
| nested-zero | 0.163647 | 0.208916 |
| signed-negative | 0.167427 | 0.186696 |
| signed-positive | 0.187675 | 0.521631 |

All three conditional solves passed feasibility/KKT, affine/original pairing,
and the finer original-equation concentration check. The worst negative roundoff
in those replays was about -2.4e-13 pmol/L. The third conditional fit is worse than
the independently checked historical fallback; the campaign retains the fallback
unless subsequent constrained fitting improves it. These are qualification
results, not evidence that the continued outer fits have converged.

A bounded one-call campaign smoke check also retained a useful local probe from
the primary positive endpoint: NMSE 0.172692, with no negative insulin on the
finer replay grid. Its original/affine loss discrepancy was below 7e-13. The
four-second smoke audit was intentionally incomplete; this point demonstrates
remaining feasible improvement, not convergence. Completed receipts and reporting
were then replayed without repeating fitting. Delta still starts from the same
three historical endpoints in both arms; these smoke probes do not seed it.

## Run on Delta

Upload from the Mac:

```bash
scp /Users/yuanzhangxiao/Projects/autoformalism/output/insulin-positive-continuation-delivery-20261106/insulin-positive-continuation-bundle.tar.gz \
  yxiao2@login.delta.ncsa.illinois.edu:/projects/bibo/yxiao2/
```

On Delta:

```bash
set -euo pipefail
AF_REPO=$(mktemp -d /projects/bibo/yxiao2/repos/insulin-positive-continuation.XXXXXX)
tar -xzf /projects/bibo/yxiao2/insulin-positive-continuation-bundle.tar.gz -C "$AF_REPO"
cd "$AF_REPO"
sha256sum -c SHA256SUMS --quiet
export AF_INSULIN_POSITIVE_CONTINUATION_ROOT=/work/hdd/bibo/yxiao2/phase_c/insulin-positive-continuation-v1
bash scripts/hpc/start_insulin_positive_continuation_delta.sh run
printf 'Keep this source directory: %s\n' "$AF_REPO"
```

The bundle contains the compact authenticated training intake and saved endpoints.
Defaults use the existing native project Python/CasADi environment,
`bibo-delta-cpu`, one CPU and 8 GB per worker, and at most two concurrent workers.
Workers request two hours including persistence/replay ceilings; the report
requests one hour. Login preparation does no fitting or test-suite execution.
Six tasks are frozen before submission, with separate durable reservations.

In a new terminal, recover the recorded repository rather than an empty directory:

```bash
export AF_INSULIN_POSITIVE_CONTINUATION_ROOT=/work/hdd/bibo/yxiao2/phase_c/insulin-positive-continuation-v1
AF_REPO=$(jq -er '.value.repo' "$AF_INSULIN_POSITIVE_CONTINUATION_ROOT/launcher_execution.json")
bash "$AF_REPO/scripts/hpc/start_insulin_positive_continuation_delta.sh" status
squeue -u "$USER" -o '%.22i %.42j %.12T %R'
# Once the workers finish:
bash "$AF_REPO/scripts/hpc/start_insulin_positive_continuation_delta.sh" inspect
```

Download from the Mac:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/insulin-positive-continuation-v1/inspection.tar.gz \
  "$HOME/Downloads/insulin-positive-continuation-inspection.tar.gz"
```

Source/runtime/launcher identities are sealed. Completed and interrupted phases
do not acquire new budgets on resume. Ambiguous scheduler replies require receipt
reconciliation rather than automatic duplicate submission. Existing campaign
outputs are neither modified nor overwritten.

## Returned Delta results

All six tasks completed. Each unconstrained continuation improved training NMSE
and retained a negative insulin tail. The primary constrained fit achieved
0.16556 versus 0.16164 in its matched free arm and passed all 38,416 finer-grid
concentration checks within tolerance. Its three shifted meal peaks remained
36, 29 and 30 minutes early. Every outer search reached its 1,800-second limit.

On the 150 g trajectory the primary constrained fit reaches zero at minute 300
with a net insulin rate of −0.802 pmol/L/min. A one-minute diagnostic extension
with zero ingestion and external insulin takes it to −0.794 pmol/L. The
observed concentration at minute 300 is 22.62 pmol/L. The paired results and
equation-level witnesses are in the [construction handoff](../work/agent-comms/2026-11-07-insulin-positive-continuation-results.md).
