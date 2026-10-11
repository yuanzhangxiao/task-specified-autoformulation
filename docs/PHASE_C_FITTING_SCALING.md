# M26: guarded scaling and recovery from original generic starts

The approved follow-up tests whether the scaling advantage seen at saved M25
endpoints survives from the original generic initializations. It adds an opt-in
experiment, `phase-c-generic-scaling-1`. Production defaults, historical fitting
protocols, equations, prompts and datasets are unchanged.

The [M26 results review](PHASE_C_FITTING_M26_RESULTS_2026-10-11.md) records all
nine completed tasks: fixed and guarded scaling recover all three original
starts; iterative Jacobian scaling recovers two. Fixed and guarded remain
paired alternatives pending broader qualification.

## Scaling principle

Keep the training/public-domain coordinate units introduced earlier in Phase C.
For a search block starting at parameter/initializer vector theta_b, optimize
dimensionless displacements q with theta = theta_b + S q. S is diagonal and
fixed by the existing coordinate contract. Exact terminal-output profiling still
eliminates applicable gains; the Jacobian below is then the derivative of the
profiled residual with respect to the remaining q coordinates.

For m normalized training residuals, let J_q = dr/dq and define

```text
a_j   = ||J_q[:, j]||_2 / sqrt(m)
a_ref = median of the strictly positive a_j
d_j   = clip((a_ref / max(a_j, epsilon*a_ref))**alpha, d_min, d_max)
```

The guarded arm supplies d to the optimizer as `x_scale`. It uses alpha = 0.5,
epsilon = 1e-6, d_min = 0.1 and d_max = 10. These are predeclared numerical
safeguards, not claimed optimal hyperparameters. Alpha = 0 reproduces fixed
scaling; alpha = 1 fully balances column norms before clipping. With alpha = 0.5,
the adjustment is deliberately partial. An all-zero Jacobian uses identity
scaling and records zero information. Weak columns are reported; scaling never
certifies that their parameters are identifiable.

Relative sensitivities cancel any common residual multiplier or repeated rows.
Consistent changes of physical units leave J_q and the guarded adjustment
unchanged. Norms are calculated after removing a common matrix magnitude to
avoid overflow. These invariances concern the scaling rule; they do not promise
bitwise identical numerical integrations or optimization on every problem.
Column scaling cannot remove correlations between nearly dependent columns.

Compute d from the first *charged, cached* evaluation of each search block, then
freeze it throughout that block. Recompute at the next continuation/restart
anchor. This avoids rapidly changing the trust-region geometry within a solve.
No extra rollout is performed solely to calculate d. Record the scale vector,
relative column norms, weak columns, base units and policy in checkpoints.

## Matched comparison

The three arms are:

| Arm | Additional optimizer scale |
|---|---|
| `jacobian` | SciPy's iterative Jacobian-column scaling |
| `fixed_coordinates` | Identity in the existing dimensionless coordinates |
| `guarded_relative` | Bounded relative sensitivity adjustment above |

Every M26 arm uses the same residual and Jacobian divided by sqrt(m), so the
least-squares objective is half the mean squared normalized residual. The
additional multiplier is 1. Absolute gradient stopping and cost-change stopping
are disabled; native step stopping remains at xtol = 1e-12. Independent numerical
accuracy, finite budgets and the continuation rule provide the other stops.
Disabling absolute gradient stopping prevents a small numerical loss scale from
prematurely ending polishing; it does not certify a stationary point.

This common normalization/stopping change removes the old 1e6 polishing
amplification. **Historical M25 and M26 are therefore not an isolated one-factor
scaling comparison.** Scaling differences *within M26* are matched. Existing
callers retain their original amplification and stopping defaults. Unit tests
also exercise fixed and guarded physical search paths under different residual
multipliers, without making that a fourth benchmark arm.

## Sources and training boundary

The original frozen development input digest is
`ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09`.
The exporter verifies the M25 source bundle and extracts this original data;
it does not import M24/M25 fitted endpoints, scores or derivative-audit decisions.
All three original generic starts are mandatory. Each initial incumbent is that
generic start, not a saved warm solution.

The qualification problem remains the known nonlinear alien-device skeleton:
six states, thirteen dynamic coefficients and five shared hidden initial values,
with sixteen training and four retrospective validation trajectories. Exact
bounded profiling removes four terminal output gains, leaving fourteen outer
unknowns. No initial values are fitted to validation observations.

Before any arm runs, freeze three further starts per original seed using the
existing domain-only stratified generator, seeded by the training-problem hash.
Finite positive domains are sampled logarithmically; other domains use a bounded
window in the existing coordinate units. The complete numerical vectors are
identical across scaling arms. Check bounds and distinctness in the *optimized*
coordinates, so changing only eliminated gains cannot masquerade as a restart.

The fitting function accepts only the allowlisted TrainingProblem plus starts
and numerical policy. Retention, restart decisions and stopping cannot access
validation observations or reference coefficients. Seal the final backend before
retrospective prediction/coefficient/hidden-initial scoring. No test data,
scientific critic or LLM is used.

## Continuation, retention and budgets

Each task gets at most six search blocks of 40 started residual/Jacobian
evaluations or 800 seconds each: at most 240 evaluations and 4,800 search seconds.
Each block resets optimizer internal state and centers coordinates at its anchor.
Continuation restores the fitted vector, not a SciPy trust-region state.

Continue a branch only when all of the following hold:

1. Its preceding block stopped because of its evaluation/time budget.
2. Its best training loss improved by at least 0.1% over the last ten completed
   evaluations, relative to max(the earlier best loss, the target loss).
3. Its endpoint has a usable independent rollout check.
4. It has used fewer than two consecutive blocks.
5. At least one block remains reserved for every unvisited predetermined start.

A native stop, failed verification, insufficient progress evidence or plateau
advances to the next frozen start. The current branch can explore a worse point,
but it cannot erase the globally retained, independently verified incumbent.
If all starts plateau, unused ceiling is not spent by repeating them. Thus budgets
are equal maxima; actual work is adaptive and must be reported. A maximum of two
blocks per branch is an experimental allocation rule, not a universal optimum.

Profile numerical failure may invoke joint rollout fitting, but only with the
same block's remaining calls and time. It cannot reset the allowance. A fallback
does not justify extending the failed profiled branch.

Before search and after each block, use independent DOP853/Radau rollout checks
of the original equations. Retain candidates by the existing conservative
two-solver loss comparison. A target stop requires both aggregate losses at
most 1e-12, strict numerical agreement, and worst-trajectory NMSE at most 1e-11.
A tighter verification is available for uncertain or near-target points. Poor
fit, numerical uncertainty and weak sensitivity are separate outcomes.

Each ordinary/tighter check has a 180-second ceiling; final sensitivity of all
eighteen coefficients/initials has another 180 seconds. Retrospective evaluation
has 300 seconds. The worst configured search + check + sensitivity + evaluation
allowance is about 130 minutes, plus setup and reporting; reserve 2.5 hours per
Delta CPU array task. There are nine tasks, default concurrency three, one CPU
and 16 GB per task. No GPU is requested. Record setup, verification, search,
sensitivity and evaluation costs separately from one another.

Every completed operation and branch decision is sealed and reused on resume.
An interrupted operation with unknown consumption is not relaunched: charge its
full allowance, report incomplete usage and retain the prior verified incumbent.
Its unverified partial checkpoint cannot displace that incumbent. Unknown
scheduler submissions require reconciliation, not automatic duplicate submission.
The plan binds source/runtime identities, policy, input digest and all starts.

## Interpretation and qualification

The primary outcomes are independent training accuracy, retrospective validation
accuracy, coefficient/initial recovery, evaluations, time and recovery by start.
Report all nine cells, including failures. Final joint sensitivity and alternative
vectors produce the existing fitting-assessment fields; they are not calibrated
confidence probabilities, minimized profile likelihoods or global uniqueness
certificates. A small NMSE does not establish correct coefficients.

`status: complete` means a finished task with a retained usable point; use
`accuracy_verified` and the fitting assessment to distinguish successful recovery
from a completed poor fit. Distinct restart vectors do not guarantee different
attraction basins. This remains a finite test on one known skeleton.

Qualification covers extreme common multipliers, physical-unit changes, weak/zero
Jacobian columns, charged initial evaluation, exact profiling, continuation,
reserved restarts, independent retention, interrupted accounting, fallback
budgets, immutable resume, training-only boundaries and scheduler idempotence.
A two-state nonlinear ODE smoke passes all three arms in seven evaluations each,
with train NMSE about 1.49e-12 and maximum parameter absolute error 1.83e-6 under
its looser smoke target. This is an interface/integration test, not evidence of
benchmark robustness or a scaling winner. Benchmark fitting runs on Delta.

## Delta commands

Upload `transfers/phase-c-fitting-m26.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c`. The bundle includes only the committed runtime
and frozen portable inputs; no older remote campaign is required.

```bash
bash <<'BASH'
set -euo pipefail
cd /work/hdd/bibo/yxiao2/phase_c
tar -xzf phase-c-fitting-m26.tar.gz
bash phase-c-fitting-m26/scripts/hpc/submit_phase_c_fitting_scaling_delta.sh
BASH
```

Default root: `/work/hdd/bibo/yxiao2/phase_c/fitting-scaling-v1`.
`AF_CONCURRENCY=2` may be set before submission if sharing the CPU allocation;
three is only the experiment's requested concurrency, not a cluster job limit.
The script checks transfer hashes and dependencies, freezes the plan, then
submits preparation, nine fitting tasks, and a report job. Repeating a successful
submission returns its recorded jobs. Do not change an existing plan's settings.

Inspect and collect the sealed results, partial results and logs with:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m26/scripts/hpc/inspect_phase_c_fitting_scaling_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/fitting-scaling-v1
```

Download the resulting `review-*.tar.gz` for inspection. This milestone does not
promote guarded scaling to the production fitting policy; the larger benchmark
comparison must establish whether its safeguards help initialization robustness.
