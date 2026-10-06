# M14: profile conditionally linear output gains during rollout fitting

Protocol: `phase-c-profiled-output-1`.
Configuration: `configs/phase_c_profiled_output_v1.json`.
This implements the approved follow-up to the
[M13 review](PHASE_C_FITTING_M13_RESULTS_2026-10-06.md). Production fitter defaults,
construction, benchmark data and public prompts are unchanged.

## Question and scope

The M13 joint rollout control recovered two of three original generic starts.
The third suppressed one output gain and retained inaccurate coefficients.
Adding the tested start portfolios did not resolve that failure. Here we test
whether solving the output gains conditionally removes an avoidable source of
parameter coupling. It can still legitimately choose a zero gain for a poor
latent trajectory: this is not a guarantee of global recovery.

Use the unchanged sealed M12/M13 development inputs:
`ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09`.
The diagnostic supplies correct alien-device hard equations and anchored
coordinates: six states, 13 dynamic coefficients, five shared latent initials,
16 training trajectories and four validation trajectories. All three original
starts remain in the denominator, including the difficult third start. Correct
structure/coordinates are explicit assistance, not a construction result.

| Arm | Method |
|---|---|
| `rollout_only` | Unchanged joint forward-sensitivity TRF control, 18 unknowns |
| `profiled_rollout` | Bounded linear least squares for four output gains at every outer evaluation; scaled TRF over the remaining 14 unknowns |

Both still estimate all 18 quantities. There are six one-CPU tasks, no GPU or
LLM calls. Each gets the same 1,200-second fitting ceiling, including symbolic
setup, integration, optimization and a 180-second reserve for training
certification. Independent final evaluation has a separate 300-second allowance.
Maximum residual calls remain 900. Equal ceilings do not require equal consumed
time; report elapsed cost and recovery separately.

## Conditional linearity, without observed derivatives

In this case the observed state is `y=x5`. Its four signed features depend on
latent states, and the latent equations do not depend on `y` or these gains:

\[
\dot z=f(z,u;\beta),\qquad
\dot y=-d_5y+\sum_{j=0}^{3}g_j\phi_j(z),\quad g_j\ge0.
\]

The outer vector includes `d5`, the other latent-dynamics parameters and all five
latent initials. For each training trajectory and a proposed outer vector,
integrate the latent states and filtered features:

\[
\dot b=-d_5b,\quad b(0)=y(0),\qquad
\dot v_j=-d_5v_j+\phi_j(z),\quad v_j(0)=0.
\]

Then `y=b+sum(g_j*v_j)` is the original free rollout, up to numerical integration
error. Stack every training observation into a design matrix `A` and offset `b`,
using the original training normalization. Solve

\[
g^*(\beta)=\arg\min_{\ell\le g\le h}
 \tfrac12\|A(\beta)g+b(\beta)-y_{\rm train}\|^2.
\]

This is bounded linear least squares across trajectories, with a single shared
gain vector. It does not estimate observed derivatives, fit latent labels, reset
observed states during rollout or relax the dynamics. Original term signs and
parameter domains are preserved; signed feature values themselves can change
with latent state. No reference-centered bounds are introduced.

The implementation also supports a certified output of the form
`y'=a(z,u,beta)*y+c(z,u,beta)+Phi(z,u,beta)*g`: integrate
`b'=a*b+c` and `v'=a*v+Phi`. It requires one identity-observed terminal state.
CasADi dependency checks on the restricted compiled model certify absence of
output feedback, joint affinity in the selected gains, gain-independent remaining
dynamics/output coefficient, and gain-independent causal initializers. Gains are
found from expressions, not parameter-name conventions. A model that cannot be
certified is rejected. This milestone does not add general multi-output profiling
or profiling through feedback loops.

## Derivatives and numerical safeguards

Forward sensitivities differentiate latent states and filtered features with
respect to the 14 outer unknowns. The outer residual Jacobian also differentiates
the inner optimum. It is not correct to hold the fitted gains fixed in this step.
For the free gain columns `A_F`, a direction satisfies

\[
(A_F^TA_F)\,dg_F=-A_F^T\bigl((dA)g+db\bigr)-(dA_F)^Tr,
\qquad dr=(dA)g+db+A_Fdg_F.
\]

Active gains remain at their original bounds. The implementation uses an SVD of
column-normalized features, without explicitly forming or inverting the normal
matrix. The inner solve uses SciPy BVLS (`tol=1e-12`, at most 100 iterations).
A column-normalized singular-value ratio at or below `1e-12`, zero columns,
nonfinite values, unsuccessful bounded solves or unavailable sensitivities make
the point unavailable. No zero Jacobian substitutes for a failed point.
The derivative is exact on a fixed, regular active set, subject to numerical
integration/linear-solve errors. At active-set changes the residual can be
nonsmooth. Active masks, singular values and solver optimality are recorded.

Integration retains RK45, `rtol=1e-8`, `atol=1e-10`, all original samples and all
public input corners. Outer scaling, TRF tolerances, stagnation thresholds and
the 30-second sensitivity-point ceiling match the control. No new restarts,
collocation stage, stiff solver or prior fitted vector is added. The control's
ordinary initial screen remains unchanged; the profiled arm starts by integrating
its features and solving its gain block at the original outer vector. Its
original gains are logged for comparison but intentionally replaced by the inner
optimum. This is a comparison of complete algorithms, not equal solver call counts.

Although the nonlinear search is smaller, the augmented integration is larger:
10 filtered/latent states times 15 (state plus 14 sensitivities) gives 150 scalar
integration variables versus 6 times 19 = 114 for joint sensitivities. A speedup
is therefore an empirical question. Both arrays record their actual costs.

## Checkpoints, evaluation and resume

Every completed full-training evaluation can publish a durable best parameter
vector. Partial trajectory results cannot select an incumbent. The worker logs
all outer attempts, solved gains, inner diagnostics and completed integration
counts. Supervisor receipts hash the structure certificate, training history,
accounting and best checkpoint in addition to the result. An interrupted worker
can contribute its saved complete checkpoint only after termination is confirmed.
Unconfirmed cleanup blocks evaluation and further work. The existing fit envelope
and scheduler journals forbid silently restarting spent budgets.

Training certification uses independent Radau and DOP853 evaluation of the
original complete equations, not the filtered representation. The thresholds
remain mean worst-method training NMSE `1e-6`, every trajectory `1e-5`, and
normalized maximum solver disagreement `1e-5`. The final selected backend is
sealed before validation/reference data are opened by the separate evaluator.
Final gates are train/validation NMSE `0.01`, maximum coefficient relative error
1%, and maximum latent-initial absolute error `0.01`. Prediction and parameter
recovery remain separate outcomes. Test data remain unopened.

Review all six results, including failures, and compare matched starts on:
training/validation NMSE; coefficient and initial errors; certified early stop;
fit time; outer calls; active output gains; and inner conditioning. A successful
inner solve alone does not certify identifiability, mechanism recovery or global
optimization. Broader cases and constructed, imperfect equations remain future
qualification steps before production integration.

## Delta commands

Upload `phase-c-fitting-m14.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c/`.
The bundle includes pinned source, tests, config and unchanged sealed inputs.
It uses the existing Delta Python/CasADi environments; no installation is needed.

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m14"
tar -xzf "$AF_BASE/phase-c-fitting-m14.tar.gz" -C "$AF_BASE/code/fitting-m14"
export AF_PROFILED_ROOT="$AF_BASE/fitting-profiled-output-v1"
bash "$AF_BASE/code/fitting-m14/scripts/hpc/submit_phase_c_profiled_output_delta.sh"
BASH
```

The six tasks request one CPU, 16 GB and 35 minutes each, at most two running
concurrently. This covers fitting, separate final replay and orchestration.
Preparation runs focused tests; the report runs after the array, including after
failures. Existing roots and submission receipts are preserved. Do not blindly
resubmit after an uncertain scheduler acknowledgement.

Check and package the results:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m14/scripts/hpc/inspect_phase_c_profiled_output_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/fitting-profiled-output-v1
```

The inspector prints the explicit root, scheduler status, six per-task summaries
and the review archive path. Download the printed archive. A complete campaign
means terminal records exist, not that all models recovered.

## Local qualification

The hard-case symbolic certificate identifies exactly the four output gains and
14 outer unknowns. No hard-case refitting has been performed locally. Numerical
regression checks use a small, independently integrated nonlinear two-state
control with two bounded gains and one shared latent initial. Both arms pass
independent training/validation replay and recover all its parameters within
`2e-6` absolute error. Their fitting times are approximately 5.2 and 4.1 seconds;
this tiny smoke is a correctness check, not evidence of benchmark speedup.

Tests cover unsafe feedback/nonlinearity/initializer dependence, identity mapping,
full-equation versus filtered trajectories/sensitivities, finite-difference outer
Jacobians, interior/lower/upper active sets, rank/nonfinite rejection, budgets,
checkpoint retention, cleanup uncertainty, training-only boundaries, all-start
rosters, sealed evaluation and idempotent scheduler submissions.

Verification completed on 2026-10-06: the portable preparation suite passes all
71 tests. Full `pytest -q -n 4` records 4,130 passes and eight optional Torch skips;
five provenance checks encountered source drift because final source edits
occurred during that run. All five pass on an isolated rerun with the source
frozen, accounting for all 4,143 collected tests. Changed Python files pass Ruff
and `git diff --check` is clean. Repository-wide `ruff check .` retains 37 existing
findings in unrelated, untracked `analysis/claude` files, which were not modified.
