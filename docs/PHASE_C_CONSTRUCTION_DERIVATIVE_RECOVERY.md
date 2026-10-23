# Construction fitting: derivative recovery and eight continuations

This isolated milestone continues the eight unchanged T2-easy models from
`construction-local-recovery-1`. That experiment authenticated every retained
endpoint and resolved all six disputed affine/Radau comparisons. However, all
eight searches were blocked by inconclusive derivative checks: 130 of 197
directions passed, 67 were inconclusive, and none was classified as a definite
derivative failure. No search occurred. The source results remain immutable.

## Numerical changes

The new local policy distinguishes physical coordinate scales from the current
coefficient value. Ordinary gains and initial values near zero can use a feasible
inward stencil of measurable size. Parameters in potentially singular nonlinear
expressions retain conservative local scales. Probe sizes still shrink within a
declared local radius; no wide-excursion derivative test is restored. The checks
retain numerical uncertainty, incomplete coverage and all unsuccessful attempts.

For ordinary coordinates the physical scale is `max(abs(value), optimizer_unit)`;
for sensitive nonzero coordinates it is `min(abs(value), optimizer_unit)`.
The maximum displacement remains `1e-4` times that scale, also limited by the
declared domain. This fixes both nearly zero gains and large coefficients whose
old absolute probes were too small to measure. Optimizer scaling is unchanged.

An inconclusive numerical check can trigger a bounded higher-precision retry.
The original matrix-norm heuristic alone missed derivative-sensitive calculations:
one saved coordinate's relative finite-difference discrepancy fell from about 94%
to 0.077% when its transitions used higher precision, at the same local steps.
That exploratory result motivates escalation; it does not qualify other columns.
Both attempts share one diagnostic allowance and remain separately reported.

Only initialization coordinates proven inactive for all supplied training
boundaries may be omitted from optimizer coordinates. Their original values are
retained when reconstructing the complete model parameter vector. The proof must
also exclude their use in dynamics, outputs and dependent initializations. A small
numerical sensitivity is insufficient. Training inactivity does not establish
identifiability or irrelevance under a different observation contract.

## Experiment and retention

The portable export authenticates the preceding campaign's terminal records and
operation receipts, and imports its eight selected, verified incumbents. The six
verification-only probes remain historical evidence; they do not select starts
and are not rerun. Historical costs and new operation costs remain separate.

Each model has one new bounded continuation in a distinct campaign directory.
Qualification, search, independent original-equation replay and final diagnosis
are separate operations. A changed numerical policy requires fresh checks; a
previous verification is not silently transferred to new numerical execution.
Qualification selects the precision mode before the incumbent is reverified;
search, candidate verification and final diagnosis use that same selected mode.
An unverified candidate cannot displace the verified incumbent. Optimizer stopping,
training error, derivative reliability, local convergence and parameter information
remain separate conclusions. This run cannot certify a global optimum or infer
which scientific mechanism should change.

The campaign uses only training data, with unchanged equations, parameter domains,
initializer contracts and loss normalization. It makes no LLM calls, uses no GPU
and does not access validation or test data. Durable operation journals preserve
partial work and charge interrupted allowances; repeating the commands does not
renew consumed budgets. A new milestone allowance is recorded explicitly.

The delivered configuration allows 600 seconds each for qualification, search and
final diagnosis, 180 seconds per independent verification, and at most 1,000 search
calls. Qualification's two precision attempts share its allowance. The diagnostic
allowance covers the more expensive precision checks; it does not relax acceptance
thresholds or guarantee that every saved endpoint will qualify.

## Representative saved-endpoint checks

Local qualification used unchanged saved parameters and no optimization. Final
checks covered every active coordinate in each selected endpoint:

| Model index | Resolved / checked | Precision | Search qualified |
| --- | ---: | --- | --- |
| 0 | 27 / 27 | escalated | yes |
| 3 | 26 / 31 | escalated | no; five inconclusive directions |
| 5 | 5 / 14 | escalated | no; nine inconclusive directions |
| 7 | 27 / 27 | guarded | yes |

There were no omitted directions or definite derivative failures in these final
checks. The remaining inconclusive cases involve weak sensitivities and, for
model 5, a tiny reciprocal time constant. Higher-precision transitions alone do
not remove all uncertainty in the residual differences. These cases remain
numerically unresolved, not scientifically rejected. Submitting all eight tasks
does not promise that all eight will enter optimization: each independently
qualifies under the cluster runtime before spending its search allowance.

Model 0's large coefficient previously received an absolute step of `0.00015`
despite magnitude approximately `274000`. A predetermined `1e-4` relative step,
followed by shrinking and unchanged acceptance checks, resolved that direction.
The final four qualification runs took approximately 271, 465, 271 and 171 seconds
respectively on a local machine concurrently running repository tests; these are
diagnostic overhead records, not controlled hardware-performance comparisons.

Exact symbolic inventory of all eight models removed one training-inactive
`init_X_insulin_action_c` coordinate from each of models 1 and 2. Those coefficients
multiply initial time, which is zero for every supplied training boundary; they
are absent from the dynamics and observation equations. Saved values remain in
the full model, with no claim about a different initial-time contract.

The independent tiny synthetic continuation reached training NMSE `2.06e-13`,
and repeating it performed no numerical work. This is a software smoke test,
not evidence that the eight scientific models have converged. Final focused
tests passed (171 tests), including incorrect derivatives, kinks, sensitive small
parameters, schema failures, retention, budget accounting and deterministic resume.

## Delta commands

Use the supplied portable bundle containing frozen source, the authenticated
training-only export and the offline precision dependency. After extracting it
into a new code directory, run:

```bash
sha256sum --quiet -c CONTENTS.sha256
export AF_PLATFORM=delta
export AF_DERIVATIVE_RECOVERY_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-derivative-recovery-v1
bash scripts/hpc/start_construction_derivative_recovery.sh run
```

The default native interpreter is
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`, with the CasADi overlay
`/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps`. Set `AF_PYTHON` and
`AF_CASADI_ROOT` before preparation if these locations differ. Preparation only
validates the export and runtime; numerical checks execute in the CPU workers.
The eight-task array permits two concurrent tasks. Each requests one CPU, 8 GB
and one hour; the dependent report requests fifteen minutes.

```bash
bash scripts/hpc/start_construction_derivative_recovery.sh status
bash scripts/hpc/start_construction_derivative_recovery.sh inspect
```

The download is `$AF_DERIVATIVE_RECOVERY_ROOT/inspection.tar.gz`. Preserve the
source, runtime and root identity while jobs run. Reconcile an ambiguous scheduler
reply with its saved submission intent rather than submitting a duplicate.
