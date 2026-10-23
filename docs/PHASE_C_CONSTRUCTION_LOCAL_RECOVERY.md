# Construction fitting: local derivative checks and continuation

This opt-in milestone follows `construction-fitting-reliability-1`. It keeps the
same eight T2 models, training observations, input interpolation, causal
initializers, parameter domains and loss normalization. It changes numerical
execution and how existing checkpoints are used. It does not change equations,
open validation/test data, or turn on automatic scientific revision.

## What the returned run established

All 16 tasks returned an independently verified retained endpoint, but none
established local optimization convergence. Only four tasks entered optimization.
Eight derivative audits exhausted their 120-second allowance; two failed on
large finite-difference probes, and two remained numerically inconclusive.
The large probes included steps 1,024 times the base increment. Affine dependence
on states does not imply affine dependence on parameters: reciprocal timescales
and matrix exponentials can be highly nonlinear in the fitted coordinates.
Those probes could leave the local regime and even overflow while small probes
were finite. They are removed from the new local policy.

The best verified training NMSEs across the two old arms, in frozen model order,
were 2.024695, 2.071627, 2.071624, 1.812509, 1.168412, 1.789900, 1.042719 and
1.188070. These are checkpoints, not estimates of the model family's minimum
possible loss. In particular the reduced-coordinate result near 2.38 cannot be
best for model 1, because its equivalent original-coordinate model already
achieved 2.071627 within the same parameter domains. The new run shares this
useful starting point across representations.

Six other saved endpoints remain disputed by independent solver agreement.
Small diagnostic matrix calculations showed that double-precision affine
exponentiation itself can lose accuracy for the extreme generators involved.
Tightening Radau alone did not eliminate the discrepancies. Neither solver's
output is accepted merely because its NMSE is smaller.

## Local derivative assessment

For normalized training residuals r(q), the numerical check compares the supplied
Jacobian column J_j with finite differences at the same fitted point. It uses
central differences when both local probes are feasible and otherwise uses a
bound-aware one-sided stencil. Two shrinking scales test consistency; failed
numerical trials can trigger additional shrinking within a finite cap. There is
no mandatory large-step test, and no enlargement beyond the declared local cap.
Positive coordinates near zero receive steps scaled to their current magnitude
rather than a large optimizer unit. A probe that rounds back to the original
floating-point number is not a derivative measurement.

The audit retains successful and failed probe records, partial coverage, elapsed
cost, column discrepancies and an objective-gradient uncertainty allowance.
Malformed interfaces are errors; a typed nonfinite numerical trial may be retried
at a smaller scale. An expired allowance remains expired on resume. Missing
columns never silently count as passed.

Search permission and parameter information remain separate. An almost
insensitive unresolved column can permit search only under the declared small
gradient-uncertainty rule; it cannot certify that its coefficient is identifiable.
A kink, a material discrepancy or incomplete coverage withholds qualification.
These are local numerical diagnostics, not rigorous confidence intervals or a
global optimality proof. All old policies remain the default outside this run.

## Stable affine propagation

For a certified affine model and each piecewise-linear forcing interval, augment
states with forcing values, forcing slopes and a constant. The interval is then
an autonomous linear system w' = G(theta)w, so

    w(t + dt) = exp(dt G(theta)) w(t).

The new opt-in transition policy uses the ordinary double-precision backend for
moderate dimensionless generators. Above a frozen norm threshold it evaluates
both the exponential and its parameter derivatives with a bounded multiprecision
Taylor/scaling-and-squaring recurrence. Two precision levels must agree before
conversion to double precision. Matrix size, derivative count, series length,
scaling count and elapsed time are bounded. Backend counters are reported.

The norm threshold is an escalation heuristic, not an error certificate. Agreement
of two precisions does not prove correct equations or a correct solver. Every
candidate that might be retained still undergoes independent original-equation
Radau verification. The frozen replay here uses rtol=1e-11, atol=1e-13 once per
allowance, avoiding repeated slow tolerance ladders. Acceptance remains a maximum
normalized prediction gap of 1e-7 and NMSE gap at most
max(1e-12, 1e-8 times the larger NMSE).

The new dependency is mpmath 1.3.0 (`autoformalism[precision]`). Portable bundles
include its Python package and license offline; no cluster installation is needed.
The launcher seals its version and source digest as part of runtime identity.

## Frozen experiment

There are 14 independent CPU tasks:

- Eight main tasks, one per model, in the exactly reduced gain coordinates. Each
  first re-verifies its two original-coordinate historical endpoints under the
  new numerical policy, picks the lower verified training loss, qualifies that
  start, and performs at most one bounded continuation. A proposed improvement
  must pass independent replay before it replaces the current verified incumbent.
- Six verification-only tasks for the previously disputed endpoints. These have
  no search budget and cannot enter main start selection. They isolate whether
  propagation precision resolves the old discrepancy.

Main tasks allow 600 numerical seconds and at most 1,000 oracle calls for search,
300 seconds for each qualification/diagnosis and 180 seconds for each independent
verification. The per-task scheduler ceiling is one hour, with finite overhead
allowances inside operation journals. Limits are ceilings, not promised duration.
The report requests 15 minutes; array concurrency is two, one CPU and 8 GB per
task. No GPU or live LLM is used.

Source receipts are authenticated before export. The portable input contains
training data and typed endpoint/provenance records, not free-form responses or
held-out arrays. Historical cost is counted once per model, separate from new
cost and replay-only cost. Missing historical cost remains unknown. Repeating
commands reuses content-bound receipts, never renews an interrupted operation's
allowance. The source campaign remains read-only.

Execution completion, rollout verification, training accuracy, optimizer stopping
history, local convergence and parameter information are distinct report fields.
A high-loss point may be locally converged. A low-loss point may remain poorly
constrained. This campaign can improve numerical evidence, but cannot establish a
global minimum or justify a specific nonlinear mechanism by itself.

## Run on Delta

Extract the delivered archive in a new code directory, then from its root:

```bash
sha256sum --quiet -c CONTENTS.sha256
export AF_PLATFORM=delta
export AF_LOCAL_RECOVERY_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-local-recovery-v1
bash scripts/hpc/start_construction_local_recovery.sh run
```

The default runtime is
`/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`, with the CasADi overlay
`/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps`. Set AF_PYTHON and
AF_CASADI_ROOT explicitly before preparation if these locations differ.

```bash
bash scripts/hpc/start_construction_local_recovery.sh status
bash scripts/hpc/start_construction_local_recovery.sh inspect
```

Download `$AF_LOCAL_RECOVERY_ROOT/inspection.tar.gz`. Keep source and runtime
unchanged while tasks run. An ambiguous scheduler reply requires reconciliation
against its saved submission intent rather than a blind new submission.
ACES remains optional through AF_PLATFORM=aces and a distinct explicit root.
