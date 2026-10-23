# Construction fitting: precision-aware diagnosis and certified reductions

This opt-in milestone follows the eight returned `construction-derivative-recovery-1`
continuations. All eight retained endpoints independently replayed, six entered
search, and five improved slightly. None established optimization adequacy. The
largest relative training NMSE improvement was approximately 0.0404%. Source
archive SHA-256 is
`d791050ab496a1780324e8adc7adf539c521ff48f445695d5a680d4d1a161fca`.
The current eight models are the same definitions as the eight parents in the
separate construction comparison; its sixteen children and fitted histories are
not part of this continuation roster.

## Diagnostic changes

Search permission and final diagnostic precision have different purposes. A
relative derivative comparison can be sufficient to try a bounded optimization
step while remaining too uncertain to establish stationarity. The new final
policy continues shrinking local stencils after relative agreement when their
projected-gradient uncertainty is still material. It retains the least uncertain
supported pair, all attempts and all costs; it stops on a numerical evaluation
floor, the finite probe limit or the operation deadline. It never enlarges probes,
relaxes tolerances, or treats an inconclusive check as a wrong scientific model.
The qualification policy retains the preceding search-safety behavior.

Final diagnostics allow at most twelve local scales within their existing finite
allowance. This is particularly relevant to model 7: the old relative check
stopped while the finite-difference truncation error was still decreasing. It does
not promise to fix models 3 and 5, whose saved checks encountered a numerical
roundoff floor. Higher precision in matrix transitions alone does not make all
surrounding float64 arithmetic exact.

Parameter information now includes uncertainty in the Jacobian. Per-column error
envelopes bound a Frobenius-norm perturbation, which also bounds the spectral norm.
Singular-value perturbation bounds provide an interval for the thresholded local
rank, accounting for uncertainty in the relative threshold itself. A rank label
is qualified only when that interval resolves it. These envelopes are empirical
numerical diagnostics, not rigorous floating-point error bounds, statistical
confidence intervals or proofs of structural identifiability. Original gain
factorizations and removed coordinates are never claimed identified.

## Exact coordinate reductions

The implementation must establish dependencies in the complete assembled model,
including process consumers, observation expressions, scoped initializers and
constraints, before omitting a coordinate. A parameter appearing only in unused
process definitions can be frozen at its valid saved value. Definitions remain in
the original model, including any eager evaluation and domain obligations. Tiny
numerical sensitivities alone do not authorize removal.

Private reciprocal decay sums can also admit an exact quotient. For example,

```text
U' = (1+gain4)*Uii + I_mem - (1/tau13 + gain6/tau6)*U
```

has one effective decay coefficient when the three parameters in that sum are
exclusive to this contribution and have the supported domains. The transformation
preserves the attainable coefficient domain, supplies a differentiable valid lift
to all original parameters, and uses the lift's Jacobian during fitting. Explicit
bounds or extra consumers that invalidate the certificate cause a visible refusal.
The separate source coefficient remains `1+gain4 >= 1`; it cannot be replaced by
an unrestricted nonnegative coefficient. No scientific equation, shared parameter
tie, initialization rule or benchmark value is revised.

## Saved-model structural inventory

The symbolic checks on the eight original models give:

| Model | Previous optimizer coordinates | New coordinates |
| --- | ---: | ---: |
| 0 | 27 | 27 |
| 1 | 15 | 15 |
| 2 | 12 | 12 |
| 3 | 31 | 29 |
| 4 | 31 | 29 |
| 5 | 14 | 14 |
| 6 | 38 | 34 |
| 7 | 27 | 27 |

Model 3 has a private insulin decay; model 4 a private disposal decay; model 6
has separate private memory and disposal decays. The existing inactive initializer
coordinates in models 1/2 remain anchored. No additional unused process parameter
was found in the eight originals.

Separate static checks of Astra's saved children confirm that child 07 can freeze
its unused `tau_term_6_0_77cba2d645`, while children 08/09 can also freeze
`tau_term_8_0_aeb84ebe30`. Shared receiving coefficients remain tied. These checks
exercise the reduction capability; the children are not silently added to the
new eight-model numerical roster.

At the saved endpoints, all eight models also passed numerical checks of the
original versus lifted affine generator and declared initial values across all
128 training trajectories. Initial values were unchanged. The largest scaled
generator difference was `1.04e-16` (model 3); all other generators matched exactly
in this check. This supports the coordinate transformation, not a convergence
claim. A fresh independent original-equation rollout remains a separate gate.

Model 6's saved reduced endpoint passed that independent check: affine training
NMSE was `1.0426953072526404`, versus `1.0426953072526377` under original-equation
Radau replay. The maximum normalized prediction difference was `3.90e-12`. Its
new local diagnosis passed the derivative comparison but found a projected
gradient of `0.4623079` with upper envelope `0.4623255`, so stationarity remains
unsupported. The thresholded rank interval resolved to `[24,24]` among 34 active
optimizer coordinates. This is conditional local numerical rank, not an
identifiability theorem. No parameter optimization was performed in this check.

Concurrent full-suite testing caused the first local 180-second replay checks to
time out. The representative retry used an explicit 600-second local replay
allowance; model 6 used approximately 80 CPU seconds for replay and 156 CPU seconds
for diagnosis. These inspection retries are separate from the frozen Delta
campaign and do not modify its 180-second verification allowance.

Model 7 also independently replayed: `1.1875895021235798` versus
`1.1875895021235914`, with maximum normalized prediction discrepancy `2.65e-11`.
Its stricter derivative audit withheld stationarity and rank conclusions. For
`af_process_gain_1_0`, the finest central differences lost side consistency while
earlier differences agreed more closely. The worsening sequence is consistent
with evaluation noise amplified by division by a shrinking step. The audit's
`failed` status does not establish an incorrect analytic derivative or a scientific
model defect; it fails the numerical consistency gate. A future refinement may
label such deterioration more specifically, but must not turn earlier relative
agreement into a strong confidence claim or hide a possible inner-scale kink.
This endpoint used guarded transition precision after passing search qualification.
A possible subsequent experiment is a final-diagnostic multiprecision retry at
the identical endpoint, charged within the same diagnostic allowance. This run
does not add that adaptive policy or relax the gradient tolerance.

## Campaign and interpretation

Protocol `construction-precision-reduction-1` imports all eight latest verified
incumbents and authenticated historical receipts into a new root. It runs:

1. Coordinate certification and derivative qualification.
2. Independent verification of the incumbent under the selected execution policy.
3. A bounded continuation only if qualification and verification pass.
4. Independent verification of a proposed improved endpoint.
5. Final diagnosis of the exact retained endpoint, including unchanged incumbents.

Diagnosis from a rejected or unselected endpoint cannot be attached to the retained
one. Reports distinguish small-step (`xtol`) stops, gradient stops, budget limits
and unstarted searches. Original and new costs remain separate; interruption or
repeating the launcher cannot renew an operation's allowance. Exact parameter
values, training residual definitions and original-equation replays remain bound
together. No optimization adequacy or global optimality is inferred merely from
a successful solver return.

The saved effective vector is reused directly after verification. Reconstructing
it by reducing the lifted original parameters can introduce one-ulp differences
under the nonlinear coordinate map, which would otherwise invalidate exact
endpoint identity checks. Positive-rate trials with nonrepresentable floating
lifts are rejected numerically so search can continue; zero and negative rates
remain domain errors.

Each task has 600 seconds for qualification, 600 for search, 600 for final diagnosis,
180 per original-equation verification and at most 1,000 search evaluations, with
bounded operation overhead. The eight-task array requests one CPU, 8 GB and one
hour per task, at most two tasks concurrently. No GPU, LLM, validation or test
access is needed. This isolated experiment changes no production fitting defaults.

The exact construction/domain conflicts documented in
[the paired construction handoff](PHASE_C_PAIRED_CONSTRUCTION_NUMERICAL_HANDOFF.md)
remain separate evidence. They may justify construction repair before convergence;
these continuations neither remove conflicting historical models nor turn residual
patterns into a prescribed nonlinear equation revision.

## Local validation

The final implementation passed 279 focused fitting, reduction, campaign and
launcher tests. A repository-wide parallel run on an earlier frozen snapshot
finished with 9,022 passes, 28 failures, 45 setup errors and nine skips. The new
test failures from that earlier snapshot are covered by the final focused run;
all remaining affected test groups passed on isolated reruns (59 diagnostic tests
and seven numerical/runtime tests). Their original failures were consistent with
short wall-clock allowances exhausted under concurrent load. This is not a claim
that one uninterrupted full-suite run was green.

`ruff check .` passed on the final committed-source snapshot. The working checkout
also contains 37 unrelated lint findings in untracked `analysis/claude` files;
those files are excluded from this change. Local schema preparation, reporting
and inspection exercised the native launcher with all eight authenticated inputs
without submitting jobs. The portable bundle is checked separately after packaging.

## Delta execution

Use the supplied portable bundle, which contains committed source, authenticated
training-only inputs and the offline precision dependency. Extract into a fresh
code directory, verify `CONTENTS.sha256`, then run:

```bash
export AF_PLATFORM=delta
export AF_ACCOUNT=bibo-delta-cpu
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_PRECISION_REDUCTION_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-precision-reduction-v1
bash scripts/hpc/start_construction_precision_reduction.sh run
```

Preparation authenticates the input schemas and runtime. Scientific operations
run only in the CPU workers. Keep the source/runtime identity unchanged throughout
this campaign. The numerical follow-up intentionally uses a fresh root rather than
rewriting the previous campaign.

```bash
bash scripts/hpc/start_construction_precision_reduction.sh status
bash scripts/hpc/start_construction_precision_reduction.sh inspect
```

The inspection archive is `$AF_PRECISION_REDUCTION_ROOT/inspection.tar.gz`.
After extraction, rerunning `run` with the same identity resumes the existing
submission; an ambiguous scheduler response must be reconciled with its recorded
intent instead of blindly submitting a duplicate.
