# Fitter investigation closeout — 2026-09-14

The final bounded comparison completed all nine arms and passed its numerical
gate. No arm reached the predeclared practical band: both training and validation
NMSE at most 0.1, with passing numerical replay. Following the user's agreed
stopping rule, this closes additional algorithm exploration on the difficult
case. There are no follow-up runs or budget increases from this report.

This is an operational freeze for continuing pre-fitting work, not a claim that
parameter fitting is solved for arbitrary proposed models. Existing experiment
backends and settings remain unchanged. This document changes no runtime code,
configuration, benchmark, prompt, or acceptance threshold.

## Final evidence

The experiment implementation is pinned to commit
`24a259653ded427838897fa2462fd48b945158ba` on
`codex/fitter-final-alternatives-v1`. Its
[protocol](https://github.com/yuanzhangxiao/task-specified-autoformulation/blob/24a259653ded427838897fa2462fd48b945158ba/docs/FITTER_FINAL_ALTERNATIVES.md)
is pinned to that commit. Results below transcribe the
user-supplied completed Delta summary for `fitter-final-alternatives-v1`; values
are rounded for readability. Raw experiment artifacts remain outside Git.

All reference starts were ordinary. Fixed-shape controls retain the reference
nonlinear shapes and known hidden initial conditions while fitting 26 weights
and rates. Free-shape controls fit all 48 parameters, retaining the correct
skeleton and known hidden initials. The smaller model has three states and 13
parameters, with training-fitted causal initial maps instead of reference hidden
boundaries. It is a separate model hypothesis, not a matched numerical ablation.

| Family | Route | C accepted | Train NMSE | Validation NMSE |
| --- | --- | --- | ---: | ---: |
| Fixed shapes | Exact-Hessian collocation and refinement | No | 0.278746 | 0.467162 |
| Fixed shapes | Direct sensitivity multistart | — | 0.235698 | 0.343674 |
| Fixed shapes | Horizon continuation | — | 0.663797 | 0.694463 |
| Fixed shapes | Limited-memory collocation and refinement | No | 0.129160 | 0.354922 |
| Free shapes | Exact-Hessian collocation and refinement | No | 0.583302 | 0.627940 |
| Free shapes | Direct sensitivity multistart | — | 0.115230 | 0.140271 |
| Free shapes | Horizon continuation | — | 0.412310 | 0.351809 |
| Free shapes | Limited-memory collocation and refinement | No | 0.583309 | 0.627799 |
| Smaller model | Exact-Hessian collocation and refinement | Yes | 0.342350 | 0.417946 |

Every final parameter set passed BDF/Radau replay agreement. This checks
consistency of numerical integration at those parameters; it does not establish
accurate prediction, scientific validity, unique parameters, or a global optimum.
Every row missed the strict (1e-4), good (0.01), and practical (0.1) bands. Those
bands require both split scores and are descriptive, not scientific certification.

Direct multistart made a substantial improvement for free shapes relative to the
same-family collocation control. It still missed the frozen practical threshold,
and this single opened case does not justify switching the general backend.
Limited-memory collocation improved the fixed-shape route's final fit but did
not produce an accepted C solution. The aggregate summary does not attribute
that improvement specifically to collocation rather than retained checkpoints,
screening, or subsequent refinement. Horizon continuation was inconsistent
across the two families.

The smaller model's accepted C solution and poor prediction fit show why solver
acceptance and recovery must stay separate. Its shared/causal initialization has
a training NMSE lower bound of 0.00561563 for indistinguishable public
preparations (`train_000`, `train_014`, and `train_015`). Its achieved training
NMSE, 0.342350, is about 61 times that bound. Missing initial-condition
information imposes a real limitation, but this bound alone does not explain
the much larger observed error. It also does not promise attainability of the
bound by this particular smaller skeleton.

## What can and cannot be concluded

Earlier same-skeleton synthetic and favorable-start audits demonstrated that
excellent output recovery is attainable, including for the difficult reference
family. The present ordinary-start routes did not find such fits within the
specified resources. Correct equations, known hidden boundaries, and even fixed
reference shapes were insufficient for reliable ordinary-start recovery here.
This is a remaining fitting limitation; poor fits alone cannot prove that a
proposed skeleton is wrong or that no useful parameter set exists.

Collocation remains useful on the easier tested problems. In the earlier paired
three-start, two-noise comparison, the moderate family recovered 6/6 with either
ordinary sensitivity fitting or C followed by sensitivity refinement. The
separated-timescale family improved from 4/6 to 6/6 with the C route. One native
C failure was recovered downstream in that campaign. These are small controlled
tests, not a general success-rate estimate or evidence that every C call succeeds.

## Scope of the freeze

- Keep every active pre-fitting comparison's pinned backend, initialization
  contract, starts, tolerances, budgets, and failure accounting unchanged.
- In particular, the matched construction pilot retains its existing general
  bounded-rollout fitter in both arms, as specified in
  [PREFIT_MATCHED_CONSTRUCTION.md](PREFIT_MATCHED_CONSTRUCTION.md). The current
  sensitivity-transfer adapter's `v01` restriction does not justify substituting
  it into the pilot's multiple-target case.
- Retain the existing collocation/refinement route where already configured.
  Do not replace it with alternating optimization, direct multistart, a new
  Hessian policy, or the smaller model on the strength of this diagnostic.
- Keep failed fits, poor fits, and unavailable derivatives visible. Preserve
  valid incumbents and distinguish numerical verification from prediction
  quality and native optimizer convergence.
- Keep this difficult case in limitations and result denominators. Do not
  relabel it as structurally impossible, silently exclude it, or relax the
  threshold after observing the results.
- Keep private reference equations, starts, hidden boundaries, and diagnostic
  results outside proposer/judge feedback. Only the established public evidence
  and fitting interfaces enter pre-fitting experiments.

The current [fresh construction audit](PREFIT_CONSTRUCTION_AUDIT.md) stops before
fitting and can proceed independently. The next project effort is pre-fitting
construction and its already planned comparisons. A concrete correctness or
integration defect can still be fixed
with appropriate versioning; reopening algorithm selection or this difficult
case requires a new explicit decision, not an automatic continuation.
