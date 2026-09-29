# Fitting findings to revisit in Phase C

Recorded 2026-09-28. **Decision: defer further fitting development and experiments
until the Phase C fitting-improvement stage. The immediate priority is to qualify
the new datasets and their public information contracts.** This note preserves
the evidence and proposed work; it does not change the production fitter.

See [reference qualification](PHASE_C_REFERENCE_QUALIFICATION.md), the
[frozen CSTR follow-up](PHASE_C_CSTR_REFINEMENT.md), and the
[Phase C handoff](PHASE_C_START_HERE.md).

## Result worth preserving

With supplied correct CSTR equations, a generic numerical start, and a revised
hidden-state initializer, the fitter achieved independent free-rollout NMSE
**0.00014821 on training and 0.00006713 on validation**. Parameters and the shared
initialization map were fitted using training only. This is a strong numerical
result on an assisted, known-equation problem. It is not end-to-end discovery,
unique parameter recovery, or evidence that all latent states were recovered.

The change combined nonnegative concentration initialization with better-scaled
initializer coordinates. A useful intermediate collocation checkpoint survived
the time limit and was selected by actual training rollouts. The subsequent
derivative-free polling did **not** improve that winning checkpoint.

## Experiment identity and scope

- Protocol: `phase-c-cstr-refinement-1`; source commit
  `e763dd3ffc05b25d447a95aab9c2b4504994b17a`.
- ACES root:
  `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-cstr-refinement-v1`.
- Plan SHA-256:
  `f5d96009add1519f71222708a6fcb206698462ac2b5cacf24206f349a890a5f0`.
- Reviewed `cstr-refinement-review.tar.gz` SHA-256:
  `f8fc41d305e2d4f85bda717017711b1e317a1a172f76254a0da59c0939b368ca`.
- Local review copy: `artifacts/cstr-refinement-review-20260928/`.
  The review verified 72 artifact seals, 14 freeze identities and 14 backend
  result links. Large artifacts remain outside Git.
- Fourteen completed attempts: two tiers, two starts, three common arms, and
  two additional hard-tier initializer comparisons. Sixteen training and four
  validation trajectories per case. No LLM, pruning, GPU, or test-data evaluation.
- Equations were evaluator-supplied. `reference_near` used 0.8 times reference
  dynamic parameters and reference initializer guesses. `generic` used frozen
  guesses; changing initializer coordinates preserved its original initial
  concentration 0.25 and jacket temperature 345 K.

Most original array workers failed before optimization because their shared
environment could not import dependencies. Recovery used an isolated group
environment, preserved the one completed result, and ran the missing tasks.
Receipts are under `submission/env-recovery-1/`. The logs did not establish the
underlying filesystem cause. An import failure is not a numerical failure.

## All results, including unsuccessful starts

Each entry below is **training / validation NMSE from independent replay**.
All arms retain the same equations and development data. Standard budgets were
120 seconds for initialization, 180 seconds for refinement and 240 residual
calls. `wide_long` used 300 / 900 seconds and 1,200 calls. The initializer arm
has the standard budget and local polling.

| Tier and start | `wide_standard` | `local_standard` | `wide_long` | `local_physical_initials` |
|---|---:|---:|---:|---:|
| Easy, reference-near | 4.209e-10 / 1.542e-10 | 4.209e-10 / 1.542e-10 | 4.219e-10 / 1.541e-10 | Not run |
| Easy, generic | 0.942982 / 0.689101 | 0.847470 / 0.520805 | 0.747761 / 0.372513 | Not run |
| Hard, reference-near | 0.166691 / 0.180110 | 0.024296 / 0.014129 | 0.159940 / 0.174282 | 0.021618 / 0.005345 |
| Hard, generic | 0.799855 / 0.408068 | 0.799433 / 0.402587 | 0.799236 / 0.398897 | **0.00014821 / 0.00006713** |

The diagnostic threshold required both split NMSEs <= 0.01 and independent
solver agreement. Four attempts passed: the three easy/reference-near arms and
the hard/generic initializer arm. These are not four independent benchmark cases.
All fourteen reported exhausted refinement budgets. Only the three easy/near
collocation initializers reported convergence; the others stopped on time or
iteration limits. Completion, numerical accuracy and convergence are different
facts.

The combined initializer change reduced hard/generic validation error by about
6,000 times relative to `local_standard`. Local polling also helped hard/near.
Simply adding budget did not make the generic starts reliably accurate. There
are no repeated numerical seeds or broad-family comparisons here.

## What the initializer change actually does

The original hard maps were unrestricted affine functions:

```text
C(0)  = a_C + b_C * (T(0) - 350)
Tj(0) = a_j + b_j * (T(0) - 350)
```

For the revised maps, take training initial-temperature anchors
`L = 365.131295254 K`, `D = max(max_train_T0 - L, 1 K) = 5 K`, and
`w = clip((T(0)-L)/D, 0, 1)`:

```text
C(0)  = (1-w)*c_lo + w*c_hi,     c_lo >= 0, c_hi >= 0
Tj(0) = T(0) + offset + change*(T(0)-L)/D
```

All coefficients remain shared and training-fitted; validation uses the frozen
map. The jacket map is an equivalent affine representation in deviation
coordinates. The concentration map additionally changes the admissible domain
and saturates outside the training anchor range. That saturation is a modeling
choice, not a pure numerical reparameterization. Positivity and coordinate
changes were bundled, so this experiment cannot isolate their individual gains.

Nonnegative `C(0)` does not establish positivity of arbitrary state trajectories.
The jacket initializer output was not separately constrained. This assisted
experiment used known physical roles; it does not establish that an LLM inferred
the roles or justified the constraints correctly.

## Why retaining intermediate checkpoints mattered

In `cstr_hard_generic_local_physical_initials`, the initializer hit its 120-second
wall-clock limit. Its retained `best_feasible` checkpoint was iteration 121 at
105.026 seconds, with maximum collocation constraint violation `3.442e-7`.
Subsequent full training-rollout screening reported:

| Screened point | Rollout cost |
|---|---:|
| Best feasible collocation checkpoint | 0.3568915191 |
| Least-violation checkpoint | 0.3588743744 |
| Latest checkpoint | 0.4917552577 |
| Original generic start | 9235.433177 |

These are the fitter's rollout costs, not NMSEs or collocation objectives. The
final parameters equal the best-feasible checkpoint exactly. Local polling made
36 calls (35 valid, one timeout), within 46 total refinement residual calls, and
left that cost unchanged. Therefore the strongest result supports preserving and
screening intermediate iterates; it does not demonstrate a polling improvement.
Checkpoint retention and rollout screening were already implemented before this
experiment and should be preserved during future refactoring.

Independent Radau and DOP853 replay agreed to about `1.22e-7` training temperature
standard deviations at the retained point. The reported errors use freely
integrated trajectories, not optimized collocation state samples.

## What low observable error does not establish

For the winning hard/generic fit, some quantities remain far from the reference:

| Quantity | Reference | Fitted |
|---|---:|---:|
| Activation coefficient | 25 | 25.0162 |
| Heat exchange coefficient | 2.5 | 2.7805 |
| Jacket exchange coefficient | 1.5 | 2.0922 |
| C(0), at T(0)=L | 0.26193 | 1.37765 |
| Tj(0), at T(0)=L | 347.56565 K | 261.41810 K |

All four validation runs have T(0)=L. Thus the validation result tests changed
inputs at that observed initial condition; it does not test arbitrary hidden
initial preparations. Parameter and hidden-state errors can compensate in the
observable temperature. These results do not prove structural nonidentifiability,
but they rule out describing this fitted endpoint as accurate full-state recovery.

## Ideas agreed for the later fitting stage

1. **Center and scale optimizer coordinates.** For an invertible affine change
   `theta = mu + D*z`, the Hessian at corresponding points is
   `H_z = D^T H_theta D`. Suitable scaling can improve conditioning even when
   variables have no physical units. It is not guaranteed for arbitrary scales;
   translation alone does not change the Hessian at corresponding points.
   Centering a regressor also changes the intercept/slope parameter basis and can
   reduce their correlation. No Hessian condition number was measured here.
   Scaling cannot create missing information or remove nonidentifiability.
2. **Use justified domains.** A verified concentration can be nonnegative; a
   signed deviation need not be. An LLM-suggested domain is a hypothesis until
   supported by the public contract or a verified modeled definition. Checking
   that a quote exists cannot prove that it entails the proposed constraint.
   Keep physical state domains, parameter domains, and numerical safeguards
   distinct. Current proposal constraints are normally marked proposer/soft;
   typed parameter roles and legacy paths need explicit provenance review before
   claiming this rule is enforced everywhere.
3. **Preserve useful intermediate iterates and compare actual rollouts.** Keep
   feasible, least-violation and latest checkpoints; select on full training
   rollout evidence, retain budget/convergence status, and independently replay
   the final point. A timeout does not erase an already useful solution.

Polling means trying nearby parameter vectors in selected directions and
retaining a point when its training rollout improves. It does not mean repeatedly
checking job status. The local profile starts with smaller steps and alternates
coordinate and seeded orthogonal directions; the original profile starts with a
wide sweep. Neither supplies a stationarity or global-optimality certificate.

## Proposed integration work, explicitly deferred

- Separate scaling-only, positivity-only and combined initializer changes under
  equal budgets and equivalent initial guesses. Make extrapolation behavior
  explicit. Include multiple starts and another benchmark family.
- Introduce general coordinate transforms with recorded inverse maps, transformed
  bounds and resume identity. Derive scales from training/public information;
  do not hard-code CSTR reference parameters into discovery defaults.
- Audit domain provenance and test signed latent deviations as counterexamples
  to automatic positivity. Distinguish initial-value and whole-path constraints.
- Preserve checkpoint accounting, guard-aware rollout behavior and independent
  replay. Compare error, physical validity, success frequency and actual cost.
- Qualify broader benefit before changing default profiles. A single successful
  assisted case is insufficient for a production-default claim.

No new cluster experiment is required to archive these findings. Before resuming
fitting work, finish dataset qualification: public input semantics, reference
replay, available initialization information, and faithful representation under
the public modeling interface. The 260/260 numerical and 40/40 projection audit
passes do not settle alien-device hidden preparation or Dalla Man gastric
bookkeeping. Those are separate dataset/interface questions.
