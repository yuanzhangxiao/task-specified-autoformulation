# Phase C M3: optimization size, failures and coefficient recovery

Detailed archive review, 2026-10-01 Hawaii time. This supplements the
[summary review](PHASE_C_FITTING_STRATEGIES_RESULTS_2026-10-01.md) and
[frozen protocol](PHASE_C_FITTING_STRATEGIES.md). No new fitting, test access,
benchmark modification or production-default change was performed.

## Evidence and verification

Source: `review-20261002-014532.tar.gz`, extracted locally under
`artifacts/phase-c-fitting-m3-review-20261002-014532` (untracked experiment data).
Plan identity:
`65a0a6db305e19b1d6a6d1ffcb0ac20f4574f46694368627d6e36b36bbe0c466`.
All 296 JSON content seals were verified. All 72 result records match their
summary rows and their backend digests. There are 71 complete independent
replays and one unavailable replay, `cstr_hard_s2_adaptive_shooting`.

Reference parameters were read only for this post-selection evaluation. They
were not passed to the optimizer, used to choose checkpoints or used to stop
fitting. The following diagnostics do not change the frozen accuracy criteria.

## Did rollout-only fitting avoid difficult trajectories?

Large output errors occurred. Fast/slow start 1 began at training NMSE 16.07.
CSTR-hard start 2 began at 0.522, evaluated a trial at 11.22, and ultimately
retained a point with training NMSE 0.000379 and validation NMSE 0.000183.
CSTR-easy start 2 evaluated a trial at 9.70. A trial point need not be accepted
by the optimizer or retained as the incumbent.

No saved completed rollout-only evaluation reported an integration failure:

| Strategy | Evaluated full-training records | Timeout records | Failed records |
|---|---:|---:|---:|
| Rollout only | 298 | 0 | 0 |
| Collocation then rollout | 227 | 1 | 2 |
| Direct collocation | 266 | 9 | 1 |
| Adaptive multiple shooting | 198 | 1 | 0 |

These are saved training-oracle records, not native NLP iteration counts or
complete accounting of interrupted in-flight work. Four rollout-only fits
(easy start 0 and all hard starts) hit the outer time limit. An interrupted
evaluation can lack a finished record. Absence of reported failures does not
establish absence of internal step rejection, stiffness or large latent-state
excursions. The three reported failures in other arms say that the required
integration step became smaller than floating-point time resolution; this is
a numerical failure, not by itself proof of physical instability.

## Physical unknowns and optimization constraints

Every case has one measured output. Coefficients and fitted hidden initials
are shared across training trajectories. Observed initial values are fixed.

| Case | ODE states | Dynamic coefficients | Fitted hidden initials | Rollout unknowns | Training trajectories | Scalar observation residuals |
|---|---:|---:|---:|---:|---:|---:|
| Linear control | 2 | 3 | 1 | 4 | 3 | 363 |
| Cubic, fast/slow or saturating control (each) | 2 | 4 | 1 | 5 | 3 | 363 |
| CSTR easy | 3 | 7 | 0 | 7 | 16 | 4,816 |
| CSTR hard | 3 | 7 | 2 | 9 | 16 | 4,816 |

CSTR easy still has three dynamic states. Its auxiliary initial readings fix
initial concentration and jacket temperature. Hard fits those two shared
initial values. It does not introduce separate unknown initials for each run.

Rollout fitting uses physical parameter bounds but no explicit ODE equality
constraints or path-state inequalities. Integration enforces dynamics to its
numerical tolerance. Controls bound `a,b,c` (and `d`, where present) to
`[0.001,100]`; the saturating parameter `q` is in `[0.001,10]`. Control hidden
initial velocity is unbounded. CSTR's seven dynamic coefficients are
nonnegative. Hard additionally bounds initial concentration below by zero;
initial jacket temperature is unbounded. Thus scalar bound-inequality counts
are 6, 8, 7 and 8 for linear, other controls, easy and hard respectively.

The CSTR RHS guards `max(T,250)` and `max(C,0)` are part of the supplied
equations, not explicit optimizer constraints on the state trajectory.

Trajectory transcription introduces many more unknowns. First-mesh counts:

| Case | Collocation variables | Collocation dynamic equalities | Shooting variables | Shooting continuity equalities | Bound inequalities in either NLP |
|---|---:|---:|---:|---:|---:|
| Linear | 1,444 | 1,440 | 116 | 112 | 6 |
| Cubic | 1,445 | 1,440 | 121 | 116 | 8 |
| Fast/slow | 1,445 | 1,440 | 97 | 92 | 8 |
| Saturating | 1,445 | 1,440 | 113 | 108 | 8 |
| CSTR easy | 28,807 | 28,800 | 958 | 951 | 7 |
| CSTR hard | 28,809 | 28,800 | 960 | 951 | 8 |

Two-stage Radau has two state vectors per interval: CSTR therefore introduces
`2 * 3 * 300 * 16 = 28,800` trajectory variables. The first boundary is
expressed through initialization, not an additional independent state vector.
Hard collocation later grows to 36,009 and 45,033 variables. The hybrid's first
stage also uses the dense Radau transcription before reducing to rollout
optimization. These counts include sparse, locally coupled variables; dimension
alone does not determine difficulty. Bound inequalities need not all be active.

## Objective geometry

Let `eta=(theta,alpha)` contain dynamic coefficients and shared initialization
parameters. For trajectory `j`, integration produces
`x_j'=f(x_j,u_j;theta)`, with `x_j(0)=i(y_j(0),u_j(0);alpha)`. Training minimizes

```text
L(eta) = (1/N) sum_{j,i,k} [(h_k(x_j(t_i;eta))-y_{jik}) / s_k]^2,
```

where `s_k` is the training-channel standard deviation and `N` counts observed
values. SciPy's one-half sum of squared residuals has the same minimizer after
constant rescaling. There is no extra coefficient regularizer or reference prior.

The loss is quadratic in residuals, not in coefficients or initials. Even a
linear ODE has an exponential dependence on its rate parameters. CSTR also has
Arrhenius nonlinearity and parameter/state coupling. The guarded RHS is only
piecewise differentiable. Rollout uses bounded trust-region reflective least
squares with forward sensitivities to build local approximations. No global
landscape survey or count of local minima has been performed.

In these cases, direct collocation has a quadratic observation objective in
nodal observed states, but nonlinear (sometimes bilinear) dynamic equality
constraints. Its full optimization problem is not a convex quadratic program.
Shooting has nonlinear integration maps in both prediction and continuity.

Reference output-Jacobian checks have full local numerical rank. For CSTR the
column-normalized smallest/largest singular-value ratios are 0.00433 (easy)
and 0.00248 (hard), corresponding to condition numbers about 231 and 403.
Their Gram-matrix condition numbers are about 53,300 and 162,000. These indicate
weak local directions; they are neither the actual optimizer Hessian condition
numbers nor a proof of global identifiability or nonidentifiability.

## Coefficient accuracy, separated from initial-value accuracy

For each endpoint define
`E_coeff = 100 * max_k abs(theta_hat_k-theta_ref_k)/abs(theta_ref_k)`
over dynamic coefficients only. All reference coefficients are nonzero.
Report the median and worst error across starts, not the best run.

| Strategy | Controls: median / worst (%) | CSTR easy: median / worst (%) | CSTR hard: median / worst (%) |
|---|---:|---:|---:|
| Rollout only | 0.000120 / 0.00286 | 0.00481 / 0.148 | 6.77 / 35.8 |
| Collocation then rollout | 0.0000772 / 0.000522 | 28.5 / 71.5 | 99.9 / 169.7 |
| Direct collocation | 0.00668 / 0.0146 | 95.7 / 100.0 | 97.8 / 127.9 |
| Adaptive multiple shooting | 0.000996 / 43.2 | 98.5 / 99.1 | 87.3 / 94.0 (2/3 available) |

There are 12 control endpoints and three starts per CSTR tier per strategy.
Shooting has only two returned hard-CSTR coefficient vectors; its unavailable
third vector must remain a failure/unavailable outcome, not a silently excluded
success. At a diagnostic threshold of every coefficient within 1%, pass counts
are respectively 12/12, 12/12, 12/12, 11/12 on controls; 3/3, 1/3, 0/3, 0/3 on
CSTR easy; and 0/3 for every strategy on CSTR hard. This diagnostic threshold
does not replace the original CSTR output criterion.

For hard rollout-only starts 0/1/2, absolute initial-concentration errors are
0.000407 / 0.001566 / 0.008407, and initial jacket-temperature errors are
0.0742 / 0.4102 / 3.1315 K. Absolute temperature errors avoid dependence on
the arbitrary zero used for a relative-temperature percentage.

The last three saved rollout-only trial points show maximum coefficient errors
decreasing toward the hard-CSTR time limit:

| Start | Third-last (%) | Second-last (%) | Last (%) |
|---|---:|---:|---:|
| 0 | 10.36 | 3.96 | 1.36 |
| 1 | 17.04 | 14.36 | 6.77 |
| 2 | 42.12 | 39.73 | 35.78 |

Training losses also decrease across those points. This makes unfinished
optimization a supported explanation for at least part of the remaining error.
It does not prove that additional time recovers every parameter or that every
parameter is globally identifiable.

## Interpretation and follow-up

- Rollout-only leads on CSTR output and coefficient accuracy. The hybrid is
  slightly more precise on the small controls, where both are already excellent.
- The setting is favorable: correct fixed equations, known coefficient domains,
  dense noiseless observations, fixed observation maps, multiple excited
  trajectories, and only four to nine physical unknowns. Arbitrary learned
  equations, larger hidden-state systems and noisy data are not covered.
- CSTR is nevertheless computationally nontrivial: each sensitivity evaluation
  integrates many trajectories and all hard rollout fits consume the budget.
  Small parameter dimension does not make these evaluations cheap.
- Hard collocation repeatedly uses its approximately 40--45-second native
  stage allowance while building progressively larger NLPs. Graph construction
  alone takes roughly 7, 9 and 11 seconds across meshes; recorded checkpoints
  reach only about 10--12, 5--7 and 3--5 IPOPT iterations. The current strategy
  can refine a mesh before adequately optimizing its coarse problem. This is
  evidence about this implementation/budget, not intrinsic method inferiority.
- Next separate two questions: allow a bounded additional budget on hard-CSTR
  starts to measure continuing coefficient recovery; and broaden identifiable,
  correct-equation tests to farther starts, noisy/sparse observations and more
  difficult dynamics. Consider holding the coarse collocation mesh longer and
  reusing graph work before adaptive refinement. No new campaign is launched
  or production default promoted by this review.

Primary numerical references:
[SciPy least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
and [CasADi trajectory transcriptions](https://web.casadi.org/docs/#direct-multiple-shooting).
