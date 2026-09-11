# Smooth composites in the forward-sensitivity backend

The sensitivity adapter previously rejected any occurrence of `abs`, `min`, or
`max`. It now checks complete state equations and observation mappings after
expanding algebraic process aliases. A small set of structural certificates
allows a differentiable composite to contain a nondifferentiable primitive.
This does not introduce smoothing, alter the proposed scientific function, or
claim general support for hybrid or nonsmooth dynamical systems.

## Supported certificates

| Form | Certificate and derivative behavior |
| --- | --- |
| `w*g/(c+abs(g))` | `c` is a literal at least the existing division epsilon (`1e-12`); the numerator contains an exact matching multiplicative `g`. The ratio is C1, including `g=0`. Its derivative with respect to `g` is `c/(c+abs(g))**2`. Other factors and the argument must themselves pass the expression audit. |
| `abs(g)**n`, positive even integer `n` | Compile the exact identity `g**n`, so higher derivatives at zero are also correct. |
| `max(g,0)**2` or `min(g,0)**2`, either argument order | C1 rectifiers, with first derivatives `2*max(g,0)` and `2*min(g,0)` respectively. The derivative is zero at the crossing. |
| `min(g,g,...)`, `max(g,g,...)` | Identical branches reduce exactly to `g`. |
| Literal-only `abs`, `min`, `max` | Evaluate the constant using the explicit whitelist. |
| Existing `softplus(g)` translation | The function is mathematically smooth, but its stable `max`/`abs` implementation has only its first AD derivative certified at zero. It therefore uses the same conservative solver-curvature policy. |

The exact candidate expression `f/(1+abs(f))` has derivative
`1/(1+abs(f))**2`, with value **1 at zero**. Its second derivative has different
one-sided limits there, so the expression is C1 but not C2. Multiplication by a
separate coefficient and either order of the denominator's addition are allowed.
The matcher uses structural equality, not numerical sampling or an LLM judgment.
For example, a process `q=f+a`, another process `r=abs(q)`, and output
`q/(1+r)` receive the same certificate after substitution. Expanded expression
size is bounded to prevent exponential process substitution from exhausting
resources. Candidate source trees and public schemas remain unchanged.

## First derivatives versus solver curvature

Forward parameter sensitivities still integrate
`S' = F_x*S + F_theta`, and observation sensitivities remain `H_x*S + H_theta`.
Those local first derivatives are obtained by AD of the certified complete
expressions. No arbitrary derivative of a bare `abs(0)` is asserted.

When a C1-only composite appears in an expanded **state equation**, the Jacobian
of the augmented state-and-sensitivity ODE would require unavailable second
derivatives. The adapter does not construct that AD Jacobian. It lets the stiff
ODE solver approximate its internal Newton matrix numerically. This does not
replace the analytic first parameter sensitivities with finite differences of
whole fitted trajectories. If only an observation contains the C1 composite,
the state-and-sensitivity ODE keeps its existing AD Jacobian.
[SciPy documents the numerical Newton-Jacobian option](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

If either the dynamics or observations contain a C1-only composite, the shared
collocation/shooting initializer uses IPOPT's `limited-memory` Hessian option.
It does not give IPOPT a fabricated exact second derivative at a kink in the
first derivative. Ordinary models retain the existing exact-Hessian setting.
[CasADi documents the Hessian approximation setting](https://web.casadi.org/python-api/).

The symbolic audit records certificate locations, the crossing policy, the
augmented solver Jacobian policy, and the initializer Hessian policy.
The fit retains this audit both in its result and in `sensitivity_audit.json`,
written before initialization so a timeout does not discard the policy record.
Source/runtime fingerprints change with the implementation, so old frozen
campaigns must keep their pinned checkout; new campaigns must be prepared afresh.

## Rejected cases and limits

Bare `abs(g)`, unequal-branch `min`/`max`, and uncertified combinations such as
`1/(1+abs(g))` fail before optimization. The error says that classical crossing
derivatives are uncertified and event/generalized sensitivities are unsupported.
The adapter does not assume that a trajectory will stay on one branch merely
because its initial state lies there. This is a backend limitation, not a claim
that the proposed scientific function is invalid.

`sqrt`, `log`, and negative/variable powers retain their conservative rejection
policy. A separate positive-domain and boundary-derivative certificate would be
needed to broaden them. A parameter-valued offset in `g/(c+abs(g))` is also not
covered by this milestone's literal-offset certificate. Existing guarded
division semantics and other legacy function translations are unchanged; these
new certificates are not a global smooth-domain proof for all supported syntax.
Finite integration, production replay, optimization convergence, and scientific
model adequacy remain separate checks.

## Verification

Focused tests cover the exact candidate observation across zero, analytic local
and trajectory derivatives at zero, parameter-dependent process aliases,
weighted ratios, even absolute-value powers, squared rectifiers, and rejected
crossings/domains. A state-equation crossing is checked with both Radau and BDF
against production rollouts and independent central parameter differences at two
steps. A complete train-only collocation/sensitivity fit recovers a known scalar
coefficient from the C1 observation and verifies a separate validation rollout.
All controls are synthetic; no benchmark, hidden labels, test metrics, or LLM
calls enter this validation.

Local verification: full repository suite **1,287 passed, 3 skipped** (optional
PyTorch unavailable); after adding persistent audit output, all **42 focused
sensitivity, collocation, and node-start tests passed**. Ruff and diff checks
passed. The exact-observation fit recovered `a=0.5` within `1e-6` with validation
NMSE below `1e-9`. These controls verify implementation, not recovery of the
current proposed benchmark model; that still requires the next frozen campaign.
