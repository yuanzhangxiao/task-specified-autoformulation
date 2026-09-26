# Equation-first comparison of the shortlisted T1 cases

Fifteen frozen models were inspected: the new named canonical T1-easy Full R6 candidate; both historical Full round-12 perturbed-obfuscated candidates; canonical-obfuscated R9 repaired and refitted-control endpoints; canonical-named repaired R2; and all three matching Sol repetitions in each of these three cells. No new model was fitted, revised, or rolled out. This audit does not select a winner using new intervention scores.

**The strongest equation-based lead is perturbed-obfuscated T1, comparing both Full seeds with all three Sol repetitions.** Our two candidates have stable conditional dynamics and positive delayed meal pathways. Sol 0/1 have specific directional/feedback defects; Sol 2 has a credible delayed pathway and remains a strong comparator. Conversely, the very low-error named R6 and R9 refitted control have wrong source/sink signs. Their low NMSE does not make them the preferred mechanism demonstrations.

## What was verified

Every model compiles under its actual lowered/public initialization context. All 15 saved parameter vectors respect their declared domains and explicit bounds. The expanded, parameter-substituted symbolic equations agree with the production restricted evaluator in 30 RHS comparisons, maximum absolute discrepancy 2.98e-14. This checks reconstruction and parameter admissibility, not scientific correctness or every state constraint.

The audit distinguishes:

1. Analytic/schema validity and allowed parameter domains.
2. A syntactic dependency versus an appreciable fitted contribution.
3. An outer sign versus the net derivative of the complete fitted law.
4. An incremental response versus the absolute trajectory and latent-state interpretation.
5. Public task requirements versus post-hoc physiological interpretation.

T1 requires a causal meal/input-response mechanism; it does not explicitly require insulin secretion, every supplied auxiliary, or recovery of the complete physiological system. Anonymous prompts do not disclose physiological channel meanings. Decoding v01=Gp, v02=EGP, v03=Uii, v04=E, v05=Gt, u01=meal input is analyst-only here; it must not become hidden-reference feedback to an anonymous proposer. Several of our states have unspecified units and no nonnegativity declaration: a negative initializer is a scientific-interpretation concern, not automatically a violated runtime constraint.

## Canonical named: the low-error R6 has the worse balance structure

The new `cell00_seed0_full_c1v0s1` selected R6 endpoint has critic and shared processes enabled, scientific verifier disabled. Saved train/validation NMSE is .03323/.01289. Its equations, with rounded fitted coefficients, are

```text
M'  = 0.1 u - 0.0333333 M;  M(0)=0
Gp' = M - 0.529480 Gp + 0.856270 Gt
      - 16.653092 EGP + 10.559978 Uii + 491.059456 E - 0.043886 u.
```

Production reduces plasma glucose, utilization and excretion increase it, and a positive meal has an immediate negative contribution before M accumulates. The latent meal filter itself is positive and stable; that does not repair these target terms. At fixed other states/channels, the derivatives are exactly the coefficients displayed. None resulted from a parameter crossing its declared admissible domain.

All three matching named Sol models instead contain

```text
Gp' = positive meal appearance + EGP - Uii - E - 0.065 Gp + 0.079 Gt,
```

with positive stomach-to-gut pathways and zero meal-depot initial values. These are the expected canonical plasma-balance signs and exchange coefficients. Therefore the equations do not support labeling R6 mechanistically superior to these Sol models.

There is a narrower Sol weakness worth recording. Its dose-reference variable differs between repetitions: Sol 0 uses `D' = 0.5 u (u-D)`; Sol 1 uses `D'=u`; Sol 2 uses `D'=10 tanh(u)(u-D)`. These are different memories of pulse amplitude/cumulative dose, not equivalent encodings of a meal event. The existing reference generator sets its meal-size reference to remaining stomach content plus the new meal at each event. Repeated meals following stomach clearance can discriminate these dose-memory rules. This does not imply that R6, with its wrong balance signs, will be more accurate. The pulse-versus-jump input representation must remain explicit.

## Canonical obfuscated: R9's predictive success is not a sign-recovery success

The low-error refitted R9 control has

```text
w'  = -0.140206 w
x'  = u - 0.0186386 x
Gp' = 0.101234 x + 1.26234e-6 w - 0.0866328 Gp + 0.116439 Gt
      - 1.911554 EGP + 2.940452 Uii + 145.985584 E - 0.435588 u.
```

It repeats the production/utilization/excretion errors and adds negative direct meal feedthrough. Its fitted w(0)=-2.728 million contributes approximately -3.44 to Gp', because the gain is tiny; state magnitude alone is not the correct influence measure. It cannot be presented as the clean model that wins by correct signs.

Assisted R9 and R2 have better outer source/sink directions. However, fitted production gains are only 1e-10 and 2.26e-27 respectively; the production pathway is effectively inactive. Their meal-effect decay times are approximately 200 and 203 minutes. Repaired R9's negative initialized fast component contributes about -5.94 initially, despite its positive outer gain, while its slow component contributes +2.85. Repaired R2 is the cleaner positive meal-memory representation of the two, but remains an approximation with almost absent EGP response. An omitted optional auxiliary is not itself a deterministic T1 failure; the mechanistic consequence must be stated as restricted sensitivity, not invented prompt noncompliance.

Matching Sol repetitions are heterogeneous:

- Sol 0 has no additional meal state and has EGP coefficient -18.9703. It is a poor direct-production interpretation, despite good prediction using supplied tissue/production trajectories. The auxiliary channels themselves contain history, so absence of a latent state alone does not prove inability to predict delayed observations.
- Sol 1 has a positive three-stage meal chain, but EGP coefficient -.26519 and excretion coefficient +67.1343. The meal pathway is more credible than the physiological balance.
- Sol 2 has approximately canonical production, tissue and plasma terms, but a signed meal basis `0.0898 q_slow - 0.2505 q_fast + 0.1771 q_delay`. For a positive isolated input impulse with the auxiliary trajectories fixed, the immediate change in this meal-source contribution is `(0.0898-0.2505) dose = -0.1607 dose`, because q_delay has not yet accumulated. This is a net early negative response, not a judgment based on the sign of one internal coordinate. It is a useful conditional mechanism probe; regenerating physical auxiliaries can change the total glucose response.

Thus neither side is uniformly mechanistically correct in this cell. Correcting source/sink operators improves one property without guaranteeing the complete fitted law or initial state is appropriate.

## Perturbed obfuscated: a substantive positive-memory contrast

Both historical Full round-12 models have nonnegative incremental meal response and stable linear state dynamics when supplied auxiliaries are held fixed. These statements follow from the actual fitted matrices, not names or graph reachability: off-diagonal state coefficients and meal-input gains are nonnegative, all eigenvalues have negative real part, and the target is a state component. Therefore `delta Gp(t) >= 0` for an added nonnegative meal forcing with identical initial state and auxiliary histories. This is an incremental positivity statement, not absolute state positivity from the fitted starts.

Seed 0's principal meal path is

```text
z'  = 0.144788 u - 0.0155015 z
Gp' includes +0.421027 z.
```

It has faster positive meal filters and a Gp-to-m-to-Gp feedback loop. Although that loop is positive, the coupled fitted eigenvalues are approximately -.01749 and -.26022 per minute: this is not an unstable feedback loop. The slow z time constant is 64.5 minutes. Another w state has decay time about 3,864 minutes and negligible meal gain 5.03e-19; it functions mainly as a fitted initial transient, not a meaningful active meal mechanism. Seed 0 ignores all four supplied physiological auxiliaries.

Seed 1 has

```text
y'  = 0.769211 u - 0.0151359 y
Gp' = 0.0705976 y - 0.0149814 Gp
      + 0.0288349 w + 0.0825629 x + 45.3119 z + 4.09e-8 u,
```

where w,x,z are EGP-driven filters. The meal-memory time constant is 66.1 minutes; target relaxation is 66.7 minutes. All state eigenvalues are negative. It omits direct tissue-glucose, utilization and excretion dependencies, and one EGP memory decays over about 2,380 minutes. Both Full seeds have some negative latent initial values. These are stable task-oriented reduced models, not recovered physiological mass-balance models.

The three matching Sol models provide distinct comparisons:

| Sol repetition | Equation finding | Consequence with auxiliary histories fixed |
|---|---|---|
| 0 | No latent meal state; `Gp'=-100 tanh(P)`, where the coefficient of u in P is +1.624626e-5 | `partial Gp'/partial u = -0.001624626 sech(P)^2 < 0`: direct meal response has the wrong direction |
| 1 | No latent meal state; `Gp'=100 tanh(P)` with positive local Gp feedback | At the public initial state with u=0, `partial Gp'/partial Gp = +0.0662824 /min`; local perturbations amplify rather than relax |
| 2 | Positive three-stage meal pathway, zero initial meal states, negative target self-coupling | A plausible stable delayed-response competitor; the first-two-repetitions criticism does not apply |

Sol 0 also has positive local target feedback at that same boundary, +.0211615 /min. Sol 0/1's tanh caps the derivative at 100 in magnitude; it does not ensure homeostasis, nonnegative glucose, or small long-horizon error. This is consistent with, but does not uniquely attribute, their large historical rollout errors. No new trajectory-based causal attribution was performed.

Sol 2's fitted equations are

```text
F1' = u - 0.05 F1
F2' = 0.05 F1 - 0.05 F2
H'  = 0.05 F2 - 0.0105 H
Gp' = 0.413 H - 0.179832 Gp + 1.905621 EGP - 0.0161790 Gt + 29.495689.
```

Its negative Gt coefficient must not be dismissed using the canonical rule. In the perturbed reference, tissue-return flux is

`J(Gt)=0.079 Gt [1-0.25 tanh((Gt-Gtb)/(0.2 Gtb))]`.

At basal Gtb its derivative is -.01975, even though the flux itself is positive. Sol 2's -.01618 coefficient can approximate this local derivative. It does not recover the nonlinear global flux, but it is not evidence that the model simply confused a source with a sink. This is why fitted net derivatives and operating conditions matter more than counting minus signs.

## What interventions the equations justify

There is no theorem that a better skeleton must yield a better fitted trajectory somewhere in the particular admissible input range. Coefficients, initializers, approximation error and supplied mediators can dominate. Equations do provide falsifiable response predictions.

The next defensible case is **perturbed-obfuscated T1**, retaining both Full seeds and all three Sol repetitions. Separate two questions:

1. **Conditional mechanism diagnostic:** hold auxiliary histories and initial states identical, add an input pulse, and measure the sign, delay and decay of the incremental target response. Our two models have a provably nonnegative incremental meal pathway; Sol 0 has a negative direct response. This is a component-level input probe. Holding physiological auxiliaries fixed is not automatically a feasible whole-patient intervention and must not be labeled as one.
2. **Complete physical schedule test:** regenerate the reference and every permitted auxiliary consistently for every schedule, then replay all five fitted endpoints unchanged. Use a small predeclared grid spanning meal magnitude, spacing and washout; report every outcome. This tests whether the conditional mechanism advantage actually translates to target accuracy. Sol 2 must remain included, and both our long-lived fitted memories may fail washout.

A proposed compact grid is single meals of 30/60/120 g at minute 60, fixed-total 60 g split equally with 15/30/60/120-minute gaps, plus a long-washout pair of 60 g meals at minutes 60 and 540 (horizon 900 minutes). The long-washout case is a new stress test, not presumed representative. Freeze any final schedule list and metrics before generating its outcomes; do not keep extending the grid until a winner appears. These are post-hoc exploratory diagnostics, separate from the fixed benchmark test suite. This audit did not run that grid.

For named canonical Sol, repeated meals after washout can additionally probe the cumulative/latched dose-reference implementations. Since named R6 has worse balance signs, this is an audit of dose encoding, not a promised win by our method.

## Files and verification

Generated analysis files, uncommitted, are under `artifacts/t1-equation-comparison-2026-09-25/`:

- `EQUATIONS.md`: all 15 fitted state/process systems, initial values and direct derivatives.
- `models.json`: exact frozen candidates, parameter vectors and source identities; external test-score fields removed.
- `audit.json`: input hashes, reconstructed equations, graph dependencies, direct derivatives, conditional stability/positivity checks, domain results and 30 production-RHS comparisons.
- `audit.py`: restricted-AST inspection orchestration; no production runtime edits.

Focused regression tests: 20 passed in 1.36 s. Whole-repository Ruff still reports the same 37 existing findings under `analysis/claude`; unrelated files were unchanged. All 30 actual-model numerical reconstruction checks passed. No fitting, new rollouts, live LLM calls, remote sessions or test-trajectory access occurred. Reference-generator code was read only to interpret the existing perturbation and dose-reference rules; it was not fed to a proposer or selection process.
