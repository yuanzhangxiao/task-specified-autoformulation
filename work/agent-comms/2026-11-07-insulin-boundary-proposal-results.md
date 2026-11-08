# Candidate 15 depletion proposal and feedback for the next revision

ACES job 2314569 completed its one GPT-OSS-120B call. Construction rejected the
reply with `UNKNOWN_FORMULA_SYMBOL` before the insulin boundary check. The
worker exited 0, stderr was empty, the reply finished with `stop`, and the
provider recorded 5,726 prompt tokens and 1,340 completion tokens in 14.10
seconds. The official result contains zero admitted models. The relevant
fitted reference is Orion’s constrained training NMSE of 0.165562.

## Reproducing the rejection

The new state’s equation used `τ_rise` and `τ_fall`. Its parameter declarations
used `tau_rise` and `tau_fall`. Replaying the cached reply reproduced the error
code and details exactly.

On a separate diagnostic copy, substituting those two formula identifiers
allowed the patch to compile and pass the three public path checks. The
existing runtime derived the sources from the equations. It added I and
dropped the unused insulin_decay reference from the insulin RHS source list,
and recognized the constant process referenced in the new state’s RHS.
This copy returned `eligible_for_refit` with the data and boundary checks both
deferred. The point diagnostic lacks a value for the new latent state at
minute 300. All subsequent calculations use this spelling-corrected copy.

## The proposed equations

Let P be the inherited insulin production process, X the insulin-action
state, r the supplied meal rate, J the existing infusion process, and
[z]₊ = max(z, 0). The new state D is called `direct_ingestion`.

```text
P  = k_G Gp + k_X X
I′ = c_I r + [P]₊ − g₂ [I]₊ + g₇ J + D
D′ = C {sigmoid((t − s)/τ_r) − sigmoid((t − e)/τ_f)}
C  = 154.27042889042568
```

The edit keeps the task-required I → X → U path. Taking the positive part of
P addresses the specific negative production term in the previous fit. The
self-term also changes. At positive I the inherited assembled self coefficient
was +0.01826 per minute, while the proposed coefficient at the inherited
g₂ is −1 per minute.

At the previous 150 g depletion point, I, r and external insulin are zero and
P is −0.801736 pmol/L/min. The new equation therefore gives **I′ = D** there.
D enters the equation directly. Its nonnegative trajectory-specific initial
value alone gives no sign bound for D later. The infusion process also has a
real-valued coefficient, so a sign argument for arbitrary nonnegative external
insulin would need to account for that coefficient.

## Two checks on the claimed behavior

**Meal timing and magnitude.** The new state's forcing depends on absolute
time and globally shared parameters. Every schedule, including a no-meal
control, gives exactly the same D′ at the same time. Changing D(0) shifts D
vertically. It cannot align this added pulse with meals shifted by 60 or 120
minutes. Other terms in the full model retain meal dependence.

The number C is the earlier fitted contribution of c_I r integrated over a
90 g meal. The observed total insulin increase was 1.97 pmol/L. The proposal
keeps c_I r and also uses this integrated quantity as a fixed coefficient in
D′. Since D enters I′ as a flux, using that concentration increment in D′
would require an additional time scaling to give it the appropriate units.

The sigmoid difference drives **the derivative of D**. With s = 0, e = 10,
τ_r = τ_f = 0.5 min as described in the reply and D(0) = 0, D accumulates to
1,435.77 at minute 10 and 1,489.24 at minute 300. A fixed-parameter replay of
the 90 g train_003 schedule with those settings and the inherited fit gives a
ten-minute insulin rise of **1,295.12 pmol/L**, versus the observed **1.97**.
The actual runtime guesses for s, e and both widths are all 1.5. At those
guesses the two sigmoid terms cancel and D stays at its initial value.

**Positivity.** A second fixed-parameter replay uses the 150 g train_005
schedule, the inherited shared coefficients and that trajectory’s fitted
initials. For the four new parameters set s = 0, e = 10, τ_r = 20 min and
τ_f = 1 min, with D(0) = 0. The start precedes the end and both widths are
strictly positive. Under this permitted choice the wider rise and narrower
fall let D become negative. Insulin moves from **+26.10 pmol/L at minute 31
to −1.53 at minute 32**. Radau and DOP853 agree within 0.000021 pmol/L for
insulin across the sampled times. The D trajectory also agrees with the
analytic integral of its two sigmoid terms.

These fixed-parameter examples test specific scientific claims. The second is
a counterexample to the claimed positivity guarantee across admissible
parameters. Fitting could choose other values. The attainable NMSE of this
family is unknown.

## Feedback for the next proposer call

1. Use the parameter names exactly as declared. Give any time constant used
   as a divisor a strictly positive domain.
2. Revisit the direct meal term in the complete insulin balance. Its inherited
   90 g contribution is +154.27 pmol/L, while the observed ten-minute rise is
   +1.97. Treat these numbers as evidence about the current fit.
3. If introducing a meal-dependent process, identify how the supplied meal
   rate or its causal downstream states generate the process. Explain how its
   response shifts with the meal, varies with dose and behaves on a no-meal
   schedule. Distinguish a flux from the integral of that flux.
4. Evaluate the entire insulin RHS at I=0. For each latent term whose sign the
   argument needs, show how its own dynamics and initialization provide that
   sign. Account for the permitted domains of the production and infusion
   coefficients. Preserve the public delayed insulin-to-disposal pathway.

The next construction experiment should repair the scientific response before
it goes to Orion for optimization. The spelling substitution is suitable for
an unambiguous, logged normalization in a future implementation. The historical
one-call run has a terminal rejection and its generation allowance was spent.

## Records

- Implementation `2e19c663662acb502737d4dc126d3e5d4d6276a9`.
- Returned archive SHA-256 `88697eb07139366ba989d7c80e01709591c484b7d94d5c5ea563907b1015ae83`.
- Plan `e240ab4f446d031f80f89c82eb263a433126b45368ef9d630a162e98e54bb57c`.
- Local analysis `output/insulin-boundary-result-20261107`, including
  `REPORT.html`, the visible reply, the spelling-only replay, trace data and
  the reproducible `audit.py` script.
- The audit used the two public training schedules named above. It made zero
  optimizer or provider calls and opened no held-out data.
