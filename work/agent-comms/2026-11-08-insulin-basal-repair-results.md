# Candidate 15 basal secretion repair

ACES job 2315743 completed one GPT-OSS-120B call on two H100s. The reply
introduced a shared nonnegative basal secretion rate and used
`direct_ingestion` without declaring it. The fitted parent has no such state
or process. It appeared in the earlier rejected draft. The runtime rejected
the new reply in the insulin equation with `UNKNOWN_FORMULA_SYMBOL`. There
were zero spelling substitutions. No child model, refit handoff or new fitted
NMSE was produced.

The request contained the deterministic feedback about the earlier clock
proposal. The new undefined-name diagnostic was generated after the sole
proposer call. It has yet to be presented to the proposer in a correction turn.

## Equation audit

Let $r$ be the supplied meal rate, $v$ insulin infusion, $X$ insulin-action
memory, $B$ the new basal rate and $D$ the unresolved direct-ingestion term.
At the inherited primary constrained coefficients, the fitted parent and
the returned expression are

$$
\begin{aligned}
\dot I_{\rm parent}
  &=1.714116r + 0.000509944G_p - 1.072581X + 0.018264I + 8.838773v,\\
\dot I_{\rm reply}
  &=1.714116r + 0.000509944G_p - 1.072581X + B - I + 8.838773v + D.
\end{aligned}
$$

The replacement uses the signed production process directly. Its written
explanation treats the remaining production as nonnegative. The equation
also changes the net self coefficient from +0.018264 to −1 per minute.
The reply lists insulin decay as a source but uses $I$ directly. After a
valid equation has been supplied, the deterministic parser can recover that
source metadata.

At the parent's recorded 150 g depletion point, $I=r=v=0$, $G_p=240.84609$
and $X=0.86199$. The glucose and action terms give
$+0.122818-0.924554=-0.801736$ pmol/L/min. The returned balance therefore
reduces to $\dot I=B+D-0.801736$. With $D=0$, $B$ must be at least 0.801736
to point inward or be tangent at this particular state. Positivity of $B$
alone gives no such bound. The archived run stopped before its boundary check.

## Fixed-parameter diagnostics

Two separate analyst copies make the ambiguity explicit. The first removes
the final $+D$ and its source-list entry. The second restores the prior
draft's state, initializer, constant process and four clock parameters,
while keeping the newly returned insulin equation. Both copies pass the
three public path checks and receive deferred training-consistency and
boundary-sign assessments. Neither copy is an official proposer response.

Each replay uses the same 16 public training schedules and fitted initials.
The restored clock uses start 0, end 10, rise width 20 and fall width 1 minute,
with $D(0)=0$, as in the previous counterexample. The quantities below
are predictions with frozen values. Brackets give unscaled MAD. A negative
sample is below −0.0001 pmol/L.

| Completion | $B$ | Trajectories with a negative sample | Minimum insulin per trajectory, median [MAD] (pmol/L) | Absolute peak-time error on three shifted 90 g meals, median [MAD] (min) |
| --- | ---: | ---: | ---: | ---: |
| Omit $D$ | 0 | 16/16 | −0.484 [0.102] | 95 [0] |
| Omit $D$ | 0.1 | 16/16 | −0.384 [0.102] | 95 [0] |
| Omit $D$ | 1.5 (runtime guess) | 0/16 | 1.008 [0.102] | 95 [0] |
| Restore previous $D$ | 1.5 | 16/16 | −580.424 [0.098] | 149 [59] |

With $D$ omitted and $B=0.1$, the 150 g replay reaches −0.471 pmol/L at
minute 13, when observed insulin is 47.013. At $B=1.5$, the lowest sampled
insulin over all 16 trajectories is +0.0136. On 90 g meals at minutes 0, 60
and 120, the peaks occur at 6, 66 and 126 minutes, reaching 26.96, 27.72
and 27.87 pmol/L. Observations peak at 101, 161 and 221 minutes, at 301.21
pmol/L. The fitted parent's median timing error was 30 [1] minutes on
these same schedules, with a training NMSE of 0.165562 across all targets.

When the old clock is restored, $I(31)=21.94$ and $I(32)=-5.50$ on the
150 g trajectory. The observation at minute 32 is 440.42. On the no-meal
control the restored clock produces a peak of 744.88 pmol/L at minute 12,
while the observed insulin stays at 25.59. This clock's rate depends only
on absolute time and shared parameters. Moving or removing the meal cannot
translate its rate curve. The full insulin balance still responds to the
supplied meal through its other terms.

## Next construction step

Present the specific unresolved reference together with the signed depletion
calculation. Ask the proposer to explicitly remove $D$, define a replacement,
or reuse the previous state with its complete declarations. Require the
assembled $\dot I$ at zero under the proposed domains, and an account of the
delayed meal response. Reapply the public path checks and training witnesses
to the completed edit. Orion can evaluate a refit once construction produces
a defined candidate. The attained loss and best possible fit of either
diagnostic completion remain unknown.

## Verification

The prepared prompt, returned plan, execution record and cached call reconcile.
Offline replay reproduced the exact diagnostic and its details. The worker
exited 0 with empty stderr. The provider finished with stop after 10.77 s and
reported 6,377 input and 818 completion tokens. The preflight recorded the
full scientific context. DOP853 on the compiled diagnostic models and Radau
on independently assembled equations agreed within 0.000012 pmol/L across
all sampled insulin values. The analysis made no provider or optimizer calls
and used public training data only.

- Construction used commit `0d2aba7b79d66ae172f9739bfe5959232e9efd47`.
- The returned archive SHA-256 is `3f7d352fe6afaa1426c58211b6673769ff76bd64249c3301ebe331e0972cb8ab`.
- The plan SHA-256 is `743c066e297595c6951ad36a8e0d96a4dabb9c76a9d169dea069f26715f52358`.
- The terminal receipt SHA-256 is `6022e0833710c8520b7ee6b793bc7c1b3e4adc35670f0c1d5c2f0bb064399fa0`.
- Local [equation and trajectory report](../../output/insulin-feedback-repair-result-20261108/REPORT.html),
  [audit and per-trajectory results](../../output/insulin-feedback-repair-result-20261108/audit.json)
  and [draft feedback for the next proposer turn](../../output/insulin-feedback-repair-result-20261108/next-repair-feedback.txt).
