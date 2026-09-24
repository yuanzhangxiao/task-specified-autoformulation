# Dalla demonstration: salvage existing structures before another long search

## Decision

A restart from round zero is unnecessary to rescue a demonstration. Completed
models and fitted vectors remain usable even if the constructor/controller was
older. Their exact generating protocol and any additional fitting must be stated.
They are not evidence that the latest complete pipeline produced that endpoint.

The fresh campaign has valid prior fits through R11, three completed proposal/fit
visits at R12, and three provider failures at R12 that preserved their incumbents.
The failures occurred while vLLM loaded the Harmony vocabulary, before generation.
The dispatcher blocks R13, so there are no subsequent R13/R14 results corrupted by
those failures in the supplied snapshot. Recovering the missed proposals could
change subsequent search choices; skipping them is a different recorded execution
path, not an exact replay of an uninterrupted run. Historical records must remain.

The useful distinction is between an unsatisfactory **fitted vector** and an
inadequate **family of equations**. Prior reviews established that the saved
vectors did not supply the desired mechanism/intervention example. That conclusion
does not establish that every parameterization of those structures is inadequate.

## Specific structures worth rescuing

| Priority | Original model | Saved training / validation NMSE | Why inspect or refit it |
| --- | --- | --- | --- |
| 1 | T1-easy, perturbed named, Full seed 1, R4 | 36.2287 / 37.8674 | A two-stage meal chain and adjustable balance terms already exist. Most coefficients remain 0.1; Gp damping is 4.4; latent initials are zero. Fifteen residual calls and a budget stop do not establish the best achievable fit. |
| 2 | Same lineage, R13 | 0.042353 / 0.059328 | Stronger existing fit and the same core absorption chain, with additional parallel states. A useful refit control and a source of compatible training-fitted starts. |
| 3 | T1-easy, canonical obfuscated, Brief-only seed 1, R9 trial | 0.019131 / 0.017761 | All directly supplied balance channels and two meal filters are present. The six balance coefficients are real-valued, so their fitted signs are not structurally forced. |
| Repair candidate | T1-hard, Full seed 0, v7 R15–17 patches | Trial scores around 0.2355 / 0.3494 | The intended new delay parameter collided with an existing coefficient. A fresh parameter name would permit the intended independent timescale, but would constitute a corrected patch and a new fit. |

Exact model IDs in the original exported inventory:

- R4: `bc125e1216f406afd69df24c6494bce6f4d77a2e96d187a3704341e6622b8e14`.
- R13: `9d60ddc59cf797f44ef7fc3cbe9505b041245187a4c0046ac99f96246b7c73cf`.
- Brief R9: `4d630f90e574895a0046686a7e0f1380bf99cb1dc154e399bc9ff33f360d660d`.

The R4 active structure, with coefficient names abbreviated, is

```text
dX/dt = a * meal - b * X
dY/dt = c * X - d * Y
dGp/dt = km * Y + kh * EGP - ku * Uii - ke * E - kt * Gt - kp * Gp
```

There is also a disconnected M filter. The four filter coefficients are
nonnegative; the six balance coefficients are real. In particular, the displayed
minus sign before kt does not force negative tissue coupling: kt may be negative.
This is a plausible reduced-model layout, not an exact reference skeleton. Its
exchange terms are linear, while the perturbed benchmark can require nonlinear
exchange for faithful behavior. That limitation remains even after a good refit.

The R4 record is valuable precisely because a low-NMSE-only shortlist would miss
it. Its default-like vector is evidence to investigate optimization, not proof
that no optimization happened or that a better fit will recover the mechanism.

A declaration-level audit confirms that R13 supplies a compatible learned start
for all 13 R4 parameters, including its three latent initializers. There are no
new or changed parameter declarations and no changed surviving boundary rules;
four R13-only parameters are removed. This is a starting vector for a new R4 fit,
not a transfer of R13's score: R13 has additional terms in the Gp equation.

For T1-easy, the public contract supplies EGP, Uii, E and Gt trajectories over the
entire horizon. A meal-intervention demonstration must specify how these supplied
channels are set under each condition. Success would establish the reduced model's
conditional glucose response; it would not establish autonomous recovery of the
insulin system. Changing meal alone while freezing those channels is a different
probe from regenerating all channels under a physiological meal intervention.

## Saved runtime rejections

Replayed all 21 saved v7 raw patches from the final archive against their exact
historical parent using current strict declaration checks. No LLM calls or fits
were made by this audit. Citation verification was not reconstructed because the
original displayed reference catalog is absent; this is mechanical admissibility
only, with no claimed verified evidence.

- Thirteen previously accepted patches remain mechanically accepted.
- Four previously accepted patches now reject conflicting declarations: T1-hard
  seed 0 R15/R16/R17 and T2-easy seed 1 R16.
- All four historically rejected attempts still reject. One redundantly supplies
  a latent initializer for the observed Gp state. The three other attempts replace
  the glucose observation with a flux expression while leaving the Gp state behind.
  They cannot simply be admitted as faithful dynamic-target models.

Thus this saved trace contains no ready-to-use patch incorrectly excluded by
the current checks. It does contain useful intended revisions whose declarations
need correction. A bounded corrective proposer call is more defensible than
silently assigning a new meaning to an occupied parameter name.

## T2 is a structural repair problem

The combined old inventories contain 315 distinct exported request/vector keys.
The insulin screen covers 62 older T2 records and all 51 fresh T2 trial fits.
These are correlated saved candidates, not 113 independent experiments.

In 100 records, I is an observed state whose equation refers only to I and
external infusion. At fixed I(0) and infusion, changing a meal cannot affect I.
The other thirteen records share a meal process whose fitted contribution to I
is negative. These are not meal-insensitive, but they still lack a positive source
for the observed meal-associated rise when infusion is zero.

Parameter-only fitting cannot add the missing input dependence in the first
group or reverse the constrained negative meal contribution in the second.
T2 should therefore be lower priority for an immediate positive demonstration.
If revisited, a proposer should receive the public training counterexample and
current equation, not a private-reference prescription for an insulin equation.

## Bounded local refit check

Two unchanged-equation trials were frozen before fitting: Brief R9 and Full R13.
Each uses the existing sibling-fit adapter and its original profile: 120 seconds
of collocation plus 180 seconds of screening/refinement, starting with the complete
saved vector, including initializer parameters. No domain, equation, initializer
policy, public data, historical result, or production fitter was changed.

| Model | Original training / validation | Local refit training / validation | Outcome |
| --- | --- | --- | --- |
| Brief R9 | 0.019131 / 0.017761 | 0.033659 / 0.027771 | Worse; keep the original model. |
| Full R13 | 0.042353 / 0.059328 | 0.042353 / 0.059328 | Original vector retained exactly; no improvement. |

Both larger collocation subprocesses exited with code -11. Their fallback
screening also timed out when evaluating the known finite original vectors under
the ten-second screening limit. Both allocated trials ended with 15 residual
calls and exhausted budgets; neither reports verified optimizer convergence.
These results are not a clean test of optimizing the saved basins and do not
demonstrate that the equation families cannot improve. The original vectors and
historical scores remain intact. R4 has not yet been refitted.

Execution provenance: the initial freeze used source commit
`79c525294512cdec53b3b3f74f6813985ca43842`. Concurrent source changes were detected
before starting Full, so that exact source was restored as a read-only execution
snapshot. Two Full harness launches failed, first because the snapshot lacked a
worker script and then because launching from stdin was incompatible with
multiprocessing spawn. These failures concern this local diagnostic harness,
not the model. Their consumed receipts were preserved; the reported Full result
is from a separately identified recovery with the complete snapshot and a
file-based main guard. No prior result was overwritten or budget silently reset.

All saved-result resume checks returned identical results with the numerical
backend patched to fail if invoked. The interrupted harness receipt correctly
becomes an interrupted terminal result rather than restarting fitting. Records
are in `artifacts/dalla-salvage-20260924/`.

## Next experiment

1. Prioritize a small fixed-structure CPU rescue of R4, with R13 and Brief R9 as
   controls. Keep the original candidate families, parameter domains and causal
   initialization contracts. Include a compatible later-model start for R4 only
   with an explicit declaration/initializer compatibility audit.
2. Use an execution environment where the collocation backend passes its smoke
   test. Verify each known finite parent on the complete training split before
   optimization; a short screening timeout is not a reason to discard that parent.
   Any enlarged allocation or screening limit belongs to a separately recorded
   diagnostic, not an edit to an existing frozen campaign.
3. Compare the original parent and all new fits, retaining the best eligible
   candidate without overwriting old results. Select numerical vectors using
   development data, not the diagnostic intervention outcomes.
4. For a promising fit, replay every original training/validation trajectory and
   the already defined intervention family with fixed parameters. Check initial
   equilibrium, input-response timing and direction, and both absolute curves
   and effects relative to the same model's matched control. Perturbed models
   require their own perturbed reference dynamics.
5. If fixed-structure refitting does not deliver a credible example, use one
   short targeted T1 repair branch addressing a demonstrated missing mechanism
   or parameter-declaration conflict. A new 15-round six-lineage run is not the
   first rescue experiment.

These are exploratory demonstration candidates selected after inspecting existing
experiments. A successful refit can illustrate intervention-capable modeling; it
does not by itself establish benchmark-wide superiority or erase earlier failures.
No confirmed positive intervention demonstration is claimed by this shortlist.

## Handoff files and checks

The portable package is
`artifacts/dalla-salvage-20260924/dalla-salvage-candidates-20260924.tar.gz`.
It contains the three unchanged requests, original fitted parameters, public
briefs and target contracts, and the two public train/validation datasets.
It does not launch a new experiment automatically. Its SHA256 is
`b20d82b0cbb2cce36f3eddef127b54524183c31f82998d091c01e7bd9c051c63`.

The local audit records are `saved-patch-audit.json`, `t2-structural-screen.json`,
and `r13-to-r4-start-audit.json` under `artifacts/dalla-salvage-20260924/`.
Generated records and experiment outputs are excluded from Git.

Verification: 62 tests passed in `test_sibling_fit.py`, `test_public_fitting.py`
and `test_review_integrity.py`. The public-fit collocation smoke passed with
training NMSE 5.09e-17, validation NMSE 7.46e-18 and unchanged resume. This small
smoke does not establish that the larger Dalla collocation subprocesses work.
Whole-tree `ruff check .` reports 37 existing issues in unrelated
`analysis/claude/` files. No production implementation or benchmark was changed.
