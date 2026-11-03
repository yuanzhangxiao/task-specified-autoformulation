# Construction portfolio checkpoint continuation: results review

3 November 2026. Review of `construction-portfolio-checkpoint-1`, submitted
from commit `0ee602d97268c4b9b1e6cb221b4724fc32aed524`.

## Conclusion

The compact-checkpoint change worked operationally: persistence consumes much
less of the search allowance. All four retained improvements pass the saved
independent original-equation replay checks. However, the additional fitting
produces less than 0.1% relative reduction in training NMSE for every model, and
none of the outer searches converged. These results do not establish the best
achievable fit of any equation family or justify attributing all remaining error
to the equations.

The next experiment should target parameter boundaries and correlated parameter
coordinates before allocating another identical continuation. Models 12/15 are
the first priority; models 10/14 need bounded diagnostics before more expensive
full searches.

## Evidence and authentication

- Received archive: `/Users/yuanzhangxiao/Downloads/inspection (1).tar.gz`.
- Archive SHA256:
  `5273db8b9908ad615432c9cc130812fb2d82ef0323b21947358e46df49d2a4f5`.
- Local extraction: `output/construction-portfolio-checkpoint-review-20261103`.
- Delta root:
  `/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-checkpoint-v1`.
- Recomputed frozen-plan/input identities, roster and all four terminal results
  from authenticated operation receipts. Every terminal reconciliation was exact;
  all saved summary rows equal the reconstructed rows.
- No fitting, new LLM calls, validation access or test access was performed in
  this review. Independent replay results below were computed by the returned
  campaign, not newly recomputed locally.

## Numerical outcomes

These are full-training reconstructions with training-estimated initial roots,
not new causal-prefix forecasts or held-out results.

| Model | Starting NMSE | Retained NMSE | Relative reduction | Search termination |
| --- | ---: | ---: | ---: | --- |
| 10 | 1.712941252 | 1.712229230 | 0.04157% | 1,800 numerical seconds |
| 12 | 0.436484876 | 0.436354216 | 0.02993% | 1,800 numerical seconds |
| 14 | 1.688725401 | 1.688704861 | 0.00122% | 1,800 numerical seconds |
| 15 | 0.438860884 | 0.438459830 | 0.09139% | 2,000 objective evaluations |

All four report `improved_verified` and `optimization_converged=false`.
`complete` denotes a terminal campaign record, not optimization convergence.
The retained independent-solver prediction discrepancies are, respectively,
1.95e-10, 1.27e-9, 2.27e-10 and 1.05e-9 in the verification metric. Loss
discrepancies are at most 1.98e-11.

There are small tradeoffs among targets, not uniformly improving trajectories:

| Model | Gp NMSE, before → after | I NMSE, before → after | U NMSE, before → after |
| --- | --- | --- | --- |
| 10 | 0.660994 → 0.663062 | 1.007989 → 1.006398 | 3.469841 → 3.467228 |
| 12 | 0.269622 → 0.269863 | 0.540479 → 0.541157 | 0.499354 → 0.498042 |
| 14 | 0.624068 → 0.625620 | 1.000707 → 1.000789 | 3.441402 → 3.439706 |
| 15 | 0.244456 → 0.240262 | 0.550397 → 0.554270 | 0.521730 → 0.520848 |

## Persistence and usable numerical budget

| Model | Previous callback seconds | Current callback seconds | Current numerical-active seconds | Objective attempts |
| --- | ---: | ---: | ---: | ---: |
| 10 | 292.2 | 27.1 | 1800.0 | 224 |
| 12 | 1172.8 | 107.5 | 1800.0 | 1918 |
| 14 | 269.1 | 12.7 | 1800.0 | 242 |
| 15 | 1178.9 | 100.3 | 1784.0 | 2000 |

No callback or checkpoint-write failures were recorded. Current durable writes
number 76/348/68/445. Models 12/15 now use nearly their full numerical allowance,
instead of spending about twenty minutes in callbacks. This is measured
operational improvement, not a controlled optimizer speedup comparison: the
new runs start from previous retained endpoints, and filesystem conditions can
differ.

Models 10/14 still finish no complete outer Powell iteration; their accepted
history contains the seed only, although better trial points were retained.
The 24-coordinate residual evaluations are expensive. Model 12 completes six
outer iterations; model 15 completes ten before the evaluation cap. Conditional
root optimality for a given shared vector does not establish outer optimality.
Profiled searches reject invalid inner trials (16 each for models 12/15), rather
than treating them as usable fits.

## Parameter patterns worth testing

### Model 12: absorption-state decay approaches its allowed zero boundary

The meal state satisfies

`A_meal' = r_meal - k_A_decay * A_meal`,

and contributes `k_A_to_Gp * A_meal` to the glucose balance. The retained
`k_A_decay` falls from 4.01730e-5 to 3.53125e-9, while `k_A_to_Gp` stays near
0.0177. Over the observed horizon this approaches a cumulative-input state.
Zero decay is allowed by the original nonnegative domain, so an explicit
zero-decay fit and one-sided profile are legitimate original-family controls.
The current boundary receipt tests `k_Uii=0`; it does not test this decay boundary.

This trend suggests testing whether appreciable decay improves the profiled
training objective. It does not prove that zero is optimal or that the state
must be changed by construction.

### Model 15: opposing changes in absorption coefficients and latent scale

The meal block is

`A' = k_abs_rate * r_meal - k_abs * A`,

`Gp' = r_meal - U + k_abs * A`.

`k_abs` decreases from 2.74381e-4 to 1.38389e-4, while `k_abs_rate` rises from
59.9307 to 120.9675 and the fitted meal-state initials approximately double.
The product changes much less: 0.0164438 → 0.0167406. This is a specific
parameter-coupling clue, not a proof of structural unidentifiability.

For strictly positive `k_abs`, define `B = k_abs * A` and
`g = k_abs * k_abs_rate`. The same model becomes

`B' = g * r_meal - k_abs * B`,

`Gp' = r_meal - U + B`.

The inverse must reconstruct `k_abs_rate = g / k_abs` and every
`A(0) = B(0) / k_abs`, with finite values and all original domains respected.
This offers a physically output-scaled coordinate system for a matched-budget
comparison; it does not add a new mechanism.

**The zero boundary is different:** for finite original coefficients and
initials, `k_abs=0` requires `g=0` and every `B(0)=0`. Allowing positive `g`
or `B(0)` at zero would add limiting models obtainable only by diverging
original coefficients/initials. Such points cannot be promoted as valid
reparameterizations. Keep a separate exact original zero-boundary control and
verify positive-branch candidates after lifting to the original equations.

### Models 10/14: nearly saturated input law and costly outer search

Both use `Vmax_ins * Gp**2 / (Km_ins**2 + Gp**2)`. Retained `Km_ins` values
are about 3.82e-7 and 5.65e-6, respectively. On the observed glucose range,
this law is effectively saturated, so small local changes in `Km_ins` need
not reveal a useful search direction.

A bounded diagnostic should compare meaningfully separated values using
public training-glucose scales, reoptimizing relevant nuisance coefficients.
These are alternative parameter starts in the same family, not suggestions
to impose a known physiological parameter. An exact `Km_ins=0` simplification
must account for the original expression being undefined at `Gp=0`; it is not
an unconditional symbolic equivalence. No global parameter-confidence claim
is supported by the present residual-only qualifications.

## Recommended next milestone

1. Keep compact persistence and original-equation verification.
2. Certify the positive-branch meal-state coordinate transform for model 15,
   including all trajectory initials, inverse domains and boundary handling.
   Compare it with the existing coordinates from the same saved start and with
   matched numerical/call allowances; include a small predetermined set of
   distinct decay starts. Retain the incumbent if nothing improves.
3. Profile model 12's decay with an exact allowed zero-boundary control and
   distinct positive decay values, reoptimizing the nuisance variables.
4. Run bounded saturation-scale diagnostics for models 10/14 before committing
   to another full 24-coordinate continuation.
5. Record objective progress, termination, nuisance-solve adequacy, finite lifted
   parameters and independent replay. If the diagnostics still cannot establish
   fitting adequacy, report unresolved optimization rather than a certified
   equation-family limitation.

These checks can eventually support construction feedback about persistent
training mismatches. They cannot yet justify prescribing nonlinear terms,
additional latent states or a scientifically preferred alternative equation.
Any retained improvement needs a new causal-prefix evaluation before a
generalization claim.

## Review verification

Authenticated result reconstruction passed for all four tasks. The relevant
fitting, campaign, launcher and report suite passed: **304 tests**, including
actual-ODE campaign/resume smoke coverage. Repository-wide Ruff retains the existing 37 unrelated
findings in `analysis/claude`. This review changes only this handoff document;
no experiment or production-fitting behavior is changed.
