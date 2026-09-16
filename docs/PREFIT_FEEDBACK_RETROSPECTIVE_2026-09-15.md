# Retrospective evidence review, 2026-09-15

This is the explicitly requested ground-truth-assisted analysis of the anonymous
hard/opaque case. It is separate from proposer execution. Private equations and
the generator were inspected; only TRAIN trajectories were regenerated. No test
trajectories or test metrics were evaluated. None of the private structure below
is supplied to the proposer or encoded as a runtime rule.

## Verified reference

The Phase-B generator uses
`data_raw/benchmark6_alien_device/private/selected_system_spec.json`.
Regenerating its sixteen TRAIN protocols and applying the hard/opaque public
projection reproduces the experiment's exact training CSV SHA:
`ffaa9ff0b947d555b057eebd890dc5ef6f42e38e38e4b25f696b0f2f7c24b1e8`.
The rendered public prompt SHA also matches:
`2a5bcedbdc0d3c115a783e7d820c1a5aba638edc21f4bd59205b6374837f799e`.
Both match the supplied `review-context.json`. No benchmark asset was modified.

The reference has five coupled latent states, damped skew coupling, bounded
nonlinear terms (including biased tanh and products), a saturating input drive,
and a dynamic output driven by the latent system. The output does not itself
drive those reference latent equations. The current candidate has two latent
states and uses a quadratic output-to-memory interaction.

These facts do not require a reduced model to have five latent states, copy tanh,
or match latent coordinates. They suggest what observable contrasts are useful;
they do not by themselves identify the correct reduced equation.

## Most informative public contrasts

1. **Dynamics with zero external input.** The reference zero-input output ranges
   approximately from -9.795 to 0.819 over t=0..60, with five sampled reversals
   under the exporter's common amplitude tolerance. Reversal times are about
   2.5, 17.9, 28.5, 43.2 and 53.9. Continued internal motion must be represented;
   externally driven response alone cannot explain it. Compare the fitted
   prediction's range, reversal timing and late-time residual against this
   measured curve. These are finite-horizon observations, not proof of a limit
   cycle or of a particular defective term.
2. **Input-off memory and phase.** Pulse trials provide a shared initial segment
   followed by positive/negative perturbations and a long input-off interval.
   Compare response timing and persistence to the zero-input baseline. This
   distinguishes an amplitude mismatch from lost memory or phase mismatch more
   usefully than a single total NMSE. A change to one static input gain will not
   necessarily address a missing internal timescale.
3. **Amplitude and sign contrasts.** Matched positive/negative pulses and
   different pulse/step amplitudes allow baseline-subtracted response comparisons.
   Sublinear response can motivate a saturating hypothesis; asymmetry can motivate
   an offset or signed-response hypothesis. Neither observation uniquely selects
   tanh, a square, or a particular source variable. Compare candidate predictions
   on the same trials before choosing a change.
4. **Frequency-dependent response.** The training sine periods 8, 16 and 32 and
   chirp supply distinct memory tests. Compare lag, reversals and amplitude in
   aligned input/output records, including persistent patterns at successive
   fitted parameter vectors. Fewer reversals or phase drift in a prediction can
   motivate a timescale/coupling hypothesis. A topology change may be required;
   the current one-RHS scope cannot express every useful hypothesis.

Only the reference observations were regenerated for these findings. The exact
saved first-continuation prediction arrays were not available locally. Therefore
this review does not claim which of these contrasts has the largest candidate
error, or that the candidate has no reversals. The actual residual packet is
needed for that comparison. The new v2 policy uses that packet without inserting
these retrospective conclusions.

## An initialization information limit

The TRAIN protocols `zero`, `initial_shift_a` and `initial_shift_b` all have
u(t)=0 and observed y(0)=0. Their hidden initial states differ and their future
outputs differ. The public projection contains no covariate identifying the
shift. Under our contract, a deterministic model initialized only from public
initial observations and inputs must predict the same curve for all three.
Neither more optimizer time nor a different deterministic RHS removes that
information deficit. Trajectory IDs are not admissible explanatory inputs.

For this group, the best unconstrained shared curve under squared loss is its
pointwise mean. With global training variance 14.6361171864, the group's minimum
average NMSE is 0.0299557796. Its contribution to the all-sixteen-trajectory
objective is at least 0.00561670867 (3/16 times the group value). This is a lower
bound for the present deterministic information contract, not an attained ODE
fit or a lower bound of 0.827. Most of the current error remains unexplained by
this particular information limitation.

The same fact can be detected from public TRAIN data alone by grouping identical
available initial conditions and complete supplied input schedules, then measuring
within-group output disagreement. It should eventually be a separate data and
initialization diagnostic. Potential contract changes include an observed warm-up
history, an initial-condition covariate, or predictive uncertainty over latent
initial states. None is implemented here; validation/test hidden initial values
remain unavailable and unfitted.

## Feedback design implication

Use an observation → hypothesis → predicted improvement chain. For example,
*if the actual fitted curve loses the measured input-off reversals*, ask for a
change that could preserve delayed internal response, naming the relevant
trajectory/window evidence and the expected improvement in timing. Do not tell
the proposer that the true system uses five latent states or tanh. Do not present
an optimization time limit as the cause of a residual pattern. Treat a request
for topology or initialization information as a distinct outcome rather than
forcing it into an incompatible function slot.
