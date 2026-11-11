# Latent neural ODE baseline

On 2026-11-11 the user asked which part of our method's gain comes from latent
states and which from the task specification, with two comparisons in mind:
latent states without the specification, and the specification without latent
states. No external baseline we had could hold a state it does not measure, so
the first comparison had no external model. This baseline is that model: a
conventional latent-state model that never reads the specification.

## Method

`src/autoformalism/baselines/latent_ode.py`. The measured targets `y` are
augmented with latent coordinates `h`; one network gives the time derivative of
the whole state from the state and the supplied channels `u` (measured inputs
and supplied auxiliaries):

    x = [y, h],   dx/dt = F([x, u(t)]) / tau,   h(t0) = E([y(t0), u(t0)])

- Augmented neural ODE (Dupont et al., 2019) for the state; the latent ODE's
  encoder (Rubanova et al., 2019), restricted to the first observation so that
  a rollout is causal, for the latent initial state; inputs entering the
  derivative as in continuous-time neural state-space identification.
- `F`: two hidden layers of 64 tanh units; its last layer starts at a tenth of
  PyTorch's default scale. `E`: one hidden layer of 32 tanh units.
- Targets and supplied channels are standardized with training statistics, and
  the channels are interpolated linearly between samples, as every other
  model's are. `tau` is a tenth of the training horizon.
- Training: full batch over the training trajectories, free rollouts from the
  first sample by fixed-step RK4 (two steps per sample interval), mean squared
  error of the standardized targets, Adam at 1e-3, gradient norm clipped at 1,
  2,000 iterations. Every 25 iterations the validation free-rollout error is
  measured and the best checkpoint kept (early stopping).
- Selection: for each cell and seed, the latent dimension (2, 4 or 8) whose
  checkpoint has the lowest validation free-rollout NMSE when replayed in numpy
  with an adaptive solver (Radau, rtol 1e-8, atol 1e-10), one sample interval at
  a time. That replay is also how the model is scored.

The settings are common neural ODE defaults. They were fixed in
`configs/latent_ode_intervention_v1.json` before any model was trained and
before any intervention score was seen, and frozen on 2026-11-11 on the user's
sign-off. Public training and validation data only.

## Scope and run

The intervention study's two cells, Phase C CSTR-hard and T2-hard, seeds 0, 1
and 2: 18 training tasks (two cells, three seeds, three latent dimensions), run
on the Mac with `scripts/run_latent_ode_baseline.py` (`plan`, `train`,
`select`); about 0.23 s per iteration, so about eight minutes per task.

In the scoring (`scripts/score_baseline_interventions.py`, branch
`phase-c-intervention-scoring`) its latent coordinates are passed to the
study's training-calibrated probe, as our models' are, so its hidden-process
rows can be complete where the other baselines' cannot.
