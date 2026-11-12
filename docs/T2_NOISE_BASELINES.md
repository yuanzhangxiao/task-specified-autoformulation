# External baselines on the T2-hard measurement-noise study

The study supplies four public acquisitions of the T2-hard cell: the same
sixteen training and four validation trajectories with measurement noise of 0,
1, 5 and 10% of each target's clean training SD (`acquisition_00` to `_03`).
Each method fits every acquisition with seeds 0, 1 and 2 (twelve outcomes), is
frozen with every outcome, failures included, and only then forecasts the blind
queries. The analyst scores the forecasts; no test data or answers are read here.

## Decisions

- **No smoothing.** Noisy observations are used as delivered. Each method keeps
  the derivative rule of its frozen plan (`numpy.gradient` within each
  trajectory for SINDy, PySR and LLM-SR's table; fourth-order differences in
  LLM-ODE's arrays), the same at every noise level (decided with the user,
  2026-11-12 HST).
- **Every level is refitted**, the 0% acquisition included, although its
  trajectories equal the Phase C T2-hard cell's: its files are a different
  release, and every level then runs the same code.
- **Roster.** Primary: latent-state SINDy, LLM-SR in its expression form (its
  picks are equations of the present, so a blind forecast needs no
  transcription), LLM-ODE with the task description, D3 and HDTwinGen. Labelled
  extras: LLM-ODE with its published prompt, specification-informed SINDy,
  SINDy, PySR and the latent neural ODE.

## Plans

Each plan is its frozen CSTR-hard plan with the four acquisitions in place of
the reactor cell and the noise release's receipt; nothing else changes
(`tests/test_t2_noise_plans.py`).

| Plan (`configs/…`) | Runner | Tasks |
|---|---|---|
| `t2_noise_llm_sr_expression_hosted_120b_v1` | `phase_c_llm_sr.py` | 12 (24 target searches) |
| `t2_noise_llm_ode_hosted_120b_v1` | `phase_c_llm_ode.py` | 12 |
| `t2_noise_llm_ode_published_prompt_hosted_120b_v1` | `phase_c_llm_ode.py` | 12 |
| `t2_noise_d3_hosted_120b_v2` | `phase_c_d3.py` | 12 |
| `t2_noise_hdtwingen_hosted_120b_v1` | `phase_b_nine_case_d3.py` | 12 |
| `t2_noise_spec_sindy_hosted_120b_v1` | `phase_b_nine_case_spec_sindy.py` | 12 |
| `t2_noise_classical_jetstream_cpu_v1` | `phase_c_classical.py` | 24 (SINDy and PySR) |
| `t2_noise_latent_ode_v1` | `run_latent_ode_baseline.py` | 36 (three latent sizes) |

On js2 the launchers read the noise release at `~/af/release/t2-noise-fit`,
the extracted public ZIP. SINDy and PySR need the VM to themselves.

## Handoff

`src/claude_noise_adapter.py` is the handoff's adapter. Its `fit` never
searches: it reads one acquisition and seed's sealed pick from the method's own
run, checks that the run read exactly that acquisition's public files, writes
`model.bin` and `fit_receipt.json`, and reports the validation free-rollout
NMSE through the same reader that forecasts
(`autoformalism.rebuttal.noise_exchange`). Before any noise fit, each reader
reproduced its method's recorded validation error on the T2-hard picks: SINDy,
PySR, LLM-ODE under both prompts and LLM-SR's expression form within 1e-7, D3,
the latent ODE and an HDTwinGen hybrid pick exactly. Fit and predict refuse to
run unless the checkout is clean and at the configuration's revision.

## Latent-state SINDy

No established latent-state SINDy fits this interface: an autoencoder that reads
the current observation cannot recover a hidden state, and a delay-coordinate
encoder needs readings the forecast does not receive. This baseline is our
formulation (`autoformalism.baselines.latent_sindy`,
`configs/t2_noise_latent_sindy_v1.json`), approved by the user on 2026-11-12
(HST) and frozen before any full fit:

    x = [y, h],   dx/dt = Xi^T theta([x, u(t)]) / tau,   h(t0) = A [y(t0), u(t0), 1]

- `y` are the two targets, standardized with training statistics; `h` are
  `k = 1, 2 or 3` latent coordinates, each with its own sparse equation; `u` are
  the measured inputs and the supplied auxiliary, linear between samples,
  centred and divided by their largest training deviation.
- `theta` is the degree-2 polynomial library: the constant, every coordinate and
  every product of two. `tau` is a tenth of the training horizon.
- `A` is a linear initializer of the latent part from the first readings and
  inputs, fitted on training trajectories only; a forecast is causal.
- Fitting matches RK4 rollouts of every training trajectory (Adam, learning rate
  0.01, clipping 1.0): 2,000 iterations with an L1 penalty of 0.001 on `Xi`,
  then, for each threshold 0.01, 0.03 and 0.1, the smaller coefficients fixed at
  zero and the rest refitted for 500 iterations without the penalty. A diverged
  step returns to the last finite parameters with a tenth of the step, at most
  three times.
- Validation free-rollout error (Radau replay) stops each phase early and, per
  seed, chooses `k` and the threshold. The chosen equations are recorded in
  closed form.

A 20-iteration feasibility check before freezing changed only the input scaling
(standardized pulses had made the first rollouts diverge) and the divergence
recovery; no setting was chosen from a validation score.
