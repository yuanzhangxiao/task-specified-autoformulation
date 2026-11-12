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
