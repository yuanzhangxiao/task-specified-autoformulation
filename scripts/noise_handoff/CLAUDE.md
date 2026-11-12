# Run external baselines on the T2-hard measurement-noise study

Please run the external baselines on the four supplied Phase C Dalla Man T2-hard acquisitions. Start with genuine latent-state SINDy. Also run the established LLM-SR, LLM-ODE, D3 and HDTwinGen configurations. Keep the variants we have already agreed to compare under distinct method IDs. Label observed-state SINDy, specification-informed SINDy and the learned latent neural ODE separately whenever included. For latent-state SINDy I want an explicit evolving unobserved state and a causal initializer that works from the information below. Give the exact formulation if you introduce one. A delay-coordinate implementation that needs future readings at test time cannot enter this comparison under that initialization.

## Fit from the public attachment

Unzip `t2-noise-fit.zip` on the machine running the baseline. These commands use the baseline environment's Python 3.11 or newer. Run the Python commands from inside the extracted directory.

~~~bash
unzip -q t2-noise-fit.zip
cd t2-noise-fit
python -m noise_handoff.exchange verify
~~~

| Directory under `public/` | Measurement noise SD |
|---|---:|
| `acquisition_00` | 0% |
| `acquisition_01` | 1% |
| `acquisition_02` | 5% |
| `acquisition_03` | 10% |

Use seeds 0, 1 and 2 in every cell. Each method has 12 terminal outcomes. Use the packaged files as supplied, including at 0%. Each directory contains Full's exact `train.json`, `val.json`, `specification.json` and `proposer_prompt.txt`. JSON public files contain the original `sha256` and `value` envelope. A split's trajectories are in `value.rows`. `release.json` gives ACES file hashes and plan identities. All four levels share 16 training and four validation trajectories, each with 301 samples from 0 to 300 minutes. The seeds repeat search on the same corrupted observations.

Fit each cell using only that cell's public files. Use noisy training and validation measurements exactly as delivered. Estimate any needed derivatives or smoothing parameters from the permitted observations within trajectories. Use per-condition training statistics, keep train and validation trajectories intact, and preserve the method's established model-selection rule and budget across noise levels. Do not pool levels, infer the clean observations from a paired file, add fresh noise, or substitute earlier Phase B/C scores. If reusing a frozen fit, verify the four input hashes, baseline version, budget, settings and seed. For a deterministic method, identify identical repeats. For LLM methods, preserve the prescribed prompt access, checkpoint the search and cache requests. Keep credentials and raw provider responses out of the handoff.

The two targets are `Gp` (mg/kg) and `I` (pmol/L). The supplied auxiliary is `Uii`. The known inputs are `meal_rate_g_per_min` and `insulin_pmol_per_kg_min`. Interpolate supplied channels linearly through the original knots. The meal input is already a rate. The observation noise uses 0, 0.01, 0.05 or 0.10 times each measured channel's clean training SD. Constant Uii and the external inputs are identical across noise levels. Train hidden-state initializers on training only. Each forecast receives two initial target readings, the allowed auxiliary and input paths, and the clock. Use the rollout's own states thereafter. The test interface supplies no later target measurements, burn-in window, or hidden simulator coordinates.

## Connect the existing implementation

Keep each baseline's native search and representation. On this Mac the relevant checkouts are under `~/Projects/`. The CSTR launch configs and fixed cell rosters need a T2-noise input adapter. Use the four acquisition identities directly and bind their supplied hashes. Earlier CSTR experiment launchers target different cells.

| Existing checkout | Useful code |
|---|---|
| `autoformalism-cstr-hard` or `autoformalism-phase-c-baselines` | `scripts/phase_c_llm_sr.py`, `scripts/phase_c_llm_ode.py`, `scripts/phase_c_d3.py`, and `scripts/phase_c_classical.py` |
| `autoformalism-cstr-hard-hybrid` or `autoformalism-phase-b-baselines` | `scripts/phase_b_nine_case_d3.py` handles the HDTwinGen hybrid variant. `scripts/phase_b_nine_case_spec_sindy.py` implements the specification-informed SINDy variant. Reuse method code with the T2 inputs. |
| `autoformalism-cstr-hard` | `src/autoformalism/baselines/latent_ode.py` is the separate augmented neural ODE. Locate or implement the actual latent-state SINDy method in the baseline checkout. |
| `autoformalism-intervention-scoring` | Reuse native model readers and rollout adapters from `scripts/score_baseline_interventions.py`. Use this packet's forecasting inputs and scoring commands. |

Implement one trusted module `claude_noise_adapter.py`, importable from the fitting environment. Put method code, selection and native checkpoint handling in your checkout. The supplied `example_adapter.py` demonstrates the file and function interface with an initial-value persistence predictor for transport testing only.

With a Phase C baseline checkout on `PYTHONPATH`, call `from noise_handoff.native import baseline_cell` and then `cell = baseline_cell(task, cell_dir)` inside the fitting adapter. This returns the baseline project's native `PhaseCBaselineCell` with its training/validation datasets, estimated derivatives, context, prompt, hard tier and checked acquisition identity. Pass that object to the existing method or its task runner. For existing forecast readers, `noise_handoff.native.blind_trajectory(inputs)` returns a compatible `Trajectory`. For models exported in Full's restricted ODE grammar, `noise_handoff.native.compiled_forecast(endpoint, inputs, solver)` runs Full's own backend directly. The endpoint has `candidate`, `context`, `parameters` and `initials`. A native map or hybrid network should use its own reader. Return just the two target channels from any reader that also emits latent coordinates.

1. `fit(task, cell_dir, task_dir, config) -> dict`. Read only the specified cell. Run or resume the method for `task["seed"]`. Write one self-contained native model file `task_dir/model.bin` (a ZIP is fine) with all fitted equations, weights, training scalers, initializers and readouts. Write `task_dir/fit_receipt.json` including `cell`, `seed`, and `public_files` with the hashes actually read, plus the code/configuration, native selection metric, solver, budget, observed costs and result provenance. For a fitted model return `{"status":"ready","validation_nmse": number}`. Report validation free-rollout MSE divided by the condition's training variance per target, averaged over both targets and the four trajectories. Record any different native selection objective separately. For a terminal failure write the receipt and return `{"status":"no_model","reason":"specific cause"}`. Interruptions and transient execution errors should raise and resume from native checkpoints.
2. `predict(model_path, inputs, solver, config) -> {"Gp": sequence, "I": sequence}`. Load the frozen artifact. `inputs` contains `time`, scalar `targets0`, and keyed sampled `auxiliaries` and `external_inputs`. Return both target forecasts in their physical units for every time, including time zero. Each call is a fresh initialization. For `continuous_ode`, supply independent `Radau` and `BDF` integrations at rtol 1e-8, atol 1e-10. Follow Full's piecewise-linear forcing, restart at forcing slope changes, carry state continuously, and use a 120-second deadline per trajectory/solver. Reuse `simulate_trajectory(..., reset_observed_states=False)` where the grammar fits. If the integration fails, raise a descriptive `ValueError` or `TimeoutError`. For `discrete_map`, solver is `native`: run the original recursive sample-step update. D3 and HDTwinGen use their fitted increments, including native hybrid weights. Do not multiply an increment by dt or reinterpret it as an ODE. Use reviewed model readers and restricted expressions.

For each method, create `configs/METHOD.json` with its stable method ID, `semantics` (`continuous_ode` or `discrete_map`), `implementation_revision`, `budget` and `selection_rule`. Use `latent_sindy`, `llm_sr`, `llm_ode`, `d3` and `hdtwingen` for the five primary configurations (60 endpoints). Include the actual settings, search versions, model/prompt if used, native solver, initialization, package versions and selection criteria. Pin the adapter and imported method code before fitting. Use a versioned module or checkout to preserve previously frozen adapters. This example launches all twelve tasks once the adapter and config exist. Change `method` to the next declared identifier to repeat. For a cluster array, append `--index "$SLURM_ARRAY_TASK_ID"` to `fit` with array indices 0–11. Existing verified terminal tasks are skipped on resume.

~~~bash
method=latent_sindy
python -m noise_handoff.exchange plan --config "configs/$method.json" --adapter claude_noise_adapter --work "runs/$method"
python -m noise_handoff.exchange fit --work "runs/$method"
python -m noise_handoff.exchange freeze --work "runs/$method"
~~~

Rehearse the frozen model reader against the reported public validation rollout. Return `runs/METHOD/freeze.json` for each completed method with all 12 outcomes, including failures, before receiving blind inputs. Preserve the native models, sanitized receipts and forecast adapter. Use validation for all selection. Keep held-out results away from any continuing fits or tuning.

## Forecast after freeze

I will provide `t2-noise-blind.zip` for forecasting after receiving that method's 12-row freeze. Keep this ZIP outside any still-running fitting process. From inside `t2-noise-fit`, put the ZIP in the parent directory and run:

~~~bash
unzip -qo ../t2-noise-blind.zip -d ..
method=latent_sindy
python -m noise_handoff.exchange predict --work "runs/$method" --queries ../t2-noise-blind/queries.json --out "forecasts/$method"
tar -czf "return-$method.tar.gz" "runs/$method/plan.json" "runs/$method/freeze.json" "forecasts/$method"
~~~

Send `return-METHOD.tar.gz`. Preserve the native model artifacts and sanitized fit receipts for reproducibility. Return elapsed time, compute, calls and tokens when available. The analyst will score the six held-out intervention/control pairs against the clean targets with Full's original metric code.
