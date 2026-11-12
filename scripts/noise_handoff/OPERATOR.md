# External baseline results on Full's T2 noise panel

1. Send Claude `t2-noise-fit.zip` and `SEND_TO_CLAUDE.md`. The ZIP contains the same instructions as `CLAUDE.md`. The four public cells and three algorithm seeds make twelve fitted endpoints per method. Claude implements the small adapter in the baseline checkout and runs its native methods.
2. Save the first public-data-only `freeze.json` Claude returns for each method under `first-freeze/METHOD.json`. Once that method's 12 outcomes are frozen, send `t2-noise-blind.zip` to its forecasting process. Other methods can continue to fit using the public ZIP. The blind file supplies time, commands, the available auxiliary and the initial target measurements for each paired rollout. The private grader holds the target answers.
3. Keep `t2-noise-grade.zip` with the analyst. Extract each forecast return under `returned/`. Check the returned freeze against the saved copy and score with the original. These commands assume the three folders below and the grader ZIP are siblings. The project's existing virtual environment has the required NumPy version. On another machine, use Python 3.11+ with `numpy==2.3.5` from `requirements.txt`.

~~~bash
method=latent_sindy
mkdir -p returned
tar -xzf "return-$method.tar.gz" -C returned
cmp "first-freeze/$method.json" "returned/runs/$method/freeze.json"
unzip -qo t2-noise-grade.zip
cd t2-noise-grade
python -m noise_handoff.grade --freeze "../first-freeze/$method.json" --forecasts "../returned/forecasts/$method" --out "../scores/$method"
~~~

The grader reads numeric forecasts and hashes. It checks all twelve frozen endpoints, forecast-to-checkpoint bindings, condition-specific input identities, and the six complete forecast pairs for each available model. An absent file is an incomplete return. A documented unavailable model or explicit numerical failure retains its row. The continuous-time predictor's independent Radau/BDF forecasts pass the original Full solver-agreement test before scoring. A native discrete map takes its original updates on the one-minute grid and carries the `discrete_map` label.

The six clean reference pairs are from Full's frozen T2 acquisition test. They share the 301-point grid and physical interventions across noise levels. Full's acquisition function supplies noisy initial target measurements per condition. The perturbed and control trajectories in each pair share a noise draw. No target feedback follows initialization. Both solver settings and reference identities are recorded with the private key.

`scores.csv` reports one row per cell and seed. Use `common_scale_test_nmse` for comparing noise levels. It uses each target's clean training variance in every cell. `test_nmse` uses the current condition's noisy training variance. Both measure mean squared error against clean perturbed targets, averaging equally over targets and test cases. `response_rse` measures the error in the predicted perturbed-minus-control response against the simulator response using trapezoidal integration. It averages only non-negligible reference effects. The evaluator also retains per-target MSE, signed peaks, areas, peak timing, numerical statuses and each unavailable denominator in `scores.json`. Show failure counts and individual seeds alongside available-case summaries. Join to Full by acquisition ID and seed using the independent acquisition test outputs.

Each ZIP includes `SHA256SUMS`. From its extracted top-level directory, use `shasum -a 256 -c SHA256SUMS` on macOS or `sha256sum -c SHA256SUMS` on Linux.
