# Candidate 15 insulin revision

This experiment asks whether fitting the direct ingestion coefficient improves the
timing and size of the insulin response in the T2-easy model. It starts from
Orion’s independently replayed endpoint on the 16 public training trajectories.
The public task requires a meal-to-glucose pathway and a delayed
insulin-to-disposal pathway. The direct meal-to-insulin term is a construction
choice. [Orion’s returned results](../work/agent-comms/2026-11-05-construction-meal-reliability-results.md)

## Matched fits on Delta

The parent uses the coefficient one. The child fits one global nonnegative
coefficient in the insulin balance,

```text
I' = c_I * meal_rate_g_per_min + a * Gp - lambda * I + b * external_insulin
```

Both arms start at the saved parent parameters and training initial states.
At `c_I=1` the worker checks the predicted values across the full training
array and the saved loss. At each outer evaluation the parent jointly solves
for two shared forcing coefficients and 32 trajectory initial values. The
child adds one column to that bounded least-squares problem. Both arms optimize
the same six outer coordinates within Orion's positive absorption-rate and
relaxation-time parameterization. The new gain can reach zero.

Each of the two CPU jobs receives 1,800 active search seconds and up to 2,000
objective calls. The workers record nesting, numerical qualification, profiled
start, search, original-equation endpoint replay and retained-endpoint
diagnostics separately. Recorded reservations govern resume after interruption.
Results include the retained fit even if a later stage stops early.

In the local preflight, fixing the outer coordinates at Orion's warm fit and
solving the linear parameters gave the following verified training results.

| Conditional fit | Direct gain | Overall NMSE | Glucose | Insulin | Utilization | Minutes early, shifted 90 g meals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent | 1.0000 | 0.42387 | 0.19320 | 0.73459 | 0.34381 | 91, 91, 91 |
| Child | 1.3044 | 0.39373 | 0.19559 | 0.61018 | 0.37542 | 91, 91, 91 |

For the meal beginning at minute zero the larger fitted gain raised the sample
peak from 114.3 to 141.1. The observed peak was 301.2. The two Delta fits
determine whether refitting the outer dynamics improves on that result.

## Run and retrieve

The delivery bundles the code with a sealed projection of the reviewed
training inputs, numerical endpoint, public task and admissibility evidence.
The source plan and warm terminal are pinned to the November 5 reliability run.
The archive is generated locally under
`output/insulin-meal-control-delivery-20261105/insulin-meal-control-v1.tar.gz`.

From the Mac:

```bash
scp /Users/yuanzhangxiao/Projects/autoformalism/output/insulin-meal-control-delivery-20261105/insulin-meal-control-v1.tar.gz yxiao2@login.delta.ncsa.illinois.edu:~/
```

On Delta:

```bash
bash <<'BASH'
set -euo pipefail
cd /projects/bibo/yxiao2/repos
test -d insulin-meal-control-v1 || tar -xzf "$HOME/insulin-meal-control-v1.tar.gz"
cd insulin-meal-control-v1
sha256sum --quiet -c SHA256SUMS
bash scripts/hpc/start_insulin_meal_control_delta.sh run
BASH
```

Two CPU array tasks may run concurrently. A dependent report runs after both
finish. Repeating `run` from the same verified checkout reuses confirmed
scheduler receipts. For status and packaging, use the installed path in any
Delta shell.

```bash
AF_WORKER=/projects/bibo/yxiao2/repos/insulin-meal-control-v1/scripts/hpc/start_insulin_meal_control_delta.sh
bash "$AF_WORKER" status
bash "$AF_WORKER" inspect
```

From the Mac, after both workers finish:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/insulin-meal-control-v1/inspection.tar.gz "$HOME/Downloads/insulin-meal-control-inspection.tar.gz"
```

`SUMMARY.md` gives the gain, overall and channel NMSEs, peak times and heights
across all single-meal trajectories, and errors on the no-meal controls. Group
summaries use medians and unscaled MAD. A separate conditional diagnostic
starts from observed insulin at meal end, supplies interpolated observed glucose,
and propagates the fitted insulin balance to the observed peak. Treat that
calculation as a diagnostic on training data. All fitting here uses the fitted
training initials.

## One conditional proposer call

We predeclare the follow-up rule as a diagnostic choice. Both arms must finish
qualified searches with verified endpoints and measured residuals. On each of
`train_003`, `train_006` and `train_007`, the child must have a unique sampled
peak more than one local sampling interval earlier than the observed peak.
The observed peak must follow meal completion. The checker verifies the 90 g
triangle and zero external insulin. Ambiguous or missing witnesses defer the
decision. Height and the glucose, utilization and no-meal errors are reported
beside that decision so tradeoffs remain visible.

Only if the report gives `persistent_early_peak`, prepare the request on Delta.
Preparation recomputes the decision from the original terminal receipts. The
proposer sees the fitted child's equations, shared coefficients, public
requirements, three observed-versus-fitted peak comparisons, and the conditional
insulin calculations. It chooses one replacement for the insulin balance or
glucose-driven insulin production. It can add a connected process or latent
state with an explicit initializer. It must explain the preserved mechanism and
predict the change in timing and height.

```bash
cd /projects/bibo/yxiao2/repos/insulin-meal-control-v1
export PYTHONPATH="$PWD/src:$PWD:/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps"
AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
AF_FITS=/work/hdd/bibo/yxiao2/phase_c/insulin-meal-control-v1
AF_REV=/work/hdd/bibo/yxiao2/phase_c/insulin-meal-proposer-v1
"$AF_PYTHON" scripts/propose_meal_insulin_revision.py prepare --campaign "$AF_FITS" --root "$AF_REV" --model-size 120b
"$AF_PYTHON" scripts/propose_meal_insulin_revision.py preview --root "$AF_REV" --html "$AF_REV/PROMPT.html"
```

Use the same code bundle and prepared revision directory on the serving host.
The default choice is GPT-OSS-120B, with `--model-size 20b` available at
preparation. In the existing vLLM environment on an allocated two-GPU node,
the following starts the 120B server in its own shell if needed. Set `HF_HOME`
to the existing model cache and use the cluster's existing offline Harmony
vocabulary environment as in the earlier construction runs.

```bash
vllm serve openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a --host 127.0.0.1 --port 8000 --max-model-len 32768 --max-num-seqs 1 --gpu-memory-utilization 0.90 --tensor-parallel-size 2
```

On that same node, from the extracted code directory with the project Python
environment active, set `AF_REV` to the prepared revision directory:

```bash
PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" python scripts/propose_meal_insulin_revision.py run --root "$AF_REV" --endpoint http://127.0.0.1:8000
```

The proposer has one generation attempt. Resume uses the recorded attempt and
cached response. Acceptance requires static compilation, public mechanism paths,
connected new components and the current deterministic training checks. An
accepted `result.json` carries the typed child and the numerical work order for
Orion. A revision with new states or nonlinear dynamics needs a fitting backend
qualified for those equations. After an independently verified refit,
`compare_refit_replay` scores its residual array against the first child under
the same training scales, dose panel and no-meal controls. Held-out forecasting
remains a separate experiment under the saved eight-unit prefix.
