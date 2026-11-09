# Further external baselines on the nine-case Phase B comparison

Three more external baselines run on the nine cases of the Phase B main
comparison, with the same training and validation trajectories and the same
public prompts the compared methods saw:

| Method | What runs | Selection | Requests |
|---|---|---|---|
| D3 as published | D3's released propose-reflect loop, prompts and settings, white-box mode (as in [Phase C](PHASE_C_D3.md)) | lowest one-step validation loss (D3's rule) | ≤ 1,053 |
| HDTwinGen | the same loop with HDTwinGen's prompts, which allow black-box multilayer perceptrons beside white-box terms | lowest one-step validation loss (the loop's rule) | ≤ 1,053 |
| Spec-informed SINDy | one request designs up to eight library features from the public prompt; then the compared SINDy | lowest validation rollout NMSE over the compared SINDy's 12 thresholds | 27 |

Each runs three repetitions (0, 1, 2) of the nine cells with gpt-oss-120b on
the Jetstream2 hosted endpoint, sending no reasoning effort. Nothing here reads
test data, a private reference, or anything the compared methods produced.

## Inputs

The comparison's campaign (plan `cdfbd68d…`) recorded the SHA-256 of every
public file it read. `configs/phase_b_nine_case_inputs_v1.json` holds the
digests of each cell's `train.csv`, `validation.csv` and `proposer_prompt.txt`.
`scripts/stage_phase_b_nine_case.py stage` copies each file from the first
local source whose copy has the recorded digest and writes a sealed
`receipt.json`. The two obfuscated T1 tables were kept locally before their
input column was renamed from `u06` to `u01`; such a file is accepted only if
renaming that column in the header line gives the recorded digest. Nothing
else is transformed. No manifest is staged, because the registered manifest
also fingerprints the sealed test table; cells are read with the registered
loader's own table reader, so derivative estimates are the ones the compared
regression baselines used. D3 and HDTwinGen get the validation context the
Phase B D3 campaign built; spec-informed SINDy gets the compared SINDy's.

```bash
python scripts/stage_phase_b_nine_case.py stage \
  --source /path/to/checkout \
  --source /path/to/checkout/artifacts/reference-integrity-audit-2026-09-26-v3/public-development \
  --output /path/to/phase-b-nine-case-v1
python scripts/stage_phase_b_nine_case.py verify --root /path/to/phase-b-nine-case-v1
```

## D3 and HDTwinGen

Both use `baselines/d3_upstream.py`, the port of D3's released loop: a first
model, then per generation a reflection on the 16 best models so far (each
with its validation loss, per-state losses and fitted parameters) and a new
model; at most 20 generations, stopping after 20 without a lower loss;
temperature 0.7, top-p 0.95. Each model's parameters are fitted with Adam
(learning rate 0.01, up to 2,000 epochs, validation every 10, patience 100)
on the one-step update `x_next = x + f`, with no dt, as both releases fit and
validate. The sealed result replays the selected model in numpy, requires
its one-step validation NMSE to match the fit's within relative 1e-6, and
reports the recursive validation rollout: targets start at their observed
initial values and are never reset, supplied auxiliaries and inputs are read
at every step, NMSE uses the training standard deviation per target.

HDTwinGen's release (`samholt/HDTwinGen`, `a7cd335`, class `NSDT`) runs the
same loop with the same settings; D3's released prompts carry the same
black-box wording. D3 runs here in its white-box mode, as on Phase C; the
HDTwinGen campaign runs the hybrid mode. Its prompts are HDTwinGen's, changed
only where they describe code, and they state the update the fitting uses.

HDTwinGen's model writes PyTorch code. We do not execute model-written code,
so a model declares each network in a fixed form instead
(`baselines/hybrid_networks.py`): a name, its inputs as expressions in the
restricted grammar of the observed states, inputs, parameters and `t` (used as
written, without rescaling), 1 to 3 hidden layers of 2 to 64 units, and one of
tanh, relu, sigmoid, softplus, elu or silu. A network has one output, which
equations and algebraics use as a plain symbol; a model declares at most four.
The harness builds the network with PyTorch's default initialization under the
fit's seed, fits its weights with the parameters by the same Adam fit, keeps
them at the same early-stopping checkpoint, and replays them in numpy for the
rollout. As upstream's `get_model_parameters` does, the reflection shows only
the scalar parameters, not the weights.

Departures, recorded in every sealed plan:

- models are equations in a restricted grammar, and networks are declared in
  the fixed form above, instead of executed code;
- the prompts change only where they describe code, and state `x + f`;
- a generation without a usable model is recorded as failed, and while no
  model exists the next generation asks for a first model again;
- the fit runs in float64 on a CPU over all of a cell's trajectories at once,
  where upstream uses float32 batches of up to 1,000 trajectories;
- D3's feature acquisition is off: the cells have no features to acquire.

## Spec-informed SINDy

`llm_feature_sindy` (`baselines/runner.py`) asks the model, in one request
that shows the cell's public prompt, for up to `feature_count` algebraic
features of the observed channels, inputs and `t`, written in the restricted
grammar. Features beyond the count, malformed ones, duplicates and copies of
a single channel are not used. SINDy fits its usual library (constant, linear,
pairwise products and tanh of every channel) plus the features, over the
compared SINDy's 12 thresholds, and keeps the one with the lowest validation
rollout NMSE from the shared evaluator. The fit runs in a child process under
the compared SINDy's 1,800 s limit; a fit past it is `timed_out`. The request
is made once and its reply kept, so an outage or a rerun does not repeat it.
A reply outside the format leaves the library without designed features,
which the result records. Features are algebraic in current values, so this
baseline cannot introduce latent states or memory.

## Running on Jetstream2

From a clean checkout on the VM, with the staged inputs copied to
`~/af/release/phase-b-nine-case-v1`:

```bash
bash scripts/jetstream/run_phase_b_hosted.sh d3 configs/phase_b_nine_case_d3_smoke_v1.json phase-b-nine-case-d3-smoke-v1 1
bash scripts/jetstream/run_phase_b_hosted.sh hdtwingen configs/phase_b_nine_case_hdtwingen_smoke_v1.json phase-b-nine-case-hdtwingen-smoke-v1 1
bash scripts/jetstream/run_phase_b_hosted.sh spec_sindy configs/phase_b_nine_case_spec_sindy_smoke_v1.json phase-b-nine-case-spec-sindy-smoke-v1 2
```

The smokes make 3, 5 and 2 requests. The full plans
(`configs/phase_b_nine_case_*_hosted_120b_v1.json`) run the same way. A task
an outage stops exits with status 3 and resumes on rerun; a finished task is
never repeated. Each run writes `report.json` when its tasks have ended.

Measured locally on the staged cells with scripted replies: on T2-hard a
white-box fit took 15 s and a fit with two 2×32 networks 22 s, both replays
matching the fit's one-step loss, so the model's requests dominate a task's
time. Spec-informed SINDy on T1-easy, with two designed features that did not
change the selected model, reproduced the compared SINDy's validation NMSE
(1.714) in 209 s; on CSTR it reached the 1,800 s limit, as the compared SINDy
did in all three of its repetitions.

## Scope

These are validation results, selected by each method's own rule and scored
by free rollout. No test data have been opened. Literature-model fits and test
scoring are separate steps.
