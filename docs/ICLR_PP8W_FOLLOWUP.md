# PP8W follow-up: local analyses and dedicated Delta searches

## Frozen experiment

This follow-up uses Full with the 20B proposer, 15 visits (initial construction
plus 14 revisions), final pruning, the existing fitter, and validation/complexity
selection. A finite fitted initial Full model is required, matching the paired
anchor's handling of initialization failures. Seeds are 0, 1, and 2.

| Condition | Training trajectories | Noise SD / clean training SD | Sampling | Searches |
|---|---:|---:|---|---:|
| CSTR-hard, reduced data | 4 | 0 | 0.1 min | 3 |
| T2-hard, reduced data | 4 | 0 | 1 min | 3 |
| CSTR-hard, matched control | 16 | 0 | 0.1 min | 3 |
| T2-hard, matched control | 16 | 0 | 1 min | 3 |
| CSTR-hard, noisy | 16 | 0.10 | 0.1 min | 3 |
| CSTR-hard, sparse | 16 | 0 | 0.4 min plus input corners | 3 |

The 4-trajectory subsets are fixed by public input schedules: CSTR includes the
baseline and one upward perturbation of each input; T2 includes fasting, a 90 g
meal, a shifted 90 g meal, and a 90 g meal with insulin. These are search seeds,
not independent data resamples. Validation retains its original full roster.
The fitter recomputes normalization from the available training data; held-out
evaluation uses the common clean 16-trajectory training scales.

For acquisition arms, noise or sampling changes apply to training and validation.
Input commands are preserved and forcing corners remain sampled. Noise is
Gaussian with the frozen stress-study seed; initial target measurements are
also noisy. The new training-count comparison is conditional on retaining the
full validation set, not a claim that only four total trajectories are needed.

The 120B critic is hosted locally on Delta. Because its serving backend differs
from the earlier Jetstream critic, six fresh 16-trajectory control searches are
included. Both LLMs use pinned model revisions and the existing pinned vLLM
image. No external baseline or hosted Jetstream call is part of this launch.

## Delta launch

The delivery archive is `iclr-pp8w-delta-20261118.tar.gz`. From the Mac:

```bash
scp /Users/yuanzhangxiao/Projects/autoformalism/output/iclr-pp8w-followup-20261118/iclr-pp8w-delta-20261118.tar.gz delta:/projects/bibo/yxiao2/
ssh delta
```

In the Delta shell, extract into a new source directory, then launch:

```bash
mkdir -p /projects/bibo/yxiao2/repos
tar --keep-old-files -xzf /projects/bibo/yxiao2/iclr-pp8w-delta-20261118.tar.gz -C /projects/bibo/yxiao2/repos
cd /projects/bibo/yxiao2/repos/iclr-pp8w-delta-20261118
bash scripts/hpc/iclr_pp8w_delta.sh run
```

Use the existing extracted directory for subsequent checks; do not re-extract
over an active frozen source tree. Preparation verifies the data hashes, pinned
image, and offline model revisions. It freezes the runtime on Delta, rather
than reusing a Mac-prepared search plan. The archive includes the offline Harmony
vocabulary. It contains public train/validation data only, with no held-out
reference trajectories or API credentials.

Defaults are the existing `autoformalism-v21` Python environment, the CasADi
dependency directory, the project Hugging Face cache, and
`/projects/bibo/yxiao2/containers/vllm-openai-v0.27.1.sif`. Environment overrides:
`AF_PYTHON`, `AF_CASADI_ROOT`, `AF_HF_HOME`, `AF_VLLM_IMAGE`, `AF_ACCOUNT`,
`AF_PP8W_ROOT`, and `AF_CONCURRENCY` (default 4).

The controller requests four A40 GPUs, 128 GB RAM, and 24 hours. It first runs
the existing 14 known calibration pairs in both orders with five repetitions
(140 orientations). All original calibration gates must pass before production
searches can start. This is a local transport revalidation, not a new independent
calibration dataset. If calibration fails, the launcher stops and reports the
failure; it does not switch back to Jetstream.

After calibration, each search block requests one A40 GPU, eight CPU cores,
64 GB RAM, and six hours. Up to four blocks run concurrently, for at most eight
GPUs including the critic. The controller submits unfinished blocks and supports
up to three resumable array waves. An allocation ending can require another
submission; queue time and completion within 24 hours are not guaranteed.

The critic service has a campaign-specific credential stored with mode 0600;
it is never placed in command arguments or result packages. It is shared only
by this campaign's workers. The launcher does not stop or modify other jobs.

```bash
bash scripts/hpc/iclr_pp8w_delta.sh status
squeue -u "$USER" -o '%.18i %.30j %.2t %.10M %.10l %.6D %R'
```

Logs and receipts are under:

```text
/work/hdd/bibo/yxiao2/phase_c/iclr-pp8w-20261118/
  calibration/summary.json
  logs/
  runtime/critic-service.json
  runtime/critic-*.log
  runtime/proposer-*.log
  search/results/
  search/pruning/
  search/frozen_models.json
  completed.json
```

After an allocation ends, repeat `submit` to resume unfinished work:

```bash
bash scripts/hpc/iclr_pp8w_delta.sh submit
```

An active controller is reused; an already complete campaign creates no jobs.
Do not delete started-call markers or reset failed scientific outcomes. The
controller's final export is separate from held-out scoring. After all search
endpoints are frozen, score CSTR on the existing 24-intervention panel and T2 on
the existing six-case stress panel. Test scores must not influence search,
pruning, selection, or choosing which failed seeds to rerun.

The post-freeze adapter is `scripts/iclr_pp8w_evaluation.py`. It refuses a partial
roster and checks every model's public input hashes against the follow-up design.
It scores CSTR targets and the three declared mechanisms, and reuses the existing
six-case glucose scorer. The primary noisy condition keeps the existing stress
protocol's noisy initial target observation, with the same noise draw for the
intervention and its paired control. Three additional evaluations of those same
noisy-CSTR models use clean initial observations. This secondary diagnostic
separates forecast initialization error from effects of noisy training; it is
reported separately and never used for model selection. There are 18 discovery
searches and 21 post-freeze evaluations, not 21 searches.

Once `completed.json` exists, copy the frozen export to the Mac and evaluate:

```bash
cd /Users/yuanzhangxiao/Projects/autoformalism
scp delta:/work/hdd/bibo/yxiao2/phase_c/iclr-pp8w-20261118/search/frozen_models.json output/iclr-pp8w-followup-20261118/delta_frozen_models.json
export PYTHONPATH="/Users/yuanzhangxiao/Projects/autoformalism-term-timing/src:$PWD/src:$PWD:/Users/yuanzhangxiao/Projects/autoformalism-term-timing"
.venv/bin/python -m scripts.iclr_pp8w_evaluation freeze \
  --models output/iclr-pp8w-followup-20261118/delta_frozen_models.json \
  --design output/iclr-pp8w-followup-20261118/design \
  --panels output/iclr-pp8w-followup-20261118/anytime/plan.json \
  --root output/iclr-pp8w-followup-20261118/delta-evaluation
.venv/bin/python -m scripts.iclr_pp8w_evaluation run \
  --root output/iclr-pp8w-followup-20261118/delta-evaluation --workers 4
.venv/bin/python -m scripts.iclr_pp8w_evaluation report \
  --root output/iclr-pp8w-followup-20261118/delta-evaluation
```

The analyst ZIP also includes the exact borrowed evaluator dependencies for a
portable local replay. The source archive intended for Delta contains no saved
held-out responses, so evaluation remains a separate post-freeze operation.

The calibration source is expected at
`/work/hdd/bibo/yxiao2/phase_b/judge-jetstream-calibration-inputs-v1.json`.
Alternatively set `AF_CALIBRATION_INPUTS`. If absent, preparation attempts the
existing public-case export from `judge-hybrid-consensus-validation-v1` and the
Phase B public inputs; it fails explicitly if those assets are unavailable.

## Analyses that do not require LLM calls

`scripts/iclr_anytime_response.py` evaluates the recorded validation-selected
incumbents at visits 1, 5, 10, and 15, plus the final pruned endpoint. The initial
scope is 60 Full runs: the six physical anchor runs and 54 additional physical
and synthetic stress runs. The corrected CSTR-easy campaign and other historical
benchmark campaigns are outside this budget analysis. This gives 300 planned
endpoints, including unavailable models in the denominator.

The source is the authenticated scientific-history export. Existing frozen
reference responses are reused. CSTR uses the 24 interventions, glucose uses
the six-case panel, and synthetic systems use their three-case panels. The final
six physical anchor endpoints must reproduce all four previously reported scores
to within 1e-8 before the cost report is produced. No test-based checkpoint is
selected, and these results do not define an adaptive stopping policy.

Within a physical run, NMSE and response RSE are medians across its panel;
directional accuracy and shape error are means. Across seeds, all use
median [worst, best]. Synthetic NMSE retains the existing mean across its three
cases. CSTR RSE retains the original sample-wise convention; the glucose and
synthetic panels retain their existing conventions. These panel scores should
not be pooled into a single cross-domain NMSE.

`scripts/iclr_anytime_cost_export.py` is a read-only standard-library export of
the original cost receipts. `scripts/report_iclr_anytime.py` authenticates each
visit's result hash against its scientific history, joins recorded time and
tokens, verifies final score parity, and produces tables plus PNG/PDF figures.
Time includes recorded proposal, critic, fitting and pruning activity, excluding
queues, server startup, this evaluation, and unrecorded work before interrupted
attempts. Missing timing receipts remain missing, not zero. Token counts are
observed proposer usage, not total critic-plus-proposer usage.

`scripts/iclr_pp8w_reference_diagnostic.py` fits the existing supplied CSTR
equations on the four-trajectory, 16-trajectory, and 10%-noise conditions. Three
optimizer starts use training data only; validation selects one fit per
condition. All three selected models are frozen before opening the existing
24-intervention panel. The reported score is for one selected fit, not a
three-seed discovery statistic. Equations, parameter scales and bounds are
privileged information; this is neither an equal-information baseline nor a
lower bound on discovery error. The sparse-grid condition is omitted because
the existing supplied-structure fitter requires the anchor's uniform time grid.

## Verification and limits

Relevant tests cover preserved observations/splits, deterministic subsets,
unchanged prompts, local-only transport and credential redaction, failed
calibration gates, endpoint selection without test metrics, continuous metrics,
cost provenance, and resumable Delta submission. A local preparation smoke test
must freeze all 18 blocks with the local critic and finite-initial-model rule.
Actual Delta asset availability, vLLM startup, local calibration success, and GPU
runtime can only be verified after authenticated access and a scheduled job.
