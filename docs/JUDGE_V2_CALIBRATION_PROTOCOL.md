# Scientific judge v2 calibration protocol

## Purpose

This prospective experiment tests whether the scientific-only v2 judge is
repeatable and whether it reliably lowers its score after a controlled
scientific degradation. It uses only completed development candidates, public
task semantics, and runtime-valid mutations. No fitting, test data, hidden
trajectory, or private reference model enters a judge request.

The unmodified member is labeled `baseline`, not `valid`: the selected gpt-oss
models are runtime-valid but are not assumed to be scientifically correct.

## Matched mutations

Each of the five selected candidates receives four deterministic-valid changes:

1. add the public meal input as an explicit sink in the `Gp` balance;
2. duplicate one existing top-level `Gp` flux;
3. add a plausibly named but disconnected meal mechanism; and
4. add an unjustified one-sided latent accumulator.

Both pair members receive neutral hashed candidate IDs, no lineage marker, and
the same neutral change summary. Mutation names and labels never enter judge
requests. The mutations target source/sink semantics, nonredundancy, mechanism
coupling, and latent-state justification without encoding the hidden benchmark
model. The builder compiles every baseline and mutation before writing it.

Five controlled Ollama sampling seeds judge both members of each pair. With 20
pairs, the complete experiment contains 200 scored rows, sharded over five
GPUs. Within a seed, the same baseline is reused across its four mutations, so
the cache reduces this to 125 unique provider requests without compromising
matched comparisons or repeatability measurement.

## Delta execution

After updating the checkout, build the pair file on the login node:

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-v21

CAL=/work/hdd/bibo/yxiao2/phase_b/judge-v2-calibration-v2
mkdir -p "$CAL" logs

/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python \
  scripts/build_v2_judge_calibration_pairs.py \
  --runs-root /work/hdd/bibo/yxiao2/phase_b/local-generation-weighted-v1/runs \
  --data-root /projects/bibo/yxiao2/phase_b/inputs/public \
  --output "$CAL/pairs.jsonl"
```

The builder should report 20 pairs. Submit all configuration values with the
job so an expired login session cannot affect execution:

```bash
sbatch --account=bibo-delta-gpu --partition=gpuA40x4 --time=02:00:00 \
  --array=0-4 \
  --export=ALL,AF_CALIBRATION_ROOT=/work/hdd/bibo/yxiao2/phase_b/judge-v2-calibration-v2,AF_REPETITIONS=5,AF_SHARD_COUNT=5,AF_JUDGE_SEED_BASE=1000 \
  scripts/hpc/phase_b_v2_judge_calibration.slurm
```

After every shard completes, merge and analyze:

```bash
CAL=/work/hdd/bibo/yxiao2/phase_b/judge-v2-calibration-v2
PY=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python

"$PY" scripts/merge_adversarial_scores.py \
  --inputs "$CAL"/shards/shard_*/adversarial_judge_scores.csv \
  --output "$CAL/judge_scores.csv" \
  --expected 200

"$PY" scripts/analyze_adversarial_judge.py \
  --scores "$CAL/judge_scores.csv" \
  --output "$CAL/judge_metrics.json"
```

Primary outputs are `judge_metrics.json` and
`adversarial_judge_summary.md`. Interpret paired preference as the fraction of
calls where the baseline scores above its controlled mutation. Repeat SD
measures sampling variability for an identical candidate. Category margins
show whether each defect affects the intended rubric component.

## Decision rule

Do not increase judge influence merely because the average paired margin is
positive. Before another weighted-selection experiment, require:

- high repeatability for identical candidates;
- consistently positive margins for every mutation family;
- a low false-preference rate;
- meaningful rather than near-zero score separation; and
- justifications that cite the mutated equation or dependency instead of
  rescoring deterministic validity.

## Stronger-judge comparison

Use the exact same blinded `pairs.jsonl` and unchanged v2 prompt to isolate the
judge-model effect. The hosted comparison defaults to one repetition: 40 scored
rows and 25 unique provider requests because identical baselines are cached.
Set `AF_JUDGE_MODEL_SPEC` at submission time rather than hard-coding a model in
the repository. The OpenAI API key must be present in the submitted environment
but must never be written to a command, script, or log.

```bash
export AF_JUDGE_MODEL_SPEC=openai:gpt-5.6
export AF_HOSTED_OUTPUT_ROOT="$CAL/stronger-judge"

sbatch --account=CPU_ACCOUNT --partition=cpu --array=0-4 \
  --export=ALL,AF_CALIBRATION_ROOT="$CAL",AF_HOSTED_OUTPUT_ROOT="$AF_HOSTED_OUTPUT_ROOT",AF_JUDGE_MODEL_SPEC="$AF_JUDGE_MODEL_SPEC",AF_REPETITIONS=1,AF_SHARD_COUNT=5 \
  scripts/hpc/phase_b_hosted_judge_calibration.slurm
```

Replace `CPU_ACCOUNT` with an authorized Delta CPU allocation. Before
submission, confirm model access with a minimal authorized API request. Official
OpenAI documentation should be consulted for the exact model identifier
available to the account; the runner intentionally accepts any
`provider:model` specification.

When no hosted API credential is available, a larger model from the same local
family provides a capacity-controlled comparison. Download it once into the
persistent shared Ollama store; never let array tasks pull the model
concurrently:

```bash
sbatch --account=bibo-delta-gpu --partition=gpuA40x4 \
  --export=ALL,AF_LOCAL_MODEL=gpt-oss:120b \
  scripts/hpc/ollama_pull_model.slurm
```

After the download succeeds, run a one-shard, one-repetition smoke comparison
with four A40 GPUs. Then use all five shards and five repetitions only after
model loading and structured output are confirmed. The local calibration
launcher accepts `AF_LOCAL_MODEL`; command-line resource requests can override
its one-GPU default for the 120B model.
