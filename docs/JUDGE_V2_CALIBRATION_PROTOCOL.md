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

1. reverse the complete proposed `Gp` dynamics, swapping all claimed sources
   and sinks;
2. duplicate the complete proposed `Gp` balance, double-counting every flux;
3. add a plausibly named but disconnected meal mechanism; and
4. add an unjustified one-sided latent accumulator.

These mutations target source/sink semantics, nonredundancy, mechanism
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

CAL=/work/hdd/bibo/yxiao2/phase_b/judge-v2-calibration-v1
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
  --export=ALL,AF_CALIBRATION_ROOT=/work/hdd/bibo/yxiao2/phase_b/judge-v2-calibration-v1,AF_REPETITIONS=5,AF_SHARD_COUNT=5,AF_JUDGE_SEED_BASE=1000 \
  scripts/hpc/phase_b_v2_judge_calibration.slurm
```

After every shard completes, merge and analyze:

```bash
CAL=/work/hdd/bibo/yxiao2/phase_b/judge-v2-calibration-v1
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
