# GPT-OSS proposer transport calibration

## Purpose

This development experiment selects a maximum output-token budget for the
high-reasoning GPT-OSS-120B proposer. It does not evaluate the scientific judge,
fit parameters, open test data, or use private benchmark information.

The 4,096-token search setting is retained as the failed control. The matched
alternatives are 8,192 and 12,288 tokens. Reasoning effort, temperature, model,
public prompt, deterministic target contract, request seed, response schema,
and retry count remain fixed.

## Frozen matrix

The matrix contains two public Phase-B cells, one named/easy and one
opaque/hard, with three seeds. Each of the six round-zero requests is evaluated
under all three output budgets. A single vLLM server allocation runs the three
conditions for a matched request, avoiding repeated model startup.

Every request has empty beam feedback. This isolates structured proposal
transport and initial candidate validity. Later feedback-rich rounds remain a
separate search integration test after an operating point is selected.

## Measurements and selection

For each budget the analyzer reports:

- structured-response success;
- first-attempt structured-response success;
- deterministic expression and candidate validity;
- deterministic public-target-contract pass rate;
- provider attempts, input/output tokens, and latency;
- output-budget utilization;
- attempts terminating because the output budget was exhausted;
- reasoning-character count when vLLM exposes reasoning text.

The selected operating point is the smallest budget passing every predeclared
gate in `configs/phase_b_proposer_transport_calibration_v1.json`. Candidate
scientific quality has no hidden gold label in this experiment and is not
silently converted into a tuning score.

The GPT-5.6 raw-data-agent mean output-token count is retained as descriptive
resource context only. It does not enter any gate because that agent received a
different task and could use hosted code execution. Exact GPT-5.6 reasoning
tokens remain pending an offline audit of the saved raw response caches.

## Interpretation

Passing selects a proposer transport budget, not a production method result.
The selected budget must next be used in a checkpointed search smoke test that
includes feedback-rich rounds, fitting, deterministic target enforcement, and
the independently frozen low-reasoning scientific judge. A failure at all three
budgets means the experiment selects no operating point; thresholds are not
relaxed after observing the calls.

## ACES replication

The same frozen matrix can run independently on two ACES H100 GPUs per task.
The ACES path changes only the serving hardware and tensor-parallel topology;
the public prompts, target contracts, proposer model, high reasoning level,
sampling seeds, output budgets, retry limit, schema, and analysis gates are
unchanged. The result is an independent transport replication, not a bitwise
duplicate of the four-A40 Delta run. Wall time and GPU-hours must therefore be
reported by platform.

The launcher uses the ordinary ACES `gpu` partition, requests
`--gres=gpu:h100:2`, and permits two array elements at once. A dependent CPU
job first creates or verifies the pinned vLLM image in project storage and
prefetches the model into a shared scratch Hugging Face cache. The array thus
does not race to download the 120B weights. A final dependent CPU job runs the
unchanged analyzer.

From the ACES repository checkout:

```bash
git pull --ff-only
export AF_REPO_ROOT="$PWD"
export AF_PYTHON="$PWD/.venv/bin/python"
export AF_PUBLIC_DATA_ROOT="$PROJECT/phase_b/inputs/public-prompt-v3"
export AF_OUTPUT_ROOT="$SCRATCH/phase_b/proposer-transport-calibration-v1-aces-h100x2"

test -x "$AF_PYTHON"
test -d "$AF_PUBLIC_DATA_ROOT"
bash scripts/hpc/submit_phase_b_proposer_transport_calibration_aces.sh
```

If `sbatch` reports an ACES controller socket timeout, inspect `squeue` before
resubmitting; the scheduler may have accepted the job despite the client-side
timeout. The submitter refuses to overwrite an existing submission manifest,
which also protects against accidental duplicate launches.
