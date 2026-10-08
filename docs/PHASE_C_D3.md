# Phase C D3: as published, with gpt-oss-120b

Phase C runs D3 as its authors released it, with one intended change: the model
is gpt-oss-120b, the free model the other baselines and our method use. The
upstream reference is github.com/samholt/DataDrivenDiscovery at
`ee86212dfd5935bb0c9626eaa0570223ff7ecf1c` (`agents.py`, class `D3`;
`config/config.yaml`; `utils/prompts.py`). Phase B's D3 used our own one-call
adapter at a smaller budget; its results stay as they were scored.

Code: `src/autoformalism/baselines/d3_upstream.py` (the loop and prompts),
`src/autoformalism/rebuttal/phase_c_d3.py` (plans, requests, resume).

## The loop

- Generation 0 asks for a first model.
- Every later generation makes two requests in one conversation:
  1. The model reflects on the best models so far. It sees up to 16, each with its validation loss, the loss on every state, and its fitted parameter values, sorted so the lowest loss comes last. It also sees the best model after every earlier generation.
  2. It then writes a new model using its reflection.
- Each new model is fitted with D3's native protocol: Adam, teacher-forced one-step updates `x_next = x + f`, early stopping on validation. Its validation loss is the one-step squared error over every state it models. A model already fitted is not fitted again.
- After each generation only the 16 lowest losses are kept.
- The search ends after 20 generations, or after 20 without a lower best loss.
- **Selection is D3's own rule:** the model with the lowest validation loss is returned.
- The recursive rollout is computed for that model and reported. It is never shown to the model, and it never selects.

## Settings

| Setting | Upstream | Phase C plan |
|---|---|---|
| Model | GPT-4 (1106-preview) | gpt-oss-120b, hosted |
| Generations, patience | 20, 20 | 20, 20 |
| Models shown for reflection | the 16 best, plus the best of each generation | same |
| Temperature | 0.7 in the paper and the configuration; the code sends 0 instead | 0.7, as the paper states |
| top-p | 0.95 | 0.95 |
| Reasoning effort | none: GPT-4 has no such setting | none sent, so the server default applies, as for LLM-SR and LLM-ODE |
| Output limit | none set; GPT-4 Turbo stops at 4,096 tokens | 8,192 tokens, because a reasoning model's thinking counts against the limit |
| Attempts per request | 1; it retries only when a prompt exceeds the context, which these prompts never approach | 1 |
| Fitting | Adam, learning rate 0.01, up to 2,000 epochs, validation every 10 epochs, patience 100 checks | the same, in float64 on CPU, full batch (upstream's 1,000-trajectory batches would take these cells whole) |
| Seeds | 10 | 2 |

## Where it cannot be upstream's

The sealed plan records these departures.

1. **Equations instead of code.** D3 writes PyTorch code that its harness executes. We never execute model-written code, so the model writes the same content as equations in our restricted grammar, with each parameter's starting value. Neural-network components cannot be expressed, so this is D3's white-box mode.
2. **Prompt wording.** The prompts change only where they describe code or black-box components. They also state the update the fitting actually uses. Upstream's prompt says the model is used with an ODE solver, but its fitting adds the output to the current state, with no dt.
3. **Failed generations.** Upstream stops when its first model fails to run. Here a generation that yields no usable model is recorded as failed. While no model exists, the next generation asks for a first model again.
4. **No feature acquisition.** A Phase C cell has no features to acquire.

## Requests and resume

- **Requests:** a task makes at most 39 requests (1 + 2 × 19). The 16-task full run makes at most 624.
- **Replies are kept:** every reply is kept under `results/<task>/conversation/`. Every finished generation is checkpointed. A resumed task re-reads what it has and asks only for what is left.
- **Hosted service:** requests to the hosted service ask for fresh answers. Any reply the gateway replays is counted as a cache hit.
- **Outages:** an outage is waited out for six hours. Past that, the task stops without a result, and running it again resumes it.

## Plans

- `configs/phase_c_d3_smoke_v2.json`: two generations on one cell, so both requests are exercised. It makes 3 requests. Frozen; not a result.
- `configs/phase_c_d3_hosted_120b_v2.json`: 8 roster cells × 2 seeds. **Proposed, pending review.**
- `configs/phase_c_d3_smoke_vm_20b_v2.json`: the smoke plan, served on the GPU VM.

Run from a clean checkout on the VM:

```bash
bash scripts/jetstream/run_phase_c_hosted.sh d3 configs/phase_c_d3_smoke_v2.json phase-c-d3-smoke-v2
PYTHONPATH=src ~/af/venv/bin/python scripts/phase_c_d3.py report --root ~/af/runs/phase-c-d3-smoke-v2
```
