# Phase C LLM-SR: as published, with gpt-oss-120b

Phase C runs LLM-SR as its authors released it, with one intended change: the
model is gpt-oss-120b on the Jetstream2 hosted service, the free model the other
baselines and our method use. The upstream reference is
github.com/deep-symbolic-mathematics/LLM-SR at
`41c212312df6c16d936c9cb395356a62774c47e3`. Their `pipeline.main` runs
unmodified; we supply the data, the specification file and a transport to the
endpoint.

Code: `src/autoformalism/rebuttal/llm_sr_driver.py` (the search),
`src/autoformalism/rebuttal/llm_sr_campaign.py` (plans, sealing),
`scripts/phase_c_llm_sr.py`, `scripts/jetstream/run_phase_c_hosted.sh`.

## The search

- LLM-SR learns one equation per run, so every target of a cell is its own search. The eight roster cells have eleven targets: T2 easy has three, T2 hard two, the others one.
- In each search their islands evolve equation programs. Each prompt asks for four samples. Their evaluator fits each program's parameters to the training derivatives (fourth-order finite differences) and scores it by negative mean squared error, within 30 seconds.
- **Selection is LLM-SR's own rule:** the sample its evaluator scored highest. We convert that program to our grammar and refit its coefficients on the same training derivatives, because their evaluator discards the fitted values. Then the cell's system is rolled out on validation data. The rollout error is reported and chooses nothing.

## Settings

| Setting | Upstream | Phase C plan |
|---|---|---|
| Model | their local engine | gpt-oss-120b, hosted |
| Islands, samples per prompt | 10, 4 | 10, 4 |
| Evaluation limit, island reset | 30 s, every 4 h | the same |
| Sampling controls | none set | none set; no reasoning effort, so the server default applies |
| New tokens per sample | 512 | 4,096 (declared adaptation) |
| A function header split over several lines | read as the start of the body | read as one line (declared adaptation) |
| Samples per target | 10,000, set in `main.py` | 10,000 |
| Seeds | none set; each run samples afresh | 2 runs per target |

## Where it departs from upstream

The sealed plan records these.

1. **Reasoning-model adaptation.** gpt-oss reasons before it answers, within the same token limit, and splits its function header over several lines. At upstream's 512 tokens every sample stopped while still reasoning, and upstream's reader kept nothing of a split header. Measured on 2026-10-05; see the plan's rationale.
2. **Transport.** Their engine applies a chat template to the prompt as one user turn; we send it as one chat message. The hosted gateway replays stored answers, so every request asks for a fresh one, as their engine always generates one.
3. **Specification.** LLM-SR's specifications embed a problem description, so the public task specification is supplied, as its own protocol does.

## Every target at once

- Each target is searched by its own process. The launcher starts all 22 searches (11 targets × 2 seeds) at once, then seals each task once its targets have finished. In turn, the three-target T2 easy cell alone would take about six days instead of two.
- This changes only the orchestration. Each target was always a separate LLM-SR run.
- A finished search leaves `search.json` beside its samples and is kept. A search that stopped part-way (a reboot, a killed process) starts over, because LLM-SR cannot resume; the stopped attempt is kept beside it as `llmsr-<target>.interrupted-N`.
- Before each search the service is asked which models it serves. One that cannot answer is asked again every minute, for as long as a search would wait for it: six hours on the hosted service. On 2026-10-09 two D3 tasks were lost to one slow answer at their start.
- A search that the hosted service stopped, by failing for longer than six hours, is searched again from the start when its task is sealed.
- Their sampler counts samples on its class, which only a new process resets; their runner gives each problem a new process. A process that searches several targets in turn resets the count before each search. Before this was fixed, a second target searched in the same process stopped at once and kept LLM-SR's starting program. No reported result used that path: Phase B reported no LLM-SR results, and the budget pilot's cell has one target.

## Time and resources

From the budget pilot (T1 easy, one seed), on 2026-10-08:
- A sample takes about 17.6 s: 11.7 s waiting for the model and 5.9 s in their evaluator. That is about 206 samples an hour, so a 10,000-sample search takes about two days.
- A search uses about a third of a core on average. 22 at once use about 7–8 of js2's 16 cores and keep about 22 requests in flight at the hosted service. If the service slows under that load, the searches slow with it.
- The full set is 220,000 samples, about 55,000 requests.

## Plans

- `configs/phase_c_llm_sr_smoke_v3.json`: the T2 hard cell, whose two targets exercise the separate searches and the assembly; 8 samples per target, 4 requests. Frozen; not a result.
- `configs/phase_c_llm_sr_hosted_120b_v1.json`: 8 roster cells × 2 seeds at 10,000 samples per target. **Proposed, pending review.**
- The budget pilot (`configs/phase_c_llm_sr_budget_pilot_v2.json`) runs from its own checkout. It is not part of the full set.

A run reads its checkout's code until it ends, so run from a clean checkout that no other run is using:

```bash
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_smoke_v3.json phase-c-llm-sr-smoke-v3 2
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_hosted_120b_v1.json phase-c-llm-sr-hosted-120b-v1 22
```

The last argument is how many target searches run at once. On the GPU VM, `run_phase_c_tasks.sh` still searches a task's targets one after another.
