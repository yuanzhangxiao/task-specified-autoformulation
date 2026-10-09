# Phase C LLM-SR: as published, with gpt-oss-120b

Phase C runs LLM-SR as its authors published it, with one intended change: the
model is gpt-oss-120b on the Jetstream2 hosted service, the free model the other
baselines and our method use. The upstream reference is
github.com/deep-symbolic-mathematics/LLM-SR at
`41c212312df6c16d936c9cb395356a62774c47e3`, and the paper is Shojaee et al.,
ICLR 2025 (arXiv 2404.18400). Their `pipeline.main` runs unmodified; we supply
the data, the specification file and a transport to the endpoint.

Code: `src/autoformalism/rebuttal/llm_sr_driver.py` (the search),
`src/autoformalism/rebuttal/llm_sr_campaign.py` (plans, data, sealing),
`src/autoformalism/rebuttal/llm_sr_upstream.py` (reading the selected program),
`scripts/phase_c_llm_sr.py`, `scripts/jetstream/run_phase_c_hosted.sh`.

## The data: PySR's regression table

LLM-SR is symbolic regression, as PySR is: it fits an equation of the current
variables to derivative labels, with no notion of time. Its own datasets are
tables of variables and a label column. So it takes the data as our PySR
baseline does:

- the same training rows, from every training trajectory;
- the same channels at each row, in PySR's order (targets first, then
  auxiliaries, external inputs and fixed covariates);
- the same labels: each target's derivative, estimated with `numpy.gradient`
  within each trajectory, as for PySR and SINDy.

What LLM-SR does not share with PySR is that its programs can read the arrays
along their rows, and in time order they did. In the budget pilot (T1 easy,
time order, 3,532 scored samples), about 89% of programs, and all of the 100
best, filtered or shifted the meal series along the rows to stand in for the
unobserved gut compartments. Such a program is a function of a whole series,
not an equation of the current variables, and cannot be rolled out. In that
pilot the best program was a filter by sample 200, and all ten best were by
sample 400. So the rows are given in a random order, fixed by the plan's seed
and the same for both repetitions, as PySR's two seeds fit one table.

The limits are the method's, as for PySR: the derivatives are estimated from
the observations, and only observed channels enter the equation.

## The search

- LLM-SR learns one equation per run, so every target of a cell is its own search. The eight roster cells have eleven targets: T2 easy has three, T2 hard two, the others one.
- In each search their islands evolve equation programs. Each prompt asks for four samples. Their evaluator fits each program's ten parameters to the labels by BFGS and scores it by negative mean squared error, within 30 seconds.
- **Selection is LLM-SR's own rule:** the sample its evaluator scored highest. Then the cell's system is rolled out on validation data. The rollout error is reported and chooses nothing.

## Reading the selected program

Their evaluator discards the fitted coefficients, and our evaluator reads one
restricted expression, so the selected program is read into one:

- Nothing is executed. A reader walks the program's syntax tree and knows a fixed set of constructs.
- Guards with one outcome are read through. These are checks on the length of `params`, which always has ten entries, on the inputs' shapes, which are all equal, and conversions such as `np.asarray`. Copies of `params`, helper functions and loops over coefficients are read through the same way.
- The coefficients are refitted with their evaluator's own call: BFGS from all ones on the same rows. A branch on a coefficient, such as `if K == 0: K = 1e-6`, follows each trial value, as in theirs.
- What depends on the rows or on the data's values is refused with a named reason. Examples: a filter along the rows, a reduction over them, a branch on the data, and a guard that changes values where the data are exactly zero.
- Read this way, 377 of the 380 pilot programs that use only the current row convert, against 38 before.
- The sealed result keeps LLM-SR's score beside the refitted training error, so any gap between their program and our reading of it shows.

## Settings

| Setting | Paper | Released code | Phase C plan |
|---|---|---|---|
| Model | Mixtral-8x7B, GPT-3.5 | their local engine | gpt-oss-120b, hosted |
| Data | tables of variables and labels | the problem's CSV | PySR's table, rows in a seeded random order |
| Temperature | 0.8 | none sent by their client (so 1.0) | 0.8 (paper) |
| Samples per prompt, examples per prompt | 4, 2 | 4, 2 | 4, 2 |
| Islands, island reset | 10, every 4 h | 10, every 4 h | the same |
| Cluster-sampling period N (T0 = 0.1) | 10,000 | 30,000 | 10,000 (paper) |
| Evaluators | 4 | 1, run one sample at a time either way | 4 (paper; changes nothing) |
| Evaluation limit, memory | 30 s, 2 GB | 30 s, no memory limit | 30 s, no limit (their sandbox) |
| Parameters, optimizer | 10, BFGS | 10, BFGS | the same |
| Samples per target | about 2,500 iterations × 4 | 10,000, set in `main.py` | 10,000 |
| New tokens per sample | not stated | 512 | 4,096 (declared adaptation) |
| A function header split over several lines | — | read as the start of the body | read as one line (declared adaptation) |
| Reasoning effort | — | — | none sent; the server default applies |
| Seeds | — | none set; each run samples afresh | 2 runs per target |

## Where it departs from the released code

The sealed plan records each, and a report states them.

1. **Reasoning-model adaptation.** gpt-oss reasons before it answers, within the same token limit, and splits its function header over several lines. At upstream's 512 tokens every sample stopped while still reasoning, and upstream's reader kept nothing of a split header. Measured on 2026-10-05; see the plan's rationale.
2. **Transport.** Their engine applies a chat template to the prompt as one user turn; we send it as one chat message. The hosted gateway replays stored answers, so every request asks for a fresh one, as their engine always generates one.
3. **Specification.** LLM-SR's specifications embed a problem description, so the public task specification is supplied, as its own protocol does. The starting program names its result after the target's derivative (`dI` for target I), as upstream's `dv` does.
4. **Data.** PySR's regression table, its rows in a seeded random order (above).
5. **The paper's settings.** Temperature 0.8, cluster-sampling period 10,000 and 4 evaluators, where the released code differs. Each is one of their own configuration values; the search code is unchanged.

## Every target at once

- Each target is searched by its own process. The launcher starts all 22 searches (11 targets × 2 seeds) at once, then seals each task once its targets have finished. In turn, the three-target T2 easy cell alone would take about six days instead of two.
- This changes only the orchestration. Each target was always a separate LLM-SR run.
- A finished search leaves `search.json` beside its samples and is kept. A search that stopped part-way (a reboot, a killed process) starts over, because LLM-SR cannot resume; the stopped attempt is kept beside it as `llmsr-<target>.interrupted-N`.
- Before each search the service is asked which models it serves. One that cannot answer is asked again every minute, for as long as a search would wait for it: six hours on the hosted service. On 2026-10-09 two D3 tasks were lost to one slow answer at their start.
- A search that the hosted service stopped, by failing for longer than six hours, is searched again from the start when its task is sealed.
- Their sampler counts samples on its class, which only a new process resets; their runner gives each problem a new process. A process that searches several targets in turn resets the count before each search. Before this was fixed, a second target searched in the same process stopped at once and kept LLM-SR's starting program. No reported result used that path: Phase B reported no LLM-SR results, and the budget pilot's cell has one target.

## Time and resources

From the budget pilot (T1 easy, one seed, time order), on 2026-10-08:
- A sample takes about 17.6 s: 11.7 s waiting for the model and 5.9 s in their evaluator. That is about 206 samples an hour, so a 10,000-sample search takes about two days.
- A search uses about a third of a core on average. 22 at once use about 7–8 of js2's 16 cores and keep about 22 requests in flight at the hosted service. If the service slows under that load, the searches slow with it.
- The full set is 220,000 samples, about 55,000 requests.

The table has the same rows as the time-ordered arrays, so the evaluator's cost per sample should not change.

## Plans

- `configs/phase_c_llm_sr_smoke_v4.json`: the full plan's settings on the budget pilot's cell (T1 easy) and the two-target T2 hard cell, 400 samples per target, at most 300 requests. It shows whether the programs LLM-SR ranks highest on the shuffled table are equations our reader converts. Frozen; not a result.
- `configs/phase_c_llm_sr_smoke_v3.json`: the T2 hard cell in time order, 8 samples per target. Frozen; ran on 2026-10-09 and sealed "inexpressible": both picks read along the rows.
- `configs/phase_c_llm_sr_hosted_120b_v1.json`: 8 roster cells × 2 seeds at 10,000 samples per target, with PySR's table and the paper's settings. **Proposed, pending review.**
- The budget pilot (`configs/phase_c_llm_sr_budget_pilot_v2.json`) runs in time order from its own checkout. Only its pace is used.

A run reads its checkout's code until it ends, so run from a clean checkout that no other run is using:

```bash
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_smoke_v4.json phase-c-llm-sr-smoke-v4 3
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_hosted_120b_v1.json phase-c-llm-sr-hosted-120b-v1 22
```

The last argument is how many target searches run at once. On the GPU VM, `run_phase_c_tasks.sh` still searches a task's targets one after another.
