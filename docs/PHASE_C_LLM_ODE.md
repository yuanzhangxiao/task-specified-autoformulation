# Phase C LLM-ODE: as published, with gpt-oss-120b

Phase C runs LLM-ODE's own search with one intended change: the model is
gpt-oss-120b on the Jetstream2 hosted service, the free model the other
baselines and our method use. The upstream reference is
github.com/gryaklab/llm-ode at `31667d9ac948e8f411cd2f07726d7744784b5d5a`
(MIT), and the paper is Bideh and Gryak, GECCO 2026 (arXiv 2603.20910). Their
per-variable searcher runs unmodified. We supply the data, the declared prompt
addition and a transport to the endpoint, and choose the system by the rule in
their released code.

Code: `src/autoformalism/rebuttal/llm_ode_driver.py` (the search and the
selection), `src/autoformalism/rebuttal/llm_ode_campaign.py` (plans, data,
sealing), `src/autoformalism/rebuttal/llm_ode_upstream.py` (loading the
checkout, reading an equation without running it),
`src/autoformalism/rebuttal/phase_c_vendored_campaign.py` (the plan),
`scripts/phase_c_llm_ode.py`, `scripts/jetstream/run_phase_c_hosted.sh`.

## The search

- Upstream's `LlmOde` builds its data by integrating a known system, which we do not have. It is only a loop over `LlmOdeEquation`, one searcher per variable, so we drive those directly on the observed trajectories, in its order: every searcher steps once per iteration.
- **One search per target.** Upstream's systems are autonomous: every variable is a state. A cell supplies auxiliaries, measured inputs and fixed covariates over the horizon, so they are variables every equation may use, and only the targets are searched. The eight roster cells have eleven targets: T2 easy has three (Gp, I, U), T2 hard two (Gp, I), the others one.
- In each iteration every island gets a prompt with 8 equation skeletons, sorted worst to best, and asks for 3 new ones with every constant written C. Their code fits the constants by BFGS on the training derivatives and scores each equation by its training derivative error. Every 5 iterations each island keeps its better half, then trades copies of 2 programs with another island.
- Each target keeps a Pareto front: equation size (SymPy node count) against the derivative error on the validation split.
- **Derivatives.** Upstream differentiates with findiff at fourth order. We use the same stencil within each trajectory and stack the rows. The regression is pointwise, so stacking is sound; differentiating across a join between trajectories would not be.

## Choosing the system: the released code's rule

Decided on 2026-10-17. This replaces the rule the adapter had before: the
lowest validation error alone, the Phase B fallback for when no usable selector
was found.

**The released code** (`LlmOde._store_results`):
- assembles every system in the product of the per-variable fronts;
- rolls each out from the training initial state over the whole training trajectory, t from 0 to 10: the half its equations were fitted on and the half it validates on;
- keeps the system with the lowest mean squared error in the data's own units, pooled over every variable and time point;
- computes the test error beside it and only reports it;
- gives each rollout 2 seconds, scores the systems in 30 processes and has no cap on the product.

**The paper's rule** (Eq. 2) takes the knee of the systems' size-against-error
front, subject to T ≥ h·T_max with h = 0.1 and T = −MSE. It is not in the
released code. As written it cannot be met: with T ≤ 0, it asks for an error
below a tenth of the best one.

**Here** the same rule runs on the development data:
- Every development trajectory, training and validation, is rolled out from its own first values: 16 and 4 trajectories, or 6 and 3 in the basin cells.
- The rollout is the shared evaluator's: RK45, rtol 1e-7, atol 1e-9, with the supplied channels held to their observations, as on test.
- The targets' squared errors are pooled over every trajectory and time point, in the data's own units.
- In the cells with two or three targets, a target with larger values counts for more, as upstream's variables do. The training standard deviations are Gp 50 and I 112 in both T2 cells, and U 1.9 in T2 easy. So I's error dominates, and U's barely counts.
- A rollout may take 30 seconds per trajectory, a tenth of the frozen evaluator's 300 for a test trajectory. Upstream's 2 seconds were for looser tolerances on 100-point trajectories. A system whose rollout fails, or takes longer, cannot be chosen, as upstream leaves it.
- The whole product is scored; there is no cap. The old cap of 10,000 could have turned a finished T2 easy search into a failure.
- **Stopping early changes no choice.** Squared errors only add, so once a system's errors so far exceed the lowest error a finished system has, it cannot be the lowest, and it is stopped there.
- Eight processes score the systems.
- A tie goes to the first system in product order, best derivative fits first. Upstream's sort leaves ties in no set order.
- Recorded beside the choice, from the same rollouts: the training and validation errors normalized as the evaluator normalizes them. They choose nothing.
- `systems.json` lists every system, how it fared and its error. It takes the place of the table upstream writes after its last iteration.

**Cost.** On a laptop, rolling out one T2 easy system over one trajectory takes
0.25 to 0.4 s, because the solver restarts wherever an input's slope changes,
and the insulin input changes slope every minute (300 restarts). A test product
of 125 similar three-target systems took 4 minutes in 8 processes. In the
smokes on js2 on 2026-10-18 (20 iterations), choosing took 2 to 21 seconds in
the one-target cells, 18 and 60 seconds in T2 hard (154 and 168 systems), and 6
and 9 minutes in T2 easy (1,344 and 2,288 systems); most systems were stopped
early.

## The prompt: two campaigns

The paper withholds variable meanings and system descriptions to limit recall
of memorized equations (Sec. 4.1). Every other method compared here, our own
included, is given the public task description. Decided on 2026-10-17:

- **The main comparison gives LLM-ODE the description**, as a declared adaptation. Every prompt ends with:
  - the description's public sections (A and B, or A to D in the basin layouts);
  - the variables' names (`x_0 = Gp, ...`);
  - which derivative the search is proposing.

  Upstream's instructions, examples, operators, fitting and selection are unchanged.
- **A second campaign runs the prompt as published.** It names the variables x_0 to x_n and describes nothing. It is reported separately, with its own label.

## Upstream reads each reply with SymPy

Upstream's `validate_hypothesis` passes every reply line to `sympy.sympify`,
which evaluates the text as Python. This happens inside upstream's search, in
its own processes on js2.

Accepted on 2026-10-17, as LLM-SR's search runs every program it scores:
- The search is upstream's and is not modified.
- js2 holds no credentials.
- Our code never evaluates the text. The chosen equations are read by a syntax-tree reader (`llm_ode_upstream.to_state_equations`) into the restricted baseline grammar, and rolled out by the shared evaluator.

## Settings

| Setting | Paper | Released code | Phase C plan |
|---|---|---|---|
| Model | Qwen3-30B-A3B-Instruct-2507, Ministral-3-14B-Instruct-2512, Olmo-3.1-32B-Instruct, on vLLM | the first model the server lists | gpt-oss-120b, hosted |
| Iterations | 200 | 200 | 200 |
| Islands, programs per island | 4, 2 | 4, 2 | the same |
| Examples and new equations per prompt | 8, 3 | 8, 3 | the same |
| Trim and mix | every 5 iterations, 2 programs | the same | the same |
| Output tokens per reply | not stated | 1,024 | 1,024 |
| Temperature, reasoning effort | not stated | none sent | none sent; the server's defaults apply |
| Prompt | no variable meanings, no description | the same | description appended (main campaign); as published (second campaign) |
| Derivatives | findiff, fourth order | the same | the same stencil, within each trajectory |
| Selection | knee of the front (Eq. 2, h = 0.1) | lowest rollout error over the training trajectory | the released code's rule, over every development trajectory |
| Rollout limit | not stated | 2 s, solve_ivp defaults | 30 s per trajectory, the shared evaluator's tolerances |
| Seeds | not stated | none set | 2 runs per cell, unseeded |
| Held-out scoring | NMSE from a held-out initial state | the same, reported | the shared rollout NMSE on test trajectories |

On 2026-10-06, eight replies of the hosted gpt-oss-120b to this prompt with the
description finished within 1,024 tokens, at 452 to 916 (394 to 820 of them
reasoning), and all 24 of their equations parsed. In the smokes on 2026-10-18
(every cell, 20 iterations), 46 of the 880 replies to the prompt with the
description reached the limit, 43 of them in the two detention cells (33 and 10
of 80), and 7 of the 880 to the published prompt; the model's reasoning counts
toward it. Upstream reads the first three lines of a reply, and 30 of the 53
cut-off replies had written their three equations first, so about 2% of the
description campaign's proposed equations were lost. The limit stays
upstream's, decided on 2026-10-21.

## Where it departs from the released code

The sealed plan records each, and a report states them.

1. **Model.** gpt-oss-120b on the hosted service, a reasoning model, where the paper used three instruct models it served itself.
2. **Data.** Observed trajectories rather than an integrated known system, with only the targets searched and each trajectory differentiated on its own (above).
3. **Selection.** The released code's rule, on every development trajectory, with the shared solver and a 30-second limit (above). Test data are never read.
4. **Prompt.** The main campaign appends the description (above).
5. **Transport.** None of these changes a prompt or a reply.
   - Every request names the plan's model; the gateway lists several, where upstream's own server served one.
   - Every request asks the gateway for a fresh answer; the gateway replays a stored one to a repeated request, and upstream's server samples afresh.
   - A request the server fails is retried for up to 6 hours. Upstream's client retries twice within seconds and raises.
   - At a task's start, a model list the service cannot give is asked for again every minute, for as long as a search would wait. On 2026-10-09 two D3 tasks were lost to one slow answer at their start.
6. **Logs.** Upstream's `main.py` logs every prompt and reply at INFO to `llm_ode.log`; the driver does the same, in each task's directory.

## Requests, records and resume

- Each campaign makes 200 iterations × 4 islands × 22 searches (11 targets × 2 seeds) = 17,600 requests; both campaigns make 35,200.
- A task's directory, `results/<index>/`, holds:
  - `llm_calls.jsonl`: each call's usage, why it stopped and whether it was replayed;
  - `llm_ode.log`: upstream's log of every prompt and reply, and the driver's progress;
  - `<target>.complexity_pf.jsonl`: upstream's front after every iteration;
  - `systems.json`: every assembled system and its error;
  - `result.json` and `native-selection.json`: the sealed result and the chosen model, which the frozen evaluator reads.
- A search cannot resume part-way. A task that stopped (a reboot, a killed process) starts over, with its attempt kept beside it as `<index>.interrupted-N`. A finished task is never repeated.
- An outage longer than 6 hours ends the task as `endpoint_unavailable`, an infrastructure failure, not a result.

## Plans

- `configs/phase_c_llm_ode_smoke_v1.json`: the main campaign's plan on every roster cell, one seed, 20 iterations: 880 requests. Frozen; not a result. It checks:
  - the hosted route;
  - how often a reply reaches the 1,024-token limit;
  - how many reply lines upstream can read;
  - the conversion of the chosen system to the evaluator's grammar;
  - the selection's time and the sealed result.

  It also measures an iteration's time in each cell.

  Ran on 2026-10-18 and passed:
  - all eight tasks sealed complete in 15 minutes;
  - 880 requests, no cache hits, no failed requests;
  - every system in the products converted to the evaluator's grammar;
  - about 99% of the lines upstream reads were equations; lowercase `abs`, which SymPy reads as `Abs`, was the only unusual name;
  - 46 replies reached the limit (above);
  - an iteration took 9 to 15 seconds for each target searched.
- `configs/phase_c_llm_ode_published_prompt_smoke_v1.json`: the same smoke with the prompt as published. Frozen; not a result. Ran beside it on 2026-10-18 and passed: all eight tasks sealed complete in 19 minutes; 880 requests, no cache hits, no failed requests; 7 replies reached the limit.
- `configs/phase_c_llm_ode_hosted_120b_v1.json`: the main campaign, 8 roster cells × 2 seeds at 200 iterations. Frozen on 2026-10-21, after both smokes passed and the user signed off. No setting changed; the budget's rationale now gives the smokes' measurements.
- `configs/phase_c_llm_ode_published_prompt_hosted_120b_v1.json`: the second campaign, the same with the prompt as published. Frozen with it.

Run from a clean checkout that no other run is using:

```bash
bash scripts/jetstream/run_phase_c_hosted.sh llm_ode configs/phase_c_llm_ode_smoke_v1.json phase-c-llm-ode-smoke-v1 8
bash scripts/jetstream/run_phase_c_hosted.sh llm_ode configs/phase_c_llm_ode_published_prompt_smoke_v1.json phase-c-llm-ode-published-prompt-smoke-v1 8
bash scripts/jetstream/run_phase_c_hosted.sh llm_ode configs/phase_c_llm_ode_hosted_120b_v1.json phase-c-llm-ode-hosted-120b-v1 8
bash scripts/jetstream/run_phase_c_hosted.sh llm_ode configs/phase_c_llm_ode_published_prompt_hosted_120b_v1.json phase-c-llm-ode-published-prompt-hosted-120b-v1 8
```

The last argument is how many tasks run at once. The two full campaigns run
side by side, 8 tasks each, as the smokes did. When every task has ended, the
launcher writes the report to `report.json` in the run's directory.

## Limitations

- **Selection time grows with the product of the fronts.** After the smokes' 20 iterations the fronts held 8 to 16 equations per target, and T2 easy's 1,344 and 2,288 systems took 6 and 9 minutes. The full run's fronts will be larger: at 25 equations per target, T2 easy has 15,625 systems, about an hour at the smokes' rate, longer if fewer stop early.
- **Wall-clock limits.** The 30-second limit is wall time, as upstream's alarm is. On a heavily loaded VM a slow system could exceed it and become unselectable. Typical rollouts take under a second; in the published prompt's smoke, 58 of T2 easy's 2,288 systems exceeded the limit.
- **Replies cut off.** The model's reasoning counts toward upstream's 1,024-token limit. In the smokes it cut off 5% of the replies with the description (41% in the coupled detention cell) and 0.8% of the replies to the published prompt. About 2% of the description campaign's proposed equations were lost.
- **Unseeded.** Upstream sets no seed, so the two runs per cell are independent and cannot be reproduced exactly.
- **Raw units.** Pooling the targets in their own units, as upstream does, lets the target with larger values decide the choice in the T2 cells.
- **A finished search can still be lost.** The selection runs after the search in the same process, so a reboot during selection restarts the whole task. The fronts are in upstream's per-target logs, but nothing yet reruns the selection from them.
- **A shared helper LLM-SR uses.** `llm_ode_driver.development_rollout_error`, which LLM-SR uses to roll out a pick that is an equation, lets the simulator's timeout escape rather than return a failure. LLM-ODE no longer uses it to select. It is unchanged here, as LLM-SR's run depends on it.
