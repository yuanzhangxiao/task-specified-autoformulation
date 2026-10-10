# Phase C LLM-SR: as published, with gpt-oss-120b

Phase C runs LLM-SR as its authors published it, with one intended change: the
model is gpt-oss-120b on the Jetstream2 hosted service, the free model the other
baselines and our method use. The upstream reference is
github.com/deep-symbolic-mathematics/LLM-SR at
`41c212312df6c16d936c9cb395356a62774c47e3`, and the paper is Shojaee et al.,
ICLR 2025 (arXiv 2404.18400). Their `pipeline.main` runs unmodified; we supply
the data, the specification file and a transport to the endpoint.

Code: `src/autoformalism/rebuttal/llm_sr_driver.py` (the search and recovery),
`src/autoformalism/rebuttal/llm_sr_campaign.py` (plans, data, sealing),
`src/autoformalism/rebuttal/llm_sr_programs.py` and `llm_sr_program_worker.py`
(running a selected program), `src/autoformalism/rebuttal/llm_sr_upstream.py`
(reading a program without running it), `scripts/phase_c_llm_sr.py`,
`scripts/jetstream/run_phase_c_hosted.sh`.

## The data: PySR's regression table, in time order

LLM-SR fits a function to derivative labels, as PySR does. So it takes the data
as our PySR baseline does:

- the same training rows, from every training trajectory;
- the same channels at each row, in PySR's order (targets first, then
  auxiliaries, external inputs and fixed covariates);
- the same labels: each target's derivative, estimated with `numpy.gradient`
  within each trajectory, as for PySR and SINDy.

The rows stay as the table is built: each trajectory in time order, one after
another. LLM-SR's programs see whole columns, so they can read along them, and
they do. In the budget pilot (T1 easy, time order, 3,532 scored samples), about
85% of programs and all of the 100 best did: they filtered the meal series to
stand in for the unobserved gut compartments. gpt-oss-120b wrote these filters
itself. LLM-SR's prompt holds no data or scores, and the model took the time
order from the task description.

Decided on 2026-10-09: such programs are allowed as long as they respect
causality, and the rows are not shuffled. The next section says how a pick that
reads the history is run. (Smoke v4 shuffled the rows instead; it was frozen
and never ran.)

The limits are the method's, as for PySR: the derivatives are estimated from
the observations, and only observed channels enter the function.

## The search

- LLM-SR learns one function per run, so every target of a cell is its own search. The eight roster cells have eleven targets: T2 easy has three, T2 hard two, the others one.
- In each search their islands evolve equation programs. Each prompt asks for four samples. Their evaluator fits each program's ten parameters to the labels by BFGS and scores it by negative mean squared error, within 30 seconds.
- **Selection is LLM-SR's own rule:** the sample its evaluator scored highest. Then the cell's system is rolled out on validation data. The rollout error is reported and chooses nothing.
- **The test score is the shared rollout NMSE, the same for every method.** LLM-SR's own score, the derivative error row by row, is not used in the comparison.

## Recovering the selected program

Their evaluator discards the coefficients it fits, so the pick is recovered from
its program text. With the plan's `program_rollout`, as in smoke v5 and the
full plan, each target's pick is recovered by running it:

1. **Allowlist.** The program's syntax tree is checked before anything runs.
   It may use arithmetic, control flow, helper functions and lambdas, and
   listed numpy, math and scipy names. Attribute names are listed rather than
   refused, so no path through an allowed object reaches a frame, a module's
   globals or the interpreter. Imports are limited to numpy, math, scipy and
   five of its modules (integrate, interpolate, ndimage, signal, special).
   Names with a double underscore, classes, `yield`, `eval`,
   `getattr` and `open` are refused. Anything outside the list is refused with
   a named reason. Of the pilot's 3,533 scored programs, 3,532 pass, including
   all of the 100 best; the one refused reads `Gp._dt`.
2. **A separate process.** The program runs in `llm_sr_program_worker.py`:
   - It can import only numpy, math and those scipy modules; numpy and scipy may still load their own internals.
   - Its namespace holds what LLM-SR's specification module defines (`np`, `MAX_NPARAMS`, `params`) and a short list of builtins.
   - It cannot write a file, and on Linux its memory is limited to 4 GB.
   - It is killed past its time limit.
   - Requests and replies are plain JSON, so neither side reads anything that can run code.
3. **Their fit, by their call.** The coefficients are refitted exactly as their evaluator fits them: BFGS from all ones over all ten, on the same table, with the program receiving the columns of one input array. On the same machine this reproduces their fit bit for bit: in the local smoke, the refit loss equalled LLM-SR's score to every digit. On a different machine, the pilot's best program matched to seven significant digits.
4. **Equation or program.** The reader then reads the fitted program without running it.
   - If, at its fitted coefficients, the program is one equation of the current values, and that equation reproduces its outputs on the training rows, the equation goes to the shared evaluator, as PySR's do.
   - Otherwise the pick is run as the program it is.
   - In a cell where any target is run as a program, every target is, so the cell is rolled out as one system.
5. **Rollout along the grid.** Each trajectory is stepped along its own observation grid with the trapezoid (Heun) rule.
   - At each step the programs see only that trajectory's rows so far: the simulated targets and the supplied channels.
   - The output for the last row is the derivative there. A predictor step adds the next row, the derivative is taken again on it, and the two are averaged.
   - A rollout cannot see later rows or another trajectory, whatever the program does. Observed targets after the first row are never sent to the process.
   - The error is the shared evaluator's: per trajectory, the mean over targets of the mean squared error scaled by the training standard deviation, averaged over trajectories.
   - Every roster cell's grid is uniform and the same in training and validation: 1 for Dalla Man, 0.1 for the CSTR and the alien device, 4 for detention. A split on another grid is refused.
6. **Look-ahead, recorded.** The program's output on the training table is compared with its output when later rows are removed, cutting mid-way through each trajectory and after it. The largest change is recorded, and whether it exceeds rounding. It chooses nothing. How a report treats such a pick is decided before the test data open.

Why the coefficients come from running the program, not from reading it: in the
pilot, filters were often guarded by a branch on a coefficient, such as
`if alpha >= 1: gut = meal.copy()` with the filter on the other side. A reader
that cannot follow the filter refits only on the branch it can read, and settles
on a different and worse model than the one LLM-SR scored.

Checks on the pilot's data (T1 easy, validation):
- For the eight best pilot picks that are equations of the current values, the grid rollout and the shared evaluator's adaptive solver agreed within 2×10⁻⁵ to 3×10⁻³ relative.
- The pilot's best program, a two-compartment filter of the meal series with a delay, rolled out with error 0.0017.

A plan without `program_rollout`, such as the budget pilot and smokes v3 and v4,
keeps the earlier recovery: the reader refits each pick without running it, and
a pick that reads the history is refused as inexpressible.

### The exception to the no-exec rule

AGENTS.md forbids executing proposer-written text. Running LLM-SR's selected
programs is a declared exception for this baseline, agreed on 2026-10-09:
- LLM-SR's own search executes every program it scores, so the selected program has run there already.
- The exception covers only the programs LLM-SR selected, only after the allowlist, and only in the worker process.
- The repository's scan for dynamic execution (`tests/test_ode_compiler.py`) names the worker as its one exception and allows it exactly one call. Anything else in the package that executes text still fails the scan.

## Settings

| Setting | Paper | Released code | Phase C plan |
|---|---|---|---|
| Model | Mixtral-8x7B, GPT-3.5 | their local engine | gpt-oss-120b, hosted |
| Data | tables of variables and labels | the problem's CSV | PySR's table, each trajectory's rows in time order |
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
| Held-out scoring | NMSE of the label on held-out rows | none released | the shared rollout NMSE on test trajectories |

## Where it departs from the released code

The sealed plan records each, and a report states them.

1. **Reasoning-model adaptation.** gpt-oss reasons before it answers, within the same token limit, and splits its function header over several lines. At upstream's 512 tokens every sample stopped while still reasoning, and upstream's reader kept nothing of a split header. Measured on 2026-10-05; see the plan's rationale.
2. **Transport.** Their engine applies a chat template to the prompt as one user turn; we send it as one chat message. The hosted gateway replays stored answers, so every request asks for a fresh one, as their engine always generates one.
3. **Specification.** LLM-SR's specifications embed a problem description, so the public task specification is supplied, as its own protocol does. The starting program names its result after the target's derivative (`dI` for target I), as upstream's `dv` does.
4. **Data.** PySR's regression table, in time order (above).
5. **The paper's settings.** Temperature 0.8, cluster-sampling period 10,000 and 4 evaluators, where the released code differs. Each is one of their own configuration values; the search code is unchanged.
6. **Program rollout.** A pick that reads the history is run as a program and rolled out along the grid (above). Their code has no held-out scoring; the paper reports a row-by-row error on held-out rows, which no other method in the comparison is scored by.

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

Recovering a program costs seconds per target. On a laptop:
- Refitting the pilot's best program took 6 s.
- Rolling it out took about 1 s on validation and 2 s on training data. The trajectories have 61 to 601 rows.

## Plans

- `configs/phase_c_llm_sr_smoke_v5.json`: the full plan's settings on the budget pilot's cell (T1 easy) and the two-target T2 hard cell, 400 samples per target, at most 300 requests. It shows that a pick that reads the history is recovered by running it, rolled out on validation and sealed. Frozen; not a result.
- `configs/phase_c_llm_sr_smoke_v4.json`: the same with the rows shuffled. Frozen on 2026-10-09 and superseded the same day; it never ran.
- `configs/phase_c_llm_sr_smoke_v3.json`: the T2 hard cell in time order, 8 samples per target, without a program rollout. Frozen; ran on 2026-10-09 and sealed "inexpressible": both picks read along the rows.
- `configs/phase_c_llm_sr_hosted_120b_v1.json`: 8 roster cells × 2 seeds at 10,000 samples per target, with PySR's table in time order, the program rollout and the paper's settings. **Proposed, pending review.**
- The budget pilot (`configs/phase_c_llm_sr_budget_pilot_v2.json`) runs in time order from its own checkout. Only its pace is used.

A run reads its checkout's code until it ends, so run from a clean checkout that no other run is using:

```bash
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_smoke_v5.json phase-c-llm-sr-smoke-v5 3
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_hosted_120b_v1.json phase-c-llm-sr-hosted-120b-v1 22
```

The last argument is how many target searches run at once. On the GPU VM, `run_phase_c_tasks.sh` still searches a task's targets one after another.

## Limitations

- **Test scoring is not wired in yet.** The shared frozen evaluation cannot take a program pick: its adapter refuses one and names `llm_sr_programs`. The test-scoring step must call the program rollout, with the same metric and a check that the test grid matches the training grid.
- **The search sees a different history from the rollout.** In the search a program reads the whole training table, its trajectories end to end, so a filter runs on from one trajectory into the next. In a rollout each trajectory starts afresh.
- **Short histories.** A program that cannot run on the first few rows of a trajectory fails its rollout there, with the row and the reason recorded.
- **Coarse steps.** Rollouts step at the data's own interval, which is coarse for detention (4 units). The two rollouts were compared only on the Dalla Man grid.
