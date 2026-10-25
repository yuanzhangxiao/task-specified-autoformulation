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
6. **Look-ahead, recorded.** The program's output on the training table is compared with its output when later rows are removed, cutting mid-way through each trajectory and after it. The largest change is recorded, and whether it exceeds rounding. It chooses nothing.
   - Decided by the user on 2026-10-15: a pick this check flags is reported as a failure of LLM-SR, because a model cannot use future rows. It counts as a failure whatever its rollout error, and nothing is rerun or reselected.
   - The first such pick was task 5 of the full run (T2 easy, seed 1). Its dI/dt program begins with `np.gradient(I, edge_order=2)`, a central difference that reads the next row. Removing later rows changed its output by up to 93%. Its validation rollout also failed at the first row, where `np.gradient` needs three rows and the rollout has one, so the task sealed as `rollout_failed`.

Why the coefficients come from running the program, not from reading it: in the
pilot, filters were often guarded by a branch on a coefficient, such as
`if alpha >= 1: gut = meal.copy()` with the filter on the other side. A reader
that cannot follow the filter refits only on the branch it can read, and settles
on a different and worse model than the one LLM-SR scored.

Checks on the pilot's data (T1 easy, validation):
- For the eight best pilot picks that are equations of the current values, the grid rollout and the shared evaluator's adaptive solver agreed within 2×10⁻⁵ to 3×10⁻³ relative.
- The pilot's best program, a two-compartment filter of the meal series with a delay, rolled out with error 0.0017.

### Solver check on the full run (2026-10-15)

In the full run, none of the first 12 sealed tasks' picks became an equation.
In 4 tasks, the picks use only the current row, but the reader refused how
they were written: a parameter vector padded with `np.pad`, NaN guards,
`np.broadcast_arrays`, comparisons in `np.where`. In substance they are ODEs.
Like the picks that read the history, they were stepped along the data grid
rather than given the shared evaluator's adaptive solver.

`scripts/phase_c_llm_sr_solver_check.py` checks this on validation and changes
no sealed result.
- It first requires each program to give every training row the same output
  alone as with the whole table.
- It then integrates the picks with the shared evaluator's solver and settings
  (`FitConfig`: RK45, rtol 1e-7, atol 1e-9, restarts where an input's slope
  changes), in the worker's `solve` mode.
- The rollout error is the same one the grid rollout reports.

Run on the Mac against the frozen release (receipt `200deb42`), the check gave:
- The grid rollout recomputed here equalled every sealed value exactly.
- The four picks gave each row exactly the same output alone. The other seven
  complete tasks' picks did not (1.2 to 12.8 times their largest output), and
  neither did task 5's.

| Task | Sealed grid rollout | Adaptive solver |
|---|---|---|
| 8, CSTR, seed 0 | 1.27e21 | 1.32e21 |
| 9, CSTR, seed 1 | 0.620 | 2.87e20 |
| 14, detention independent, seed 0 | 1.33e-4 | 9.22e-5 |
| 15, detention independent, seed 1 | 2.75e-4 | 2.69e-4 |

CSTR seed 0's model diverges either way. CSTR seed 1's does too when solved
accurately: steps of 0.1 understate its fast growth.

**Test rollout, decided on 2026-10-22 (superseded twice below).** The user asked for the rollout closest
to the LLM-SR paper. The paper never simulates: its model is the program,
evaluated on the data's own rows. The rollout that changes that least runs the
program itself on the observation grid, feeding back its own predictions with
the trapezoid rule. That is the rollout behind the sealed validation numbers,
and the only one a history-reading pick has. So on test every pick is rolled out
on the grid, the four current-value picks included. For those four, the
adaptive solver's error is reported beside it as a sensitivity check, never as
the score; on validation the two differ only for CSTR seed 1 (0.62 against
2.9e20).

**Revised by the user the same day: results always come from an accurate
solver.** The CSTR seed 1 case showed why: on one validation trajectory its
model sits 3.7 K below an ignition point (it reads the reactant concentration
from the data, so nothing stops a runaway); solved accurately it crosses the
point and runs away, while steps of 0.1 stay just below it. Halving the step
already ignites it. So the four current-value picks (CSTR, detention
independent) are rolled out with the shared adaptive solver, as PySR's are.
A program that reads the history is defined only on the grid, its filters
advancing one row at a time, so it has no more accurate rollout of its own.
A check on 2026-10-22 found the solver matters nowhere else: re-solving
every SINDy, PySR and LLM-ODE pick on validation with Radau at rtol 1e-10
changed no NMSE by more than 0.1%.

**Decided by the user the same day: a pick that reads the history is scored by
its ODE transcription,** solved with the shared evaluator's solver, as the most
accurate evaluation. The program's output is a derivative, so stepping it once
per row is itself a coarse integration; the transcription gives the program's
filters a continuous reading under fixed rules, so that the whole model can be
solved accurately. Every LLM-SR pick is now scored as an ODE: the four
current-value picks by their exact rewrite, the other eleven by their reviewed
transcription (below). The grid rollouts stay in the sealed record. They chose
nothing, and nothing is reselected.

What follows from it:
- The transcriptions now set LLM-SR's score, not only its mechanism tests, so
  they must be reviewed and frozen before its test scoring.
- On validation every transcription scores worse than its program's grid
  rollout: 1.1 to 2.0 times the grid's error on ten picks, and 5.6 times on
  T2 hard seed 0 (0.49 against 0.087). The two rollouts differ from each other
  by 0.00023 to 0.054 of the training variance on nine picks, and by 0.33 and
  0.73 on T2 hard seed 0 and alien seed 0, whose programs use a discrete-time
  feedback device. Two things differ: the transcription keeps each filter's
  gain and mean delay but not its exact response, and it integrates the
  derivative accurately where the grid steps it once per row. Which of the two
  accounts for the gap was not measured.
- Two transcriptions fail on one training trajectory each, at a pole of
  LLM-SR's own fitted model that the grid steps over.
  - Task 3 (T1 hard, seed 1) fitted `Km_ins = -127` in its insulin term
    `-insulin_signal * Gp / (Km_ins + |Gp| + eps)`. On train_005 glucose falls
    to Gp = 127. The program's own clip then sets the rate to -1e6 just above
    that point and +1e6 just below it, so no adaptive solver can pass.
  - Task 4 (T2 easy, seed 0) runs away on train_005, as its program does on
    training (error 3.5e38), down to Gp = -1040: the pole of its
    `k_ins * X * Gp / (Km + Gp)`.
  - Neither fails on validation. On test, a trajectory that reaches such a
    pole fails, and under the general rule the failure is LLM-SR's.
- On validation, the eleven transcriptions solved with the shared evaluator's
  RK45 (rtol 1e-7, atol 1e-9) and with Radau at rtol 1e-10 agree within
  1.2e-5 relative, except alien seed 0, which RK45 cannot finish.
- Alien seed 0's transcription is stiff, through the transcription rule rather
  than LLM-SR's model. The program's command filter is an exponential moving
  average that keeps almost no memory (weight `exp(-1/tau_mem)` = 4.7e-5 per
  row). The filter rule makes it a hidden state with time constant 4.7e-6, so
  the ODE has an eigenvalue near -2.1e5. Radau solves each validation
  trajectory in under 2 s, with the same error at rtol 1e-7 and 1e-10 (2.474).
  RK45 reaches t = 6 of 60 in 120 s, so it times out at its 300 s limit. The
  user decided on 2026-10-23 that, for every method, a trajectory RK45 fails
  numerically is integrated again with Radau at the same tolerances
  ([PHASE_C_BASELINE_MECHANISMS.md](PHASE_C_BASELINE_MECHANISMS.md),
  decisions for test scoring, item 6), so this transcription is scored.

A plan without `program_rollout`, such as the budget pilot and smokes v3 and v4,
keeps the earlier recovery: the reader refits each pick without running it, and
a pick that reads the history is refused as inexpressible.

### ODE transcriptions

The public mechanism tests probe an ODE's states, and these picks are programs.
`configs/phase_c_llm_sr_transcriptions_v1.json` transcribes each sealed pick
by hand into an ODE whose hidden states are its row recursions, under fixed
rules, bound to the exact program text and coefficients. They were made for the
mechanism tests. Since the decision above, they also set the score of the
eleven picks that read the history. The rules and each transcription's
fidelity are in [PHASE_C_BASELINE_MECHANISMS.md](PHASE_C_BASELINE_MECHANISMS.md).

### The exception to the no-exec rule

AGENTS.md forbids executing proposer-written text. Running LLM-SR's selected
programs is a declared exception for this baseline, agreed on 2026-10-09:
- LLM-SR's own search executes every program it scores, so the selected program has run there already.
- The exception covers only the programs LLM-SR selected, only after the allowlist, and only in the worker process.
- The repository's scan for dynamic execution (`tests/test_ode_compiler.py`) names the worker as its one exception and allows it exactly one call. Anything else in the package that executes text still fails the scan.

## The "w/o Program" form: one expression per hypothesis

Asked for by the user on 2026-10-22, to run beside the program form as its own
labelled row, `llm_sr_expression`. The paper's ablation has this variant: "The
'w/o Program' variant, which restricts LLM hypothesis generation to
single-line mathematical expressions" (Sec. 3.5, Fig. 6, on Oscillation 2 with
GPT-3.5). Neither the paper nor the released code says how it was built, and
the code's `specs/` holds only numpy and torch program specifications. So this
is a reconstruction, declared in the plan's `hypothesis_form` block, that
changes LLM-SR in three places only:

1. **The starting function returns one expression** (`return params[0] * Gp + …`)
   instead of assigning and then returning it.
2. **Its docstring asks for one.** The evolved function's docstring, which
   upstream shows on the first function of every prompt, gains: "Write the body
   as one return statement: a single mathematical expression of the arguments
   and params, with no other statements."
3. **Only such samples are evaluated.** A subclass of their sandbox
   (`llm_sr_driver.single_expression_sandbox`) answers any other sample with
   `(None, False)`, the answer their evaluator gets from a program that
   raises, so it is never registered. "Such" means: past its docstring, the
   body is one return statement, and our non-executing reader turns its
   expression into an equation of the current values, so no loop, branch,
   helper, row access, comparison or reduction. Nothing is executed to decide
   it. Refusals are counted by reason in each search's record.

Everything else is the frozen main plan's: PySR's table in time order, the
prompts, islands, the paper's temperature and cluster period, their BFGS fit,
10,000 samples per target, and LLM-SR's own selection. Every pick is an
equation of the current values, so it is rolled out by the shared solver, as
PySR's are, and the mechanism tests read it as written, with no transcription.

Plans:
- `configs/phase_c_llm_sr_expression_smoke_v1.json`: smoke v5's cells (T1 easy;
  T2 hard, two targets) at 400 samples per target. Frozen. It shows whether
  gpt-oss writes one-line expressions often enough, and what share the form
  refuses. Ran on js2 on 2026-10-23, 00:38 to 01:07Z, from checkout `38a0dba`
  (run `phase-c-llm-sr-expression-smoke-v1`):
  - The searches worked: 1,200 samples in 300 requests, no transport failures,
    25 to 29 minutes per search.
  - The form refused 3.5% of T1 easy's Gp samples (11 read along the rows, 3
    had several statements), 30% of T2 hard's Gp samples (92 and 28) and 6.8%
    of its I samples (22 and 5). For T2 hard's glucose the model kept reaching
    for the meal history, mostly with one-line convolutions.
  - Task 0 (T1 easy) sealed complete. Its pick is an equation, its refit equals
    LLM-SR's score exactly, and its validation rollout error is 108, against
    0.0012 for the program form's pick in the main run: one expression of the
    current values cannot hold the gut's delay.
  - Task 1 (T2 hard) did not seal, through a harness bug. Its system rolls out
    on validation (error 1.75, reproduced on the Mac), but its training rollout
    fails: on train_011 glucose reaches a pole of the Gp equation at Gp = 151.
    The equation path then passed the missing training error to the sealed
    record, which requires a number, and crashed. The program path instead
    sealed such a task as `rollout_failed`, a rule that was never decided and
    never applied: no task of the main run failed its training rollout.
  - **Decided by the user on 2026-10-23:** a task whose training rollout fails
    seals complete, since the training rollout neither selects nor scores; the
    test rollout decides, as for every method. Both paths now do this: the
    result names the failing training trajectories (`training_rollout_failure`),
    the sealed training error is +infinity, and the report counts such tasks
    (`training_rollouts_failed`). A failed validation rollout still seals as
    `rollout_failed`. On the Mac, with the fix, task 1 seals complete, naming
    train_011, with validation error 1.751. A plan refuses to resume under
    changed code, so the smoke reruns from a fresh root.
  - **Rerun on js2 on 2026-10-25, 08:05 to 08:38Z, from checkout `ca45579`
    (run `phase-c-llm-sr-expression-smoke-v1-rerun`), and passed:** both tasks
    sealed complete and the report was written. 1,200 samples in 300
    requests, no transport failures. The searches drew new samples, so the
    picks differ from the first run's. The form refused 5 of T1 easy's 400
    samples and 126 of T2 hard's 800 (102 read along the rows, 24 had several
    statements). Task 0's pick, chosen as always by its training fit, runs
    away on validation (error 1.8e21). Task 1's pick scores 0.961 on
    validation. Its system fails on two training trajectories, which the
    result names (train_011, train_015); its sealed training error is
    +infinity, and the report counts one such task. The searches ran at 750
    to 910 samples an hour, two to three times smoke v5's pace.
- `configs/phase_c_llm_sr_expression_hosted_120b_v1.json`: the full roster, two
  seeds, 10,000 samples per target. Frozen on 2026-10-25 on the user's
  sign-off, after the smoke rerun passed; no setting changed. It has the main
  run's budget: 220,000 samples and about 55,000 requests. The main run's 22
  searches, all at once, took 1.5 to 4 days each (106 to 278 samples an hour);
  if the expression form keeps its smoke pace advantage, this run takes one to
  two days. It runs as `phase-c-llm-sr-expression-hosted-120b-v1`.

```bash
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_expression_smoke_v1.json phase-c-llm-sr-expression-smoke-v1-rerun 3
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_expression_hosted_120b_v1.json phase-c-llm-sr-expression-hosted-120b-v1 22
```

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
6. **Scored as an ODE.** Every pick is scored by solving an ODE with the shared evaluator's solver: the exact rewrite of a pick of the current values, or the reviewed transcription of one that reads the history (above). The program itself is run only to refit it, check it for look-ahead and record its grid rollout on validation. Their code has no held-out scoring; the paper reports a row-by-row error on held-out rows, which no other method in the comparison is scored by.

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
- The pilot finished on 2026-10-10: 10,000 samples in 54.5 hours, about 183 an hour overall.
- Smoke v5, three searches at once at the paper's temperature, ran at 260 to 390 samples an hour on 2026-10-10.
- A search uses about a third of a core on average. 22 at once use about 7–8 of js2's 16 cores and keep about 22 requests in flight at the hosted service. If the service slows under that load, the searches slow with it.
- The full set is 220,000 samples, about 55,000 requests.

Recovering a program costs seconds per target. On a laptop:
- Refitting the pilot's best program took 6 s.
- Rolling it out took about 1 s on validation and 2 s on training data. The trajectories have 61 to 601 rows.

## Plans

- `configs/phase_c_llm_sr_smoke_v5.json`: the full plan's settings on the budget pilot's cell (T1 easy) and the two-target T2 hard cell, 400 samples per target, at most 300 requests. It shows that a pick that reads the history is recovered by running it, rolled out on validation and sealed. Frozen; not a result. Ran on 2026-10-10 and passed: both tasks sealed complete with programs that read the history, rolled out on the grid; each refit loss equalled LLM-SR's score exactly, and no output changed when later rows were removed. 300 requests, no cache hits, no failed requests.
- `configs/phase_c_llm_sr_smoke_v4.json`: the same with the rows shuffled. Frozen on 2026-10-09 and superseded the same day; it never ran.
- `configs/phase_c_llm_sr_smoke_v3.json`: the T2 hard cell in time order, 8 samples per target, without a program rollout. Frozen; ran on 2026-10-09 and sealed "inexpressible": both picks read along the rows.
- `configs/phase_c_llm_sr_hosted_120b_v1.json`: 8 roster cells × 2 seeds at 10,000 samples per target, with PySR's table in time order, the program rollout and the paper's settings. Frozen on 2026-10-10, after smoke v5 passed and the user signed off; no setting changed, and the budget's rationale now gives the measured paces.
- The budget pilot (`configs/phase_c_llm_sr_budget_pilot_v2.json`) ran in time order from its own checkout and finished on 2026-10-10. Only its pace is used.

A run reads its checkout's code until it ends, so run from a clean checkout that no other run is using:

```bash
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_smoke_v5.json phase-c-llm-sr-smoke-v5 3
bash scripts/jetstream/run_phase_c_hosted.sh llm_sr configs/phase_c_llm_sr_hosted_120b_v1.json phase-c-llm-sr-hosted-120b-v1 22
```

The last argument is how many target searches run at once; the full run searches all 22 at once, then seals each task. When every task has ended, the launcher writes the report to `report.json` in the run's directory. On the GPU VM, `run_phase_c_tasks.sh` still searches a task's targets one after another.

## Limitations

- **Test scoring is not wired in yet.** The shared frozen evaluation cannot take a program pick: its adapter refuses one and names `llm_sr_programs`. The test-scoring step must solve each pick's ODE (exact rewrite or reviewed transcription) with the shared solver, reading it with the external-baseline grammar.
- **The score is a reading of the program.** For the eleven picks that read the history, the scored model is the program's transcription under fixed rules, not the program. The fidelity table states how close each one is.
- **The search sees a different history from the rollout.** In the search a program reads the whole training table, its trajectories end to end, so a filter, a lag or a simulated basin runs on from one trajectory into the next, while the derivative labels are taken within each trajectory. In a rollout each trajectory starts afresh. Measured on 2026-10-23 with `scripts/phase_c_llm_sr_boundary_check.py` (Mac, frozen release, sealed coefficients, training rows only), after the planning agent raised it in review. Given one trajectory at a time, each program's error against the labels (the error LLM-SR's evaluator minimizes) changes:
  - by under 1% for the three T1 picks that filter only the meal series (tasks 0 to 2), and not at all for the four current-value picks;
  - downward for six picks, where the rows just after a boundary inflated the error LLM-SR minimized: task 3 Gp 0.81 times, task 4 I 0.56, task 6 I 0.077, task 7 I 0.41, task 11 v01 0.10, task 12 h_down 0.40. In tasks 6 and 11 the rows the carried history changes (95 and 141, 2% and 1% of the table) hold at least 92% and 90% of that error;
  - upward for two, whose coefficients fit the carried history: task 10 v01 1.11 times, task 13 h_down 18 times.

  So eight of the eleven picks that read the history were fitted, and selected, on an error shaped by the trajectory boundaries, which no rollout has. Nothing marks a boundary for the program: their specification gives each variable as "a numpy array of observations", and the table has no time or trajectory column, though the task description says the data come as trajectories.

  Not rerun, decided by the user on 2026-10-23: how a method uses its training trajectories is its own, and the test runs each trajectory on its own, from its own start. The evaluator is LLM-SR's, run unmodified: one call of the program on the whole table, as in their specification, on the table PySR gets. Like fitting derivatives rather than rollouts, this is a property of the method that the test score includes.
- **Short histories.** A program that cannot run on the first few rows of a trajectory fails its rollout there, with the row and the reason recorded.
- **Coarse steps in the record.** The sealed validation rollouts step at the data's own interval, which is coarse for detention (4 units) and missed the CSTR seed 1 runaway. They stay as recorded and score nothing.
