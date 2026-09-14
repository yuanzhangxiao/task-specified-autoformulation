# Matched initial-construction pilot

Milestone 4 implements `prefit-matched-construction-1`. The configuration is
`configs/prefit_matched_construction_v1.json`. It has six matched pairs: two public
cells, three proposer seeds and two presentation arms, for twelve tasks total.

- `brief_only`: the finalized public scientific brief.
- `training_evidence`: that same brief plus the milestone-3 training packet.

Both arms use the causal-training initializer constructor, the same staged
variable/topology/function controller, equation-batch generation with bounded
atomic repair, GPT-OSS-20B revision and sampling settings, request/token budgets,
and modeling limits. The scientific judge is off in both arms. The evidence packet
does consume context and token budget; that cost is part of the treatment. Task
order alternates by cell and seed. No extra information is returned to the
proposer after fitting in this pilot.

The current sensitivity-transfer fitter accepts only target `v01`; the named
cell has multiple targets. This pilot therefore pins the existing general
bounded-rollout fitter in both arms, with one start, at most 240 optimizer
function evaluations and a 300-second cooperative numerical deadline per task.
The random seed is the frozen fit seed plus the proposer seed. Numerical settings
and dependency versions are recorded in the plan. No fitter source is changed.
The optimizer evaluation count is not a universal count of residual calls; the
underlying diagnostics are retained. A finite fit does not establish optimality,
and a poor or failed fit does not establish model infeasibility.

This is an initial-construction experiment: it performs zero post-fit repair
rounds. `best_validation_nmse` is explicitly the same sole candidate as the first
fit, not a best-of-search result. A repair-policy or fitter comparison requires
a separate freeze after examining these traces. This avoids confounding the
packet treatment with changes in routing, scientific review or numerical search.

## Frozen public boundary and execution

`freeze` copies exactly `manifest.json`, `proposer_prompt.txt`, `train.csv` and
`validation.csv` for each registered public cell. It builds the reviewed public
brief from the existing prompt/target/mechanism contracts. Their hashes must
match. The construction context comes from the public registry, so validation
observations do not contribute channel bounds or other proposal features. Only
the training split enters the packet. No test table or private system reference
is opened by the campaign.

The plan binds code, entry points, numerical dependency versions, config, seeds,
public assets and the exact packet. Freeze on the execution machine using the
same Python environment that will run the workers. Changed content requires a
new root. Source assets are copied into the frozen root, so later changes to the
original directory do not affect a verified run.

Each task has a nonblocking worker lock. Construction and fitting can drain
independently; `--arm` can select one presentation arm without waiting for the
other. The model client has one request/token budget across all construction
stages and process restarts. Cached requests are deduplicated. In-flight outcomes,
unknown token usage and reservations remain visible. Missing referenced request
records or changed checkpoint contents fail explicitly.

Successful or failed construction is terminal for that task. Fitting reconstructs
the canonical causal initializer against the frozen public context, fits its
coefficients on training, and evaluates held-out trajectories with the same map.
It never fits validation initials. The fit-start marker prevents an interrupted
optimizer from silently restarting with a new budget. Such an interruption is
reported as `interrupted`; a terminal fit is replayed exactly. The general fitter
uses its existing cooperative deadline; an allocation's hard wall limit remains
the outer process limit.

`summary` retains every task and both paired denominators. It includes construction
and fit status, errors, finite predictive metrics, failed trajectories, optimizer
success, topology identity/size, initialization modes, physical requests, observed
tokens, unknown usage, charged reservations and timing. Partial construction
costs are included. Paired differences are reported only when both validation
rollouts are finite and complete. No automatic scientific winner is defined.

`fit_success` describes a usable retained fit. `optimizer_success`, status and
message come from the selected start's native diagnostic. Reaching the evaluation
limit can produce a finite fit while optimizer success remains false. Reported
initial-overrides dictionaries are the fitter's explicit overrides; the canonical
initializer artifact supplies the expressions evaluated by the simulator.

## Local verification

From this commit's clean checkout and configured Python environment:

```bash
PYTHONPATH=src python -m pytest -q tests/test_prefit_construction_campaign.py tests/test_training_evidence.py tests/test_causal_initialization_construction.py tests/test_staged_topology_runner.py tests/test_staged_function_runner.py tests/test_staged_function_prefit_campaign.py
PYTHONPATH=src python scripts/smoke_prefit_construction.py
ruff check .
bash -n scripts/hpc/run_staged_topology_server.sh
bash scripts/hpc/run_staged_topology_server.sh --check-config configs/prefit_matched_construction_v1.json
```

The final focused suite passed 59 tests. The real CPU smoke constructs both arms from
prescribed provider responses on temporary synthetic observations, fits the
resulting models, and verifies exact terminal resume. Both validation NMSEs were
approximately `2.08e-16`; each arm used six cached synthetic provider responses.
This is an implementation control, not evidence of live-proposer improvement.

The full-suite run reported 1,474 passed, 39 failures with exactly the same
missing-fixture identities as milestone 3, and three optional skips. A final
reporting correction separates retained-fit success from selected-start optimizer
convergence; the subsequent 59-test focused run includes its additional regression.
Ruff, shell checks and the repeated real CPU smoke passed after that correction.

## Bounded public run

The ACES launcher `scripts/hpc/submit_prefit_construction_aces.sh` now freezes
the plan and queues CPU preparation (one hour), construction on one H100
(four-hour allocation, three-hour worker), and CPU fitting (two hours).
Preparation runs the relevant tests and real synthetic smoke, verifies the SIF,
and ensures the pinned 20B model is cached. Construction depends on successful
preparation. CPU fitting runs after construction terminates, including after an
interrupted GPU allocation, so completed models remain evaluable.

In an ACES shell, create a clean experiment checkout from the pushed branch:

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 fetch origin codex/prefit-aces-v1 && git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-aces-v1 FETCH_HEAD
AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-aces-v1 AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-aces-v1/scripts/hpc/submit_prefit_construction_aces.sh
cat /scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1/submission_manifest.json
squeue -u u.yx126462 -o '%.18i %.18j %.10T %.10M %.30R'
```

The launch defaults to the established project Python environment and the
public-only snapshot under
`/scratch/user/u.yx126462/phase_b/staged-fitter-rescue-v1-c1754fe/frozen/public`.
It validates the required files and exact public prompt hashes. Override
`AF_PUBLIC_ROOT` or `AF_PYTHON` explicitly if those installations have moved.
No current queue state was verified from the local agent: ACES SSH authentication
was unavailable. These commands are intended for the user's authenticated shell.

After the workers finish, inspect the paired metrics and all failures:

```bash
module load GCCcore/13.2.0 Python/3.11.5 && PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-aces-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-aces-v1/scripts/prefit_construction_campaign.py summary --root /scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1
```

Repeating the ordinary launch command prints the existing submission manifest.
If the summary is partial, resume only after the prior jobs have terminated:

```bash
AF_RESUME=1 AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-aces-v1 AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-aces-v1/scripts/hpc/submit_prefit_construction_aces.sh
```

Resume checks scheduler/accounting state and retains completed work and spent
budgets. It submits only CPU fitting if all construction is already terminal.
It submits nothing when all fits are terminal. A failed or ambiguous submission
leaves an intent directory and returned job IDs; inspect those records and the
queue before resolving it. Failed preparation can leave a GPU dependency queued;
inspect that failure and its dependencies before resubmitting. Do not remove an
intent simply to force another submission.

The local project checkout now contains these changes on `codex/prefit-aces-v1`.
Existing unrelated local edits were preserved. Future implementation edits use
the project checkout; the clean ACES checkout above is a pinned experiment copy.

Set `AF_REPO_ROOT` to a clean checkout of this milestone, `AF_PYTHON` to the
project Python environment, `AF_PUBLIC_ROOT` to the existing public root containing
`phase_b_v1/<cell>/`, and `AF_OUTPUT_ROOT` to a new experiment directory. Do not
point the public root at a private reference directory. Every command below is
one physical shell line.

```bash
cd "$AF_REPO_ROOT" && PYTHONPATH=src "$AF_PYTHON" scripts/prefit_construction_campaign.py freeze --config configs/prefit_matched_construction_v1.json --public-root "$AF_PUBLIC_ROOT" --output "$AF_OUTPUT_ROOT"
cd "$AF_REPO_ROOT" && PYTHONPATH=src "$AF_PYTHON" scripts/prefit_construction_campaign.py verify --root "$AF_OUTPUT_ROOT"
cd "$AF_REPO_ROOT" && PYTHONPATH=src "$AF_PYTHON" scripts/prefit_construction_campaign.py run --root "$AF_OUTPUT_ROOT" --base-url http://127.0.0.1:8000 --wall-seconds 10800
cd "$AF_REPO_ROOT" && PYTHONPATH=src "$AF_PYTHON" scripts/prefit_construction_campaign.py fit --root "$AF_OUTPUT_ROOT" --wall-seconds 7200
cd "$AF_REPO_ROOT" && PYTHONPATH=src "$AF_PYTHON" scripts/prefit_construction_campaign.py summary --root "$AF_OUTPUT_ROOT"
```

The `run` command assumes the pinned model is already served at the displayed
local endpoint; substitute its actual port. Alternatively, the existing
`scripts/hpc/run_staged_topology_server.sh` now dispatches this protocol and
starts its own server on an allocated H100. Supply its existing required
`AF_VLLM_IMAGE`, `AF_HF_HOME`, `AF_COMPUTE_CACHE_ROOT` and `AF_IPC_TMP_ROOT`
environment variables. Use a GPU allocation long enough for the configured
three-hour worker plus startup; do not reuse the short debug allocation unchanged.
The frozen image is the already verified v0.27.1 SIF with SHA-256
`9c56389af06cafbf4aa8bd825393f606ac7f2a18e1a700c5999b1ed47f7f9c1e`.
The shared launcher verifies that image and the frozen model revision. Fitting
can subsequently run on CPU without holding the model server allocation.

A worker can return a partial summary after deferring at its deadline. Resume
the same command/root to process remaining work; inspect the summary before
starting a new allocation. No live campaign was submitted as part of implementation.

## Remaining limits

The live twelve-task matrix has now run; the reporting recovery below records
its execution status. Synthetic tests do not establish which
presentation is better, calibrate the evidence-priority policy, or establish
benchmark recovery. The descriptive packet omits some samples/events and has no
measurement-noise model. Deterministic initial maps cannot recover unobserved
preparation differences that are absent from all allowed initial features.
The pinned general fitter may be too weak for difficult candidates. The named
cell's auxiliary-channel availability remains governed by its public task
contract; this pilot does not redefine that contract.

## Reporting recovery for the e7ffd12 ACES run

The first live run used commit `e7ffd121ac6f27cd55eb7dc8e65c97bafc8ddcf3`
and output root
`/scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1-e7ffd12`.
Preparation job `2126622` completed. Construction job `2126623` and fit job
`2126624` both exited with status 1 in the final summary: a failed function
construction correctly saved `initialization: null`, but the summary attempted
to read it as a dictionary. Both worker tracebacks identify that same line.
The saved records account for all twelve tasks: eleven completed constructions,
one terminal construction failure, eleven evaluated fits and one `not_run`
because construction failed. Recovering the report needs no new model calls or
fitting jobs. `evaluated` alone does not imply a successful numerical fit.

The summary now treats absent initialization as an empty mode inventory and
retains the failure and its costs in the paired denominator. Regression tests
exercise budget exhaustion during both function and initializer construction.

Use the new `report` command to inspect a previously frozen run with corrected
reporting code. It verifies the sealed plan and results, task identities, public
asset hashes, evidence packet and provider-request provenance, including missing
cached requests. It writes JSON only to stdout and does not alter `summary.json`
or any experiment artifact. The `reporting` block records both the frozen
execution identity and current reporting identity. A different reporting runtime
does not authorize resuming the run: `verify`, `run`, `fit` and `summary` retain
the exact frozen execution checks.

Create a separate reporting checkout; leave the original experiment checkout and
plan intact. Each command below is one physical shell line:

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 fetch origin codex/prefit-aces-v1 && git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-report-v1 FETCH_HEAD
module load GCCcore/13.2.0 Python/3.11.5 && PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-report-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-report-v1/scripts/prefit_construction_campaign.py report --root /scratch/user/u.yx126462/phase_b/prefit-training-evidence-v1-e7ffd12
```

Reporting regression and ACES submission checks can run without a GPU:

```bash
PYTHONPATH=src python -m pytest -q tests/test_prefit_construction_campaign.py tests/test_prefit_aces_submission.py
PYTHONPATH=src python scripts/smoke_prefit_construction.py
ruff check .
```
