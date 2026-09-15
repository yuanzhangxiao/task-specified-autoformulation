# Saved-checkpoint execution recovery

This is the authorized execution repair of `fitter-scaled-alternating-v1`, not
another collocation comparison or a change to production fitting defaults.
Both source screening workers timed out before creating their evaluation
directories; both refinements performed zero calls. Joint retained its original
point; alternating retained an improved surrogate point. Neither establishes
physical output recovery.

Original files and charged stage budgets remain untouched. The new campaign
records the explicit recovery authorization, original freeze, stage outcomes,
common units, and initializer metadata/arrays. It refuses sources with an
existing screening calls directory, screened best point, or nonzero refinement
calls. Existing recovery intents consume their stage budget on resume; running
the same command cannot silently buy another attempt.

## Execution contract

1. A CPU preparation job binds the original commit to its source hash and
   confirms unchanged fitting, expression, data, schema and relevant campaign
   modules. It copies Python, its standard/shared libraries, and the required
   installed packages to local storage. Exact source Python/package versions and
   local numerical import origins must match. `-S` prevents editable `.pth`
   files from redirecting imports to a shared checkout.
   Bootstrap commands use the configured Python with site loading disabled;
   they do not assume a particular `/usr/bin/python3` version.
   Dependency discovery reads installed metadata without importing the numerical
   stack. It selects the mandatory dependency closure of the project's runtime
   imports and CasADi, honoring extras-directory precedence and Python/platform
   markers. Optional GPU, plotting and development extras are not selected.
   Only the selected distributions' recorded library/data/metadata files are
   copied, including native support libraries. Console scripts outside package
   roots, caches and `.pth` redirects are excluded. Missing inventories or
   incompatible installed dependencies stop preparation rather than silently
   falling back to copying the entire environment. Discovery uses installed
   `packaging`, or pip's bundled parser; it installs nothing.
2. An independent small synthetic smoke exercises screening, sensitivity,
   replay and deterministic resume. Successful preparation publishes one
   hash-checked runtime/code/input tar bundle. No benchmark point is optimized
   during preparation.
3. Two CPU workers unpack the bundle on node-local storage and verify inputs
   before numerical budgets start. Ordinary-start screening runs first in its
   own process. Checkpoint screening independently validates its metadata and
   reads only its parameter array `q`. A bad checkpoint cannot block the
   ordinary point. An identical point is `duplicate_ordinary`, without another
   rollout.
4. The lowest finite full-training rollout cost selects the sensitivity start.
   Refinement retains the original equations, parameter domains, fixed hidden
   boundaries, units, sparse sensitivities, ODE tolerances, and stopping rules
   (`ftol=None`, `xtol=gtol=1e-8`). Its failure cannot erase a better screened
   point.
5. Independent tight Radau/BDF replays score training and validation output
   `v01`. Validation never selects parameters. Replay agreement is numerical
   consistency, not accuracy. Strict/good/practical still require both NMSEs
   at most `1e-4` / `0.01` / `0.1`, plus replay agreement.

There are no new collocation nodes, initializer iterations, random starts,
models, test reads, latent-trajectory labels, LLM calls or ACES jobs. Known
hidden initial boundaries remain isolated from proposer/judge feedback.

Each screened point retains its original 60-second numerical cap, at most two
per arm, within the original 240-second screening ceiling. Refinement retains
118 evaluations, a 180-second cooperative per-evaluation cap and a 3,360-second
hard stage cap. Each replay retains 240 numerical seconds. Worker startup and
model construction have a separate 180-second hard cap. These explicitly
authorized recovery timings are not a controlled speed comparison with the
interrupted experiment.

Flushed markers precede imports, model construction, checkpoint loading, and
rollouts. A ready handshake starts the numerical clock. The parent records
startup, numerical work, worker exit and publication separately. If no valid
point exists, it skips refinement and replay before launching those workers.
Startup failure/timeout, numerical timeout, no feasible evaluation and replay
disagreement remain distinct.

Local results and best evaluations are atomically published periodically by a
separate supervised process. A stage's budget claim must reach durable storage
before its worker starts. Publication is capped at 60 seconds; a failure stops
subsequent stages without blocking numerical deadline supervision. SIGKILL has
a bounded follow-up wait. An uninterruptible process stops the task.

## Delta commands

Use a clean checkout pinned to this implementation commit. Retain the original
commit in its Git object database for the numerical-source audit.

```bash
bash scripts/hpc/submit_scaled_recovery_delta.sh
```

Defaults:

- Source: `/work/hdd/bibo/yxiao2/phase_b/fitter-scaled-alternating-v1`.
- New output: `/work/hdd/bibo/yxiao2/phase_b/fitter-scaled-recovery-v2`.
- Python: `/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`.
- CasADi: `/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps`.

The launcher submits preparation (one CPU, 8 GB, 45 minutes), two dependent
fit tasks (one CPU and 16 GB each, 2.5 hours including staging), and a summary.
Submission is recorded incrementally and never duplicated automatically.
Failed preparation blocks both fit workers.

The v1 preparation job `22089715` timed out while copying the entire venv
`site-packages` directory. It published no freeze and started no fitting. Its
summary job `22089717` then raised a secondary missing-freeze error. V2 replaces
that broad copy with the dependency inventory above, retaining numerical code,
saved checkpoints and budgets. The failed v1 directory remains untouched.
An EXIT handler now attempts bounded publication of preparation progress and
errors even after an outer timeout. Summary can report blocked preparation
without requiring `freeze.json`, and does not label it a numerical fit failure.

After completion:

```bash
/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python -S \
  scripts/run_scaled_recovery.py summarize \
  --output /work/hdd/bibo/yxiao2/phase_b/fitter-scaled-recovery-v2
cat /work/hdd/bibo/yxiao2/phase_b/fitter-scaled-recovery-v2/SUMMARY.md
```

`preparation/` contains runtime-copy/import/smoke timing and source audits.
Its `dependencies.json` lists selected installed versions, file counts and
recorded byte sizes; `copy_extras.log` and `copy_site.log` retain file-copy
progress. `current.json` identifies the last preparation operation and
`result.json` records its overall outcome. These diagnostics are also printed
in the summary when preparation prevents fitting. If the filesystem prevents
even bounded diagnostic publication, use the Slurm stdout/stderr logs.
`results/task_000` is joint; `results/task_001` is alternating. Their
`supervisor-events.jsonl`, stage `events.jsonl`, `ready.json` and `result.json`
separate execution phases. Screening `calls/` directories contain physical
evaluations when those began. Original initializer records are in
`provenance/` and `points/`.

Limits: one already opened reference case does not establish general fitter
reliability. Shared storage must still work for staging and publication. A
crashed node can lose changes since the last successful publication; its
persisted intent prevents a hidden fresh budget. The copied runtime is audited
on the preparation node and still requires compatible Delta CPU nodes.
