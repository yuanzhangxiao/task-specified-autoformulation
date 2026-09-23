# Fresh Dalla Man shared-process / multi-output campaign

Protocol: `shared-multi-pruning-1`.

This is a prospective integration experiment. It starts six new Full lineages,
with seeds 0 and 1 in T1 canonical named hard, T2 canonical named easy, and T2
canonical named hard. It imports public development data, **no previous models,
parameters, rejected proposals or fit checkpoints**. Existing v8 runs remain intact.
There are 15 search visits, numbered 0–14: fresh construction, then 14 revisions.
After visit 14, each eligible retained model gets one bounded pruning experiment.

## What is combined

- Astra's general shared-process construction: an optional named law can have
  several consumers. Declared signs/conversions are assembled once. Empty or
  invalid optional process proposals fall back to ordinary construction within
  the same construction budget and variable inventory.
- Shared-law changes use the whole-model compiler. One edited process changes
  all consumers; before/after dependencies are recorded. Shared symbols denote
  the same parameter, without automatically certifying physical conservation.
- Public target requirements are visible before construction. Executable drafts
  that fail them can receive bounded repairs and continue into later visits.
- Joint-output fitting: T1-hard fits Gp; T2-easy fits Gp, I and U; T2-hard fits
  Gp and I. All use the existing `collocation-multi-target-v1` backend, including
  its single-output special case. Initialization is fitted on training only.
- Compact training-response feedback covers every target. Equations and public
  contracts remain intact; the serving tokenizer checks revision prompt length.
  Unverifiable evidence citations do not count as verified evidence. A bad
  citation alone need not discard an otherwise admissible equation patch.
- Existing parameter declarations are immutable. A conflicting redeclaration
  produces an explicit repair message. A failed revision retains the incumbent;
  an unsuccessful scientific revision can use that visit's unchanged-fit allowance.
  Delivery/preflight failures retain the incumbent without another fit and stop
  automatic dispatch for inspection.
- Search uses v8's validation tolerance (`1e-8 + 1e-6*max(abs(scores))`), then
  fewer terms, then the incumbent. This prevents numerical noise from rewarding
  extra terms. Public graph eligibility remains required.
- **The scientific critic is off.** No ground-truth insulin equation or new
  meal-to-insulin requirement is supplied. Any such dependence must be proposed
  from the public task and training evidence.

The configuration retains Astra's current general construction limits: 64
variables, 32 terms per equation, 512 total terms. These are safety ceilings,
not desired model sizes. This differs from the older small Dalla pilot; the
experiment is not an isolated old-versus-new controller comparison.

## Final pruning

The implementation reuses `process_pruning.choose`, `execute_row` and `decide`;
it does not introduce a different pruning algorithm. The parent is joined to its
saved fit receipt and sealed separately. Legal atomic removals are recompiled,
public requirements rechecked, and unreachable dependencies/initializers cleaned
up. Ranking uses only training contributions, with a single 300-second allowance.
One deletion is frozen before any child validation score is read.

There are up to two additional fits per retained parent: an unchanged refit and
the pruned candidate, with the same frozen numerical profile and compatible
retained parameters. Execution order is counterbalanced. A pruned candidate must
be Pareto smaller and have validation NMSE no worse than the best parent/control
by more than `max(1e-6, 0.01*baseline)`, exactly Astra's pruning policy. This 1%
pruning tolerance is distinct from the much tighter ordinary search tolerance.
The best parent/control remains available when pruning fails or worsens fit.

Unresolved public obligations prevent automatic deletion. Missing/failed final
fits are explicit skips, not successful pruning. The maximum allocation is 90
ordinary search fits plus 12 final comparison fits; actual consumed fits and
requests are reported. The proposer has the existing 128-request/524288-token
construction ceiling, at most three calls per ordinary revision, and an 8192-token
response cap per request. Reconstruction after a failed inventory uses a new
visit's construction ceiling; a saved draft instead receives up to three repairs.
Every physical request and numerical allowance is checkpointed.

## ACES: upload and submit

Use the source archive provided with the implementation. Upload it into group
scratch through the portal. The archive contains a `SOURCE_COMMIT` marker and
needs neither a GitHub credential nor a writable personal repository.

After extracting it, replace the source directory below with the extracted path:

```bash
export AF_REPO_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/repos/autoformalism-dalla-shared-COMMIT
export AF_COMMIT="$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")"
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-shared-multi-pruning-v1
export AF_PUBLIC_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-response-feedback-r14-v1/public
bash "$AF_REPO_ROOT/scripts/hpc/submit_fresh_shared_aces.sh" 0
```

The wrapper checks all public files before submission. It reuses the existing
Python environment, vLLM image and downloaded model cache. New source, experiment
outputs and compilation caches use group scratch; per-job IPC uses `/tmp`.
Each proposer job requests **one H100** for GPT-OSS-20B. Every fit and pruning job
uses one CPU. Only one proposer round is dispatched at a time. A successful finish
job dispatches the next round; missing results and delivery errors stop the chain.
After the final round, CPU pruning and a report job are submitted automatically.

Do not reuse a previous experiment's output root or change a frozen plan. A
submission manifest records scheduled jobs, not proof that all work completed.

## Inspect progress and results

```bash
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-shared-multi-pruning-v1
jq '{round,planned_rounds,all_rounds_submitted,jobs}' "$AF_OUTPUT_ROOT/submission_manifest.json"
squeue -u "$USER" -o '%.18i %.30j %.12T %.10M %R'
AF_IDS=$(jq -r '.jobs | [.[]] | unique | join(",")' "$AF_OUTPUT_ROOT/submission_manifest.json")
sacct -j "$AF_IDS" -X --format=JobID,JobName%32,State,ExitCode,Elapsed

module load GCCcore/13.2.0 Python/3.11.5
AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" "$AF_REPO_ROOT/scripts/fresh_shared.py" report --root "$AF_OUTPUT_ROOT"
jq '[.rows[] | {task:.task.task_id,status,pruning_choice,selected,search_parent_validation_nmse,final_validation_nmse}]' "$AF_OUTPUT_ROOT/pruning_summary.json"
```

`summary.json` retains all 90 original search visits and denominators.
`pruning_summary.json` adds final pruning decisions and complete final requests
and parameter vectors. Intermediate reports explicitly say `partial`; pending
future rounds are not failures. `pruning/<task>/` contains the parent binding,
choice, two fit receipts and decision. No test data are accessed automatically.

## Recovery after an ambiguous scheduler reply

Do not erase a submission intent or blindly resubmit the uncertain job. Read the
saved `submission-intent/round-N/STAGE-N.reply.json`, then inspect `squeue`/`sacct`.
If the job exists, use the exact same source, environment and round:

```bash
PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_fresh_shared.py" \
  --root "$AF_OUTPUT_ROOT" --round 0 --adopt prepare-0=CONFIRMED_JOB_ID
```

The command verifies scheduler ownership, job name, status and full submission
command against saved intent before adoption. It retains the raw timeout receipt,
reuses already confirmed jobs and only submits missing stages. Without a confirmed
matching job it stops. It does not cancel or duplicate old v8 jobs.

## Local / ACES verification

```bash
cd "$AF_REPO_ROOT"
PYTHONPATH=src "$AF_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_fresh_shared.py tests/test_fresh_shared_submission.py \
  tests/test_shared_process_integration.py tests/test_process_pruning_campaign.py \
  tests/test_review_multi.py tests/test_review_integrity.py tests/test_response_revision.py
PYTHONPATH=src "$AF_PYTHON" scripts/smoke_fresh_shared.py
```

The integrated smoke uses prescribed provider replies and synthetic data, with
real fitting of two targets, shared-law revision, paired pruning fits and exact
resume. It makes no live LLM calls. The scheduled CPU prepare job runs the relevant
checks and smoke before requesting usable proposer work.

## GPU sizing

OpenAI documents GPT-OSS-120B fitting on a single H100 with MXFP4 weights:
https://developers.openai.com/api/docs/models/gpt-oss-120b
https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm

The existing critic launcher requests two H100s and tensor-parallel size 2. That
is its serving configuration, not a requirement imposed by being a critic. A
second GPU provides weight/KV-cache headroom and sharding; the actual throughput
benefit depends on workload. GPT-OSS-20B is substantially smaller and the current
proposer uses one GPU. A one-H100 120B configuration has not been qualified on
this ACES image/context by this integration; no critic allocation is made here.
