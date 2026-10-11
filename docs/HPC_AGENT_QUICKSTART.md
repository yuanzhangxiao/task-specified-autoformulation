# Delta and ACES: experiment setup for a new Codex agent

This is the operational starting point for this project. It explains how to
deliver, submit, inspect and recover an experiment using the existing cluster
environments. Read the experiment's own runbook for its scientific protocol.
Examples were checked against repository launchers on 2026-10-10; paths and
accounts below belong to this user and are not universal cluster defaults.

**Working arrangement:** implement and test locally, commit and push, then give
the user self-contained upload, launch and inspection commands. The user has
asked to run remote commands themselves. Do not open ACES/Delta sessions or
submit an experiment merely to test this guide. Do not change another agent's
jobs, environment or campaign directory.

## 1. First read, then choose an existing launcher

Read [AGENTS.md](../AGENTS.md), [Phase C handoff](PHASE_C_START_HERE.md), and the
relevant parts of [project context](PROJECT_CONTEXT.md) and
[pipeline design](PIPELINE_DESIGN.md). For data, use the
[Phase C readiness contract](PHASE_C_DATASET_READINESS.md). Historical Phase B
data and corrected Phase C releases are different experimental inputs.

| Work | Current starting point | Typical platform |
|---|---|---|
| Fitting qualification, no LLM | [Fitting plan](PHASE_C_FITTING_PLAN.md), [M26 scaling study](PHASE_C_FITTING_SCALING.md) | Delta CPU |
| Variable/shared-process/topology construction | [Focused construction prompts](PHASE_C_FOCUSED_PROMPTS.md) | ACES, one H100 for the pinned local proposer |
| Inventory, retention, historical recovery | [Research operations](RESEARCH_OPERATIONS.md) | Either cluster; usually CPU/metadata only |

These are separate experiments, not one universal end-to-end launcher. Use the
launcher belonging to the plan's protocol. For example, the focused construction
study stops before functions and fitting. M26 uses known equations and no LLM.
Do not silently substitute one for the other or promote a fitting diagnostic to
the production pipeline.

Before coding, inspect the launcher, submission script, worker and relevant
tests. Existing submission code already records identities and receipts; reuse
that behavior instead of wrapping bare `sbatch` calls in an automatic retry loop.

## 2. Machines, storage and existing runtimes

| Setting | Delta | ACES |
|---|---|---|
| Username | `yxiao2` | `u.yx126462` |
| Experiment storage | `/work/hdd/bibo/yxiao2/phase_c` | `/scratch/group/p.nairr260351.000/u.yx126462` |
| Code/environment storage | `/projects/bibo/yxiao2` | Group scratch above, usually `repos/` |
| Existing scientific Python | `/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python` | `/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python` |
| Extra fitting dependencies | `/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps` | Check the experiment's existing environment |
| Recorded CPU account | `bibo-delta-cpu` | `156264627414` |
| CPU partition in current launchers | `cpu` | `cpu` |
| Local inference example | Requires a separately supported GPU launcher | `gpu`, `--gres=gpu:h100:1` |

An old commit in a virtual-environment path does not select that old source:
the worker's `PYTHONPATH` selects the pinned experiment code. Package versions
still matter and must pass the experiment's preflight. Do not upgrade a shared
environment used by active jobs, copy a macOS virtual environment to Linux, or
reinstall a large environment for every experiment.

Keep three paths distinct: **code**, **frozen inputs**, and **results**. Keep
them available at their recorded absolute paths for queued jobs and resume.
`/tmp` is for disposable node-local scratch, not sole copies of checkpoints.

### Login and transfer

The user can use the [ACES portal](https://portal-aces.hprc.tamu.edu) or
[Delta Open OnDemand](https://openondemand.delta.ncsa.illinois.edu). Terminal
commands to connect run on the user's computer, not inside another login node.

Delta uses the NCSA username/password and NCSA Duo. Ordinary SSH-key access is
not the default. Follow the official
[Delta access and transfer instructions](https://support.access-ci.org/documentation/resources/delta-cpu).
On the user's computer:

```bash
ssh yxiao2@login.delta.ncsa.illinois.edu
```

ACES uses portal-issued SSH credentials and a jump host. The
[official ACES access guide](https://hprc.tamu.edu/kb/User-Guides/ACES/Access/)
currently documents a 49-hour certificate lifetime, portal `Utilities → sshca`
or `gen_ssh_cert.sh`, and the required local SSH configuration. Once that
configuration exists, connect from the user's computer with:

```bash
ssh login.aces.hprc.tamu.edu
```

An ACES public-key failure may mean an expired certificate. A cluster login
does not grant GitHub SSH access. Do not ask for passwords/private keys in chat
or disable host-key verification. Use the official recovery procedure when
authentication fails. If the portal download proxy fails, Delta's official
transfer guide includes Globus as another transfer route.

### Lightweight environment checks on the destination cluster

On **Delta**, for the existing fitting environment:

```bash
bash <<'BASH'
set -euo pipefail
AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
test -x "$AF_PYTHON"
PYTHONPATH="$AF_CASADI_ROOT" "$AF_PYTHON" -c \
  'import sys, numpy, scipy, casadi; print(sys.version); print(numpy.__version__, scipy.__version__, casadi.__version__)'
BASH
```

On **ACES**, for the existing construction environment:

```bash
bash <<'BASH'
set -euo pipefail
module load GCCcore/13.2.0 Python/3.11.5
AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
test -x "$AF_PYTHON"
"$AF_PYTHON" -c 'import sys, numpy, scipy; print(sys.version); print(numpy.__version__, scipy.__version__)'
test -r /scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif
test -d /scratch/user/u.yx126462/huggingface-cache
showquota
BASH
```

These checks do not establish GPU/model compatibility. The pinned worker must
also check its actual GPU, driver, container, model revision and context limit
inside the allocation. Keep heavy fitting, model loading and numerical smoke
runs off login nodes. Existing workers set BLAS/OpenMP threads to match their
allocations; more numerical threads do not create more allocated CPUs.

## 3. Deliver a reproducible experiment

1. Freeze the question, cells/seeds/arms, budgets and input release before the
   run. Record whether it needs CPU, a local GPU model, or a remote API.
2. Implement in the user's local checkout. Preserve unrelated edits; run the
   required tests and relevant smoke. Commit only source, tests and docs, then
   push. Never include credentials, result payloads or datasets in the commit.
3. Deliver a small archive from that exact commit, with `SOURCE_COMMIT`,
   `SHA256SUMS`, required frozen inputs and the relevant launch/inspect scripts.
   Include all files covered by source-digest verification. Keep generated
   archives outside Git. Provide the archive's checksum separately.
4. Upload and extract into a versioned directory, without overwriting code
   used by queued/running jobs. Uploaded archives work when cluster GitHub
   authentication does not. For a Git deployment, verify a clean checkout at
   the specified full commit; do not use a moving branch as the run identity.
5. Choose a unique result root for a new protocol/configuration. Resume an
   existing root only with its original code, input identity and settings.
6. Run the campaign's prepare/verify step, then its submission wrapper. Preserve
   its printed root, plan hash, commit and scheduler job IDs.

Do not package an entire working directory or `.venv`. Do not send the proposer
private evaluator/reference files simply because they share a release archive.
Use the public-data allowlist. Test data must not enter proposal, fitting or
selection. Reference coefficients in known-structure qualification are for the
explicit retrospective evaluator, not optimization or start selection.

Verify an uploaded trusted bundle before launch, from its extracted code root:

```bash
cat SOURCE_COMMIT
sha256sum -c SHA256SUMS
```

Compare the archive checksum with the separately supplied value as well. A
checksum is an integrity check, not proof of scientific correctness. The plan's
verification remains necessary after file verification.

## 4. Worked launch examples

These examples document existing studies; onboarding alone is not a reason to
launch duplicate experiments. Use them when continuing an authorized study, or
adapt its established launcher for an explicitly requested new experiment.

### Delta: M26 CPU fitting

The M26 delivery `phase-c-fitting-m26.tar.gz` includes `scaling-inputs.json`.
A bare repository clone does not include that generated input file. Its source
commit is `c4c9bd599cfcbe2f6b85a6ac33c9591284bceb40`; retain that code for an
existing M26 campaign. See the [M26 runbook](PHASE_C_FITTING_SCALING.md).

After extracting the verified bundle into the path below, run on **Delta**:

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=/work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m26
export AF_SCALING_ROOT=/work/hdd/bibo/yxiao2/phase_c/fitting-scaling-v1
test -f "$AF_CODE/scaling-inputs.json"
export AF_ACCOUNT=bibo-delta-cpu
export AF_CONCURRENCY=3
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_scaling_delta.sh"
BASH
```

This study submits preparation, a nine-task CPU array, and reporting. Each fit
requests one CPU, 16 GB and a 2.5-hour scheduler ceiling. It makes no LLM calls.
`AF_CONCURRENCY=3` limits this array to three running tasks; it is not a Delta
account limit. Solver budgets and scheduler wall time are separate: allow time
for setup, verification and final writes when choosing a scheduler ceiling.

Inspect and create the review archive on **Delta**, including after a new login:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m26/scripts/hpc/inspect_phase_c_fitting_scaling_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/fitting-scaling-v1
```

The inspector prints the exact archive path. It can also report partial results;
creating an archive does not imply all tasks succeeded.

### ACES: focused construction

Use the pinned construction code and frozen topology-source context supplied
for that study. Set `AF_CODE` to its absolute extracted code directory **in the
same shell** before the following block. The source path below is an existing
campaign dependency; if absent, obtain the matching frozen input bundle rather
than substituting another campaign or regenerating data.

Run on **ACES**:

```bash
(
  set -euo pipefail
  : "${AF_CODE:?Set AF_CODE to the pinned construction code directory}"
  export AF_COMPARISON_STUDY=focused_prompt_confirmation
  export AF_COMPARISON_SEEDS='0 1 2 3 4'
  export AF_COMPARISON_SOURCE=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-topology-v1
  export AF_COMPARISON_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-focused-prompts-seeds5-v1
  test -f "$AF_COMPARISON_SOURCE/plan.json"
  bash "$AF_CODE/scripts/hpc/start_phase_c_construction_comparison.sh" run
)
```

The current study has eight cells × five seeds. It uses one H100 allocation
for construction episodes, followed by a CPU report; no fitting jobs. The
proposer's Slurm ceiling is 3.5 hours, with a shorter internal working window.
The account, GPU and other resource settings are in
[`submit_phase_c_baseline.py`](../scripts/submit_phase_c_baseline.py), not a
generic `AF_ACCOUNT` override. Do not silently change model size, context,
sampling or numerical settings to fit another device.

For inspection, repeat the same `AF_CODE` and study/root settings, replacing
the final launcher argument `run` with `inspect`. It prints reports and writes
`inspection.tar.gz`. Receipt location:
`submissions/<wave>/manifest.json`. If tasks remain pending after a wave has
ended, follow the [construction runbook](PHASE_C_FOCUSED_PROMPTS.md) for a new
wave name. This resumes the same plan and charged budgets. Do not start a new
wave while an earlier writer is still active or its submission is uncertain.

## 5. Inspect scheduler state and scientific completion separately

Set the root explicitly on every new terminal session. A missing variable once
produced the misleading path `//audit/summary.json`. A root that does not exist
means “wrong path or not created,” not “jobs still running.”

For **M26**, run on Delta:

```bash
(
  set -euo pipefail
  AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/fitting-scaling-v1
  test -d "$AF_ROOT"
  AF_RECEIPT="$AF_ROOT/submission_manifest.json"
  test -f "$AF_RECEIPT"
  AF_IDS=$(jq -er '.jobs | [.[]] | select(length > 0) | join(",")' "$AF_RECEIPT")
  squeue -j "$AF_IDS" -o '%.24i %.30j %.12T %.12M %R'
  sacct -j "$AF_IDS" --parsable2 \
    --format=JobID%40,JobIDRaw%40,JobName%40,State%30,ExitCode,Elapsed
)
```

For construction, use that wave's `submissions/<wave>/manifest.json` instead.
Older experiments may use `submission-intent/` or another layout: inspect
`plan.json`, the submitter and available filenames before writing a `jq` query.
Do not assume `.rows` exists in every report or silently turn missing fields
into a successful zero count.

- **Pending / Dependency:** the predecessor has not satisfied its dependency.
- **Pending / Priority:** queued behind other work; not itself a code failure.
- **QOS/association limit:** a user/account/resource cap is binding. Array task
  counts matter; one compressed array line can represent many jobs.
- **Empty `squeue`:** check accounting and artifacts; it does not prove success.
- **Slurm `COMPLETED`:** the process exited successfully. Check expected versus
  recorded scientific tasks, per-task status, unknown costs and replay results.
- **Report completed after worker failure:** possible with `afterany`; the
  report can correctly describe an incomplete campaign.

To query limits, on either cluster where accounting visibility permits:

```bash
sacctmgr -n -P show assoc where user="$USER" \
  format=Cluster,Account,Partition,QOS,DefaultQOS,MaxJobs,MaxSubmitJobs
sacctmgr -n -P show qos \
  format=Name,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPU
```

Blank association fields do not imply unlimited access; QOS, account/group and
partition limits can also apply. Historic ACES output showed 40 running and 90
submitted jobs under `normal`, with CPU/GPU caps. Re-query rather than treating
that snapshot as a current guarantee. A `%2` or `%3` array throttle is not a
cluster-wide limit. `squeue --start` provides an estimate, not a reservation.

## 6. Recover without duplicating work

| Symptom | Next step |
|---|---|
| `scheduler reply ... unconfirmed`, socket timeout, empty job ID | Preserve intent/reply files. Query `squeue`, `sacct` and `scontrol show job <id>`. Match user, submit time, full command, root, output paths and dependency against the saved intent. Only reconcile a verified match using the appropriate recovery tool. |
| `sbatch` return code 0 but stderr says failure | Do not trust return code alone. ACES has produced exactly this combination. Treat a missing valid receipt as unresolved. |
| Dependency problem after recovering a completed predecessor | Inspect whether Slurm still knows that predecessor and whether the scientific preparation passed. Use the campaign's recovery procedure; do not strip dependencies blindly. |
| Existing intent but no completed manifest | Submission may have stopped halfway. Inspect individual receipts before retrying; absence of a manifest does not prove no job exists. |
| Source/config/input digest differs | Find the correct frozen code and assets. New scientific settings require a new campaign identity. Do not overwrite hashes to make verification pass. |
| Many workers fail in under a second | Read the first worker exception and preflight logs: often an import, path, environment or validation failure. Fix that before spending another full array. |
| Provider failure / request too long | Inspect retained request status and context budget. Preserve caches and charged attempts; missing token usage remains unknown. |
| Upload fails / disk quota exceeded | Check bytes **and file counts**. ACES: `showquota`. The personal scratch inode quota has filled while hundreds of GB remained. Use the retention runbook before removing anything. |

Never delete submission intents, cached calls or `.lock` files to force a retry.
Never infer that two similarly named jobs belong to the same campaign. Record
recovered IDs with the evidence used to match them. An interrupted fit may have
a usable checkpoint; the protocol decides whether/how to continue it, without
resetting consumed budgets or losing the incumbent.

## 7. Return results and leave a usable handoff

Use the study's inspector to package plans, task outcomes, fitted parameters and
initial values, fit diagnostics, request/cost receipts and required logs. Include
failed and missing tasks. Preserve full request caches on the cluster even when
the compact review archive excludes them. Check logs for accidental credentials
before sharing; do not dump the whole environment into reports.

Download from the **user's computer**. Set `AF_REMOTE_REVIEW` to the exact Delta
archive path printed by the inspector, then:

```bash
: "${AF_REMOTE_REVIEW:?Set the absolute review archive path on Delta}"
scp "yxiao2@login.delta.ncsa.illinois.edu:$AF_REMOTE_REVIEW" \
  /Users/yuanzhangxiao/Downloads/
```

Running this inside `yxiao2@dt-login...` does not copy to the Mac: `/Users/...`
would refer to a directory on that remote host. For ACES, use the configured
`login.aces.hprc.tamu.edu` SSH host and its printed archive path, or the portal.
Keep archive checksums and at least one durable verified copy before cleanup.
See [retention rules](RESEARCH_OPERATIONS.md); do not delete another agent's
caches, environments or source directories.

Every handoff should include:

- Scientific purpose, protocol, cell/seed/arm roster and expected task count.
- Platform, absolute code/input/result roots, full commit, plan/input hashes.
- Runtime/container/model pins, account, CPU/GPU/memory/time requests, throttle.
- Submission receipt paths and IDs; which jobs are pending, failed or unknown.
- Exact launch/resume, inspection and download commands, with destination labels.
- Summary of completed checks, remaining limitations and cost accounting.

The next agent should be able to inspect an existing campaign without launching
new work, guessing a path, rebuilding an environment or reading the whole chat.

## Source of operational facts

Repository defaults come from the linked runbooks and their launchers. Live site
rules take precedence; consult the [Delta user guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/),
[Delta access/transfer guide](https://support.access-ci.org/documentation/resources/delta-cpu),
[ACES access guide](https://hprc.tamu.edu/kb/User-Guides/ACES/Access/) and
[ACES filesystem/quota guide](https://hprc.tamu.edu/kb/User-Guides/ACES/Filesystems_and_Files/).
No remote environment, quota or queue was inspected to create this document.
