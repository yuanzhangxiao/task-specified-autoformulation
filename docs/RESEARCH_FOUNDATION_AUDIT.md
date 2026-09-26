# First research-reset milestone: campaign and stage census

The objective is to establish what the historical campaign actually completed
and where construction work was spent before choosing the next algorithmic
change. Orion owns reference-generator corrections and benchmark qualification.
This milestone changes no generators, finalized prompts, fitter, proposer,
critic, controller, model selection, or historical results.

## Deliverables and interpretation

`scripts/audit_component_campaign.py` supports the frozen
`final-component-campaign-1` protocol, including matched-block shards. It uses
Python 3.10+ and the standard library; no scientific environment, API key, GPU,
or `module load` is needed.

`collect` reads the sealed plan and explicitly named campaign artifacts. The
plan and model receipts may embed public training/validation data; only metadata
projections are exported. The collector does not open separate trajectory files,
test/evaluation results, prompts, provider caches, or private reference files.
It does not compile or execute model equations or reproduce scientific checks.

Outputs:

| File | Purpose |
|---|---|
| `audit.json` | Sealed portable snapshot: complete planned rows, retained origin receipts, stage events, source-file hashes, and integrity issues |
| `SUMMARY.md`, `summary.json` | Completion, endpoint, proposal, critic and stage counts; first unresolved stage per lineage |
| `STAGES.md` | Accepted/rejected/unrecorded validator events and diagnostic categories |
| `issues.json` | Invalid seals/bindings, missing required receipts, symlinks, or observed concurrent changes |
| `job_ids.txt` | Recorded submission/worker job IDs, for a separate read-only scheduler query |

The full matrix has 160 lineages, 2,400 search visits and 192 endpoints. Shards
keep their own planned denominators. A completed scheduler allocation is not a
completed visit; a terminal critic receipt is not necessarily available advice;
a pruning receipt can be terminal while a child fit failed. A missing outer
round result with a worker/fit start or fit result is flagged as a consumed
attempt requiring inspection, not proposed as a fresh fit.

Stage events are validator decisions, not physical model calls. Rejections may
have been repaired successfully. An empty optional process proposal is a legal
outcome. Missing stage records are unknown, not zero errors. Error categories are
text heuristics with diagnostic hashes; inspect the original record before
attributing a scientific or engineering cause. Counts alone cannot establish
that an accepted model has the right mechanism or equation skeleton.

Final stage results take precedence over progress checkpoints, and sealed stage
wrappers take precedence over their inner files, avoiding duplicate events.
The raw source-file hash is recorded even where the runtime does not provide a
stage seal. The audit validates campaign seals, selected request/fit hashes,
and selected receipt/task/predecessor bindings; it does not duplicate the full
matrix/config validator, compiler, selection policy, calibration, or
data-contract verifier. The census follows the declared plan, after checking
its protocol tag, seal, bounded dimensions and unique safe task identifiers.
`matrix_contract_revalidated=false` makes this limitation machine-readable.
It is not a replacement for the original runtime's verification before resume.

## ACES: collect the current campaign

Run on an **ACES login node**, not Delta or your laptop. Set `AF_COMMIT` to the
full commit reported with this change. The following uses a new code directory
on group storage, avoiding writes to the quota-limited personal-scratch repo.
The new directory contains only the audit scripts and documentation from the
pinned commit. It does not replace the campaign's original runtime checkout.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set AF_COMMIT to the reported full audit commit first}"
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  [[ -d "$AF_GROUP" ]] || { echo 'Run this block on ACES.' >&2; exit 1; }
  AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
  mkdir -p "$AF_GROUP/repos"
  AF_GIT=$(mktemp -d "$AF_GROUP/repos/foundation-fetch.XXXXXX")
  AF_CODE=$(mktemp -d "$AF_GROUP/repos/foundation-code.XXXXXX")
  git init --bare "$AF_GIT"
  git -C "$AF_GIT" fetch --depth=1 \
    https://github.com/yuanzhangxiao/task-specified-autoformulation.git "$AF_COMMIT"
  git -C "$AF_GIT" archive "$AF_COMMIT" \
    scripts/audit_component_campaign.py scripts/component_audit_io.py \
    scripts/component_stage_inventory.py docs/RESEARCH_FOUNDATION_AUDIT.md \
    docs/BOUNDED_RESEARCH_WORK_ORDERS.md | tar -x -C "$AF_CODE"
  printf '%s\n' "$AF_COMMIT" > "$AF_CODE/SOURCE_COMMIT"
  AF_OUT=$(mktemp -d "$AF_GROUP/foundation-audit.XXXXXX")
  PYTHONDONTWRITEBYTECODE=1 "$AF_PYTHON" "$AF_CODE/scripts/audit_component_campaign.py" \
    collect --root "$AF_GROUP/final-components-v1" --output "$AF_OUT"
  cat "$AF_OUT/SUMMARY.md"
  cat "$AF_OUT/STAGES.md"
  printf '\nAudit output: %s\nAudit code: %s\n' "$AF_OUT" "$AF_CODE"
  tar -czf "$AF_OUT/portable-audit.tar.gz" \
    -C "$AF_OUT" audit.json \
    -C "$AF_CODE" scripts docs SOURCE_COMMIT
  if [[ -s "$AF_OUT/job_ids.txt" ]]; then
    AF_JOBS=$(paste -sd, "$AF_OUT/job_ids.txt")
    sacct -j "$AF_JOBS" --parsable2 \
      --format=JobID,JobName%30,State,ExitCode,Elapsed > "$AF_OUT/scheduler.psv"
  fi
)
```

The ACES command uses the existing Python 3.11 executable but imports only its
standard library, avoiding pandas/ctypes and their missing-module dependencies.
A working `python3` version 3.10+ can be substituted. This is filesystem
inspection, not a CPU fitting job; it can take several
minutes on shared storage. The script prints progress per lineage. If local
policy requires a batch allocation for filesystem audits, use an ordinary CPU
allocation and the same command. Do not request GPUs. No Slurm dependency chain
or new campaign submission is created.

The two `mktemp` code/fetch directories are intentionally retained; nothing is
deleted automatically. The lightweight code directory can be reused with a
fresh `--output` directory. An identical report is idempotent; a changed snapshot
is refused at an existing output path. If interrupted, rerun into a fresh output
directory. There are no model calls or fitting budgets to consume/reset.

The collector checks files again at the end and reports observed changes. It
does not lock the entire campaign or guarantee a globally atomic snapshot.
`issues.json` must be reviewed before acting on any suggested stage. Keep the
scheduler snapshot alongside the audit: a start marker cannot establish whether
its worker is still running. No commands here cancel, release or resubmit jobs.

## Delta and Jetstream2: inspect the portable bundle

Download `portable-audit.tar.gz` from ACES and upload it to the host where you
want to inspect it. No cluster-to-cluster authentication or hostname is assumed.
The bundle contains no dataset, equations, provider payloads or credentials.
It contains the exact audit source and its commit identifier. The source commit
can additionally be checked against the repository if independent code
provenance is required; the JSON seal checks the audit data itself.

On **Delta**, put the uploaded file at
`/work/hdd/bibo/yxiao2/phase_b/portable-audit.tar.gz`, then:

```bash
(
  set -euo pipefail
  AF_PARENT=/work/hdd/bibo/yxiao2/phase_b
  [[ -d "$AF_PARENT" ]] || { echo 'Run this block on Delta.' >&2; exit 1; }
  AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
  AF_COPY=$(mktemp -d "$AF_PARENT/foundation-review.XXXXXX")
  tar -xzf "$AF_PARENT/portable-audit.tar.gz" -C "$AF_COPY"
  PYTHONDONTWRITEBYTECODE=1 "$AF_PYTHON" "$AF_COPY/scripts/audit_component_campaign.py" \
    report --input "$AF_COPY/audit.json" --output "$AF_COPY/report"
  cat "$AF_COPY/report/SUMMARY.md"
  cat "$AF_COPY/report/STAGES.md"
)
```

On an existing **Jetstream2 VM**, upload it as `$HOME/portable-audit.tar.gz`:

```bash
(
  set -euo pipefail
  AF_COPY=$(mktemp -d "$HOME/foundation-review.XXXXXX")
  tar -xzf "$HOME/portable-audit.tar.gz" -C "$AF_COPY"
  PYTHONDONTWRITEBYTECODE=1 python3 "$AF_COPY/scripts/audit_component_campaign.py" \
    report --input "$AF_COPY/audit.json" --output "$AF_COPY/report"
  cat "$AF_COPY/report/SUMMARY.md"
  cat "$AF_COPY/report/STAGES.md"
)
```

These reproduce the same report from the saved bundle; they are not independent
re-measurements of the ACES filesystem. Do not launch a new VM just for this
small report. Jetstream2's managed GPT-OSS endpoint is not used in this milestone.
Later it can run bounded construction probes after their inputs, budgets and
success criteria are fixed. Delta/ACES CPU workers can then run matched fitter
diagnostics. Shared campaign roots must not be written by overlapping recovery
or successor runs.

For a separate component-campaign shard already on Delta, run `collect` against
its exact existing path and a new output directory. Do not guess a root, combine
shards by concatenating summaries, or mix corrected reference data into this
historical audit. A future merge must reconcile task overlap and plan identity.

## Next decision after the audit

Return `audit.json`, `SUMMARY.md`, `STAGES.md`, `issues.json`, and (from ACES)
`scheduler.psv`. The next milestone is a small, explicit work list:

1. Distinguish unfinished scheduler work from genuinely consumed failures, and
   finish the historical campaign using its existing contracts and budget.
2. Select a small representative sample at the dominant failing stages. Compare
   ordinary predecessors against independently reviewed predecessors, holding
   the public information, seed policy and model settings fixed. Measure both
   mechanical acceptance and expert-reviewed scientific decisions.
3. Pair this with Orion's benchmark/reference audit. Before changing proposer
   search, establish whether the fitter can recover adequate rollouts from a
   known admissible structure under the same observations and initialization
   rules. Keep this oracle diagnostic separate from benchmark method results.

Do not choose prompts/architectures from held-out test scores. Keep corrected
benchmark versions and historical campaign results separately labeled. Search
expansion (beam/MCTS), new routing, and larger sweeps come after the dominant
failure is identified. A benchmark may test adequacy under incomplete
specification without promising recovery of a unique textbook realization.
