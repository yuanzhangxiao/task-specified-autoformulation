# Research operations: observe, preserve, then retire

The inventory commands run metadata audits, not experiments. They do not submit jobs,
fit models, invoke an LLM, modify benchmark releases or open test payloads.
Orion's corrected benchmarks remain separate from the historical campaign.
Python 3.10+ and its standard library suffice; no scientific packages, GPU,
module load or API credential is needed for inventory. The explicit recovery
section below submits experimental workers using the original scientific runtime.

## Use one small portable tools bundle

Use the supplied `phase-c-operations.tar.gz`, containing the audit/inventory
scripts, these docs, `SOURCE_COMMIT` and `SHA256SUMS`. Upload it to your home
directory on each host. No full project clone or new VM is needed. Keep the
reported archive checksum with your copy; compare against the sender's checksum,
then verify the enclosed per-file checksums. A checksum establishes byte
agreement, not scientific correctness or an independent signature.

The supplied bundle also contains `run-aces.sh`, `run-delta.sh` and
`run-jetstream.sh`, generated verbatim from the three command blocks below.
They are convenience copies of this runbook, not separate campaign launchers.

Alternatively, in a trusted Git checkout, set `AF_COMMIT` to the supplied full
commit, then export the same small bundle. Use an existing group/project mirror;
do not fetch into a quota-limited personal-scratch repository. There is no need
for a clone per experiment. The portable bundle is simpler for this milestone.

The extraction commands use a fresh directory under `/tmp` for code and reports.
Copy the resulting result archive to durable storage before logout/node cleanup.
Reusing extracted code is fine. Each new observation needs a fresh output path;
existing reports are never overwritten. Partial inventories can be rerun with a
larger limit or narrower roots; they are not completed experiments to resume.

## ACES: campaign, scheduler and storage

Run on ACES. This scans metadata up to 250,000 entries per storage root. Shared
filesystems can be slow; if site policy requires it, run the same block in an
ordinary CPU allocation. The script does not walk other mounted filesystems or
follow symlinks. Protected roots are still counted; protection means retention.

```bash
(
  set -euo pipefail
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  [[ -d "$AF_GROUP" ]] || { echo 'Run on ACES.' >&2; exit 1; }
  AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
  AF_RUN=$(mktemp -d /tmp/phase-c-ops.XXXXXX)
  tar -xzf "$HOME/phase-c-operations.tar.gz" -C "$AF_RUN"
  (cd "$AF_RUN" && sha256sum -c SHA256SUMS)
  export PYTHONDONTWRITEBYTECODE=1
  "$AF_PYTHON" "$AF_RUN/scripts/audit_component_campaign.py" collect \
    --root "$AF_GROUP/final-components-v1" --output "$AF_RUN/campaign"
  AF_JOBS=$(paste -sd, "$AF_RUN/campaign/job_ids.txt")
  [[ -n "$AF_JOBS" ]] || { echo 'No recorded job IDs; inspect audit.' >&2; exit 1; }
  sacct -j "$AF_JOBS" --parsable2 \
    --format=JobID%40,JobIDRaw%40,JobName%40,State%30,ExitCode,Elapsed,Submit,Start,End \
    > "$AF_RUN/scheduler.psv"
  squeue -u "$USER" -o '%.30i %.32j %.12T %.12M %R' > "$AF_RUN/queue.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$AF_RUN/captured-at.txt"
  "$AF_PYTHON" "$AF_RUN/scripts/audit_component_campaign.py" report \
    --input "$AF_RUN/campaign/audit.json" --scheduler "$AF_RUN/scheduler.psv" \
    --output "$AF_RUN/reconciled"
  "$AF_PYTHON" "$AF_RUN/scripts/inventory_research_storage.py" \
    --root "$AF_GROUP" --root /scratch/user/u.yx126462 \
    --protect "$AF_GROUP/final-components-v1" \
    --protect "$AF_GROUP/repos" --protect "$AF_GROUP/final-component-inputs-v1" \
    --protect /scratch/user/u.yx126462/repos \
    --protect /scratch/user/u.yx126462/phase_b \
    --output "$AF_RUN/storage"
  cat "$AF_RUN/reconciled/SCHEDULER.md"
  AF_DELIVERY=$(mktemp -d "$AF_GROUP/foundation-followup.XXXXXX")
  tar -czf "$AF_DELIVERY/operations-results.tar.gz" -C "$AF_RUN" \
    campaign reconciled storage scheduler.psv queue.txt captured-at.txt SOURCE_COMMIT
  printf '\nDownload: %s\nTemporary working directory: %s\n' \
    "$AF_DELIVERY/operations-results.tar.gz" "$AF_RUN"
)
```

The copied live queue is an observation, not a lease: it may change immediately.
Missing `sacct` rows stay unknown. `JobIDRaw` maps array tasks to the IDs in worker
receipts; do not infer it by adding the array index to its parent ID. If accounting
retention has removed a record, inspect original logs and leave its state unknown.

Three failed critic logs in the supplied snapshot are at:

```text
/scratch/group/p.nairr260351.000/u.yx126462/final-components-v1/logs/cpu-recovery-1-critic-2162100_0.err
/scratch/group/p.nairr260351.000/u.yx126462/final-components-v1/logs/cpu-recovery-1-critic-2162100_1.err
/scratch/group/p.nairr260351.000/u.yx126462/final-components-v1/logs/cpu-recovery-1-critic-2162100_2.err
```

Inspect their terminal exception and the corresponding `.out` worker finish
messages. Treat logs as private; redact any accidentally logged credential before
sharing. They are intentionally not included automatically in the portable audit.

## Delta: inventory existing project and work storage

Upload the same tools bundle to `$HOME/phase-c-operations.tar.gz`. This does not
copy an ACES campaign to Delta or run overlapping writers against it.

```bash
(
  set -euo pipefail
  AF_PROJECT=/projects/bibo/yxiao2
  AF_WORK=/work/hdd/bibo/yxiao2
  [[ -d "$AF_PROJECT" && -d "$AF_WORK" ]] || { echo 'Run on Delta.' >&2; exit 1; }
  AF_PYTHON="$AF_PROJECT/venvs/autoformalism-v21/bin/python"
  AF_RUN=$(mktemp -d /tmp/phase-c-ops.XXXXXX)
  tar -xzf "$HOME/phase-c-operations.tar.gz" -C "$AF_RUN"
  (cd "$AF_RUN" && sha256sum -c SHA256SUMS)
  PYTHONDONTWRITEBYTECODE=1 "$AF_PYTHON" "$AF_RUN/scripts/inventory_research_storage.py" \
    --root "$AF_PROJECT" --root "$AF_WORK" \
    --protect "$AF_PROJECT/repos" --protect "$AF_PROJECT/venvs" \
    --protect "$AF_PROJECT/containers" --protect "$AF_WORK/phase_b" \
    --output "$AF_RUN/storage"
  squeue -u "$USER" -o '%.30i %.32j %.12T %.12M %R' > "$AF_RUN/queue.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$AF_RUN/captured-at.txt"
  AF_DELIVERY=$(mktemp -d "$AF_WORK/foundation-storage.XXXXXX")
  tar -czf "$AF_DELIVERY/storage-results.tar.gz" -C "$AF_RUN" \
    storage queue.txt captured-at.txt SOURCE_COMMIT
  printf '\nDownload: %s\n' "$AF_DELIVERY/storage-results.tar.gz"
)
```

## Jetstream2: inventory an existing VM

Run inside the VM, not on the laptop. This scans the current user's home;
volumes mounted elsewhere must be supplied explicitly using another `--root`.
Do not launch a VM or call the managed GPT-OSS service just for this audit.

```bash
(
  set -euo pipefail
  AF_RUN=$(mktemp -d /tmp/phase-c-ops.XXXXXX)
  tar -xzf "$HOME/phase-c-operations.tar.gz" -C "$AF_RUN"
  (cd "$AF_RUN" && sha256sum -c SHA256SUMS)
  PYTHONDONTWRITEBYTECODE=1 python3 "$AF_RUN/scripts/inventory_research_storage.py" \
    --root "$HOME" --protect "$HOME/.ssh" --protect "$HOME/.config" \
    --output "$AF_RUN/storage"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$AF_RUN/captured-at.txt"
  AF_DELIVERY=$(mktemp -d "$HOME/foundation-storage.XXXXXX")
  tar -czf "$AF_DELIVERY/storage-results.tar.gz" -C "$AF_RUN" \
    storage captured-at.txt SOURCE_COMMIT
  printf '\nDownload: %s\n' "$AF_DELIVERY/storage-results.tar.gz"
)
```

Reports contain path names and file metadata, not file contents. Keep them
private. Counts are not the provider's quota meter: check both byte and inode
quotas (`showquota` on ACES, `quota` on Delta), allocation expiry, snapshots and
detached volumes separately. A truncated or unreadable tree is incomplete, never
zero usage; `--max-entries` can be increased for a deliberate second pass.

## Retention and retirement rules

| Class | Default treatment | Evidence needed before removing a working copy |
|---|---|---|
| Active campaign, claims, partial calls and referenced runtime | Keep at original paths | Campaign closure or a validated migration; no active writers; budgets and resume identities preserved |
| Scientific archive | Keep two verified copies in independent storage | Plans, all candidates/fits/failures, initialization, source/environment/data versions, LLM/cost/selection records and evaluation identities are complete; checksums and restore test pass |
| Duplicate checkout/export or environment | Review and consolidate | Exact version captured; no active job or frozen command references its path; remaining copy can reproduce the runtime |
| Downloaded weights/container image | Review for reproducibility | Pinned revision/digest and confirmed replacement source; offline/retention needs considered |
| Generated Python/test/lint cache | First cleanup candidates | No active writer or needed diagnostic record; reviewed exact paths, never a broad wildcard delete |
| Unclassified directory, partial archive, unknown file | Keep | An owner assigns purpose and proves archive/dependency status |

`inventory.json` deliberately marks every bucket `deletion_eligible=false`.
The metadata scan cannot establish which old-looking directories hold unique
results or which environments running jobs require. Names and ages alone cannot
certify safe deletion. There is no delete mode or automatically scheduled cleanup.

The next cleanup deliverable is an explicit manifest of exact paths, measured
size/file count, owner, dependency check, archive checksums, restore evidence and
proposed action. Preserve relative paths when archiving many small files; verify
contents before releasing originals. Archives reduce inode use but may not save
many bytes for already compressed arrays or images. Never publish credentials
with source, logs or scientific archives; preserve required configuration through
a separate secret-management process.

Repository-root `output/`, `tmp/`, `transfers/` and `backups/` are ignored as local
generated/staging directories, alongside existing `artifacts/` and `work/` rules.
This avoids accidentally committing outputs and nested checkouts; it neither
deletes them nor backs them up. Source code and historical protocol modules remain
tracked. Move an intended permanent source contribution to a reviewed source path
instead of forcing a generated directory into Git.

Retire obsolete source in a separate small change after identifying callers,
fixtures and frozen source-hash dependencies. First identify active entry points
in [PHASE_C_START_HERE.md](PHASE_C_START_HERE.md); do not reorganize Orion's files
or switch the shared checkout during his benchmark work. New tasks can start
with that handoff; old tasks can be archived after decisions and evidence are
preserved. Conversation archival does not free cluster storage.

## Platform expiry reminders

Check your actual allocation, not only directory age. As documented on
2026-09-26: [ACES project scratch](https://hprc.tamu.edu/kb/User-Guides/ACES/Filesystems_and_Files/)
is not backed up and is removed 90 days after allocation expiration;
[Delta](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html)
offers a 30-day data-management grace period for expired projects, not permanent
archival storage; [Jetstream Exosphere](https://docs.jetstream-cloud.org/ui/exo/manage/)
deletes the root disk when deleting an instance, including volume-backed roots.
Do not use any of these expiry windows as the sole backup plan.

## ACES: bounded continuation after the confirmed critic lock collision

The 2026-09-26 follow-up reconciled all 88 worker identities. All three failed
recovery critics raised `RuntimeError: public fit directory is in use` at
`component_critic.review_one`'s shared review-cache lock. This is a concurrency
failure before that operation's provider request, not evidence of a Jetstream
outage. Different lineage claims can reach the same cached review. Never remove
`.lock` files to resolve this: a live process holds the OS lock, and unlinking
the file can defeat exclusion. Closing the process's file descriptor releases it.

The original source can resume with **one critic worker across the campaign**.
This is an allocation workaround, not a fix for future parallel critics. Keep
the original runtime because its source and launcher hashes are part of the
frozen identity. There is no need to clone or fetch code, restage public inputs,
reauthorize the critic, or restart completed models. Existing authorization,
cache entries and consumed allowances remain in use.

Run the following on ACES in a Bash terminal. It submits one bounded wave with
four proposer GPUs, one CPU critic, 16 fit workers and eight pruning workers.
The latter can work on the 72 pruning frontiers while the critic unblocks the
88 other lineages. Subsequent fits and revisions require the other services,
which is why resuming only the critic would not finish the campaign. Worker
counts change concurrency, not the frozen model/fit/review allowances.

```bash
(
  set -euo pipefail
  export AF_REPO_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/repos/final-components-34378d1
  export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/final-components-v1
  export AF_COMMIT=34378d1f1da5eefb2c1f977b440346afab29a33f
  [[ -f "$AF_REPO_ROOT/SOURCE_COMMIT" ]] || { echo 'Run on ACES with the original source archive.' >&2; exit 1; }
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]] || { echo 'Source commit differs; stop.' >&2; exit 1; }

  AF_QUEUE=$(squeue --noheader --user "$USER" --format='%j')
  if printf '%s\n' "$AF_QUEUE" | awk '/^component-(propose|critic|fit|prune)$/ {found=1} END {exit !found}'; then
    echo 'Component workers are still queued/running; inspect squeue before adding this wave.' >&2
    exit 1
  fi

  module load GCCcore/13.2.0 Python/3.11.5
  export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
  export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
  export AF_VLLM_IMAGE=/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif
  export AF_HF_HOME=/scratch/user/u.yx126462/huggingface-cache
  export AF_COMPUTE_CACHE_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/component-runtime-cache
  export AF_IPC_TMP_ROOT=/tmp/af-ipc-u.yx126462
  [[ -f "$AF_VLLM_IMAGE" && -d "$AF_HF_HOME" ]] || { echo 'Original image or model cache is missing; stop.' >&2; exit 1; }
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/component_campaign.py" verify --root "$AF_OUTPUT_ROOT"

  if [[ -z "${AF_JETSTREAM_API_KEY:-}" ]]; then
    read -r -s -p 'Jetstream API key (hidden): ' AF_JETSTREAM_API_KEY
    printf '\n'
    export AF_JETSTREAM_API_KEY
  fi
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_component_campaign.py" \
    --root "$AF_OUTPUT_ROOT" --wave serial-critic-recovery-1 \
    --proposers 4 --critics 1 --fits 16 --pruning 8
)
```

The submitter verifies the complete frozen runtime and saved critic authorization
before submitting. Repeating this wave name reuses its recorded job IDs; it does
not request another allocation after they finish. An uncertain scheduler reply
stops for inspection; preserve its intent and receipt. Do not start another
critic manually or through a different wave while this one is active. The queue
name check detects the known launcher jobs, not arbitrary shell/API clients.
Do not send the API key or enable shell tracing.

### After `serial-critic-recovery-1`

The 2026-09-27 recovery snapshot verifies all 29 allocations completed normally.
The critic-disabled endpoint backlog is cleared (71 complete, one skipped);
all 88 critic-enabled lineages still require more rounds. For the next bounded
wave, use the same original-source command block above with
`--wave serial-critic-recovery-2 --proposers 4 --critics 1 --fits 16 --pruning 1`.
One pruning service keeps the stage available without repeating eight workers
against an empty current pruning frontier. The critic remains singular across
the campaign. These are allocation choices, not changes to the frozen scientific
budgets. Verify the live queue and source first, as in the command block.

Do not rerun the old wave name expecting new allocations. Do not prequeue a
sequence of overlapping critic waves. Inspect the new snapshot after the bounded
wave ends. At the observed 146 critic operations per five-hour active window,
595 remaining round rows plus up to 88 final reviews suggest roughly 24 further
critic processing hours if throughput stays similar, before queue delays and
final fitting/pruning dependencies. Cache reuse, request lengths and failures can
change this substantially; it is a planning estimate, not a completion deadline.

The same snapshot's actual quota meter reports personal scratch at
247,723/250,000 files and 244.2 GiB/1 TiB: file count, not bytes, is the immediate
constraint. The allocation-wide group meter is 411,299/500,000 files and
239.6 GiB/5 TiB. Continue writing campaign artifacts and generated runtime caches
under the existing group paths. Keep the referenced personal Python environment,
model cache and container. Cleanup requires the reviewed path/dependency checks
above; this checkpoint does not authorize deletion.

Each allocation lasts at most 6h30; workers stop before their next bounded unit
would overrun the working window. Queue time and complete campaign convergence
are not guaranteed. Refresh the existing report and operations audit after the
wave; unfinished work remains explicit. This command does not export endpoints,
open tests, change benchmark files or create a new campaign.

The accompanying storage observation found 247,719 entries under personal
scratch. The group scan stopped at its 250,000-entry scan budget with no recorded
read errors, so its byte/file counts are lower bounds. Check `showquota` for
actual byte and inode headroom before continuing. A zero generated-cache total
does not establish that caches are absent: protected roots take precedence in
classification. Use the full `operations-results.tar.gz` directory breakdown to
prepare exact archive candidates; the pasted totals alone cannot justify deletion.
