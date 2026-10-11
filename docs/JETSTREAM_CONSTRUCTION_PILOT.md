# Jetstream2 focused construction pilot

The four independent serving probes passed on October 11, 2026. Their saved
requests, schemas, usage and model/container provenance were independently
checked. This pilot tests the complete variable-to-topology workflow on that VM.
Serving compatibility does not establish science or equal outputs across hosts.

## Scope and provenance

Eight fresh Full-specification constructions, seed 0: Dalla Man T1-easy,
T1-hard, T2-easy, T2-hard, CSTR, alien device, coupled basins, independent basins.
The sealed public source is the same one supplied in the ACES five-seed bundle.
No saved inventories or topology proposals become starting models.

The constructor, `minimal-focused-construction-1` prompts, public checks,
sampling and per-episode budgets are unchanged: up to 128 physical requests /
524,288 charged tokens per construction in a three-hour worker window. Existing
local recovery and up to five global repairs apply. No interaction functions,
fitting, scientific critic calls or test data. An incomplete topology is a
retained terminal outcome, not automatically retried with a fresh budget.
Interrupted or unstarted work can resume from checkpoints.

The new plan records `jetstream2-h100x1-docker` and the Docker image digest,
instead of ACES and its SIF digest. `execution.json` binds the plan, launcher
source, compatibility evidence and serving record. Compatibility costs are
separate from construction costs. Existing ACES/compatibility records remain.

## Run on the H100 VM

Upload and extract the delivered `phase-c-js2-pilot-<commit>.tar.gz` on
`exouser@149.165.152.222`. From that extracted directory:

```bash
sha256sum -c MANIFEST.sha256 > manifest-check.log &&
bash scripts/jetstream/run_construction.sh run
tail -f "$HOME/af-construction/focused-seed0-v1/launcher.log"
```

This reuses `~/af-construction/venv` and the running `af-construction-vllm`
container. It does not install packages, download weights or start another
server. It validates the owned container's command, mounts, offline weights and
loopback binding, rechecks the four compatibility outcomes, and reads live
model/version metadata. It shares the compatibility worker lock to prevent
concurrent execution. Storage preflight requires 2 GiB and 10,000 free entries;
these are startup checks, not reservations or project quota accounting.

Control-C exits `tail` only; the worker continues after SSH disconnects.
Repeat `run` to resume the same plan. Do not edit its installed source or plan
in place. Every launch log is retained; `launcher.log` points to the latest.

```bash
bash scripts/jetstream/run_construction.sh status
```

A zero worker exit means the bounded worker completed, not that all topologies
passed. Check `all_tasks_terminal` and `status_counts` in `summary.json`.

## Inspect and download

After the worker stops, from the delivered directory on the VM:

```bash
bash scripts/jetstream/run_construction.sh inspect
```

This regenerates the overall, variable and shared-process reports, and packages
all saved calls, feedback, drafts, runtime records and logs. No new LLM calls.
Model weights and compiler caches are excluded.

Then run **on the Mac, outside SSH**:

```bash
scp exouser@149.165.152.222:/home/exouser/af-construction/focused-seed0-v1/inspection.tar.gz \
  /Users/yuanzhangxiao/Downloads/jetstream-pilot-inspection.tar.gz
```

The server remains running. Stopping it uses the original compatibility
launcher's `stop` command; that does not shut down the VM. No follow-up campaign
is automatic. This one-seed pilot tests workflow operation; broader quality
claims require multi-seed results and equation inspection.
