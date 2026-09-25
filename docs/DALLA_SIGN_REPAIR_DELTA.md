# Delta alternative for the R4 sign-repair pilot

The ACES sign review serves **GPT-OSS-20B**, not the larger scientific critic.
It permits three proposal/assessment pairs (six physical calls). H100 is the
original serving placement, not a mathematical requirement of checking signs.
The Delta alternative uses the existing one-A40 server path with vLLM 0.27.1,
the same model revision, prompts, seeds, context limit and request budgets.
Actual A40 startup and resource adequacy still need confirmation on Delta.

The CPU allocation is 4h30 because one array task runs repaired and unchanged
arms sequentially. Each arm may run a rescue refit, a pruning fit and an
unchanged pruning-control refit: up to six fits total. Each fit has a nominal
300-second initializer and 900-second refinement allowance; starting-vector
replays, recovery and scoring add overhead. The Slurm limit is a ceiling, not
a requested duration of actual work. This handoff preserves both fitting budgets
and the allocation ceiling until measured durations support reducing it.

Specific GPU requests restrict eligible resources, and a longer wall limit can
reduce backfill opportunities. Neither proves the current cause of waiting.
In the reported queue, review waited for Priority and fitting waited for
Dependency. Delta has its own queue and allocations; moving does not guarantee
an earlier start. Delta documents one-GPU requests on shared A40 nodes:
https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html

## Avoid running the same pilot twice

On ACES, only if review 2162092 is still pending, hold it while trying Delta:

```bash
bash <<'BASH'
set -euo pipefail
af_state=$(squeue -h -j 2162092 -o '%T')
[[ "$af_state" == PENDING ]] || {
  echo "Review is '$af_state', not pending. Inspect ACES before launching a duplicate."
  exit 1
}
scontrol hold 2162092
squeue -j 2162092,2162093,2162094 -o '%.18i %.25j %.10T %R'
BASH
```

This retains the preparation and dependent CPU jobs. If Delta fails, first
ensure its review is stopped, then release ACES with `scontrol release 2162092`.
No launcher cancels or changes another campaign.

## Upload and launch

The supplied `dalla-sign-delta-COMMIT.tar.gz` includes a pinned code archive,
`SOURCE_COMMIT`, and the **original rescue packet**, containing public training
and validation only. No test data, reference equations or GPU weights are
included. The packet is not the v1 sign-repair output. Only the
preselected `full_perturbed_r4` endpoint is reviewed/refitted.

Upload the archive into `/work/hdd/bibo/yxiao2/phase_b`, then run, replacing
`COMMIT` with the supplied short identifier:

```bash
bash <<'BASH'
set -euo pipefail
AF_PACKAGE=dalla-sign-delta-COMMIT
mkdir -p /projects/bibo/yxiao2/repos
tar -xzf "/work/hdd/bibo/yxiao2/phase_b/$AF_PACKAGE.tar.gz" \
  -C /projects/bibo/yxiao2/repos
bash "/projects/bibo/yxiao2/repos/$AF_PACKAGE/scripts/hpc/launch_dalla_sign_delta.sh"
BASH
```

The launcher checks the existing Python and container before submitting. Defaults:

- Python: `/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python` (`AF_PYTHON`).
- SIF: `/projects/bibo/yxiao2/containers/vllm-openai-v0.27.1.sif` (`AF_VLLM_IMAGE`).
- HF cache: `/projects/bibo/yxiao2/huggingface-cache` (`AF_HF_HOME`).
- Accounts: `bibo-delta-cpu` / `bibo-delta-gpu` (`AF_CPU_ACCOUNT` / `AF_GPU_ACCOUNT`).
- Output: `/work/hdd/bibo/yxiao2/phase_b/dalla-sign-repair-delta-v2-r4`
  (`AF_OUTPUT_ROOT`).

Image hashing during submission can take a little time. CPU preparation checks
the actual image digest and vLLM version, runs regressions and a synthetic smoke,
and caches the pinned 20B revision if missing. Production fitting runs only in
the CPU fit job. The jobs request:

| Stage | Resources | Slurm limit |
|---|---|---|
| prepare | 1 CPU, 8 GB | 1 hour |
| review | 1 A40, 8 CPUs, 64 GB | 1h15 |
| fit | 1 CPU, 16 GB, one array task | 4h30 |
| report | 1 CPU, 4 GB | 15 minutes |

All use the `projects&work` filesystem constraint. No ACES module commands run
on Delta. The existing site worker retains its historical `_aces.sh` filename
but dispatches modules by the frozen platform. Preparation failure blocks review
and fitting. Review failure blocks fitting; report runs after the chain terminates.

`--delta` creates a separate sealed plan and records the real local SIF digest;
it does not resume or overwrite the ACES plan. Cross-hardware responses and
fitting timings need not be identical. This is a demonstration pilot, not an
extra replicate to select by intervention performance. Source bytes, public data,
configuration, code identity and scheduler commands are pinned. Repeating the
same submission reuses confirmed IDs. Unconfirmed scheduler responses require
inspection and verified `--adopt STAGE=JOB_ID`; never erase their receipts and
submit blindly. The ACES-specific recovery helper is not a Delta recovery tool.

## Check results

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_b/dalla-sign-repair-delta-v2-r4
AF_IDS=$(jq -er '.jobs | [.[]] | join(",")' "$AF_ROOT/submission_manifest.json")
sacct -j "$AF_IDS" -X --format=JobID,JobName%28,State,ExitCode,Elapsed
jq '{status,expected,rows}' "$AF_ROOT/summary.json"
```

The summary appears after report execution. Inspect `logs/prepare-*.err` and
`logs/review-*.err`, plus `runtime/server-*.log`, on failure. Download the final
`models.json` for equation and trajectory inspection.
