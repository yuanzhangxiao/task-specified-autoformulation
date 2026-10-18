# Saved initialization equality-feedback retest

This opt-in milestone follows the ACES provider-recovery audit. It fixes a
presentation-sensitive progress fingerprint and surfaces the conditional equality
behind each of the two unresolved saved T2-easy cases. It does not change public
benchmark prompts, numerical evidence, scientific acceptance predicates, or
interaction functions.

## Changes

`saved-topology-initialization-4` uses `topology-initialization-repair-4` and
`diagnostic-semantics-3`. Retained trajectory-witness identities use structured
mathematical topology instead of the entire draft hash. Editing a process's
`scientific_meaning` or a variable/term's `scientific_role` alone cannot turn an
unchanged model into mathematical progress. Full draft hashes remain available
for artifact provenance. Historical policies retain their original identities.

The repair packet presents up to two supported model/data conflicts before the
full feedback and public background. Each summary connects the current model,
its initial-information rules, matched arguments or forcing, and actual witness
values. The underlying observations, uncertainty information, and assumptions
remain in the request. Unsupported or incomplete conditions do not produce an
equality claim. No particular missing dependency is chosen by the runtime.

The two selected starts illustrate different restrictions:

- Seed 1: an algebraic disposal readout has identical arguments at the initial
  instant, yet its two recorded initial outputs differ. Changing a later ODE
  argument cannot change a state initialized to the same shared constant.
- Seed 2: the coupled insulin/memory subsystem has identical initial information
  and matched external forcing, yet the recorded insulin values differ at time
  8. Feedback through another equally initialized state does not distinguish
  these trajectories under shared deterministic equations and unique solutions.

These are necessary-condition conflicts under explicit hypotheses, not fitted
error measurements or proof of scientific correctness. Functions, signs,
conservation and overlap still need interaction-stage evaluation.

## Provenance and budget

The freezer selects one or two explicit IDs from a sealed initializer-v3 provider
recovery. It validates the recovery proof, selected event/call chains (including
the inherited request offset), final drafts, initializer records and accounting.
It retains old conflicts, historical costs and source hashes. Seed 0 is excluded
because it no longer has an actionable current conflict.

This is a **new experiment**, with at most 10 new physical requests and 262,144
charged tokens per case: at most 20 requests / 524,288 tokens for the two-case
plan. Historical usage and unknown-usage reservations are retained separately;
they are neither refunded nor treated as new usage. New calls are cached and
checkpointed. Resume cannot renew the allowance. A provider failure terminates
availability without advancing scientific stagnation or automatic retry.

One ACES H100 runs the cases serially with the existing pinned GPT-OSS-20B model,
vLLM image, serving settings and verified offline Harmony vocabulary. CPU
preflight runs in a short CPU allocation because ACES login nodes need not expose
a container runtime. No fitting, test-data access, model promotion or automatic
follow-up is included.

## Prepare and inspect

After source changes are complete, freeze the explicit saved cases:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/saved_initialization_aces.py freeze-equality \
  --source output/initialization-provider-recovery-analysis-20261017/inspection \
  --root output/initialization-equality-inputs \
  --task-id cell02_seed1_full_actionable_initialization_focused \
  --task-id cell02_seed2_full_actionable_initialization_focused
```

The delivered source-pinned bundle includes this plan, vocabulary and a
`run_retest.sh` wrapper. Upload it to the ACES group directory, extract it, and
verify `MANIFEST.sha256`. From its extracted directory:

```bash
srun --account=156264627414 --partition=cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=00:30:00 --export=ALL --exclude=ac042 \
  bash "$PWD/run_retest.sh" run
bash run_retest.sh status
bash run_retest.sh inspect
```

Output defaults to
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-initialization-equality-v1`.
Download its `inspection.tar.gz` after completion. Keep the same root and wave
when checking or resuming; do not overwrite a historical campaign.
