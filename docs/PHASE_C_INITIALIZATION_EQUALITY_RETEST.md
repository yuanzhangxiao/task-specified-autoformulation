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


## Returned ACES audit (2026-10-17)

Job 2225153 completed with worker exit zero. The returned plan matches
`f23173ba553a87afadb7dc565940c983d48f0db3b9bb3b02c5d7d73105874105`
and source `dca7f8d47b6d59d8abe8bc4e510d32c8a032c845`. The inspection archive
SHA-256 is `8a73a4b2790ffb431c370e45b9f990e00154910a7a0f06fe46698611e94ef3da`.
All 13 generation responses finish normally, with 276,720 observed/charged tokens
and no unknown usage. Historical costs remain separate. CUDA and offline Harmony
preflights pass; no fitting or test access occurred.

| Saved case | Calls | New tokens | Final conditional conflict | Stop |
| --- | ---: | ---: | --- | --- |
| T2-easy seed 1 | 3 | 63,043 | Initial algebraic U readout | Three repeated repairs |
| T2-easy seed 2 | 10 | 213,677 | Coupled I / I_mem subsystem | Call allowance exhausted |

Both final models are structurally eligible and have complete initialization
records. Neither resolves its current conditional conflict. Eight other diagnostic
records per case remain deferred; those are not passes. Every request includes
the intended leading equality explanation.

Seed 1 retains `U = f(Uii) + g(X)` and `X(0) = c`, with one shared constant.
For train_011 and train_012, sealed training evidence gives `Uii(0) = 1` in both,
but `U(0) = 1.9890358776874768` and `1.8350720852593723`. The proposer repeatedly
labels these U measurements as unequal Uii readings and disputes the equality,
without editing the model. The runtime correctly preserves the conflict and stops
after three identical replies. The prompts assert matched Uii but omit its actual
paired values; subsequent feedback does not correct that specific channel
confusion. Accepted reply transactions do not imply accepted scientific claims.

Seed 2 retains `I_mem(0) = h(I(0))`. The train_000/train_011 pair has equal
`I(0) = 25.5878134642966` and matched complete external-forcing histories, yet
`I(8) = 25.5878134642966` versus `48.42727474869684`. All I/I_mem dependencies
remain inside that equally initialized subsystem and its matched forcing. One
accepted edit changes an internal process argument from I to I_mem; this does not
introduce distinguishing information. Four attempted `observed_initial` declarations
for latent I_mem are rejected because no measured-initial binding exists. One reply
explicitly understands the initializer equality restriction but proposes that
unsupported initialization mode. No retained initializer changes.

The V4 description-only progress fix behaves correctly: seed 2 event 004 changes
only role text and remains semantically unchanged with the same question ID.
A separate residual defect remains: events 003 and 004 have identical base
scientific attempt signatures, but appending an explicit unchanged initialization
versus `null` produces different final fingerprints. The unchanged counter resets
to one, contradicting the instruction that redundant restatements do not reset it.
Distinct rejected mathematical alternatives should still keep distinct identities.

Recommended next milestone, not implemented in this audit:

1. Present numeric paired conditioning values beside the observed difference,
   with explicit channel and time labels. Use general evidence formatting rather
   than a benchmark-specific scientific correction.
2. For an unsupported observed initializer, name the state, absence of a measured
   initial binding, current retained rule, and legal editable initial-information
   choices. The proposer must choose the scientific dependence.
3. Canonicalize unchanged initializer submissions for repetition counting while
   retaining distinct rejected alternatives. Verify on saved replies before a new
   separately budgeted two-case test.

These are two selected difficult saved cases, not an estimate of all-seed success
or a controlled causal comparison of prompts. No function construction, fitted
accuracy or full scientific consistency is established. The local review at
`output/initialization-equality-analysis-20261017/REPORT.html` links all 13 exact
requests/replies; generated responses and evidence remain outside Git.
