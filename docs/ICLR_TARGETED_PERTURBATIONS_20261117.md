# Targeted perturbation searches — 17 November 2026

Run Full with the anchor method on the two approved public benchmark variants.
There are six independent fresh lineages: two cells × seeds 0, 1, 2, each with
15 visits. The ACES array permits all six H100 allocations to run concurrently.
Slurm still controls their start times. Each allocation requests one H100,
8 CPUs, 64 GB and 12 hours; the worker reserves time for shutdown and checkpointing.
An unfinished lineage resumes from the same plan and cached calls in a later wave.

## Scientific design

- Proposer: the same pinned GPT-OSS-20B weights, serving image, generation settings,
  construction prompts and search policy as the anchor/T1 run 2353678.
- Critic: the same calibrated Jetstream critic and authorization. The only
  inference credential prompt is hidden; no key is written into this delivery.
- Fitting, parent selection and pruning: unchanged from the anchor method.
  An unsuccessful fitted initial construction has the anchor Full disposition;
  it is not silently replaced by a new seed or extra budget.
- Runtime: the search launcher requires the anchor's Python/package versions.
  It uses the existing anchor activation and CasADi overlay rather than making
  another environment. GPU compiler caches live in job-local temporary storage.
- Perturbed CSTR-hard: use the four supplied, byte-pinned public files. The
  original public scientific prompt and hidden-preparation rule are unchanged.
- Perturbed T1-easy: retain the original training/validation arrays and all
  specification fields except the explicitly approved `public_prompt` change.
  The new plasma–tissue exchange clause is visible to proposer and critic. Its
  deterministic graph prerequisite checks a `Gt`-to-`Gp` pathway. This does not
  certify a correct quantitative exchange law, signs or mechanism recovery.
- No test data, private perturbation formula, analyst diagnostics, intervention
  outcomes or reference-fit parameters enter construction. The supplied archive
  is processed through a fixed public-file allowlist, not extracted wholesale.

The input SHA-256s are in `configs/iclr_targeted_perturbations_v1.json`.
`scripts/iclr_targeted_perturbations.py stage` reproducibly stages the authorized
prompt variant without modifying the source release. The original T1 prompt
remains a separate comparison on identical numerical data. These are targeted
follow-up experiments, not a previously unseen benchmark selection.

## ACES search

Upload the delivered `.tar.gz` to
`/scratch/group/p.nairr260351.000/u.yx126462`. Extract it there, then use its
`run_aces.sh` entry point. The final delivery message provides the exact versioned
filename. Run the launch command by itself, wait for the hidden critic-key prompt,
and paste the key there if it was not already set in the environment.

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/BUNDLE/run_aces.sh run
```

`run` verifies the archive files, offline vocabulary, cached 20B weights, anchor
settings and calibrated critic, prepares the plan and submits all six blocks as
one array with `%6`. Repeating `run` with the same wave returns its existing job
ID. It does not submit a second copy. Check progress with:

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/BUNDLE/run_aces.sh status
```

Once the allocation has finished, if some visits or pruning dispositions remain
pending, resume the same root with:

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/BUNDLE/run_aces.sh run wave2
```

This schedules only unfinished blocks and preserves budgets, proposals, fits and
critic receipts. A running previous wave or an uncertain Slurm acknowledgment
blocks another submission. For the latter, inspect `squeue`, `sacct` and the saved
`submissions/wave*/array.reply.json`; a socket timeout can occur after Slurm
accepted the job. Do not delete the submission intent or create a new campaign
to bypass the guard. The launcher's `prepare` action performs preflight without
submitting jobs or making inference requests.

After all 90 visit dispositions and six pruning dispositions are present:

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/BUNDLE/run_aces.sh finish
```

Download:

```text
/scratch/group/p.nairr260351.000/u.yx126462/iclr-targeted-perturbations-v1/targeted-perturbation-results.zip
```

This freezes the endpoint roster and exports native models, fitted parameters,
causal initializers, per-seed validation scores, all endpoint/visit receipts,
critic authorization, runtime and scheduler records, and an internal SHA-256
manifest. Failed seeds remain in the roster. `finish` refuses an incomplete run;
it makes no fitting or LLM calls. There is no automatic follow-up wave.

Overrides: `AF_TARGETED_ROOT`, `AF_GROUP`, `AF_ANCHOR_ACTIVATION`,
`AF_ANCHOR_REFERENCE`, `AF_JUDGE_GATE_ROOT`, `AF_PYTHON`, `AF_CASADI_ROOT`,
`AF_HF_HOME`, `AF_VLLM_IMAGE`, `AF_GPU_ACCOUNT`. Keep a resumed campaign on the
same code and numerical environment. Use the existing group scratch space, not
the inode-constrained home directory. Check `showquota` if extraction or durable
checkpoint writes fail; preserve experiment results and the input archive.

## Optional Delta CPU reference fits

The handoff also requests fitting the **canonical, privileged CSTR equations** to
the new perturbed public data. These are separate numerical controls, not LLM
searches or discovered models. They can run on Delta while the six ACES GPU
searches are pending/running. This avoids porting the anchor search runtime to
another platform during the rebuttal window.

The original reference fitter's default behavior is preserved. The CSTR-only
extension supports the original ±0.5 log-multiplier bounds and the expanded
±log(4) bounds used in the T1 sensitivity study. There are three profiles:

1. Original bounds and original starts.
2. Expanded bounds with the same original starts.
3. Expanded bounds with starts distributed over the inner 80% of each interval.

Each uses starts 0, 1, 2 (RNG seeds 800–802), up to 50 function evaluations,
shared physical coefficients and shared hidden preparation parameters. Held-out
hidden initials are not fitted separately. Training sets the normalization;
validation chooses the lowest-NMSE numerically eligible start within each fixed
profile. Every start, failure, fitted coefficient and bound hit is retained.
RK4 two/four-substep agreement is checked as in the original reference fitter;
optimizer convergence and numerical agreement remain separate diagnostics.

Upload the same delivery to `/work/hdd/bibo/yxiao2/phase_c`, extract it there, and run:

```bash
AF_PLATFORM=delta bash /work/hdd/bibo/yxiao2/phase_c/BUNDLE/run_refits.sh run
AF_PLATFORM=delta bash /work/hdd/bibo/yxiao2/phase_c/BUNDLE/run_refits.sh status
```

The launcher uses `/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`, the
existing `fitter-methods-v1-deps` overlay, account `bibo-delta-cpu`, partition
`cpu`, and a nine-task array with up to nine concurrent tasks. Each requests
2 CPUs, 16 GB and four hours. `AF_PYTHON`, `AF_CASADI_ROOT`, `AF_CPU_ACCOUNT` and
`AF_REFERENCE_ROOT` can override paths/account before preparation. The actual
numerical versions are sealed; changing them during resume is rejected.

After completion:

```bash
AF_PLATFORM=delta bash /work/hdd/bibo/yxiao2/phase_c/BUNDLE/run_refits.sh finish
```

Download:

```text
/work/hdd/bibo/yxiao2/phase_c/iclr-targeted-reference-v1/perturbed-cstr-reference-fits.zip
```

Alternatively, run the same entry point on ACES with `AF_PLATFORM=aces`; it uses
the anchor activation and CPU account `156264627414`. Do not run both platforms
as additional, selectively reported starts. A `wave2` resumes unfinished starts
on the chosen platform. These controls never call the proposer or critic.

## Implementation and limits

### Recovery of the first delivery's missing fitter identity file

The first delivery (`16fdae7`, array `2373558`) constructed all six initial
models and obtained their critic reviews, but every fit raised `FileNotFoundError`
for `scripts/run_phase_b_fitter_diagnostic.py`. The fitter reads that file while
constructing its runtime identity, before starting collocation or refinement.
The import-based bundle collector had omitted this path-based dependency.
Consequently all 84 subsequent visits were recorded as
`shared_initial_unavailable`, and the six frozen endpoints have no NMSEs. This
is an infrastructure failure, not evidence about perturbation recovery.

The corrected collector explicitly includes that file and the optional timed
fitter worker. The launcher exercises the identity read before submission. An
isolated delivery test reproduces the old failure, then runs a real multi-target
collocation fit and verifies unchanged resume using only packaged files.
Future fit receipts include the exception message in the portable export.

Use the corrected bundle's `run_recovery.sh` entry point:

```bash
bash /scratch/group/p.nairr260351.000/u.yx126462/BUNDLE/run_recovery.sh run
bash /scratch/group/p.nairr260351.000/u.yx126462/BUNDLE/run_recovery.sh status
bash /scratch/group/p.nairr260351.000/u.yx126462/BUNDLE/run_recovery.sh finish
```

It reads `iclr-targeted-perturbations-v1` and prepares
`iclr-targeted-perturbations-recovered-v1`. It preserves the original evaluation
freeze, results and cached calls. The new root copies the same plan, public
inputs, proposals, critic reviews, call journals and fit freezes byte-for-byte.
The retry is allowed only after authenticating all six exact missing-file
failures, checking that no backend work exists, and verifying that the remaining
visits were skipped. It rejects other failures, changed inputs and used fitting
budgets. This does not modify the fitting algorithm or the Full search rules.

Preparation makes no LLM or optimizer calls. The scheduled run retries fitting
the six saved proposals, then continues their original 15-visit searches; it does
not regenerate the initial proposals or repeat their critic reviews. Historical
proposal/critic cost remains in the reused journals, with explicit provenance in
`packaging_recovery.json`. The failed attempt's allocated GPU time remains a
separate cost in the original archive. If a retry fails numerically, the original
fitted-start rule still applies. No successful numerical outcome is promised.

Download the recovered results from:

```text
/scratch/group/p.nairr260351.000/u.yx126462/iclr-targeted-perturbations-recovered-v1/targeted-perturbation-results.zip
```

`run_recovery.sh run wave2` resumes unfinished work with the usual uncertainty
and active-job guards. `AF_TARGETED_RECOVER_FROM` and `AF_TARGETED_ROOT` override
the source and destination; they must be disjoint. Never remove the old freeze
or reset its failed fit files. The optional CSTR reference-fit campaign is
separate and is not rerun by this recovery.

The change adds public-input staging, exact method/configuration checks, the
approved T1 operational binding, six-way submission and frozen exports. The
optional CPU launcher binds its separate numerical plan and exact code/runtime.
The shared reference fitter adds optional CSTR bounds/initial-start settings;
its defaults, canonical equations and integration procedure remain unchanged.

Offline checks exercise unchanged-data staging, private-file exclusion, sealed
resume, critic/calibration identity, both positive and negative graph cases,
concurrent submission, uncertain acknowledgments, complete/failing endpoint
exports, and actual synthetic reference fits. Live cluster execution and result
quality remain to be established by these runs. A graph pass is not mechanism
recovery, and a finite reference fit is not evidence of an optimized global
minimum. Intervention scoring remains with the analyst after freezing.
