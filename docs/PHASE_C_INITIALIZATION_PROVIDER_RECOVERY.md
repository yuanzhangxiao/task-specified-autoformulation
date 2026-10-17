# ACES initialization retest: provider recovery

The uploaded ACES job 2224147 produced nine HTTP 500 failures and no proposer
responses. Its three saved T2-easy models and initialization declarations are
unchanged. Every measured request passed context preflight (19,735–22,620 input
tokens against a 24,000-token input cap). This confirms preflight acceptance;
actual generation and scientific effectiveness remain untested.

The server failed in Harmony chat rendering while loading the GPT-OSS vocabulary.
CUDA, model discovery and ordinary tokenization had passed; none exercised that
rendering path. The vocabulary is a separate asset from the Hugging Face model
snapshot. See the [official vLLM GPT-OSS recipe](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md).

## Serving fix

`scripts/harmony_vocab.py` explicitly provisions and SHA-256-verifies
`o200k_base.tiktoken`. The delivery bundle includes that file; it is not committed
to Git. Preflight and batch launchers bind the verified asset read-only into the
pinned image using `TIKTOKEN_ENCODINGS_BASE`. They load Harmony and render a tiny
conversation on CPU, making zero generation requests. The launch check runs
before submission; the batch check runs before CUDA and server startup. Neither
launcher downloads assets on compute nodes.

Execution identity version 2 binds the asset hash, helper and launcher hashes,
image, model revision and serving settings. The worker requires the successful
Harmony receipt alongside the existing CUDA/runtime receipt. Missing or corrupt
assets and failed rendering block launch. Historical execution seals cannot be
overwritten by the new launcher.

## Recovery and accounting

`aces-provider-recovery-1` is an operational continuation of the scientific
`saved-topology-initialization-3` experiment. It does not alter the scientific
prompt policy, checks, model starts or initializer choices. A new root embeds
hash-bound evidence from the failed predecessor. The freezer accepts only cases
whose saved calls all failed at the provider without a response and whose draft
and initializer handoff remain unchanged. It recovers the original scientific
feedback and repetition state, rather than treating HTTP errors as scientific
edits.

| Seed | Failed requests retained | Charged tokens retained | Further request cap | Tokens remaining |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3 | 91,298 | 7 | 170,846 |
| 1 | 3 | 85,891 | 7 | 176,253 |
| 2 | 3 | 89,276 | 7 | 172,868 |

No usage was returned for the failed calls. Their reservations remain charged;
zero observed tokens is not zero known cost. New physical attempt numbering
starts at 3, within the original ten-call and 262,144-token caps per case.
Reports distinguish inherited, new and cumulative costs.

A new provider failure stops the episode and allocation immediately as
`provider_unavailable`. It preserves scientific feedback and repetition counters.
Re-running the same terminal checkpoint does not resend the request or reset its
allowance. A further recovery requires explicit inspection and a new bounded
continuation; it is not an automatic follow-up.

## Reproduce and run

Freeze inputs locally after completing source changes:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/saved_initialization_aces.py freeze-recovery \
  --source output/initialization-focused-analysis-20261017/inspection \
  --root output/initialization-provider-recovery-inputs
```

The delivered bundle contains the recovery plan, complete source, verified
vocabulary, and a `run_recovery.sh` wrapper. Upload and extract it under the
writable ACES group directory. ACES documents container runtimes as available on
compute nodes, without an extra module load; a login node may have neither
`apptainer` nor `singularity` on PATH. Run the CPU preflight and submission in a
short CPU allocation, then use the login node for status and inspection:

```bash
srun --account=156264627414 --partition=cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=00:30:00 --export=ALL --exclude=ac042 \
  bash /path/to/extracted-bundle/run_recovery.sh run
bash /path/to/extracted-bundle/run_recovery.sh status
bash /path/to/extracted-bundle/run_recovery.sh inspect
```

This uses the existing bundle unchanged, including the `587f99c` delivery.
The CPU allocation ends after preparation and GPU-job submission; it does not
host inference. Keep the existing output root and wave for deterministic resume.
The updated source adds actionable stderr before preparation/submission instead
of silently failing under `set -e`; these diagnostics are not retroactively added
to the sealed `587f99c` bundle. Updated batch startup also reports a missing
runtime, and inspection explains when preparation has not created a plan yet.
Status and inspection do not require a container runtime. See the
[HPRC container guide](https://hprc.tamu.edu/kb/Software/Singularity/) and
[ACES batch guide](https://hprc.tamu.edu/kb/User-Guides/ACES/Batch/).

The wrapper selects a new output root,
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-initialization-provider-recovery-v1`,
and the bundled recovery input directory. It leaves the failed campaign intact.
One H100 allocation processes the same three cases serially; at most 21 new
requests are permitted. There is no fitting, model promotion, automatic follow-up
or test-data access.

## Returned recovery: job 2224673

The recovered H100 run passed CPU Harmony rendering and CUDA probes. All 17 new
HTTP requests returned responses: 16 finished normally and one exhausted its
8,192-token output allowance. Eleven replies were accepted and six rejected;
acceptance does not establish substantive progress or scientific correctness.
New measured usage is 374,857 tokens. The nine inherited failed calls retain
266,465 tokens of reservations, giving 26 cumulative attempts and a 641,322-token
cumulative budget charge. Inherited usage remains unknown.

All three cases have structurally eligible topology and complete initialization
declarations. None has established scientific accuracy or fitted performance.

| T2-easy seed | New calls | Accepted replies | Final scientific status |
| --- | ---: | ---: | --- |
| 0 | 3 | 2 | Review deferred; no current actionable conflict |
| 1 | 7 | 4 | Initial algebraic U mismatch; repair budget exhausted |
| 2 | 7 | 5 | Closed insulin-subsystem mismatch; repair budget exhausted |

Seed 0 adds external insulin forcing and EGP dependence, removing a real equality
restriction for its paired witnesses without changing its latent initializer.
All nine current diagnostics are deferred, and two older scalar-autonomous
questions remain unresolved after revision. This is conditional progress, not a
science pass. Function overlap and the interpretation of its negatively signed,
insulin-dependent memory term remain interaction-stage questions.

Seed 1 retains a shared initial constant for X_insulin_action and an algebraic
U readout dependent only on that state and supplied Uii. The witness pair has
identical initial arguments but U(0)=1.98903588 versus 1.83507209. The initial-time
explanation was displayed on every repair call. Changing later ODE terms cannot
remove this restriction; the proposer did not change its latent initialization
or introduce distinguishing readout information.

Seed 2 changes U from algebraic to differential, allowing its recorded initial
value and removing that initial-U conflict. Its final I/I_mem subsystem still
receives identical initial information and forcing for a pair with
I(8)=25.58781346 versus 48.42727475. Adding I_mem feedback does not distinguish
the trajectories when I_mem(0)=h(I(0)) and both initial arguments match. These
comparisons are conditional on the stated deterministic, shared-parameter,
unique-solution assumptions and recorded uncertainty bounds.

The runtime retained and reported these conflicts. Proposer errors also consumed
calls: two simultaneous replace/remove edits, two fitted coefficients in fixed
conversion fields, one no-change reply, and the truncated response. All latent
initializer declarations remained unchanged.

Two bookkeeping issues are distinct from these scientific failures. Reporting
now defines `checks_clear_scope` conditionally instead of asserting clearance for
every row. Separately, seed 1 event 007 changes only a trailing period in a
process description but resets the no-progress counter and changes its question
ID. That presentation-sensitive fingerprint remains a focused follow-up; the
saved experiment and its budgets must not be rewritten.

The offline audit verified 23 sealed files, all 17 call-to-event bindings and
event chains, final drafts, cost totals, and execution entrypoint hashes against
commit `587f99c`. Local review artifacts are in
`output/initialization-provider-recovery-analysis-20261017/REPORT.html` and
`AUDIT.json`, including exact request/reply pages. Input archive SHA-256:
`6399147b44cc34f59a2ca494763ab067d5d3e81980ac6eb8b91949231a295e70`.

Next: regress presentation-only progress offline and present each retained
witness as a concise model-specific equality before retesting the two unresolved
saved cases under an explicit new budget. No broader sweep or automatic
continuation has been launched.
