# Initializer choice and output-completion diagnostic

This is a separate diagnostic following the V6 ACES bookkeeping retest. It does
not change the production construction prompts, historical repair policies,
scientific predicates, benchmarks, or fitting. The two unchanged saved final
models remain the source of all requests and qualified training witnesses.

## Evidence motivating the experiment

The V6 retest used five physical calls and 118,450 observed tokens. Seed 1
returned three complete repetitions of an unsupported observed initializer;
seed 2 returned that invalid choice once and then an incomplete final response
with `finish_reason=length`. The latter consumed its 8,192-token output allowance.
All five requests displayed the qualified equality evidence. V6 correctly
retained both models, charged every call, and separated unavailable output from
scientific repetition. No new runtime defect was established by the audit.

The remaining questions are narrower than full topology repair: can the proposer
choose a legal, scientifically justified initializer when its response scope is
small, and does the previously incomplete request produce complete output when
repeated? These are distinct measurements. Neither proves model correctness or
identifies the cause of an output failure.

## Frozen scope and budgets

Protocol: `saved-initializer-choice-diagnostic-1`.

1. **Output completion:** two independently planned repeats of the exact last
   incomplete seed-2 HTTP request, including messages, schema, seed, temperature,
   reasoning effort and output allowance. New cache namespaces keep the attempts
   separate from historical calls and from each other. The replies are checked
   for completion and schema validity only; no scientific edit is applied.
2. **Initializer choice:** each of the two saved cases receives at most two
   attempts. The response may propose `shared_parameter` or `common_map` for its
   sole editable differential state, or explicitly report that topology changes
   or additional information are needed. Those latter decisions leave the
   scientific conflict unresolved. Every variable, process, contribution and
   fixed measured-initial rule remains frozen.

The maximum is six new physical requests and 196,608 charged tokens: 32,768 for
each completion probe and 65,536 for each initializer case. The serving model
remains pinned GPT-OSS-20B, with the existing 8,192 output-token allowance. Actual
serving-token preflight remains mandatory; requests are never silently truncated.
The units run serially on one ACES H100. An unavailable response terminates its
unit; the other independently planned units may still run. There is no hidden
retry, automatic follow-up, fitting, test access, or model promotion.

This changes both response scope and prompt presentation. It is a diagnostic,
not an equal-budget causal comparison of prompt wording.

## Prompt and interpretation

The focused request retains the complete model view, public scientific context,
variable meanings, requirements, initializer contract, and all pending qualified
equality cards and their witnesses. It omits the broad topology-edit schema,
global training statistics, and nonpending review payloads, explicitly recording
the projection. The complete historical source remains archived. On the current
two starts, the serialized user packets are about 20–22 KB instead of 70–73 KB;
these are byte sizes, not model-token counts.

The proposer owns scientific argument selection. The runtime only validates the
declared dependency rule against available initial information, preserves fixed
rules, and recomputes all retained compatibility witnesses. A valid dependency
change is saved as a diagnostic candidate in a separate result directory. A
deferred check is not a passed check, and an initializer choice alone is not an
accurate recovered model. Formulas, parameter values, and numerical fitting are
outside this experiment.

## Provenance and resume

The input freezer replays the original V6 requests and events with provider and
tokenizer transports blocked. It preserves the final models, handoffs, all seen
witnesses, and nested historical cost scopes. The sealed diagnostic plan binds
source and entrypoint hashes, exact source calls, roster, budgets, and the failed
request selected for output-completion probes.

Each new call is durably cached before its outcome is processed. Events and
results bind the corresponding request and response hashes. Resume reconstructs
the same predecessor and request; an interrupted HTTP attempt remains uncertain
and charged, without being resent. Missing or mismatched call/event provenance
fails before a new generation request. Historical costs are never refunded.

## ACES operation

Upload the source-pinned bundle to group scratch. From its extracted directory:

```bash
sha256sum --quiet -c MANIFEST.sha256
srun --account=156264627414 --partition=cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=00:30:00 --export=ALL --exclude=ac042 \
  bash "$PWD/run_diagnostic.sh" run
bash run_diagnostic.sh status
bash run_diagnostic.sh inspect
```

The CPU allocation verifies Python imports, the existing container, and the
offline Harmony vocabulary before submitting one H100 job. Node-local compiler
caches avoid adding thousands of files to scratch quotas. No new model download
is requested. The output directory is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-initializer-choice-v1`.
Download its `inspection.tar.gz` after completion. Preserve the output directory
and submission wave for deterministic resume.

The report separates availability from scientific status. A Slurm failure can
mean one preplanned request had unavailable output even when the other units
completed; inspect both the per-unit table and logs. Live diagnostic results are
not available yet.
