# Readable initializer-choice retest

This opt-in milestone changes presentation for the two unchanged saved initializer
cases. It does not change topology, scientific predicates, initialization modes,
fitting, benchmark prompts, data, or historical request replay.

## Returned V1 diagnostic

The ACES run completed four requests and 56,373 observed tokens. Both independent
completion probes stopped normally and passed the schema; their replies were not
applied as model edits. Both initializer choices reported insufficient information
and retained the existing model. These outcomes establish response availability,
not scientific recovery. The server shutdown after the worker exited was cleanup.

Seed 1 treated `rule=null` as deleting an initializer, although it means no edit.
Seed 2 treated the retained shared initial value as an immutable requirement.
Both requests contained conditional equality evidence, but the request displayed
only its conditioning values rather than all permitted initial measurements.

The earlier explicit choices were made in initialization calls, not variable
construction: `initialization_000` for `X_insulin_action` and
`initialization_002` for the retained `I_mem` choice. Their initial requests had
training highlights and initialization feedback; they did not present the exact
later focused conflict in the same form. In particular, general observations must
not be described as an already displayed proof against a not-yet-chosen rule.

`U` in seed 1 is an algebraic readout of `Uii` and `X_insulin_action`, rather than
an independently initialized state. The equal initial arguments cannot produce
different initial `U` values under shared deterministic laws. This pointwise
argument does not need ODE uniqueness. In seed 2, `I` has its own measured initial
value. The later divergence concerns the closed `(I, I_mem)` subsystem with
matched initial conditions and forcing; unique solutions are relevant there.

## Changed requests

Protocol: `saved-initializer-choice-diagnostic-2`.

- Present one conditional model/data argument, readable model equations, and a
  table of all permitted recorded initial quantities for each witnessed pair.
  These values come only from sealed training evidence and index zero. They do
  not determine which scientific dependence to choose.
- Separate fixed observed-initial rules from the current editable latent rule.
  Explain that `shared_parameter` is one fitted constant shared across trajectories;
  `common_map` shares a function but can produce different trajectory initials.
- State that null preserves the current initializer. An abstention leaves the
  conflict unresolved; a legal map does not establish a correct scientific model.
- Do not repeat the full structured compatibility-question payload. Keep its
  structured evidence in the archived source and recheck every retained witness.
- Keep the uniqueness premise for propagation over time, but omit it from the
  initial-readout argument. Explain vector-valued dynamics explicitly.

The same two models and proposer seeds are retained. Each case permits at most
two initializer-only attempts, with 65,536 charged tokens per case: **four new
requests / 131,072 tokens total**. The successful completion probes are not
repeated. The entire previous diagnostic, all responses, tokenizer records,
events, and costs are embedded separately and verified by offline cache replay.
No historical cost is refunded. This is a bounded diagnostic, not a controlled
causal comparison or permission to start a larger campaign.

The new plan binds the executing source, exact request provenance and budgets.
An unavailable response terminates its unit. Resume reconstructs the request and
reuses the durable cache without resending an uncertain attempt. The candidate
initializer is saved separately; source models remain unchanged.

## ACES operation

Upload the source-pinned `phase-c-initializer-readable-<commit>.tar.gz` bundle to
`/scratch/group/p.nairr260351.000/u.yx126462`, extract it, and run from its directory:

```bash
sha256sum --quiet -c MANIFEST.sha256
srun --account=156264627414 --partition=cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=00:30:00 --export=ALL --exclude=ac042 \
  bash "$PWD/run_diagnostic.sh" run
bash run_diagnostic.sh status
bash run_diagnostic.sh inspect
```

The CPU preflight submits one H100 allocation. Its compiler cache is node-local.
The fresh output root is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-initializer-readable-v1`.
Preserve its root and submission wave for resume. Download its `inspection.tar.gz`
after completion. No live result from this retest is available yet.
