# Saved initialization evidence-feedback retest

This bounded milestone follows the two-case equality-feedback audit. It changes
presentation and repetition bookkeeping, not public benchmark prompts, scientific
predicates, allowed initialization modes, or the fitting algorithm.

## Three corrections

1. **Numeric conditioning evidence.** A qualified equality card shows recorded
   output values and matched conditioning values with explicit channel,
   trajectory and time labels. Initial readout arguments are distinguished from
   initializer-map arguments and complete forcing histories. Numbers come from
   the sealed training evidence; missing or inconsistent evidence cannot become
   a fabricated measurement. Bounded history summaries retain their qualification
   and do not substitute a few samples for the existing complete-history check.
2. **Unsupported observed initialization.** Rejected structured declarations are
   checked against the current fixed runtime initial rules. Feedback names the
   attempted state, retained initializer, missing measured-initial binding and
   available initial information. It distinguishes a permitted shared constant or
   a scientifically chosen common map from an unavailable latent measurement.
   It neither invents a measurement nor chooses a scientific dependency.
3. **Semantic repetition.** Accepted initializer restatements and null have the
   same effect. Their raw submission syntax no longer resets the repetition
   counter. State-record ordering is immaterial; multiplicity, modes and arguments
   remain represented. Genuinely different rejected alternatives retain distinct
   identities. An actual accepted initializer change remains progress.

The opt-in protocols are `saved-topology-initialization-5` and
`topology-initialization-repair-5`. V1–V4 keep their exact historical request and
stopping behavior. The new numeric display is downstream of the existing qualified
check, and does not promote an assumption challenge into a scientific pass.
Initializer and topology transactions still commit atomically.

## Scientific boundary

For the saved seed 1 case, the new display must distinguish the equal supplied
`Uii(0)` pair from the unequal observed `U(0)` pair. For seed 2, an unobserved
`I_mem` cannot be turned into a measured state merely by selecting
`observed_initial`. These are examples motivating general evidence formatting and
structured feedback; no glucose-specific predicate or edit rule is introduced.

Under shared deterministic laws and the stated uniqueness/error assumptions,
equal initial information plus matched forcing cannot yield unequal trajectories.
Which scientific dependence, representation or justified assumption should change
remains a proposer decision. Function forms, signs, overlap, conservation and
fitted accuracy still require later evaluation.

## Experiment and provenance

Freeze the two explicitly selected V4 final models, including their current
initializer records and all retained witnesses. The freezer checks sealed source
artifacts, response hashes/settings, the complete event chain, final model and
cost agreement. Historical costs remain separate and nested predecessor scopes
are preserved. Earlier files are read-only.

Each case receives an explicitly new maximum of ten calls / 262,144 charged tokens
under the same pinned GPT-OSS-20B model and serving settings. Thus the two-case cap
is 20 calls / 524,288 tokens. Resume cannot renew it. Provider failures remain
availability failures and stop the episode without advancing scientific repetition.
This is a separately budgeted repair experiment on latest saved finals, not a
same-start causal comparison with the preceding prompt version.

One ACES H100 runs both cases serially. The source-pinned bundle includes the
training-only plan and offline Harmony vocabulary. A short CPU preflight submits
the GPU job. No fitting, test access, model promotion or automatic follow-up is
included. Jetstream2 is not required.

## Prepare and inspect

The local freezer command is:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/saved_initialization_aces.py freeze-evidence \
  --source output/initialization-equality-analysis-20261017/inspection \
  --root output/initialization-evidence-inputs \
  --task-id cell02_seed1_full_actionable_initialization_focused_equality \
  --task-id cell02_seed2_full_actionable_initialization_focused_equality
```

The delivered archive has a `run_retest.sh` wrapper, a source commit and a complete
SHA-256 manifest. Upload it to the ACES group scratch directory and verify the
manifest before launching. From its extracted directory:

```bash
srun --account=156264627414 --partition=cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=00:30:00 --export=ALL --exclude=ac042 \
  bash "$PWD/run_retest.sh" run
bash run_retest.sh status
bash run_retest.sh inspect
```

The output root is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-initialization-evidence-v1`.
Download its `inspection.tar.gz` when finished. Keep the same root and wave for
status/inspection; do not overwrite the prior equality campaign.

The generated prompt review includes exact first requests, marked V4→V5 changes
on the same saved models, and specific rejected-initializer examples. Those
examples use saved replies for offline inspection and are not predicted responses.
Live effectiveness remains untested.
