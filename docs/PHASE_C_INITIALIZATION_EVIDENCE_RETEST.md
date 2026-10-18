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
is 20 calls / 524,288 tokens. Resume cannot renew it. Transport failures remain
availability failures and stop the episode without advancing scientific repetition.
The returned run exposed a separate output-truncation classification defect below.
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
The returned run and remaining limitations are recorded below.

## Returned ACES audit, 2026-10-18

The inspected source is commit `755bf12c9e86a440740c9b6637d1cfe438b1ca16`;
the sealed plan is
`a27b053e91960866c99e3ff81d773786df160ba03d489708c03fc24be68143cd`.
Job 2226484 exited zero. The archive SHA-256 is
`f84a4cf83dd93b4301ca85db63a304cfd892e4c78e8557560172ccfeacb3f7d6`.
There were 20 recorded HTTP responses, 18 normal completions and two output-length
truncations, with 444,478 observed tokens and no unknown usage. No fitting or test
access occurred. Both saved models retain complete initialization declarations and
structural eligibility, but neither clears its remaining conditional conflict.

| T2-easy case | Calls | Tokens | Remaining conflict | Recorded stop |
| --- | ---: | ---: | --- | --- |
| Seed 1 | 10 | 209,653 | Initial U readout | Repair budget exhausted |
| Seed 2 | 10 | 234,825 | Closed insulin subsystem | Three repeated repairs, including two truncated replies |

These are latest-final repair episodes with additional budgets, not same-start
fresh-construction controls. Structural eligibility is not scientific correctness,
and a serialized interaction source is not permission to promote a model.
Event and request indices below are zero-based.

### Seed 1: equal initial arguments still imply equal U

The retained topology has `U = f(Uii) + g(X_insulin_action)` and
`X_insulin_action(0) = c`, shared across trajectories. Event 003 adds Uii as a
differential variable with `dUii/dt = h()` and its recorded initial value. The
relevant recorded Uii initial values are still both 1, whereas U(0) is
1.9890358776874768 versus 1.8350720852593723. Changing the later Uii dynamics
cannot distinguish those already equal initial arguments. This is the only
accepted substantive edit; event 002 is an accepted unchanged restatement.

The concise equality/numeric card appears in requests 000–003 and disappears
from 004–009. The accepted observed-initial rule is present identically in both
`fixed_runtime_rules` and `retained_handoff`; the display helper rejects every
duplicate, even consistent copies. The full conditional diagnostic remains.
In a copied packet, removing only the redundant handoff copy restores the same
question and correctly labeled numbers without changing the model or evidence.

Later replies attempt to make Uii algebraic but retain its differential initializer
implicitly through `initialization: null` (005–008), or both remove and replace an
equation (004, 009). All are rejected atomically. Requests 005 and 006 repeat the
same failed edit; request 007 adds an unchanged U equation, resetting the attempt
counter despite proposing the same resulting topology. The V5 initializer-only
normalization does not canonicalize these redundant topology records.

### Seed 2: changing the initializer mode adds no distinguishing information

All ten transactions retain the same differential topology. The insulin subsystem
depends only on I, I_mem and the supplied meal/infusion forcing; its memory state
depends on I and itself. Event 001 clears the initializer record, and initialization
event 002 replaces the previous `I_mem(0) = g(I(0))` by a shared constant. Neither
choice distinguishes the retained pair: both have I(0)=25.5878134642966 and matched
forcing through t=8, yet I(8) is 25.5878134642966 versus 48.42727474869684.
The shared-law, uniqueness and uncertainty qualifications still apply.

All nine repair requests contain the labeled numeric evidence. Concrete unsupported
observed-initial guidance appears in 001, 004, 006 and 007. Four normal replies
nevertheless propose an unavailable observed initializer (000, 003, 005, 006);
006 repeats it immediately after the specific guidance. This is not explained by
missing feedback alone. Accepted assumption-challenge text in 004 and 007 does not
clear the conflict or change the model.

Replies 008 and 009 end with `finish_reason=length`, each at 8,192 completion
tokens; neither contains a complete JSON reply. Their 60,649 total observed tokens
are correctly charged. However, HTTP `responded` enters the scientific repair
handler, so the counter advances from 1 to 2 to 3 and labels the termination as
repeated repairs. Those two responses are availability failures, not evidence of
two repeated scientific decisions. The episode had also reached its ten-call cap,
so this defect mislabels the stopping reason rather than reducing the allowance
actually used in this run.

## Next bounded milestone

Before another scientific comparison:

1. Reconcile identical runtime/proposer initializer rules for display, while
   continuing to reject genuinely conflicting rules.
2. Keep output truncations separate from scientific rejection and stagnation;
   preserve every attempt and charge without automatic budget renewal.
3. Canonicalize rejected edits by their actual proposed effect when validly
   computable, so redundant unchanged declarations cannot reset repetition.
4. Consider dropping only inherited initializer records when an explicit accepted
   variable edit makes them inapplicable. An explicitly conflicting new handoff
   must still receive precise feedback. This is bookkeeping, not a scientific
   choice, and would not itself solve either retained conflict.

Use saved packets for offline regression checks first. Further scientific changes
to readout/initial dependencies remain proposer-owned. No new runtime implementation,
model selection, live follow-up or scientific-success claim is included in this audit.
