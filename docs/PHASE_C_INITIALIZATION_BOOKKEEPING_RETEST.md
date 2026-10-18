# Saved initialization bookkeeping retest

This milestone corrects four runtime defects exposed by the two-case V5 ACES
retest. It introduces `saved-topology-initialization-6` and
`topology-initialization-repair-6`. V1–V5 keep their historical request, acceptance,
and stopping behavior. No benchmark data, public requirements, allowed initializer
modes, fitting algorithm, or scientific compatibility predicates change.

## Runtime changes

1. **Preserve evidence cards.** An identical initializer rule can occur in the
   runtime's fixed-rule table and the retained handoff. V6 recognizes that exact
   agreement for presentation only. Conflicting rules and duplicate proposer
   records still fail closed. Current model and handoff records are not rewritten.
   The saved seed 1 requests after its `Uii` declaration regain their qualified
   equality and numerical evidence cards.
2. **Separate incomplete output from scientific repair.** An HTTP response is
   insufficient when its final output is truncated, absent, or unparseable. V6
   records the unavailable reply, preserves the draft and initializer, and stops
   without advancing scientific stagnation. All measured usage remains charged;
   unknown usage retains its reservation. Resume does not retry or reset the
   budget. Complete JSON with invalid scientific/schema edits still receives
   ordinary repair feedback.
3. **Compare rejected initializer attempts by topology effect.** Once a proposed
   topology has assembled and passed public-binding validation, an invalid
   initializer cannot commit it. V6 fingerprints that assembled alternative so
   an unchanged topology restatement does not make the same rejected attempt look
   new. Distinct dependencies, signs, multiplicities, variable types and
   initializer choices remain distinct. Earlier assembly failures keep the
   existing rejected-payload comparison.
4. **Remove only unambiguously obsolete inherited initializers.** With
   `initialization=null`, explicitly removing a differential variable or making
   it algebraic also removes its inherited initializer record. A receipt lists
   affected states. An explicitly submitted replacement handoff is validated as
   written; conflicting new choices are not silently dropped. Unrelated stale
   records remain errors. The entire transaction still commits or fails together.

The minimal system instruction remains unchanged. The editing contract describes
inherited-record cleanup. The prompt review includes exact first requests,
marked V5-to-V6 changes on the same saved starts, and saved-reply examples.

## Scope and interpretation

Neither historical model has resolved its qualified scientific conflict. In seed
1, the observed initial values of `U` differ while the current arguments of its
algebraic readout agree. In seed 2, the declared common initial information and
matching forcing still imply identical insulin trajectories under the displayed
assumptions. The runtime supplies the witnesses; the proposer must choose a
scientifically justified change. Better bookkeeping is not evidence of better
scientific recovery until the new responses are inspected.

The freezer verifies the V5 event chain, cached requests, final models, handoffs
and all charges. The two truncated seed 2 outputs remain in historical accounting
but supply no scientific edit or carried feedback. Historical labels and counters
are retained as recorded, including the V5 classification defect. Nested earlier
cost scopes remain separate. All previously seen qualified witnesses are retained.

The retest starts from the two latest V5 final models. Each receives at most ten
new calls and 262,144 charged tokens: at most 20 calls and 524,288 tokens overall.
This is a separately budgeted diagnostic experiment, not a same-start causal
comparison or a new benchmark campaign. One ACES H100 runs the cases serially.
No fitting, test-data access, model promotion or automatic follow-up is included.

## ACES operation

The source-pinned delivery includes the frozen public-only inputs, offline Harmony
vocabulary, source commit and SHA-256 manifest. Upload to group scratch, not home.
From the extracted bundle directory:

```bash
sha256sum --quiet -c MANIFEST.sha256
srun --account=156264627414 --partition=cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=00:30:00 --export=ALL --exclude=ac042 \
  bash "$PWD/run_retest.sh" run
bash run_retest.sh status
bash run_retest.sh inspect
```

The CPU allocation verifies local dependencies and submits the H100 job. The new
output directory is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-initialization-bookkeeping-v1`.
Download its `inspection.tar.gz` after completion. Preserve the same output root
and wave for status/inspection; do not overwrite older campaigns. An unavailable
response deliberately stops this bounded episode and requires inspection before
any separately authorized continuation.
