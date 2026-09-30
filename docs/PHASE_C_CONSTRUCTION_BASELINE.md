# Phase C construction baseline

This starts stage isolation while Phase B finishes endpoint processing. It is
a diagnostic baseline, not completion of the entire milestone or a final
comparison campaign.

## Scope

Use the qualified `phase-c-development-2` release, separately from Phase B:
corrected continuous input schedules and preparation contracts, including
alien-device training data. Preparation verifies the 28-cell publication receipt
and every selected public file. No private reference/diagnostic file is opened.
The delivery bundle contains the receipt and six selected public cells; it is
not a full reference archive.

The roster is named canonical Dalla T1-easy, T1-hard, T2-easy, T2-hard; named
CSTR-easy; functional alien-device-easy. Two seeds and Full/Brief-only give **24
fresh constructions and at most 24 fits**. The existing optional shared-process
stage and same-inventory fallback are enabled. Compiler/causality, generated
targets and public scientific admission rules precede fitting. Ambiguous
admission predicates remain distinct from passes.

Reuse the general shared constructor, reconstruction, causal initialization and
`collocation-multi-target-v1` fitter. The original 20B model/revision, low reasoning,
image and construction budgets are pinned in `configs/phase_c_construction_v1.json`.
Full gets training summaries; Brief-only omits them. Neither gets validation
observations, independent probe findings or reference equations. There are no
revision rounds, critic calls, pruning, test access or automatic follow-up.

The CSTR fitting improvements are not adopted here. Known-admissible-structure
fitting remains a separate foundation work item: NMSE alone cannot distinguish
proposer structure errors from optimizer limitations.

## Prediction and independent mechanisms

- NMSE: median over seeds per case, then macro median and unscaled MAD over cases.
  Terminal unavailable models receive positive infinity; pending work blocks
  aggregation. Two-seed medians are midpoints, not outlier rejection.
- Mechanisms: reuse the stronger fitted public tests from late Phase B, including
  fixed operating points, Radau/BDF agreement, and pass/fail/unresolved outcomes.
  Rebind only Dalla's declared `meal_event_g` to `meal_rate_g_per_min` channel rename.
  Public mechanism bullets must match exactly before preparation succeeds.

For K requirements, confirmed compliance is P/K; possible is (P+U)/K. Average over
seeds then cases; sample SD is across case means. JSON values are fractions.
Preserve P/F/U and availability counts. Unavailable models have unresolved
predicates (confirmed 0, possible 1). Retain the T2-hard identification limitation
even when its necessary proxy passes. These finite independent tests are stronger
than graph/annotation checks but do not certify universal scientific correctness.

`SUMMARY.md`/`summary.json` show both axes for every task. `EQUATIONS.md` includes
equations, processes, observation maps, causal initialization, fitted parameters
and independent findings. `assessment.json` retains the lowered model actually
probed. Review rejected drafts and failed constructions as well as fitted models.

## Prompt and response review

Each `results/<task>/TRACE.html` displays exact messages, schemas/settings, complete
provider responses, retries and observed runtime decisions. Text is escaped.
`trace.json`, `calls/` and `construction/` preserve the machine-readable evidence.
Calls are indexed by hash, **not time**; use step/attempt labels and stage artifacts
for context. Failed calls and unknown usage remain visible; keys are not logged.

Review variables, processes, topology, functions and initialization separately:

1. Did the displayed predecessor contain enough information?
2. Was the first reply scientifically adequate? Cite its actual equations.
3. Was it accepted, repaired or rejected, and which runtime rule acted?
4. Did repair change scientific content or only response syntax?
5. What do equations and independent probes reveal that NMSE misses?

Scientific adequacy and false rejection require review; acceptance is not an
automatic scientific label. The next stage-isolation experiment uses reviewed
predecessors at selected boundaries. Recurring failures should have reproducible
scientific, interface or numerical explanations before changing the runtime or
adding search. This pilot does not introduce another model-selection objective.

## ACES commands

From the extracted pinned source bundle with `SOURCE_COMMIT` and `public-release/`:

```bash
bash scripts/hpc/start_foundation_aces.sh closeout
bash scripts/hpc/start_foundation_aces.sh pilot
```

Closeout imports the original Phase B checkout and authorization: four CPU pruning
workers and one serialized CPU critic, with original six-hour budgets and `ac042`
excluded. It requests no search/GPU workers. More final reviews become ready as
pruning finishes. Successful allocations still require a completion audit.

The pilot requests one H100, then 24 CPU tasks (concurrency four), and a report.
Each CPU task fits one model and then independently assesses it, avoiding a second
queued array. `afterany` dependencies keep failures visible;
they do not treat failed jobs as successful. Phase B's existing critic uses the
Jetstream API; this first Phase C baseline requests no critic calls.

After completion:

```bash
bash scripts/hpc/start_foundation_aces.sh inspect-closeout
bash scripts/hpc/start_foundation_aces.sh inspect-pilot
```

Download the printed archives. Preserve source, calls and checkpoints; no cloud
files are deleted. These inspection archives contain development data/provider
responses and must not be committed to Git.

Repeated submission with the same wave reuses confirmed job IDs. Unknown scheduler
outcomes require reconciliation; do not delete intent files. If an allocation
finishes with pending work, inspect first, then submit `submit_phase_c_baseline.py
--root ROOT --wave construction-2`. Saved work and consumed budgets survive.
Interrupted fits do not silently receive another optimizer budget.

Delta/Jetstream VMs can inspect downloaded results without GPUs. This pilot is
assigned to ACES rather than duplicated across sites. Changing to managed API
inference requires a separately recorded model/serving intervention; it is not
equivalent to the pinned 20B experiment.

## Validation boundary

Only the public adapter, diagnostic coordinator/report, trace renderer, scheduler
wrappers and an explicit launcher dispatch are added. Existing scientific stages,
fit profiles, benchmark files and historical behavior remain unchanged. Tests
exercise real synthetic construction/fitting/probing, budget-preserving resume,
failure denominators, public bindings, escaped text and uncertain submissions.
Live serving, scientific adequacy and corrected-benchmark performance remain to
be assessed by this pilot.
