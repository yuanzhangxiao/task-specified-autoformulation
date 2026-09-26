# Experiment-section draft: scope and result handoff

Drafted 25 September 2026 against the attached ICLR 2027 PDF and the current
implementation. This is a **prospective test-evaluation manuscript**, not evidence
that evaluation has occurred. Protocol statements are written in paper style;
outcome-dependent benchmark test numbers and conclusions remain keyed `TBD`
fields. Section 5.4 now contains a separately labeled, measured exploratory
case study using assisted historical models.

## Files and integration

- `AUTOFORMALISM_EXPERIMENTS.tex`: section body covering setup, external baselines,
  ablations, and the familiar-model controls. It uses the paper's existing four
  bibliography keys and requires `amsmath`, `booktabs`, `graphicx`, and `natbib`.
- `AUTOFORMALISM_EXPERIMENTS_PREVIEW.tex`: standalone preview wrapper, including
  an explicit draft-status banner and a small bibliography. Compile from the
  project root. Its layout is a preview, not an ICLR style replacement.
- `output/pdf/autoformalism-experiments-draft.pdf`: generated preview, not source
  to commit or a finished test-results paper.

The source under `paper_to_revise/sections/experiment.tex` differs substantially
from the attached PDF (older benchmark roster and one-step/mean-SD reporting).
It and Orion's case-study files were left untouched. Replace the experiment
section with the new section body only after reconciling the paper's active
source and labels. Section 5.4 now contains the R9/no-specification/Sol case
study and its figure. Update appendix cross-references when integrated.

To populate a measured result, place a definition before the section input:

```latex
\providecommand{\AFSetResult}[2]{\expandafter\def\csname AFresult:#1\endcsname{#2}}
% Values below are explanatory tokens, not example measurements.
\AFSetResult{base.full.error}{<median> [<MAD>]}
\AFSetResult{base.full.mechanism}{<pass median> / <unresolved median>}
\AFSetResult{base.full.coverage}{<evaluated>/<planned>}
\input{docs/AUTOFORMALISM_EXPERIMENTS.tex}
```

Keys beginning `base`, `cv`, and `mem` name baseline, critic/verifier, and
familiar-model fields. Keys ending `sentence` require a result-supported sentence,
not just a number. Remove `TBD` statements from table captions and the preview's
draft banner only after those results exist. Never copy a validation value into
a test field. A visibly labeled validation-only version can be created separately.

## What changed from the attached Section 5

1. **Roster:** nine cases from three families, including all four canonical named
   Dalla Man T1/T2 easy/hard combinations. Seven cases are Dalla Man variants;
   they are not seven independent physical systems. The configured CSTR is named,
   contrary to the attached section's obfuscated-CSTR description.
2. **Statistics:** medians and unscaled MAD replace mean/SD or IQR as the default
   summaries. Per-case seed variation and variation across case medians are distinct.
3. **Main ablations:** critic/verifier 2-by-2 on nine cases. Sharing and pruning
   are secondary matched comparisons on the four named T1/T2 easy/hard cases.
   Full and Brief-only remain separate; no post-test prompt selection.
4. **Mechanism claims:** equation-predicate satisfaction is a scoped formal
   metric, not a percentage of scientific truth. Post-fit activity and response
   behavior remain separate. LLM plausibility scores do not provide assessment labels.
5. **Memorization claims:** use naming/dynamics controls to test reliance on
   familiar cues. Neither an error gap nor good perturbation performance proves
   whether training examples were memorized. The alien device is an additional
   domain, not proof that every constituent function is unfamiliar to an LLM.
6. **Legacy conclusions removed:** the existing figures explicitly show
   validation-selected rounds 9--12. Their NMSEs, success counts, token numbers,
   factor improvements, and ablation effects are not fresh final-campaign test
   results. Directional claims have therefore become placeholders.

The conclusion on PDF page 10 also claims reliable task-critical recovery and
better held-out-intervention accuracy. Revisit that wording after final results;
this request did not modify the conclusion or Orion's case study.

## Frozen protocol versus unfinished reporting

The nine-case search configuration has 2 seeds, 2 prompt variants, and 4
critic/verifier arms: **144 primary lineages**. Sharing-off adds 16 lineages on
the four named Dalla Man cases. Each lineage has 15 visits (initial plus 14
revisions), totaling 2,400 planned search rows. Pruning alternatives are endpoints
of the same search, not separate searches. The user's latest report still had
741 missing visits and 192 missing endpoint rows; it was not frozen for test.
This draft does not change the roster, budgets, or freeze policy.

Before replacing the placeholders:

- Complete and seal the planned endpoint roster, or explicitly revise the study
  scope prospectively; do not select an available subset by its results.
- Confirm external-baseline coverage against the exact nine cell IDs, prompt/data
  identities, and causal input permissions. Their source plan uses three
  repetitions, while ours uses two seeds. Print actual denominators; do not
  claim equal repetitions or equal token/optimizer budgets.
- Use each baseline's native fitted model. SINDy/PySR use native discovery and
  validation selection; GPT-5.6 Sol supplies its fitted constants. An optional
  structure-plus-common-fitter experiment is a separate comparison, not the
  primary agent result. D3 is native discrete increment rollout with **no extra
  time-step multiplier**. Older incompatible D3 sources cannot fill missing cells.
- Inspect scientific-check coverage. `mechanism_assessment_v1.json` supplies
  reviewed three-layer bindings for the **six easy cells**, not all nine.
  Additional T1-hard/T2 easy/hard bindings need a prospective public-specification
  assessment before filling a common nine-case mechanism table. Existing broader
  graph certificates are not interchangeable with the three-layer assessment.
  Any absent binding is unavailable evidence, not a pass or a zero score.
- Apply the same reporting predicates to all methods and all verifier treatments.
  For a required but unsupported predicate, preserve unresolved status. A
  predicate that is genuinely not an obligation is excluded by the public
  assessment contract, never by inspecting the method's performance. D3's native
  semantics must be respected by every applicable check.
- For post-fit activity, distinguish conditional model sensitivity from a
  physically realizable intervention. The numerical three-layer tool currently
  uses development data. A test-response extension is not already implemented
  merely because the main draft describes test NMSE.
- Reconcile token logs, unknown usage, and shared initial-call attribution.
  Missing usage is not zero. Tokens are counts across different model families,
  not dollars or an equal-compute budget. A no-critic run still pays proposer cost.

## Aggregation rules to use when filling tables

For metric `z`, method/prompt `m`, case `b`, and repetition `r`, compute:

1. `case_median[m,b] = median_r(z[m,b,r])` and the **unscaled** seed MAD.
2. `macro_median[m] = median_b(case_median[m,b])`.
3. `macro_MAD[m] = median_b(abs(case_median[m,b] - macro_median[m]))`.

The headline is `macro_median [macro_MAD]`, with the nine cases equally weighted.
Do not pool all rounds, all seeds, or Full and Brief-only. Only a frozen endpoint
per run enters the primary comparison. The appendix should retain individual
seed values: two seeds give a very limited estimate of stochastic variability.
Neither seed MAD nor cross-case MAD is a standard error or confidence interval.

Formal compliance for a model is `pass/(pass+fail+unresolved)`; unresolved rate
uses the same denominator. Compute case medians before the macro summary.
Do not use `pass/(pass+fail)` to hide unresolved evidence. Table 1 shows median
pass and unresolved percentages; retain both MADs and underlying counts in the
appendix. Complete workflow execution is not itself a scientific pass.

For NMSE, retain confirmed failed runs at **worst rank**, represented internally
as `+infinity`, **before** taking either median. This includes recorded model
failures, exhaustion of the method's frozen search budget, and failed target
rollouts. A complete missing case due to such failures is therefore included,
not dropped. The exported indicators are `target_status=failed` or
`terminal_status` in `{failed, timed_out}` when no NMSE was recorded. Keep the
recorded reasons; this is an operational failure classification, not a diagnosis
of its root cause. Large finite NMSEs remain finite observations.

This reporting rule supersedes the earlier available-only policy at the user's
request. A median of `+infinity` is displayed as **Failed** and its MAD is
undefined; do not calculate `infinity-infinity` or report zero spread. With a
finite median, include failures' infinite deviations in the ordinary unscaled
MAD. Report evaluated/planned and failed counts alongside every summary.

Unknown outcomes (e.g. an unexported result or interrupted worker) are not
evidence of model failure. Any such unknown prediction in a planned repetition
makes that primary case/macro statistic unavailable pending resolution. Do not
silently shrink its repetition denominator. A finite recorded NMSE remains an
observation even if a separate runtime contract check blocks graph assessment.
Graph evidence and complexity retain their own availability; this NMSE ranking
policy does not impute unmeasured predicate results or model size. Available-only
secondary metrics must state their coverage and conditional interpretation.

For paired ablations, first calculate per-block differences or ratios at the same
case, seed, and prompt; then aggregate within cases and across cases. Report
pair coverage as well as planned pair count. Keep both prompt variants separate.
For memorization, take a ratio within a matched repetition before computing its
median/MAD; a ratio of medians is not the same statistic. Zero-denominator ratios
are undefined and should be accompanied by raw NMSEs, not a tuned epsilon.

Because seven of nine cases come from Dalla Man, add a **descriptive**
family-balanced sensitivity analysis: compute each family's median of case
medians, then the median of the three family values. This is a reporting addition
to implement, not a score already present in the campaign files. Do not treat
correlated presentations as independent physical systems or report significance
from nine independent-domain trials.

## Recommended appendix tables

| Table | Rows and columns |
| --- | --- |
| Per-case external comparison | Nine cases x methods/prompts; NMSE median [MAD], formal pass/unresolved median [MAD], complexity, evaluated/planned repetitions, failures |
| Critic/verifier contrasts | Per prompt and case; both/verifier-only/critic-only/neither, paired error changes, formal checks, proposer/critic tokens separately, pair coverage |
| Sharing and pruning | Four named T1/T2 easy/hard cases x prompt; sharing on/off; no-prune parent, unchanged-refit control, proposed pruned child, validation-selected endpoint; test NMSE, removed terms/parameters/states, decision/skip status |
| Familiar-model controls | Canonical/perturbed x named/obfuscated; raw per-seed errors, paired ratios, solver/model failures, formal compliance |
| Reproducibility | Models/revisions, native baseline selection rules, fit and call caps, repetition counts, software/hardware, data and endpoint hashes, missing usage |

Do not retrofit the historical no-specification, no-latent, or refit-only runs into
the fresh component matrix. Those used different search protocols and coverage.
If retained, label them development-only exploratory results. Orion's case study
can address specification benefit without pretending those older runs are a
matched fresh test ablation.

## Interpreting familiar-model controls

The controlled factor is the named/obfuscated **presentation**, not a clean
intervention on memorization. Scientific names can supply useful prior knowledge
without verbatim retrieval. Dynamics perturbation also changes numerical
difficulty. Verify paired data hashes and schedules, compare numerical-only
controls, and report the four cells before any narrative about retrieval.

An equation-level audit of unsupported canonical terms in initial proposals
could strengthen a retrieval interpretation. It would need a prespecified
reference and scoring rule, matched prompts, and proposal artifacts. This is an
optional follow-up, not an analysis already conducted or claimed in the draft.

## Implementation sources checked

- `configs/final_component_campaign_v1.json` and `docs/FINAL_COMPONENT_CAMPAIGN.md`
- `docs/PROCESS_AWARE_PRUNING.md`
- `docs/DETERMINISTIC_MECHANISM_ASSESSMENT.md` and `configs/mechanism_assessment_v1.json`
- `docs/EXTERNAL_BASELINE_FROZEN_TEST_EVALUATION.md` and its v2 config
- `docs/RAW_DATA_AGENT_BASELINE.md`, `docs/REVIEW_BASELINE_COMPATIBILITY.md`
- `docs/PHASE_B_EXACT_BENCHMARK_PROTOCOL.md`, `docs/PHASE_B_SEMANTIC_PROMPT_AUDIT.md`
- Attached paper, Section 5 and the conclusion on pages 7--10

No fitting, LLM requests, scheduler submissions, benchmark edits, or test-data
access are performed by this writing task.

## Original draft verification

The standalone section compiles to three pages with resolved citations and table
references. All three pages were rendered and inspected for clipping and layout.
The relevant campaign and mechanism-assessment tests passed (43 tests).
`ruff check .` reports 37 pre-existing findings in unrelated `analysis/claude`
files; those files were not changed. No implementation changes were made.

Build the preview from the project root with:

```bash
mkdir -p tmp/pdfs/experiment-draft output/pdf
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=tmp/pdfs/experiment-draft \
  -jobname=autoformalism-experiments-draft docs/AUTOFORMALISM_EXPERIMENTS_PREVIEW.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=tmp/pdfs/experiment-draft \
  -jobname=autoformalism-experiments-draft docs/AUTOFORMALISM_EXPERIMENTS_PREVIEW.tex
cp tmp/pdfs/experiment-draft/autoformalism-experiments-draft.pdf output/pdf/
```

## Added interventional-discrimination case study

Section 5.4 compares assisted Brief-only R9, the historical no-specification
endpoint, and Sol repetition 0 on canonical-obfuscated T1-easy. Sol 0 is selected
by its original validation NMSE, not by the new intervention results. All three
Sol repetitions appear in the response-robustness table and supporting plots.
R9 is explicitly assisted; seeds/search budgets differ and this is not the new
matched component experiment. No new fit, rollout, provider call or test-data
access is involved in producing the section.

The section distinguishes absolute trajectory NMSE from error in the predicted
intervention effect. R9 has smaller maximum response error on the five saved
contrasts than each Sol repetition, but worse maximum absolute NMSE. The
no-specification model cannot respond to initial tissue-glucose changes because
its equations and initializers omit every auxiliary channel. T1 does not
explicitly require Gt, so this is not presented as a formal requirement failure
or a causal estimate of the specification's benefit.

Generate figures and portable data from the existing saved artifacts:

```bash
PYTHONPATH=src:. MPLCONFIGDIR=tmp/matplotlib .venv/bin/python \
  scripts/build_r9_paper_case_study.py \
  --output output/pdf/assisted-r9-case-study
```

The builder verifies 135 sealed saved trajectories, recomputes every plotted
NMSE/response score, checks reference/time-grid agreement, and selects Sol using
original validation only. It produces five figures, each as PDF/SVG/PNG:

- `r9_case_study`: main six panels (two training examples, both absolute
  tissue-glucose interventions, both own-control response differences).
- `r9_all_train`: all 16 training trajectories and all Sol repetitions.
- `r9_all_validation`: all four original validation trajectories.
- `r9_all_absolute`: all seven intervention/control trajectories.
- `r9_all_responses`: all five intervention contrasts.

`curves.csv`, `response_curves.csv`, `scores.csv`, `summary.csv` and
`plot_data.json` retain full values. `manifest.json` records source hashes and
selection rules. Override `\AFCaseStudyFigures` in LaTeX if figures are moved.
The bundle includes the existing plotting helper needed by the builder; using
`--replot` requires only its exported JSON, Python, NumPy and Matplotlib.

The plots preserve original input semantics: the historical public training
set includes meals at t=0, while the exploratory meal probes use pulses after
t=0. No input interpolation, reference data or initial conditions were changed
to improve the displayed fit. Main training examples are the same single- and
double-meal schedules used in the previous figure; the entire training grid is
provided to show the remaining errors.

The updated five-page preview compiles with resolved references and no
overfull/underfull box warnings. All pages and exported figure layouts were
visually inspected. Fourteen relevant tests pass; the full test suite was not
rerun. The added Python files pass Ruff; repository-wide Ruff still has 37
pre-existing findings in `analysis/claude`.
