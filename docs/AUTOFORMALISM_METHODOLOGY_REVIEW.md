# Review and manuscript handoff

## Current revision: Section 3 notation, conditional weights and algorithm (25 September 2026)

Reviewed the supplied `Downloads/problem_formulation.tex` and `Downloads/method.tex`
against the current fitting, verification, critic, revision and pruning paths.
The new Method uses `\mathcal M=(\Gamma,f,h,g;\psi)` throughout (923 prose words
and five numbered equations by `texcount`). It is identical
in `docs/AUTOFORMALISM_METHODOLOGY.tex` and `paper_to_revise/sections/method.tex`.
The supplied Section 3 and all implementation/experiment files are unchanged.
Local source copies are preserved in `backups/method-section-2026-09-25/`.

The main changes are:

- MAP is converted to a negative-log objective, then to a working constrained
  error/complexity surrogate. Training parameter fitting, validation selection,
  advisory critique and local pruning are distinguished from exact optimization
  of a posterior or one fixed weighted objective.
- Equation (5) now compares integrated predictions through the observation map
  `h` with target observations, pooled over trajectory/time/channel samples and
  normalized with training-only channel scales. It does not equate latent state
  coordinates with outputs or treat the full recorded data as joint noisy-state
  likelihood terms that the fitter does not use.
- Equations (6)--(8) expose an affine block of GMM weights. The analytical
  pseudoinverse solution is conditional on trajectories, their derivatives and
  all other parameters; signs/bounds require constrained least squares. Unknown
  products across nested laws are not jointly linear. The response vector is a
  derivative residual, not the observed output matrix. The production fitter
  uses collocation initialization and joint bounded rollout refinement, without
  an alternating analytical weight step or sparsity penalty.
- The deterministic gate precedes fitting and includes actual equation-derived
  scientific predicates. Explicit violations block fitting; unresolved public
  predicates are not falsely called scientifically certified. The paired judge
  supplies broader, fallible scientific critique without a numerical selection
  score or veto. Scientific interpretation is not claimed to be exclusively
  possible with an LLM.
- The main text summarizes shared-process assembly, causal initializers,
  revision and pruning. The new appendix table gives the full algorithm and
  exact search/pruning selection rules, including parent/control comparison.

### Manuscript integration

The current manuscript already inputs `sections/method.tex`. Add the following
inside its appendix when incorporating the new algorithm table:

```latex
\input{sections/appendix_method_algorithm}
```

The table requires `amsmath`, `amssymb`, `booktabs` and `tabularx`, already used by
the current manuscript. The manuscript main file and other untracked paper files
were not edited or added to this commit. In particular, the existing pruning
appendix still describes bootstrap pruning and needs separate alignment before
submission; the new table describes the implemented one-deletion comparison.
The preview uses Section 4 and equations (4)--(8), assuming the three equations
in the supplied Section 3 precede it. Actual manuscript numbering remains automatic.

### Verification and limits

- 113 relevant tests pass across component campaigns, review integrity,
  pruning, critic, multiple targets, fitted initialization, public target
  contracts and the restricted exact-derivative fitting path.
- The fresh-shared smoke test checks synthetic construction, two-output fitting,
  shared-law revision, pruning and deterministic resume without live LLM calls
  or benchmark/test trajectories.
- The standalone Method/appendix PDF compiles without unresolved references or
  overfull boxes; all three pages were rendered and visually inspected. The main
  section occupies about one and a half pages in the generic two-column preview,
  with the algorithm appendix on its own page.
- `ruff check .` reports 37 pre-existing issues in unrelated `analysis/claude`
  files. This revision changes no Python implementation files.
- No experimental efficacy or global MAP convergence is inferred from these
  code checks. The conditional analytical formula is a property of an affine
  subproblem, not a claim that every production fit is solved analytically.

The dated reviews below describe earlier drafts and remain historical context.

## Earlier revision: concise MAP-motivated Section 4 (24 September 2026)

`AUTOFORMALISM_METHODOLOGY.tex` now matches
`paper_to_revise/sections/method.tex`. The latter is the file included by the
manuscript's Section 4. Both contain the same short section with six displayed
equations (558 prose words by `texcount`); the preview wrapper numbers the section
as 4 and renders one page in its generic two-column layout. The previous untracked
manuscript section was preserved locally at
`backups/method-section-2026-09-24/method-before-map.tex` before replacement.
`AUTOFORMALISM_ALGORITHM.tex` remains unchanged.

The derivation distinguishes exact algebra from implementation choices:

- MAP supplies the starting factorization. A chosen working prior encodes
  implemented admissibility and structural complexity; it is not an estimate of
  all scientific plausibility.
- Negative logarithms yield a constrained regularized objective. Conditional
  parameter fitting uses training residuals without an L1 sparsity penalty.
- Validation-based ranking, finite-budget fitting, numerical ties and local
  pruning are stated approximations or policies, not exact MAP identities.
- Critic findings condition the proposal distribution. Critic scores do not
  become a posterior energy or a selection reward.
- Shared laws remain single definitions reused across consumers. Pruning tests
  one training-ranked deletion against the parent and an equal-budget refit.

The older method's figures were omitted from this short section because their
captions described judge-weighted selection and closed-form fitting. No other
manuscript section or figure was changed. Before submitting the whole paper,
align the abstract, introduction, comparison table in Section 3, conclusion and
pruning appendix: they still contain claims about an LLM energy prior,
closed-form fitting, Takens-style surrogate states or bootstrap pruning that do
not describe this pipeline. Existing experiment sections must also identify
which historical protocol produced their results.

Verification for this documentation revision:

- Standalone `pdflatex` compilation passes, with no unresolved references or
  overfull boxes; the final one-page PDF was rendered and visually inspected.
- `pytest` passes all 77 tests in `test_component_campaign.py`,
  `test_review_integrity.py`, `test_process_pruning.py`,
  `test_process_pruning_campaign.py` and `test_general_critic.py`.
- `scripts/smoke_fresh_shared.py` passes construction, multi-output fitting,
  pruning and deterministic resume without live LLM calls or test access.
- `ruff check .` reports the existing 37 issues in unrelated `analysis/claude`
  files; no Python implementation files changed.
- Full-manuscript `latexmk` reaches BibTeX but stops on the pre-existing duplicate
  `lecun2006tutorial` entry at line 303 of `autoformulation_dynamics.bib`.
  That bibliography is outside this edit. The standalone section builds cleanly.

The dated review below describes the earlier, longer methodology draft. It is
retained as historical context; its length and draft-specific instructions do
not supersede this revision.

## Earlier review: 23 September 2026

Reviewed 23 September 2026 against checkout `2f47b81b347e9f069b49041d933b21ad8ed0dd46`.
The original `AUTOFORMALISM_ALGORITHM.tex` remains unchanged.

## Assessment of the original

The original is a useful implementation reference, with clear causal boundaries,
output-only loss definitions, and careful distinctions between numerical success
and scientific validity. Its own scope is explicitly historical: the September 17
document primarily describes the v3–v5 review campaigns at `a067b835`, followed by
a separately versioned hybrid judge controller. It is not a current paper-length
account of every subsequently implemented module.

The main changes needed for a methodology section are:

| Topic | Treatment in the concise section |
| --- | --- |
| Length and focus | Keep model representation, construction, checks, fitting, feedback, selection, and pruning. Move solver constants, retry mechanics, judge scoring tables, protocol history, operational pseudocode, and source listings out of the main text. |
| Shared processes | Include the optional phase between variables and topology, a single law reused by consumers, declared signs/conversions, gain assembly, and coherent edits. This is more than reusing a variable name or independent functions on matching hyperedges. |
| Pruning | Describe the actual one-deletion, training-contribution ranking and paired refit procedure. Include the vector complexity criterion and the explicit 1% validation tolerance. Do not describe thresholding raw coefficients or repeated pruning to convergence. |
| Scientific critic | Describe the Milestone 4 advisory use of the established paired judge. Its score does not enter current validation selection or override a failed deterministic requirement. The older hybrid weighted-selection policy is a distinct protocol, not an implicit part of this method. |
| Public requirements | Distinguish violated, satisfied, and unresolved predicates. Some unresolved candidates remain eligible for development selection; automatic pruning is stricter and skips unresolved obligations. “All selected models are scientifically verified” would overstate the implemented contract. |
| Multiple outputs | Use channel-indexed loss notation. The repository now has `collocation-multi-target-v1`; this does not retroactively make the earlier single-target campaigns multi-output experiments. |
| Numerical interpretation | Preserve open rollouts, training-only fitting/initializers, fixed training scales, and conditional residual feedback. A timeout, executable graph, or successful replay is not proof about scientific correctness. |

The original's lengthy mathematical fitting and judge descriptions can supply
supplementary material, but their version boundaries must be retained when moving
them. In particular, the old judge-weighted selection formulas should not appear
as the selector for the newer advisory-critic experiments.

## Scope of the new section

`AUTOFORMALISM_METHODOLOGY.tex` is a drop-in section with five subsections. It
describes the implemented components and labels shared processes, pruning, and
critique as configurable. It makes no efficacy claim from the development pilots.
It is not evidence that a fresh end-to-end campaign has already run with every
module enabled.

Before attaching an experiment table, specify its actual configuration: proposer
and judge models, reasoning settings, seed count, construction/revision budgets,
fitter profile, number of rounds, and enabled modules. Preserve these distinctions:

- The earlier 15-round results were produced across versioned development
  campaigns. They are not results of a fresh run of the entire new method.
- Milestone 2 used general construction and one revision; Milestone 3 added a
  separate pruning comparison; Milestone 4 started from its selected models and
  added advisory critique plus one revision. These are sequential development
  pilots, not a matched ablation of all components.
- Whole-model revision minimizes validation error, then term count. Pruning alone
  uses the stated simplification tolerance and a larger complexity vector.
- `full` and `brief_only` remain initial-information variants. Neither is declared
  the winning default. Both receive training residual evidence during revision.
- The critic's established rubric and the new use of its advice for revision are
  distinct. Existing judge calibration does not validate routing efficacy. The
  versioned outer-factor sign correction and saved-pair recheck are documented in
  `JUDGE_SIGN_RECHECK.md`; no completed recheck or renewed calibration is assumed.
- General construction checks do not import the detention-basin-specific probes
  or bounded basin repair controller. No benchmark-specific component is added
  to this methodology.
- The present controller uses bounded incumbent/trial lineages. It is not an
  implemented MCTS or beam-search result. Alternating optimization and automatic
  unbounded fitting continuation are also outside this section.
- All reported fitting budgets and pruning tolerances are algorithm settings,
  not universal scientific constants. Unknown usage or incomplete fits must stay
  explicit in the experimental results.

The paper should report NMSE, deterministic predicate outcomes (including
unresolved cases), complexity, and cost separately. It should not collapse graph
reachability, fitted mechanism activity, and broader scientific interpretation
into an unqualified “mechanism compliance” percentage.

## Implementation sources

| Part of section | Primary sources checked |
| --- | --- |
| Model, boundaries, scoring | `schemas/public_fitting.py`, `fitting/public_fitting.py`, `fitting/initialization.py`, `docs/PHYSICAL_INITIALIZATION.md`, original algorithm Sections 2–6 |
| Shared construction and revision | `docs/SHARED_PROCESS_INTEGRATION.md`, `search/staged_topology_runner.py`, `search/staged_function_runner.py`, `search/function_dependencies.py`, `rebuttal/review_deadline_pipeline.py` |
| Numerical fitting | `fitting/collocation_sensitivity.py`, `fitting/sensitivity_probe.py`, `docs/FITTER_FREEZE.md`, `docs/DALLA_MULTI_TARGET_PILOTS.md` |
| Advisory critic and strict selection | `rebuttal/general_critic.py`, `docs/GENERAL_CALIBRATED_CRITIC.md`, `docs/JUDGE_SIGN_RECHECK.md` |
| Pruning ranking and acceptance | `pruning/process_aware.py`, `rebuttal/process_pruning.py`, `docs/PROCESS_AWARE_PRUNING.md` |

Python paths in this table are relative to `src/autoformalism/`.
Version-specific implementation and manifests take precedence over older general
design notes, some of which still describe one-step prediction or older policies.

## Use in the manuscript

The section has no preamble, title page, contents, bibliography, or external
figures. It needs `amsmath` and `amssymb`. Copy it into the manuscript or use
`\input{docs/AUTOFORMALISM_METHODOLOGY.tex}` from the repository root. Its labels
use an `af-` prefix to reduce collisions. The accompanying preview wrapper uses
a conventional 10-point two-column layout with balanced final columns. The draft
is approximately 1,000 words plus five displayed equations and occupies two pages
in this preview; the venue's class controls final length.

Build a local preview from the repository root:

```bash
mkdir -p /tmp/autoformalism-methodology-tex output/pdf
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode \
  -jobname=AUTOFORMALISM_METHODOLOGY \
  -output-directory=/tmp/autoformalism-methodology-tex \
  docs/AUTOFORMALISM_METHODOLOGY_PREVIEW.tex
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode \
  -jobname=AUTOFORMALISM_METHODOLOGY \
  -output-directory=/tmp/autoformalism-methodology-tex \
  docs/AUTOFORMALISM_METHODOLOGY_PREVIEW.tex
cp /tmp/autoformalism-methodology-tex/AUTOFORMALISM_METHODOLOGY.pdf output/pdf/
```

No cluster jobs, new model calls, fitting, or benchmark access are needed.

## Verification

The source-contract checks for process pruning, its campaign wrapper, the general
critic, shared-process integration, and the multi-target profile passed:
67 tests. The LaTeX preview compiles with shell escape disabled and has been
visually checked on both pages. Repository-wide `ruff check .` still reports
37 pre-existing findings in the unrelated untracked `analysis/claude` scripts;
these documentation changes do not modify those files or any runtime code.
