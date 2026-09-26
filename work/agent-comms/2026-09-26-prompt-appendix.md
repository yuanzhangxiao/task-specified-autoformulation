# Prompt appendix

Created `paper_to_revise/sections/appendix_prompts.tex` in response to the request
for prompts in gray code boxes. It replaces the old, untracked model-response
example at that path. No production prompt or implementation was changed.

The source snapshot is `ee6a1b003f53d586dddca68ade3932f65b20adf5`. The appendix
describes the final component campaign's shared-process, multiple-target path,
not all historical pilot protocols. Seventeen breakable, monospaced gray boxes
contain fixed prompt text, explicitly labeled excerpts, and one clearly labeled
schematic context template. Text is copied from the executable constants; only
whitespace is reflowed. Each box has a source-location comment and a
whitespace-normalized SHA-256 in the LaTeX source.

## Sources and interpretation

- `src/autoformalism/benchmarks/phase_b_public.py`: public T1, T2, CSTR and
  alien-device mechanism wording, named T1-easy channel contract, common
  requirements and plausibility instructions. T1 does not explicitly prescribe
  a specific latent compartment or insulin delay. Only T1/T2 are included.
- `src/autoformalism/search/staged_topology_prompts.py` and
  `shared_process_guidance.py`: variable identification with sharing guidance;
  an explicitly labeled excerpt of the ordinary topology/sign policy.
- `src/autoformalism/search/shared_process_contract.py`: optional process
  declarations and coordinated signs/conversions of their uses.
- `src/autoformalism/search/function_delivery.py`: effective identified-slot
  batch prompt (not the older positional-only batch prompt).
- `src/autoformalism/search/process_revision_runner.py`: bounded construction
  repair; dependency revisions are explicit, valid unrelated slots remain intact.
- `src/autoformalism/search/causal_initialization.py`: train-fitted shared
  initial values or maps from permitted initial public observations. This is not
  the obsolete fixed-number initializer prompt.
- `src/autoformalism/search/response_revision.py`: effective whole-model,
  multiple-target revision prompt with response-oriented training feedback.
  The removed small patch-count quotas are not presented as current restrictions.
- `src/autoformalism/search/fresh_shared.py` and
  `src/autoformalism/rebuttal/general_critic.py`: shared-law context and advisory
  interpretation of the critic.
- `src/autoformalism/judging/prompts.py`, as used by
  `src/autoformalism/rebuttal/repair_scientific_judge.py` and `component_critic.py`:
  atomic evidence, paired hybrid scientific assessment and their handoff.
  The old six-category benchmark-export judge and later experimental
  target-completeness extensions are not mislabeled as the current critic.

The prose distinguishes literal instructions from runtime guarantees, graph
facts from scientific verdicts, and critic advice from validation selection.
It also distinguishes Brief-only, critic-off, verifier-off and sharing-off from
historical no-specification/no-latent controls. Pruning needs no separate LLM prompt.

## Manuscript integration

The main manuscript is untracked and has unrelated ongoing edits, so it was not
rewritten or included in this commit. Its preamble already loads tcolorbox with
the `most` option. Add this after `\appendix` at the desired location:

```tex
\input{sections/appendix_prompts}
```

For another manuscript, load this in its preamble:

```tex
\usepackage[most]{tcolorbox}
```

`docs/AUTOFORMALISM_PROMPTS_PREVIEW.tex` is a standalone wrapper. From the repo root:

```bash
mkdir -p tmp/pdfs/prompt-appendix
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=tmp/pdfs/prompt-appendix \
  -jobname=autoformalism-prompt-appendix docs/AUTOFORMALISM_PROMPTS_PREVIEW.tex
# Repeat once for labels/bookmarks.
```

## Verification

- Relevant pytest suite: **167 passed** (public prompts, topology/functions,
  causal initializers, response revision, local repair, component campaign,
  hybrid judge).
- `scripts/smoke_review_multi.py --output ...`: **pass**, including actual
  multi-target fits, construction repair, fallback, deterministic resume,
  zero live LLM calls and no test-data access.
- `ruff check .`: **37 pre-existing findings** in unrelated `analysis/claude`
  files; no implementation files were edited in this task.
- Two-pass LaTeX compilation: 14 pages; no LaTeX errors, warnings, overfull or
  underfull boxes. All pages rendered and visually inspected.
- PDF preview is local at `output/pdf/autoformalism-prompt-appendix.pdf` and is
  not committed. Temporary authoring script removed after use.

Limit: these listings document the current source snapshot and selected excerpts.
They do not reproduce every run-specific user payload or JSON schema. Frozen
request records remain the authority for exact historical calls. Actual benchmark
data, prompts, fitting settings, and experiment artifacts were not modified.
