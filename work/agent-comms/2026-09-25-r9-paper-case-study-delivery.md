# R9 case study added to the experiment draft

The uploaded `AUTOFORMALISM_EXPERIMENTS.tex` was byte-identical to the tracked
project copy (SHA256 `2252db0322c647af1cd6bc8151ac74d6b9a7f034a39d2cbdba5331e08ae7790a`).
The project copy is now updated in place. Section 5.4's placeholder is replaced
by an interventional-discrimination case study, with equations, a six-panel
figure, and an all-repetition robustness table. Other prospective test fields
are left unchanged. The standalone preview and draft notes reflect that scope.

The story is deliberately about assisted historical endpoints: similar training
NMSE, an omitted auxiliary dependency that makes a response impossible, and
excessive sensitivity in validation-selected Sol 0. It states that R9's maximum
observed response error is lower while its maximum absolute trajectory error
is higher than all Sol repetitions. R9 is not called an autonomous Full result;
the unequal histories/budgets, optional Gt dependency, conditional auxiliary
availability and exploratory nature are explicit.

## Outputs

- `docs/AUTOFORMALISM_EXPERIMENTS.tex`: modified section body.
- `docs/AUTOFORMALISM_EXPERIMENTS_PREVIEW.tex`: updated preview wrapper.
- `output/pdf/autoformalism-experiments-with-case-study.pdf`: compiled preview.
- `output/pdf/assisted-r9-case-study/`: main and four supporting figures,
  each in PDF/SVG/PNG, plus full-precision CSV/JSON data and provenance manifest.
- `transfers/assisted-r9-paper-case-study-20260925.zip`: portable manuscript,
  preview, figures/data and plotting sources with SHA256SUMS.

The main figure uses the same single/double training schedules as the previous
figure (train 002: 60g@0; train 010: 30g@0, 60g@120). Every primary model has
per-trajectory NMSE below 0.044 on both. All 16 training and four original
validation curves appear in supporting grids. Both signs of the initial-Gt
intervention are displayed. Matched-control differences subtract each model's
own control, not the physical control.

`scripts/build_r9_paper_case_study.py` verifies 135 sealed saved rollouts,
recomputes absolute NMSE and response RSE, confirms shared references/time grids,
and selects Sol using original validation only. Its portable replot mode reads
the exported JSON and requires only NumPy/Matplotlib plus the bundled plotting
helper. There is no new integration, fitting, model revision, LLM call, remote
session, benchmark mutation or test-data access.

## Verification

Fourteen relevant tests pass, including new checks for own-control subtraction,
original-validation selection, mismatched/failed/nonfinite trajectories, incorrect
saved metrics and negligible response denominators. New code passes Ruff.
Repository-wide Ruff still reports the same 37 unrelated `analysis/claude`
findings. The full test suite was not rerun for this plotting/document change.
The actual 135-record build is the numerical-data smoke check.

The five-page LaTeX preview resolves references/citations and has no
overfull/underfull box warnings. Every preview page and plot layout was visually
inspected; supporting-grid title/legend spacing was corrected before packaging.
Generated figures, data, PDFs and archives remain untracked; only source,
documentation and tests are committed.
