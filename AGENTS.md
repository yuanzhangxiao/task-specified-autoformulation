# Instructions for Codex

Read these files before making architectural changes:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/PIPELINE_DESIGN.md`
3. benchmark prompt files relevant to the current task
4. the tests relevant to the component being changed

## General rules

- Work incrementally.
- Implement one milestone at a time.
- Run tests after every meaningful change.
- Do not modify benchmark data.
- Do not modify finalized benchmark prompts unless explicitly asked.
- Do not place secrets in source code, tests, notebooks, or logs.
- Do not use `eval`, `exec`, or unrestricted SymPy lambdification on
  proposer-generated text.
- Treat all LLM output as untrusted input.
- Use explicit schemas and a restricted expression grammar.
- Preserve train/validation/test separation.
- Do not expose test metrics to proposal generation or model selection.
- Every experiment must support checkpointing and deterministic resume.
- Every LLM call must be cached and logged.
- Prefer small, typed functions with docstrings.
- Add tests for successful and failing cases.
- Keep the CLI runnable independently of notebooks.

## Required verification

Before marking a task complete:

1. run `pytest`;
2. run `ruff check .`;
3. run relevant smoke tests;
4. summarize changed files;
5. identify remaining limitations.

## Git

Do not rewrite unrelated files.

Do not commit secrets, generated outputs, API responses, datasets, or
large experiment artifacts.