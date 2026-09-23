# Response-oriented feedback implementation

The user's requested change is implemented as opt-in `review-deadline-7`.
The running protocol-6 campaign and its frozen fitting profile remain intact.
Instructions and exact artifact meanings are in
`docs/RESPONSE_ORIENTED_FEEDBACK.md`.

The T2 request failure is consistent with context overflow: the supplied failed
requests contain roughly 39.2k–40.2k raw message tokens before chat formatting,
against a 32,768-token serving context. Their schema is the same as the successful
T2-hard request. The old packet had 48 trajectory/target summaries, 144 window
summaries and six detailed sample sets, rather than literally every data sample.

The proposer now receives every target's aggregate error/bias plus up to three
response examples per target. Full-grid training replay measures amplitudes,
peak timing, sampled return toward the initial value and window errors. It
reports ambiguous or censored timing explicitly. It does not estimate physical
time constants or infer a faulty term. Full residual records remain available.

Serving-tokenizer preflight reserves output space and packs optional examples
before generation. Invalid/unavailable tokenization causes zero generation
calls. HTTP delivery failure stops the visit's proposer retries, retains the
incumbent without refitting, and stops automatic future dispatch at the finish
barrier. Scientific/equation rejection retains the existing correction policy.
Only actually displayed references can resolve as cited evidence.

The ACES launcher imports the latest common completed checkpoint and defaults
to three new visits. CPU preparation previews the actual response summaries;
each GPU allocation probes the serving tokenizer endpoint. The runbook includes
optional commands to stop only pending old dispatchers and let submitted work
finish before importing. No remote sessions or jobs were opened here.

The full local pytest run passed 3,055 tests with eight optional Torch tests
skipped. Focused regression and real CPU smoke tests also pass. Ruff is clean
for the changed files; `ruff check .` reports 37 existing issues under the
untracked `analysis/claude` directory, which was left unchanged. The real serving
endpoint and improvement in scientific revisions remain ACES checks. The
software controls establish bounded delivery, multi-output fitting, preservation
and deterministic resume, not discovery quality.
