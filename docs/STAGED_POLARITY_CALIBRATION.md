# Topology-stage interaction-polarity calibration

This development-only campaign tests whether the topology proposer converts
explicit public scientific language into the required outer-weight contract:
`positive`, `negative`, or `unrestricted`. It does not ask whether a complete
interaction function is globally sign-definite.

The diagnostic exposes four public obligations for one differential target:

1. an explicitly activating external-input contribution;
2. a stabilizing self-relaxation contribution;
3. a supplied-context contribution whose direction is explicitly unspecified;
4. a source-free baseline whose sign is explicitly unspecified.

The expected source sets and polarities are frozen separately for evaluation
and are not included in the provider request. The public brief and relevant
requirement IDs are included. Terms are matched by complete source set, not by
the proposer's descriptive wording. Missing, duplicate, and additional source
sets fail the exact-coverage gate. A schema-valid but scientifically wrong sign
is scored as wrong and is not converted into a contract-repair retry.

The frozen campaign uses GPT-OSS-20B at low reasoning for three seeds on one
ACES H100. It permits at most three contract-only attempts per seed. Every
physical request is cached, task outcomes are checkpointed, and a requeued job
continues without repeating completed calls. All five predeclared response,
coverage, overall-polarity, fixed-polarity, and unrestricted-polarity gates are
exact (`1.0`) because the diagnostic language is deliberately unambiguous.

This campaign does not use a benchmark's hidden equations as labels. The two
currently studied public benchmark mechanism specifications mark regulatory
sign as unspecified, so they are not assigned artificial sign ground truth.
It performs no function generation, parameter fitting, scientific judging,
test-data access, private-reference access, or automatic model selection.

The ACES launcher is:

```bash
bash scripts/hpc/submit_staged_polarity_calibration_aces.sh
```
