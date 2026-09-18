# Dalla Man naming and dynamics comparison

This audit supports `docs/LLM_MEMORIZATION_SUBSECTION.tex`. It uses the four
T1 easy Dalla Man cells in the six-cell review campaign. It does not inspect
private equations, test observations, provider training data, or remote hosts.
No new model generation, fitting, or experimental selection is performed.

## Sources and scope

- Autoformalism: `analysis/claude/data/rd4_summary.json`, full arm, retained
  global round-12 validation scores. SHA-256:
  `16514495a2daebbade51de1bbe4872a6e16445953260463efdf2557dcdd37089`.
  Plan: `67108d6fe6b9b70b9d0a9e0e30a0aa6fbc39cccd21e3d8a8174ebeb25d6b0cf1`.
- GPT-5.6 Sol: `analysis/claude/data/summary.json`, `source_kind=raw_data_agent`,
  exact same Phase-B cell IDs, easy tier, saved-parameter validation open-rollout
  scores (`normalized_mse`). Plan:
  `602191cd2fc6acdc6dd3d60f8debf07afbc0eb3a38b5a3d9e08dfd73735a6425`.
  The source audit reports verified development-data provenance for these
  agent models. Native validation metrics and replay metrics are not mixed.
- Roster: `configs/review_deadline_v2.json`. The four cells are canonical named,
  canonical obfuscated, perturbed named, and perturbed obfuscated Dalla Man T1.
- Design: `docs/PHASE_B_EXACT_BENCHMARK_PROTOCOL.md` and
  `docs/PHASE_B_SEMANTIC_PROMPT_AUDIT.md`. Named/obfuscated pairs preserve
  numerical data up to renaming, channel roles, splits, and scientific task
  burden; names and domain cues change. These are not four independent system
  families. Different dynamics variants have different generated trajectories.
- No result is chosen by searching over rounds. Every available run in these
  cells is retained; the headline statistic is the per-cell median, not the
  best run. There are two Autoformalism runs and three external-agent runs per
  cell. Seed/repetition labels do not establish coupled randomness across
  methods. Budget and algorithm differences remain.

## Every underlying validation score

| Dynamics / presentation | Autoformalism seed 0 | Autoformalism seed 1 | GPT repetition 0 | GPT repetition 1 | GPT repetition 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Canonical / named | 0.2638198477 | 0.4446978397 | 0.5263357130 | 0.0195579166 | 0.5610854090 |
| Canonical / obfuscated | 0.2749696817 | 0.2396537639 | 0.0141723801 | 0.0453184948 | 0.0309527870 |
| Perturbed / named | 0.5386673369 | 0.3849744594 | 0.0743160739 | 0.0334849466 | 62.9707895479 |
| Perturbed / obfuscated | 0.2450713347 | 0.2436464036 | 47425.1135617494 | 78092.6482486110 | 0.0999922737 |

The public cell ID prefixes are respectively:

1. `phase_b_dalla_man_t1_canonical_named_easy`
2. `phase_b_anonymous_system_t1_canonical_obfuscated_easy`
3. `phase_b_dalla_man_t1_perturbed_named_easy`
4. `phase_b_anonymous_system_t1_perturbed_obfuscated_easy`

## Aggregation and interpretation

| Dynamics / presentation | Ours median | GPT median | Ours mean | GPT mean |
| --- | ---: | ---: | ---: | ---: |
| Canonical / named | 0.354259 | 0.526336 | 0.354259 | 0.368993 |
| Canonical / obfuscated | 0.257312 | 0.030953 | 0.257312 | 0.030148 |
| Perturbed / named | 0.461821 | 0.074316 | 0.461821 | 21.026197 |
| Perturbed / obfuscated | 0.244359 | 47425.113562 | 0.244359 | 41839.287268 |

The canonical named means are indeed close, but this does not imply statistical
equivalence. The subsection uses medians consistently across all four cells.
With two Autoformalism runs its mean and median necessarily coincide.

Obfuscated/named median ratios:

- Autoformalism, canonical: 0.726338; perturbed: 0.529120.
- GPT-5.6 Sol, canonical: 0.058808; perturbed: 638154.184701.

These ratios summarize a small, highly variable sample. In particular, the very
large perturbed ratio is driven by two runs, while another obfuscated perturbed
run scores 0.100. The named perturbed condition already has one large-error run
(62.97). The pattern is not uniform failure under obfuscation, nor proof that a
particular model was copied.

What is supported:

- Autoformalism retains similar-order validation errors in all four cells and
  has no median degradation under either naming change in this sample.
- GPT-5.6 Sol improves on the canonical obfuscated cell and is more variable in
  perturbed cells, including two very large errors under obfuscation.
- The complete pipeline shows a robustness contrast in the combined perturbation
  condition. This observation cannot be attributed exclusively to LLM memory:
  the methods also differ in construction, checks, fitting, and selection.

What is not supported:

- "GPT-5.6 is worse whenever Dalla Man is obfuscated."
- "These results prove our proposer did not memorize Dalla Man."
- "These results establish GPT-5.6 retrieved/memorized the reference model."
- "Low validation error recovers the original latent equations."

Recognizable numerical patterns, units, channel relations and general scientific
priors can survive removal of familiar names. Conversely, poor rollout behavior
can have numerical or search-related causes. Neither training-corpus membership
nor generation provenance is identified by these error measurements. Direct
claims about retrieval would require additional evidence from generated
equations or a dedicated controlled study. Validation was used for development
selection and is not an independent held-out test estimate.
