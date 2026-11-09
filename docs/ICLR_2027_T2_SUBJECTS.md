# T2-hard model selection and search history

The diagnostic evaluates the retained model at visit six (zero-based round 5) in each of the 16 primary component lineages. This is the pre-pruning search stage that supplied the historical validation table. Full and Brief-only use different construction prompts. Within each prompt and seed, the four arms cross critic on/off with scientific verifier on/off. All arms receive the task specification. Among 12 planned external repetitions, the local development plan supplies three GPT-5.6 Sol models, three SINDy models, and one PySR model. Two PySR fits and three native D3 selections are unavailable in that archive. The external methods have their own search budgets.

## What changed during one search

We traced the first Full, critic-on, verifier-on lineage (seed 0), chosen by its order in the design. At construction it had three dynamic states, plasma glucose, insulin, and a latent state `X`. The declared process named DelayedInsulinAction equaled instantaneous insulin `I`, while `X` itself already integrated insulin. Disposal depended on `I × X`.

At the next visit the search made DelayedInsulinAction a fourth dynamic state `D`. Its fitted equation was `dD/dt = 0.00023593 I − D/49.23`. The disposal term became `I × D`. At visit 2 the sign of the declared disposal expression reversed. At visit 3 the model retained this expression and fitted `dD/dt = 3.05 × 10⁻⁹ I − D/152.77`. The decay and drive determine what `D` actually carries. Its name alone cannot establish correspondence with simulator insulin action. The archive permits reconstruction of the executed edits and their validation scores. It lacks complete critique messages from which to attribute the edits to specific advice.

| Zero-based visit | Trial validation NMSE | Retained validation NMSE | Origin of retained model |
|---:|---:|---:|---:|
| 0 | 1.23451 | 1.23451 | 0 |
| 1 | 1.13152 | 1.13152 | 1 |
| 2 | 1.01067 | 1.01067 | 2 |
| 3 | 0.98904 | 0.98904 | 3 |
| 4 | 1.09478 | 0.98904 | 3 |
| 5 | 1.09587 | 0.98904 | 3 |

The file `lineage.json` contains both model roles at each visit, the exact lowered equations, initialization rules, fitted parameter vectors and source identities.

## Comparisons matched before reading intervention results

All 24 possible pairs within prompt and seed were checked. Eligibility required equal dynamic state count, parameter-count gap at most the larger of two and 20% of the smaller parameter count, validation NMSE ratio at most 1.20, and distinct functional fingerprints after accounting for renaming. Two pairs qualified.

| Prompt and seed | States in both models | Parameter counts, both-off / both-on | Validation NMSE, both-off / both-on |
|---|---:|---:|---:|
| Full 0 | 4 | 9 / 11 | 0.97586 / 0.98904 |
| Brief-only 0 | 3 | 9 / 9 | 1.01533 / 0.97468 |

The complete matched and unmatched pairs are in `matched-pairs.json`. Matching on these observables does not control all differences between equations. Both pairs compare joint removal of the critic and verifier and come from one seed.

## Reproduce locally

The packaged `subjects/models.json` is sufficient for the diagnostic replay. To reproduce its extraction from the two historical intake archives, run from the project checkout with their paths available.

```bash
python scripts/prepare_t2_hard_subjects.py \
  --inventory "$AF_COMPONENT_ZIP" --baseline-archive "$AF_BASELINE_ARCHIVE" \
  --output "$AF_OUTPUT_ROOT/subjects/models.json"
python scripts/analyze_t2_hard_selection.py \
  --inventory "$AF_COMPONENT_ZIP" --models "$AF_OUTPUT_ROOT/subjects/models.json" \
  --output "$AF_OUTPUT_ROOT/subjects"
```

`AF_COMPONENT_ZIP` is the historical `dalla-all-models.zip`, whose SHA-256 is recorded as `inventory_zip_sha256` in the sealed subjects file. `AF_BASELINE_ARCHIVE` is `baseline-validation-models.tar.gz`; the file and its sealed internal plan are also bound in that manifest.

## Optional Delta replay with native D3

On Delta, use a checkout containing these scripts and the repository Python environment. Set `AF_EVIDENCE` to the extracted package directory, and `AF_RUN` to a new writable result directory. The historical portable result ledger identifies the D3 root below and its T2-hard indices 30, 31 and 32. Its current remote availability has not been checked.

```bash
export AF_D3_ROOT=/work/hdd/bibo/yxiao2/phase_b/d3-native-vllm-120b-v1
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export PYTHONPATH="$PWD/src"
: "${AF_EVIDENCE:?set to the extracted package directory}"
: "${AF_RUN:?set to a fresh writable output directory}"
mkdir -p "$AF_RUN"
"$AF_PYTHON" scripts/prepare_t2_hard_subjects.py \
  --retained-bundle "$AF_EVIDENCE/subjects/models.json" \
  --native-d3-root "$AF_D3_ROOT" --output "$AF_RUN/models.json"
"$AF_PYTHON" scripts/evaluate_t2_latent_dynamics.py prepare \
  --models "$AF_RUN/models.json" --root "$AF_RUN/t2-action"
"$AF_PYTHON" scripts/evaluate_t2_latent_dynamics.py run \
  --root "$AF_RUN/t2-action" --workers 2
```

Use a CPU allocation for the last command. The supplement verifies the native campaign plan, saved selection and public-prompt identities. It preserves every non-D3 subject and uses the fitted discrete increments. The replay makes no model calls, performs no model refits, and never opens the original benchmark test split.
