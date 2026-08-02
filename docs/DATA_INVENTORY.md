# Phase 1 Data Inventory

## Scope and conventions

Inventory date: 2026-07-31.

`AUTOFORMALISM_DATA_ROOT` was not set in the inspection shell. This
inventory therefore uses the discovered repository directory
`data_raw/` as the data root. All paths below are relative to that
fallback root.

The inventory includes all 18 discovered public benchmark identifiers
and all three observability tiers (54 configurations), although Phase 1
initially uses only original B1, perturbed B1, obfuscated-original
case 01, obfuscated-perturbed case 01, benchmark 5, and benchmark 6.
Private and hidden files were inspected only to understand repository
boundaries; they are not pipeline inputs and must never be exposed to
the proposer, judge, candidate selection, or fitting code.

Each tier's data bundle is:

- `X_<split>.csv`: observed channel values;
- `Y_<split>.csv`: supplied derivative estimates (diagnostic/optional
  fitting data, never an additional prediction-horizon input);
- `X_<split>_clean.csv` when present: clean observed values;
- a benchmark-level `metadata_<split>.csv` or `input_<split>.csv`:
  time, external inputs, trajectory keys, and fixed covariates.

“Target” below means a channel that the task requires the candidate to
generate. “Auxiliary” means a channel explicitly permitted as an
exogenous trajectory over the prediction horizon. Some older prompts
do not label these categories directly; those classifications are
derived from their task and availability sections and are called out
under inconsistencies.

## Shared file layouts

| Layout | Train / validation / test paths | Time | Trajectory ID | Interval | Format |
|---|---|---|---|---|---|
| Dalla Man | `<benchmark>/<tier>/X_{train,val,test}.csv`, `Y_{train,val,test}.csv`, `X_{train,val,test}_clean.csv`; `<benchmark>/metadata_{train,val,test}.csv` | `metadata_*.csv: time` | none; one trajectory per split | 1.0 min | CSV |
| Obfuscated Dalla Man | `<case>/<tier>/X_{train,val,test}.csv`, `Y_{train,val,test}.csv`, `X_{train,val,test}_clean.csv`; `<case>/input_{train,val,test}.csv` | `input_*.csv: t` | none; one trajectory per split | 1.0 (manifest and data) | CSV |
| Obfuscated perturbed | same as obfuscated Dalla Man | `input_*.csv: t` | none; one trajectory per split | 1.0 observed; not declared in case manifests | CSV |
| Multi-trajectory | `<public>/<tier>/X_{train,val,test}.csv`, `Y_{train,val,test}.csv` and, for B5 only, `X_{train,val,test}_clean.csv`; `<public>/input_{train,val,test}.csv` | `input_*.csv: t` | `input_*.csv: trajectory_id`; X/Y align by row | 0.1 | CSV |

For all single-trajectory Dalla Man layouts, each split contains 301
rows. B5 has 3,612/1,204/1,806 train/validation/test rows across
12/4/6 trajectories. B6 has 7,212/2,404/3,606 rows across 12/4/6
trajectories.

## Original Dalla Man suite

Common root: `benchmark1_original_dalla_man/benchmarks/`.
Manifest: `benchmark1_original_dalla_man/manifest.json` (suite-level).
External inputs: `meal_event_g` and its alternate scaling
`meal_event_mg`, with `meal_schedule`; `body_weight_kg` is a fixed
trajectory covariate. Time is `time`. Sampling is 1.0 minute.
The runtime registry selects the numeric `meal_event_g` representation as the
single canonical intervention input. Its nonzero timestamp and value encode
meal timing and amount. The alternate numeric scaling and JSON
`meal_schedule` remain present in the source metadata but are not independent
runtime inputs.

For each row, the prompt paths are
`<root>/<ID>/<tier>/{proposer_prompt.txt,judge_prompt.txt}` and the
split paths are `<root>/<ID>/<tier>/{X,Y}_<split>.csv`,
`<root>/<ID>/<tier>/X_<split>_clean.csv`, plus
`<root>/<ID>/metadata_<split>.csv`.

| Benchmark identifier | Tier | Target channels | Supplied auxiliary channels |
|---|---|---|---|
| `original/B1_meal_appearance` | easy | `Gp` | `EGP`, `Uii`, `E`, `Gt` |
| `original/B1_meal_appearance` | medium | `Gp` | `EGP`, `Uii` |
| `original/B1_meal_appearance` | hard | `Gp` | none |
| `original/B2_absorption_action` | easy | `Gp`, `I`, `U` | `EGP`, `Uii`, `E`, `Gt` |
| `original/B2_absorption_action` | medium | `Gp`, `I`, `U` | `Uii` |
| `original/B2_absorption_action` | hard | `Gp`, `I` | none |
| `original/B3_hepatic_regulation` | easy | `Gp`, `I`, `EGP`, `U` | `Uii`, `E`, `Gt`, `Ipo` |
| `original/B3_hepatic_regulation` | medium | `Gp`, `I`, `EGP` | none |
| `original/B3_hepatic_regulation` | hard | `Gp`, `I` | none |
| `original/B4_flux_portrait` | easy | `Gp`, `I` | `Uii`, `E`, `Gt`, `Ipo` |
| `original/B4_flux_portrait` | medium | `Gp`, `I` | `Uii`, `E`, `Gt` |
| `original/B4_flux_portrait` | hard | `Gp`, `I` | none |

Metadata gaps: the suite manifest provides masks, duration, and
`dt_min`, but does not declare file paths, time column, trajectory
identifier, target/auxiliary roles, derivative policy, or row
alignment. B2–B4 use older prose that does not explicitly label
targets and auxiliaries; the table follows the prompts' statements
about which observed mechanisms must be generated and which channels
may be supplied.

## Perturbed Dalla Man suite

Common root: `benchmark2_perturbed_dalla_man/`.
Each benchmark has `<root>/<ID>/manifest.json`. External inputs,
metadata, time, trajectory convention, sampling, and split layout are
the same as the original suite.

For each row, prompt paths are
`<root>/<ID>/<tier>/{proposer_prompt.txt,judge_prompt.txt}` and split
paths are `<root>/<ID>/<tier>/{X,Y}_<split>.csv`,
`<root>/<ID>/<tier>/X_<split>_clean.csv`, plus
`<root>/<ID>/metadata_<split>.csv`.

| Benchmark identifier | Tier | Target channels | Supplied auxiliary channels |
|---|---|---|---|
| `perturbed/B1_meal_appearance` | easy | `Gp` | `EGP`, `Uii`, `E`, `Gt` |
| `perturbed/B1_meal_appearance` | medium | `Gp` | `EGP`, `Uii` |
| `perturbed/B1_meal_appearance` | hard | `Gp` | none |
| `perturbed/B2_absorption_action` | easy | `Gp`, `I`, `U` | `EGP`, `Uii`, `E`, `Gt` |
| `perturbed/B2_absorption_action` | medium | `Gp`, `I`, `U` | `Uii` |
| `perturbed/B2_absorption_action` | hard | `Gp`, `I` | none |
| `perturbed/B3_hepatic_regulation` | easy | `Gp`, `I`, `EGP`, `U` | `Uii`, `E`, `Gt`, `Ipo` |
| `perturbed/B3_hepatic_regulation` | medium | `Gp`, `I`, `EGP` | none |
| `perturbed/B3_hepatic_regulation` | hard | `Gp`, `I` | none |
| `perturbed/B4_flux_portrait` | easy | `Gp`, `I` | `Uii`, `E`, `Gt`, `Ipo` |
| `perturbed/B4_flux_portrait` | medium | `Gp`, `I` | `Uii`, `E`, `Gt` |
| `perturbed/B4_flux_portrait` | hard | `Gp`, `I` | none |

Metadata gaps: manifests do not declare file paths, time column,
trajectory identifier, row alignment, target/auxiliary roles, or
derivative policy. The B2–B4 target classifications have the same
older-prompt ambiguity as the original suite. The B3 easy proposer
file is literally named `easy/ proposer_prompt.txt` (leading space),
unlike every other tier.

## Obfuscated original Dalla Man suite

Common root: `benchmark3_obfuscated_dalla_man/public/`.
External inputs are `u01` and alternate scaling `u01_raw`;
`input_schedule` carries event timing/amount and `c01` is a fixed
trajectory covariate. Time is `t`, no trajectory ID is present, and
sampling is 1.0.

For every row the manifest is `<root>/<case>/manifest.json`; prompt
paths are `<root>/<case>/<tier>/{proposer_prompt.txt,judge_prompt.txt}`;
split paths are `<root>/<case>/<tier>/{X,Y}_<split>.csv`,
`X_<split>_clean.csv`, and `<root>/<case>/input_<split>.csv`.

| Benchmark identifier | Tier | Target channels | Supplied auxiliary channels |
|---|---|---|---|
| `obfuscated-original/case_01` | easy | `v009` | `v016`, `v012`, `v004`, `v025` |
| `obfuscated-original/case_01` | medium | `v009` | `v016`, `v012` |
| `obfuscated-original/case_01` | hard | `v009` | none |
| `obfuscated-original/case_02` | easy | `v010`, `v023`, `v028` | `v018`, `v014`, `v012`, `v004` |
| `obfuscated-original/case_02` | medium | `v010`, `v023`, `v028` | `v014` |
| `obfuscated-original/case_02` | hard | `v010`, `v023` | none |
| `obfuscated-original/case_03` | easy | `v021`, `v001`, `v002`, `v020` | `v010`, `v027`, `v006`, `v003` |
| `obfuscated-original/case_03` | medium | `v021`, `v001`, `v002` | none |
| `obfuscated-original/case_03` | hard | `v021`, `v001` | none |
| `obfuscated-original/case_04` | easy | `v014`, `v005` | `v013`, `v020`, `v024`, `v007` |
| `obfuscated-original/case_04` | medium | `v014`, `v005` | `v013`, `v020`, `v024` |
| `obfuscated-original/case_04` | hard | `v014`, `v005` | none |

Metadata gaps: manifests call every X column “observed” but do not
encode target versus auxiliary roles or file paths. Cases 02–04 use
older prose; target roles above are inferred from the mechanisms that
the prompt says must be generated. Derivative CSVs are listed in the
manifest but their permissible role is not specified.

## Obfuscated perturbed Dalla Man suite

Common root: `benchmark4_obfuscated_perturbed_dalla_man/public/`.
The actual input files contain `t`, `u01`, alternate scaling
`u01_raw`, and fixed covariate `c01`. No trajectory ID is present.
The observed sampling interval is 1.0.

For every row the manifest is `<root>/<case>/manifest.json`; prompt
paths are `<root>/<case>/<tier>/{proposer_prompt.txt,judge_prompt.txt}`;
split paths are `<root>/<case>/<tier>/{X,Y}_<split>.csv`,
`X_<split>_clean.csv`, and `<root>/<case>/input_<split>.csv`.

| Benchmark identifier | Tier | Target channels | Supplied auxiliary channels |
|---|---|---|---|
| `obfuscated-perturbed/case_01` | easy | `v015` | `v008`, `v014`, `v028`, `v026` |
| `obfuscated-perturbed/case_01` | medium | `v015` | `v008`, `v014` |
| `obfuscated-perturbed/case_01` | hard | `v015` | none |
| `obfuscated-perturbed/case_02` | easy | `v015`, `v004`, `v020` | `v032`, `v003`, `v007`, `v027` |
| `obfuscated-perturbed/case_02` | medium | `v015`, `v004`, `v020` | `v003` |
| `obfuscated-perturbed/case_02` | hard | `v015`, `v004` | none |
| `obfuscated-perturbed/case_03` | easy | `v007`, `v005`, `v002`, `v031` | `v014`, `v019`, `v032`, `v030` |
| `obfuscated-perturbed/case_03` | medium | `v007`, `v005`, `v002` | none |
| `obfuscated-perturbed/case_03` | hard | `v007`, `v005` | none |
| `obfuscated-perturbed/case_04` | easy | `v031`, `v001` | `v002`, `v033`, `v018`, `v006` |
| `obfuscated-perturbed/case_04` | medium | `v031`, `v001` | `v002`, `v033`, `v018` |
| `obfuscated-perturbed/case_04` | hard | `v031`, `v001` | none |

Metadata inconsistencies:

- case manifests omit `sampling_interval`;
- case 01 prompts and files consistently use `t` and `u01`; its
  manifest still omits the sampling interval;
- cases 02–04 prompts declare `input_schedule`, but that column is
  absent from their public input CSVs and from `input_file_columns`;
- all case 02 prompts name copied auxiliaries `v018`, `v014`, `v012`,
  `v004`, although most are absent from case 02 files;
- all case 03 prompts similarly name copied auxiliaries `v010`,
  `v027`, `v006`, `v003`;
- all case 04 prompts similarly name copied auxiliaries `v013`,
  `v020`, `v024`, `v007`;
- the table uses the channel descriptions and actual tier columns to
  state the likely intended roles, but these cases must be rejected by
  strict loading until prompt/manifest metadata are reconciled;
- derivative policy, target roles, file paths, and row alignment are
  not machine-readable.

Only case 01 is in the current implementation scope. Cases 02–04 and
their unresolved prompt/metadata issues are intentionally deferred.

## Benchmark 5: anonymous nonlinear process

Root: `benchmark5_anonymous_nonlinear_process/public/`.
Identifier: `anonymous_controlled_system_case_05`.
Manifest: `public/manifest.json`. Time is `t`, trajectory identifier is
`trajectory_id`, sampling is 0.1, and external inputs are `u01`, `u02`,
and `u03`.

For every tier, prompt paths are
`public/<tier>/{proposer_prompt.txt,judge_prompt.txt}`. Train,
validation, and test paths are
`public/<tier>/{X,Y}_{train,val,test}.csv`,
`public/<tier>/X_{train,val,test}_clean.csv`, and
`public/input_{train,val,test}.csv`.

| Tier | Target channels | Supplied auxiliary channels |
|---|---|---|
| easy | `v02` | `v01`, `v03` |
| medium | `v02` | `v01` |
| hard | `v02` | none |

Metadata gaps: `trajectory_id` and `t` occur only in the input file;
X/Y depend on positional row alignment. The manifest does not list
concrete paths or derivative policy. The tier manifest order
(`v02,v03,v01`) differs from the prompt auxiliary order
(`v01,v03`) but the sets agree.

## Benchmark 6: alien device

Root: `benchmark6_alien_device/public/`.
Identifier: `alien_device_case_06`. Manifest:
`public/manifest.json`. Time is `t`, trajectory identifier is
`trajectory_id`, sampling is 0.1, and the external input is `u01`.

For every tier, prompt paths are
`public/<tier>/{proposer_prompt.txt,judge_prompt.txt}`. Train,
validation, and test paths are
`public/<tier>/{X,Y}_{train,val,test}.csv` and
`public/input_{train,val,test}.csv`. Unlike the other suites, no
`X_*_clean.csv` files are supplied.

| Tier | Target channels | Supplied auxiliary channels |
|---|---|---|
| easy | `v02` | `v05`, `v01` |
| medium | `v02` | `v05` |
| hard | `v02` | none |

Metadata gaps: the manifest does not declare
`trajectory_id_column`, even though the input files contain
`trajectory_id`; X/Y depend on positional row alignment. It does not
list paths or derivative policy.

## Required loader stance

The implementation must not silently guess through the inconsistencies
above. A normalized, repository-owned benchmark specification should
make all roles and paths explicit. Inventory discovery may report
problems, but an experiment loader must fail before any LLM call when
files, columns, split boundaries, prompt declarations, or manifest
roles disagree. Private and hidden paths must be denied by construction.
