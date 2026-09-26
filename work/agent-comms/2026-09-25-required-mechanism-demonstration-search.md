# Search for a demonstration tied to an explicit public requirement

## Finding

The downloaded models do not currently establish the requested conjunction:
comparable good training fits, a demonstrable omission of an explicitly required
mechanism, and a held-out failure explained by that omission. This is a finding
about the inspected local evidence, not proof that such a pair cannot exist.
No paper claims or benchmark assets were changed in this inspection.

The assisted R9/no-specification comparison remains a valid missing-dependency
illustration. However, the missing tissue-glucose dependency is optional under
T1. Its omission must not be presented as a literal specification violation.

## Checks and exclusions

The exact frozen obfuscated T1 prompt in
`artifacts/t1-curves-round12-2026-09-22/inputs/ours/plan.json` requires a causal
input-timing/magnitude-to-target mechanism. Auxiliary use is optional, and
neither extra latent states nor a particular delay are mandatory. Therefore:

- The actual no-specification seed-0 model uses a meal-driven latent state. Its
  missing Gt dependence does not demonstrate omission of the required meal path.
- The named no-latent model has a direct meal-to-Gp term. Its inadequate delayed
  response cannot be described as omission of a specifically mandated T1 delay.
- Wrong conditional input direction in perturbed-obfuscated Sol 0 is a useful
  physical diagnostic, but the obfuscated prompt does not explicitly supply a
  positive pathway sign. It also already has poor free-rollout training fit.

Rechecked all 688 fitted records in the newer four-cell equation screen (177
T1-easy, 178 T1-hard, 154 T2-easy, 179 T2-hard; 1,334 total fitted/unfitted
records). These records include repeated incumbents, not independent trials.
One T1-easy fitted candidate lacks a generated meal-to-target path; its training
NMSE is 1.50785. All 178 fitted T1-hard records have a meal path. Five T2-easy
records lack the meal path, with minimum training NMSE 2.15873; T2-hard has none.
These missing-path candidates do not meet the low-training-error premise.

Source: `artifacts/dalla-all-models-review-2026-09-25/screen.json`, cross-referenced
with its source inventory and prior full equation audit. A path is a necessary
structural fact, not certification of the complete fitted mechanism.

Also screened the 96 saved external T1 subjects (eight cells, four methods,
three repetitions) in the uploaded `baseline-replay-subjects.tar.gz` for a
syntactic input-to-target path through state/process equations. All 24 Sol and
24 D3 subjects have a path; all 24 SINDy and 24 PySR subjects lack one. This
syntactic screen does not certify nonzero fitted gains, direction or memory.
Among the missing-path subjects with existing local free-rollout training
scores, the lowest training NMSE is 0.35511 (named canonical T1-hard PySR).
The three repetitions score 0.35511, 0.35633, 0.38058 on training and
0.35979, 0.36150, 0.37111 on original validation. The corresponding replayed
SINDy models have training NMSE 1.48248. They provide omission examples, but
not the desired good-training-then-intervention-failure example. Missing local
training scores for other cells were not inferred from test or native scores.

Sources: `artifacts/t1-external-intervention-replay-2026-09-25-v1/summary.json`
and `artifacts/t1-hard-external-comparison-2026-09-25/summary.json`.

## Tasks with a stronger requirement

T2 explicitly requires delayed insulin action linking I to insulin-dependent
glucose disposal. T3 requires distinct delayed peripheral and hepatic pathways.
The alien-device task explicitly requires input-driven memory. These are more
appropriate contracts for a memory-omission example than T1.

The current T2 inventory does not have a good positive Autoformalism endpoint:
minimum aggregate training NMSE is 0.68463 for easy and 0.36828 for hard. Even
the lowest individual U training NMSE among T2-easy records is 0.41343. The
historical Full/no-specification alien-device fits are also poor (training NMSE
0.472--0.943). The current local inventory provides no corresponding T3/T4
Autoformalism endpoint comparison. External models alone do not supply a
missing Autoformalism result.

## Recommendation

Keep the R9 figure as an exploratory dependency/intervention illustration. Do
not reinterpret it as evidence that an explicit required mechanism was omitted.

If the stronger benchmark claim is essential, use a bounded T2-easy experiment:
a model with a trained insulin-action state, versus an explicitly labeled
instantaneous-action ablation. Fit both on training data only, with comparable
budgets and a declared adequacy criterion; freeze both before a fixed family
of insulin pulse/spacing interventions. Regenerate reference targets and all
permitted auxiliaries jointly. Report U as well as Gp and I, every schedule,
and failures. No outcome is guaranteed by the proposed equations.

For a lag state X, X'=(I-I_b-X)/tau and U=Uii+g(X,Gt), different insulin
histories can produce different U at the same current I and Gt. An
instantaneous law U=Uii+g(I-I_b,Gt), without another memory path, cannot express
that dependence. Exact equal-current/different-history points must be checked,
not assumed from nominal pulse schedules.

This would isolate the value of the required delayed mechanism. A manually
constructed instantaneous ablation is not a no-specification proposer run;
evidence about specification-guided discovery requires an actual matched
specification ablation. No fitting, new rollouts, proposer calls, remote
sessions, or benchmark-test evaluation was performed for this search.

Verification: the 14 existing case-study/meal-probe tests pass. Repository-wide
Ruff reports the same 37 pre-existing findings; no implementation files changed.
The aggregate counts and quoted minima were recomputed from the saved JSONs.
