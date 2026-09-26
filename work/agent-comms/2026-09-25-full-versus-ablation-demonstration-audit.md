# Can Full versus ablations provide the requested demonstration?

The available local results do not yet establish a matched case in which Full,
no-specification and no-latent all fit training well, then both ablations fail
held-out interventions for equation-identifiable reasons. Two narrower stories
are supported, but their limitations differ. Weakening the comparator from Sol
to ablations does not remove the need to verify these conditions.

## Coverage checked

- The saved six-cell round-12 report and its paired endpoint audit in
  `docs/INTERVENTIONAL_DISCRIMINATION_EVIDENCE.md`.
- Independently replayed T1 Full/no-latent/no-specification endpoints in
  `artifacts/t1-curves-round12-2026-09-22/` and their exploratory interventions.
- All 217 fitted equation records in the older Dalla Full/Brief-only inventory,
  including recorded round-13/14 occurrences, for better Full training fits in
  the relevant cells. These records do not include newer ablation endpoints.
- The newer four-cell inventory has 1,334 distinct model records and 1,200
  planned visit records, all assigned to Full or Brief-only, with critic,
  verifier and shared-process variants. There is no no-specification or
  no-latent arm in this uploaded snapshot. Disabling the scientific verifier
  while retaining the specification is not the no-specification ablation.
- Canonical R9/R2 assisted repairs and their frozen comparative rollouts.
- The confirmed CSTR reference pulse defect.

No new fitting, simulation, LLM call, remote access or test-data access was
performed for this audit. It does not infer what might exist on a remote site.

## Missing dependency: strong equation argument, qualified comparison

Canonical-obfuscated T1 no-specification seed 0 has pooled training NMSE
0.043576. Its target and latent equations use meal input but none of the
supplied physiological auxiliary trajectories. Its initializers do not depend
on those auxiliary initial conditions either. Consequently,

    predicted_Gp(t) = F(meal_history, initial_Gp; fitted_parameters).

Changing physical initial tissue glucose Gt while preserving initial Gp and
meal input leaves every argument unchanged. This model must predict exactly
zero intervention effect. The trusted physical simulation instead gives a
nonzero plasma-glucose response. This is a dependency-based impossibility, not
an inference from an unfavorable NMSE or a count of latent states: this
no-specification endpoint has three latent states.

The closest well-fitted comparator is the assisted repaired Brief-only R9,
whose training NMSE is 0.041554. It contains an active positive tissue-glucose
term and captures much of the response. For -20%/+20% initial Gt, its fasting
relative squared response errors are 0.1261/0.1774, versus exactly 1/1 for the
zero-response no-specification model.

This is not an autonomous Full/no-specification ablation comparison. R9 is a
Brief-only endpoint with assisted sign repair and extra refitting, and the two
endpoints have different seeds. Absolute errors also do not uniformly favor
R9: at -20% Gt, NMSE is 0.00932 versus 0.02511; at +20% Gt the ranking reverses,
0.01944 versus 0.00788. Plot absolute trajectories alongside matched-control
differences if using this as a labeled illustrative example.

The original canonical-obfuscated Full models also ignore all physiological
auxiliaries, so they share this zero-response defect. The best training Full
record found in the 217-model historical inventory is seed 0 round 13,
training/validation NMSE 0.124109/0.172792; its compiled forcing inventory is
only `u01`. It does not solve the omitted-Gt issue. The previously evaluated
round-12 Full endpoints likewise fail this particular response probe.

Finally, T1's public requirement is a causal meal-response mechanism; it does
not mandate using every supplied auxiliary or explicitly require Gt. Omitted
Gt explains this model's limitation, but does not by itself prove violation of
the literal T1 requirement or that retaining the specification would prevent it.

## Missing delayed response: clear equations, training already poor

The saved named T1 no-latent seed-0 model reduces to

    Gp' = 1.12821256 - 0.00669917911 Gp
          + (1.47474750 + 15.6042152 exp(-17.2340450 t)) u.

It does not use the supplied physiological auxiliary trajectories. Once the
meal pulse ends, it relaxes toward 168.410568 mg/kg. If Gp is above this value,
it must decline immediately; it cannot produce the sustained post-pulse rise
in the reference. Full has additional evolving meal states and can delay its
peak. In validation_001 the reference, Full and no-latent peaks occur at
minutes 151, 143 and 91, respectively.

However, their pooled training NMSEs are 0.333330 and 0.457120; this is not a
pair of excellent training fits. Reinspection of all 16 saved training
trajectory scores found no trajectory on which both have NMSE below 0.05.
The failure is already visible during training. Validation_001 NMSE is
0.40098 versus 0.81618, which supports an expressivity/response-shape example,
not the stronger claim of indistinguishable good training fit followed by
interventional discrimination.

Do not generalize this proof to every model with no additional latent state.
The observed target remains a differential state, and supplied auxiliaries can
carry history. Some Sol endpoints demonstrate that possibility. The argument
here follows from this saved model's actual post-pulse equation.

## Other existing comparisons

Canonical-obfuscated T1 seed 1 has closer Full/no-specification training errors,
0.157364/0.171618, and validation errors 0.239654/0.337849. Both contain latent
meal paths. This is moderate predictive evidence, without the requested clean
missing-variable explanation. Seed 0 reverses the absolute ranking:
no-specification has lower training and validation error than Full.

The alien-device no-specification pairs do not have good training fits:
Full is 0.9435/0.7728 and no-specification is 0.6717/0.4722 across seeds 0/1.
They do not supply the low-training-error premise.

CSTR seed 0 originally looked promising as a rank inversion, but the attractive
validation-pulse plot used a reference trajectory whose solver missed the
input pulse. Correct integration produces a 4.36 K dip. Full's almost absent
dip matches the defective saved reference. This plot cannot support the proposed
physical mechanism story. The separate seed-1 CSTR comparison also favored
no-latent on the saved aggregate metrics.

## Recommended presentation and next decision

For existing evidence, use two explicitly narrower illustrations:

1. A missing dependency can make an intervention response impossible despite
   good training fit. The no-specification versus assisted Brief-only R9 case
   supports this, with correct provenance and both absolute/difference curves.
2. The specific no-latent equation cannot produce the observed delayed rise.
   Show its training and validation failures openly; this motivates expressive
   capacity rather than claiming a failure that emerges only after training.

For the requested clean autonomous Full/no-specification/no-latent comparison,
the missing evidence is a matched current-pipeline ablation cohort. A bounded
next experiment should select one benchmark cell from its public mechanism
contract and training suitability, retain Full and both ablations under the
same proposal/fitting budgets and seeds, inspect training fits and equations,
and freeze all endpoints before a fixed intervention grid. Prefer a public
contract in which the relevant path/memory is explicit and is not redundant
with supplied downstream mediator trajectories. T1-easy's rich auxiliaries
make a universal no-latent argument especially difficult.

No-specification means removing the specification from proposal and applicable
scientific selection checks, while retaining the data interface. Merely deleting
Gt from a chosen model is a manual dependency ablation and must be labeled as
such. Likewise, deleting a latent state without retraining a constrained model
is not a fair trained no-latent baseline. A new run may find an equally good
ablation; that outcome must be retained. No new campaign is launched here.
