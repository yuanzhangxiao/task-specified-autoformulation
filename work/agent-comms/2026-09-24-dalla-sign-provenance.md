# Sign provenance in the six Dalla rescue endpoints

## Finding

Yes: fitting can reverse the visible algebraic sign in these saved models because
the original direct coefficients are declared `role=coefficient, domain=real`,
with no numerical bounds. This is not evidence that the optimizer crossed a
nonnegative bound. All **67** selected parameters satisfy their compiled domains.
Every surviving original equation-parameter declaration is unchanged by rescue.
Negative latent initializers are distinct from negative equation coefficients.

The round-zero `review-deadline-v2` construction fit requests already contain
these six real-valued direct coefficients for both relevant lineages. Thus this
freedom predates the R4/R9/R13 revisions and the new rescue campaign. The later
revision compiler preserves inherited roles; it derives nonnegative roles only
for eligible newly introduced gains. Re-running the current fitter on a saved
request therefore does not retroactively impose fixed signs on its old gains.

## Exact examples

Names below abbreviate the complete canonical parameter IDs in the audit.
Values are from the actual retained rescue endpoint, not a fresh trial.

| Model | Saved unfitted expression | Parameter before rescue | Parameter after rescue | Assembled result |
| --- | --- | ---: | ---: | --- |
| Brief R9 | `+k1*u01` | -0.487219 | -0.435588 | Negative direct meal contribution |
| Brief R9 | `+k2*v02` | -1.858307 | -1.911554 | Negative production contribution |
| Brief R9 | `+k3*v03` | +2.756999 | +2.940452 | Positive utilization contribution; no sign reversal during rescue |
| Brief R9 | `+k4*v04` | +155.76859 | +145.98558 | Positive excretion contribution; no sign reversal during rescue |
| Brief R9 | `+k6*v01` | -0.087630 | -0.086633 | Negative target damping; a negative gain is appropriate here |
| Full R4, original start | `+k_EGP*EGP` | +0.1 | -1425.68885 | Production changes from source to sink |
| Full R4, original start | `-k_Uii*Uii` | +0.1 | -15070.11830 | Utilization changes from sink to source |
| Full R4, original start | `-k_Gt*Gt` | +0.1 | -43.33153 | Tissue term changes from sink to positive return flow |
| Full R4, R13 start | `+k_EGP*EGP` | -7.19331 | -92.34782 | Negative production already present in transferred seed |
| Full R4, R13 start | `+k_meal*Y` | -6.09465 | -304.53167 | Negative meal-chain coefficient already present |
| Full R13 | `+k_EGP*EGP` | -7.19331 | -7.19361 | Negative production persists |
| Full R13 | `-k_Uii*Uii` | -164.09475 | -164.11044 | Positive utilization persists |
| Full R13 | `+k_meal*Y` | -6.09465 | -6.09100 | Negative coefficient on one meal-related latent term persists |

R4 from R13 and Full R13 also retain negative k_Gt, hence positive assembled
Gt contributions. Neither hard endpoint has a negative equation coefficient;
their negative latent initializers, extreme timescales and structural omissions
are different questions. This analysis does not say each latent term's sign is
its overall input-response sign: multiple pathways and internal differences
must be accounted for.

R9's production and meal coefficients were already negative in its historical
R9 fit. The initial v2 model had production coefficient -0.37673 and target
damping coefficient -0.12051. Thus the recent rescue did not create all these
problems. Its utilization coefficient changed from negative in that initial
model (-0.05377) to positive by R9, while retaining the same real domain.

The physiological interpretation of anonymous channels is used here only for
post-selection analysis. It must not be silently supplied to an obfuscated
proposer. In the named task, public descriptions explicitly identify production,
utilization, and excretion; those descriptions are available to its proposer.

## A sign token is not always a sign constraint

For a contribution `s * a * f`, fixing s to +1 or -1 requires a nonnegative
outer magnitude a if fitting is to preserve that outer sign. A generic real
coefficient lets a reverse the contribution. Moreover, this does not guarantee
the sign of f itself or of its derivative with respect to a source.

The existing topology/function interface distinguishes three choices:

- `positive` or `negative`: the topology owns the outer sign; an identifiable
  scalar gain is derived as nonnegative.
- `unrestricted`: the complete function may be signed; a real direct gain is
  allowed. A visible plus/minus in its expression does not impose a numerical
  sign bound. The normalizer can deliberately convert a proposed nonnegative
  direct gain back to a real coefficient under this contract.

Similarly, `reconstructed_interactions.outer_sign` in a revised bundle is an
algebraic decomposition of its RHS. It is not evidence that the original
proposer selected a fixed topology polarity. Interpreting all reconstructed
plus signs as positivity promises would be incorrect.

Current relevant implementation:

- `staged_functions.normalize_outer_weight_reply`: fixed versus unrestricted
  gain-domain derivation.
- `construction.assess_functional_compatibility`: rejects an underived fixed
  outer gain domain; preserves legacy sign rules separately.
- `review_revision_v4._resolve` and
  `staged_multiround_feedback_campaign._effective_revision_parameter_roles`:
  inherited parameter roles survive revisions.
- `fitting.fitter._parameter_variable`: real gives unrestricted bounds,
  nonnegative sets the lower bound to zero, positive sets it above zero.
- `fitting.casadi_initializer`: likewise imposes nonnegative/positive constraints.

The original pipeline explicitly enabled proposer ownership of signs not fixed
by the public contract. The raw topology/function records are nevertheless needed
to attribute these particular real domains to an intentional `unrestricted`
choice, a construction normalization, or a historical conversion defect.
The local fitted-model inventory contains the construction fit request, not
those original provider replies. We have not proved that the runtime discarded
an explicit positive/negative topology decision.

## Next audit, before another fit

Retrieve only the original v2 construction artifacts for the two lineages.
The command below runs on ACES; it does not submit jobs or alter models. It
reports absent paths and includes only existing named files. The output path
printed by `mktemp` is the archive to download.

```bash
AF_OLD=/scratch/user/u.yx126462/phase_b/review-deadline-v2
AF_PACK=$(mktemp -d /scratch/group/p.nairr260351.000/u.yx126462/dalla-sign-origin.XXXXXX)
(
  set -euo pipefail
  cd "$AF_OLD"
  AF_FILES=()
  for AF_TASK in cell01_seed1_brief_only cell02_seed1_full; do
    for AF_FILE in proposal.json topology/result.json functions/result.json; do
      AF_REL="results/$AF_TASK/round_00/$AF_FILE"
      if [ -f "$AF_REL" ]; then
        AF_FILES+=("$AF_REL")
      else
        printf 'Missing: %s\n' "$AF_OLD/$AF_REL" >&2
      fi
    done
  done
  if [ "${#AF_FILES[@]}" -eq 0 ]; then exit 1; fi
  tar -czf "$AF_PACK/sign-origin.tar.gz" "${AF_FILES[@]}"
  printf '%s\n' "$AF_PACK/sign-origin.tar.gz"
)
```

For an explicitly fixed sign, retain its nonnegative magnitude through
construction, revisions, fitting and pruning, with provenance at each boundary.
For an intentionally unrestricted sign, poor fitted scientific behavior should
trigger model review rather than be mislabeled as an optimizer bound violation.
Do not take absolute values of all fitted coefficients: that would, for example,
destroy R9's target damping and reverse R4's tissue return flow.

## Audit artifacts

`artifacts/dalla-sign-audit-2026-09-24/audit.json` records all 67 fitted values,
roles, domains and runtime bounds, plus the two original construction requests.
The standalone diagnostic script authenticates the uploaded result and inventory
seals and checks request/candidate identities. No model was changed, fitted, or
evaluated on new trajectories. No LLM call, test access or remote session occurred.

Verification: 62 existing sign/normalization/rescue regression tests passed, as
did the synthetic rescue smoke (three toy fits, none added on resume). Ruff
continues to report 37 pre-existing issues in unrelated `analysis/claude/` files.
No production implementation or benchmark was changed.
