# Original topology signs in the Dalla rescue lineages

## Conclusion

The uploaded construction records close the provenance gap in
`2026-09-24-dalla-sign-provenance.md`. Both original topologies declare all six
terms `outer_weight_sign=unrestricted`. The named Full lineage is the ancestor
of the R4/R13 endpoints; the anonymous Brief-only lineage is the ancestor of R9.
No fixed positive/negative topology decision was lost in these original
construction handoffs. All twelve direct gains were allowed to have either sign
before fitting began.

The prior numerical audit found zero compiled-bound violations among the 67
retained parameters of the six rescue endpoints. Together, these checks locate
the problem in the admitted scientific sign decisions and their unrestricted
domains, rather than an optimizer crossing a supplied nonnegative bound.

This conclusion applies to these two original constructions and the inherited
gains traced in the prior audit; it is not a claim that every historical or
current sign-handling path is correct.

## Named Full lineage

Original task: `cell02_seed1_full`, round zero of `review-deadline-v2`.
Every row below has topology sign `unrestricted`, parameter role `coefficient`,
domain `real`, and no explicit numerical bounds.

| Source | Topology scientific role | Original function-stage expression |
| --- | --- | --- |
| meal_event_g | meal absorption contribution | `k_meal*meal_event_g` |
| EGP | endogenous glucose production | `k_EGP*EGP` |
| Uii | insulin-independent glucose utilization | `-k_Uii*Uii` |
| E | renal excretion | `-k_E*E` |
| Gt | tissue glucose exchange | `-k_Gt*Gt` |
| Gp | plasma glucose decay or clearance | `-k_Gp*Gp` |

The function proposer supplied all six coefficient declarations in its original
batch. All six functions were accepted without atomic repair or deterministic
role repair. The saved original replies already use `coefficient`; the runtime
did not convert positive/nonnegative declarations into real ones here.

Thus the proposer did encode familiar source/sink operators in several RHSs,
but did not protect their meaning with fixed topology signs or nonnegative
magnitudes. For example, `-k_Uii*Uii` remains a source when k_Uii is negative.
The R4 rescue fit was allowed to move k_EGP from +0.1 to -1425.69 and k_Uii
from +0.1 to -15070.12. Both visible sign reversals are legal under these saved
declarations and conflict with the stated source/sink interpretation.

The public named brief identifies production, utilization and excretion and
requires coherent source/sink/exchange signs. Its typed requirement, however,
has `public_pathway_sign=unspecified`; the saved polarity policy has no fixed
exact source sets. The scientific interpretation was not converted into a
machine-enforced fixed-sign obligation. This is a gap between scientific intent
and the accepted parameter contract.

Do not repair this by preserving every visible plus/minus blindly: the meaning
of tissue exchange, net rates, latent coordinates and competing pathways needs
separate review. A nonnegative outer gain also does not prove global monotonicity
of an arbitrary nonlinear function or of the complete dynamic response.

## Anonymous Brief-only lineage

Original task: `cell01_seed1_brief_only`, round zero of `review-deadline-v2`.
All six source sets (`u01`, `v02`, `v03`, `v04`, `v05`, `v01`) are unrestricted.
The final accepted functions are `k1*u01` through `k6*v01`, with real gains.

The original batch omitted all six parameter declarations; atomic repairs added
them as `coefficient`. The self term changed from `-k6*v01` to `k6*v01` in the
accepted repair. Since k6 is real, those two parameterizations permit the same
function family; their visible operators are not fixed scientific signs.

This public prompt explicitly forbids inferring an application domain and does
not disclose which channel is meal or production. Physiological names from
post-selection inspection must not become hidden repair instructions for this
anonymous arm.

## Why the deterministic sign audit passed

Both records set `proposer_owns_unfixed_signs=true`. Their policies have
`unfixed_source_ownership=proposer` and an empty `fixed_exact_source_sets` list.
Both audits report zero fixed-evidence terms and six proposer-owned terms.
`passed=true` therefore means the declared choices complied with that contract;
it does not mean physiological directions were independently certified.

The saved policy also contains `default_outer_weight_sign=unrestricted`. Under
proposer ownership, the code permits positive, negative or unrestricted; it does
not replace the accepted term's choice with the default. Whether this displayed
default influenced the model is untested. The archive contains accepted results
and request hashes, not raw HTTP response bodies or model reasoning.

## Recommended next milestone

Require an explicit sign decision consistent with the declared scientific role,
with a reason for unrestricted contributions such as signed deviations or net
fluxes. Fixed signs must produce nonnegative outer magnitudes and survive later
revisions, pruning and fitting. Publicly supported conflicts should receive
focused repair feedback; uncertain interpretations should remain explicit.

For salvage, a sign review using the public task and saved equations can create a
separate, versioned candidate and refit it using training data. Correcting signs
is a model change, not a replay of the old fit. Do not select signs from private
reference equations or intervention outcomes, and do not take absolute values
of saved fitted coefficients. Sign correction alone does not establish that the
remaining skeleton will reproduce held-out interventions.

## Verification and artifacts

- Uploaded archive SHA256:
  `cd4c1b9570b499e6a949f31e938aa899115537887f3c7e8f29b694e264f58faf`.
- Verified both proposal seals, embedded-versus-standalone stage equality, and
  each function result's source-topology hash.
- Verified both original fit requests reference the exact uploaded construction
  bundle and contain the identical candidate.
- Reconstructed both models with production `prefit_construction_audit.reconstruct`;
  both match their saved candidates and pass their deterministic certificates.
- Checked all twelve topology signs, accepted gain roles and compiled domains.
- Diagnostic artifact: `artifacts/dalla-sign-audit-2026-09-24/origin-audit.json`,
  SHA256 `5369dbd34ae2e1492a3a7284bb0d6b6ea756e4b52f0ff9f2e094b534e0476299`.
- Relevant sign, normalization and rescue regressions: 62 passed in 5.64s.
  Synthetic rescue smoke passed (three toy fits; zero additional fits on resume).
  `ruff check .` reports the same 37 pre-existing issues under `analysis/claude/`.

No production implementation or benchmark was changed. No Dalla fit, new
trajectory evaluation, LLM call, remote session or test-data access was performed.
