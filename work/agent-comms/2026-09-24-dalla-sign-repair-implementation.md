# Dalla sign repair implementation

Implemented the authorized follow-up to the sign-origin audit as the opt-in
`dalla-sign-repair-1` campaign. Full policy and ACES commands are in
`docs/DALLA_SIGN_REPAIR.md`.

- Proposer input contains public task text and symbolic equations only. Anonymous
  channels keep their public names. No fitted vector, NMSE, trajectory arrays,
  private equation or intervention result enters the review.
- Isolated real outer gains receive explicit positive/negative/unrestricted
  decisions with reasons. Fixed signs become nonnegative magnitudes. Internal
  laws, shared gains, unrelated declarations and initialization remain protected.
- Four uploaded models have eligible gains (22 total); the two hard endpoints
  have none and consume no new review or fit allocation.
- Repaired and unchanged arms reuse the rescue fitter and paired pruning.
  Changed declarations receive ordinary role-based starts; no absolute-value
  repair is applied to the fitted vector. Warm starts differ and are disclosed.
- Selected repaired endpoints are checked for surviving operators, inner laws,
  domains and fitted bounds. Pruned gains are reported separately.
- Cached physical calls and scheduler receipts preserve consumed budgets and
  verified adoption. Incomplete GPU review does not release CPU dependencies.

Verification: 94 focused tests passed; synthetic end-to-end review/paired fitting
and exact resume passed. Both sign choices compile for every eligible uploaded
gain with compatible bounded starts, without a benchmark fit. Changed Python
files pass Ruff; repository Ruff retains 37 unrelated findings. A broad pytest
run was stopped after 423 passes and five missing-Torch skips, so a full-suite
pass is not claimed. No remote sessions, submissions or live LLM calls occurred.

This milestone produces a scientifically motivated candidate for evaluation;
it does not certify that a proposed sign is correct or that an intervention
demonstration will succeed. Historical results and other active agents' edits
are preserved.
