# Follow-up: validation_001 ground truth has a 4.36 K dip

The user explicitly asked whether the ground truth is a dip or no dip. I read
the existing private canonical CSTR specification for this post-hoc diagnostic.
No private information was supplied to a proposer or selection process. No
test trajectories were opened and no benchmark or implementation file changed.

This follow-up goes beyond the preceding synthetic-integrator diagnostic:
the actual canonical reference equations were solved for
`validation_tf_mid_pulse`, with Tf=350 K until minute 10, Tf=345 K until
minute 15, then Tf=350 K again. Each forcing interval was integrated separately
with a continuous state at its boundaries. All three physical states C, T and
Tj evolved; the historical flat auxiliary arrays were not imposed.

At pulse onset the heat-balance derivative changes by exactly -5 K/min because
the feed-flow rate is 1/min and Tf drops by 5 K. The initial state is the
reference equilibrium, so a flat temperature cannot solve this ODE.

Radau and DOP853, each at rtol=1e-10, atol=1e-12 and max_step=0.025 min, agreed
within 2.19e-9 across the three state arrays. On the 0.1-minute output grid:

- Initial temperature: 365.131295254 K.
- Minimum temperature: 360.773179343 K at minute 12.7.
- Temperature decrease: 4.358115911 K.
- Temperature at minute 15: 360.803595886 K.
- Temperature at minute 30: 365.131295254 K.

Re-running the original generator reproduced the saved flat temperature to
within 5e-10 K. Instrumenting its RHS revealed five calls: approximately
0, 0.003, 0.006, 30 and 30 minutes. **None occurred during the pulse.**
The missed-forcing defect is therefore confirmed for this actual reference
trajectory, rather than only reproduced on a synthetic example.

The No latent model's approximately 0.50 K dip has the correct direction but
substantially underestimates the reference response. Full's nearly absent dip
matches an erroneous saved reference. Its lower historical NMSE on this case
cannot establish better physical prediction. Neither model has been rerun with
corrected auxiliary channels here; no corrected comparative scores are claimed.

Diagnostic outputs are under
`artifacts/cstr-reference-pulse-audit-2026-09-22/`:
`audit.json`, `reference_comparison.csv`, and
`reference_pulse_comparison.png`. These are private-reference audit artifacts,
separate from the preceding public-only model-replay package. Preserve original
benchmarks and audit all affected protocols before a separately versioned
correction and matched reruns.
