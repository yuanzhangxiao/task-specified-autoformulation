# Benchmark reference integration and integrity audit

The corrected reference generator is identified by `reference-events-2`.
This is a numerical-generator correction, not a change to discovered models,
finalized prompts, the fitter, or previously released datasets.

## Corrected defects

### CSTR and alien device: forcing discontinuities

The old Phase-B helper made one LSODA call over the full horizon. A dense
`t_eval` only requests output samples; it does not ensure that the solver
evaluates the RHS inside a short pulse. The CSTR solver could therefore remain
at equilibrium while the saved input column contained a nonzero pulse.

Both families now use a shared trusted integrator that:

- extracts every step/pulse start and end from the actual protocol, including
  every component of a multi-pulse schedule;
- integrates separately between those boundaries, carrying continuous states;
- solves to the true interval endpoint even when it is not an output sample;
- uses the left-hand forcing value to finish an interval, then the right-hand
  value in the next interval;
- caps internal steps independently of output sampling and uses tighter
  reference tolerances;
- rejects invalid/nonfinite grids, boundaries, states and incomplete solves.

States, supplied auxiliaries and derivative labels are calculated together from
the resulting trajectory. At a forcing discontinuity, ordinary derivatives
are evaluated using the right-hand input. The canonical CSTR validation
feed-temperature pulse now produces the independently confirmed 4.3581 K dip.

### Dalla Man: boundary carry and derivative consistency

Dalla Man already segmented integration at meal and infusion events. Two
separate defects were found:

1. When a forcing boundary fell between output samples, the old code carried
   the last returned sample into the next interval instead of the state at the
   actual boundary. It now always computes and carries the true endpoint.
2. After a second meal, integration used remaining stomach content plus the
   new meal as the gastric-emptying reference, but exported derivative labels
   recomputed this reference from the latest meal alone. The simulator now
   retains the actual reference used in each interval when producing labels.

Meal samples are explicitly post-jump at time zero, interior events and the
final horizon. The old implementation already made interior samples post-jump
through mutation of a shared NumPy view; it was not consistently pre-jump as
an initial source reading might suggest. The correction removes that implicit
aliasing, rather than claiming that all historical event samples were wrong.
Duplicate simultaneous meal masses are added. Meals off the output grid are
rejected because otherwise their mass would be absent from the exported event
channel. Off-grid infusion boundaries remain supported and tested.

### A remaining input-contract issue

The reference simulator treats meal amounts as exact stomach-state jumps.
Discovery-model rollout currently treats supplied numerical channels as
piecewise-linear forcing. A 90 g event sample at time zero has interpolated
area 45 on a one-minute grid; an interior event sample has area 90. Neither
continuous pulse is literally a state jump.

This update does **not** silently change the public meal encoding or teach
candidates a private stomach-state mapping. A future release needs an explicit,
consistent choice: an event-aware public model interface, or a declared
mass-preserving finite-duration input used by both generator and runtime.
Changing only a plotting interpolator or the time-zero sample is insufficient.
The audit therefore distinguishes numerical checks from release readiness.

There is also a sampled-forcing approximation for CSTR and alien schedules:
their references use exact steps/pulses (or analytic smooth signals), whereas
the generic runtime uses linear interpolation of the public input samples.
A separate diagnostic integrated the same physical equations under that exact
table interpolant, restarting at every sample knot. Across all 26 schedules,
the maximum target difference was 0.8568 K for CSTR and 0.1827 for the alien
output. These are input-representation differences, not residual solver error.
An explicit common forcing contract is needed for an end-to-end numerical
equivalence claim. This update fixes skipped reference forcing but does not
silently change the runtime's existing interpolation policy.

## Coverage and checks

`scripts/audit_reference_integrity.py` covers the complete registered Phase-B
matrix: Dalla Man T1--T4, canonical and perturbed dynamics, plus CSTR and the
alien device. Named/obfuscated and functional/opaque conditions share numeric
content; each public tier and semantic projection is checked separately.
T3/T4 are included for repository coverage even though the current paper uses
T1/T2.

With `--include-test-protocols`, the audit checks 260 protocol/configuration
instances: 26 schedules for each of eight Dalla configurations and the two
other families. These include repeated schedules across Dalla tasks, not 260
independent benchmark systems. It simulates private test schedules for generator
verification, without reading existing test files or evaluating any candidate.
Public audit exports contain training/validation only.

For every numerical instance:

1. Compare all states, derived quantities and ordinary derivative labels from
   LSODA (`rtol=1e-10`, `atol=1e-12`, maximum step 0.5) against DOP853
   (same tolerances, maximum step 0.25).
2. Repeat using half the output interval and compare the shared sample times.
3. Check Dalla glucose mass balance, gut balance between jumps, exact first
   stomach-compartment response to meal jumps, total exported meal mass and
   the identities `G=Gp/VG`, `I=Ip/VI`, `U=Uii+Uid`.
4. Read back each public CSV and compare every target, auxiliary and input to
   the corresponding reference value at the declared serialization precision.
   Verify public leakage checks, split hashes and semantic-pair equality.

The predeclared solver/grid acceptance threshold is maximum channel discrepancy
divided by `max(1, max(abs(reference_channel))) <= 2e-6`. This is a numerical
consistency threshold, not predictive NMSE. An initial audit at looser primary
tolerances failed five CSTR derivative comparisons; tolerances were tightened
instead of relaxing the threshold.

Analytical regression tests additionally cover pulses narrower than output
spacing, off-grid starts/ends, boundary events and failed/truncated solvers.
The separate detention-basin development generator already restarts at each
forcing knot and checks DOP853/Radau agreement and water-volume conservation;
it does not need the CSTR integration change.

## Run and resume

From the project checkout, using the project's Python environment:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_reference_integrity.py \
  --output-root artifacts/reference-integrity-audit-v2 \
  --include-test-protocols
```

Omit `--include-test-protocols` for development schedules only. The output
contains private reference information and must not be attached to a proposer.
The command uses CPUs only, with no fitting, GPUs or LLM calls.

Repeat the exact command to resume. Each protocol has a sealed record and
hashed NPZ checkpoint. Source files, private specifications, suite configuration,
solver settings and numerical-library versions form the run identity. Changed
identity or modified records/arrays require a new output directory; consumed
work is not silently relabeled. Public cell exports are installed atomically
after completion. `summary.json` reports failures rather than treating a
finished process as an accuracy certificate.

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_reference_integration.py tests/test_reference_audit.py \
  tests/test_phase_b_generation.py tests/test_phase_b_public.py
```

## Release and experiment implications

Existing CSVs, benchmark prompts, cached scores and experiment roots are not
rewritten. Private/public bundle writers now refuse nonempty destinations.
New manifests record `reference_generation_protocol` and solver settings;
public manifests do not disclose private protocol names through this metadata.

The production release script now targets a separate
`phase_b_reference_events_v2` directory. Do not treat generation alone as
completion of a corrected scientific release: practical-identifiability gates,
public input semantics and downstream registration still need an explicit
release decision. The audit reports `release_ready=false` while the Dalla
event contract remains unresolved. Corrected CSTR data must be kept separate
from historical CSTR data even if benchmark task IDs are unchanged.

For any corrected benchmark used in comparisons, regenerate targets and
auxiliaries jointly and rerun all methods' fitting, validation selection and
frozen evaluation consistently. Replacing only ground-truth curves under old
checkpoints is an informative diagnostic, not a corrected benchmark result.
If an existing derivative overlay uses the old multi-meal Dalla labels,
regenerate that overlay too.

Independent solvers do not prove that every physical equation is scientifically
correct. This audit checks the implemented trusted systems, finite schedules
and public projections; it does not certify every historical remote release,
arbitrary new forcing, mechanism identifiability or model-discovery performance.

See the [implementation and result handoff](../work/agent-comms/2026-09-26-reference-integrity-fix-and-audit.md)
for the completed audit counts and historical differences.
