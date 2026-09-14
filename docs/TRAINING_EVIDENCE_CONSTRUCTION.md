# Training evidence before construction

Milestone 3 adds the opt-in `prefit-training-evidence-1` packet. It accepts only
a `DatasetSplit` named `TRAIN`, validates the public target/input channels, and
hashes their actual arrays. It never opens validation, test, derivatives or
private reference equations. Reusing a split fingerprint cannot hide changed data.

The packet includes initial-target ranges over all training trajectories and a
bounded, evenly selected set of trajectory summaries. Each summary contains exact
sampled extrema, turning points, a sampled path, RMS, time-step range, input-change
brackets and coincident target differences. Counts disclose omitted trajectories
and events. Single-sample data has unavailable time-step information. Constant
channels and variation below the frozen descriptive tolerance produce no turns.

These are observations, not confidence estimates or required model components.
Input/output coincidence does not prove a causal lag. Sampled zero input does not
establish zero latent state or autonomous oscillation. Turning points may reflect
noise; their threshold is not an estimated noise model. Between-sample extrema
remain unknown. Initial variability does not establish latent identifiability.

`run_staged_topology` and `run_staged_functions` accept `training_evidence=packet`.
The exact same validated packet reaches variable, equation-topology, function and
causal-initializer requests. It is a separate `training_observations` field inside
the public brief object; finalized public scientific prose remains unchanged.
No extra topology or functional restrictions are derived from these observations.
Omitting the argument preserves the legacy prompt payload. A used packet cannot
be changed or removed while resuming the same construction root.

From this commit's checkout and configured Python environment:

```bash
PYTHONPATH=src python -m pytest -q tests/test_training_evidence.py tests/test_staged_topology_runner.py tests/test_staged_function_runner.py tests/test_causal_initialization_construction.py
PYTHONPATH=src python scripts/smoke_training_evidence.py
ruff check .
```

The focused suite passed 47 tests. The synthetic packet smoke and Ruff passed.
The next milestone freezes and compares both presentation arms; this milestone
alone does not establish that the evidence improves live construction or fitting.
