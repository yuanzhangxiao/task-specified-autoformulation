# Training residual evidence at retained parameters

This exporter supports the optional sibling experiment described in
[PREFIT_NUMERICAL_SIBLING.md](PREFIT_NUMERICAL_SIBLING.md). It performs no parameter
optimization and sets no continuation policy. Its parent is the completed first
public-fit continuation, independently of the ongoing longer diagnostic.

## What the measurements establish

The exporter replays the saved complete parameter vectors, including learned
causal initialization coefficients, using the same production equations and
numerical settings. It first checks that the current training NMSE reproduces the
stored result (relative tolerance 1e-5, absolute tolerance 1e-8). A mismatch or
failed current replay prevents the packet from reaching the proposer. This is a
reproduction check, not an independent-solver accuracy check.

For each observed training target and trajectory, the packet reports:

- input range and sampled regime: zero, constant nonzero, varying, or no inputs;
- observed and predicted initial/final values, means, ranges and sampled reversals;
- NMSE, signed bias, and maximum absolute normalized residual;
- error in early, middle and late equal-duration portions of the trajectory;
- the same measurements at the preceding fitted vector, when its replay is available.

The residual is **prediction minus observation**, divided by the target's global
training standard deviation. Every trajectory/window uses that same scale. The
aggregate NMSE weights samples equally, matching the fitting score; window scores
are descriptive rather than a new objective. Empty time windows are omitted.

For example, low early error and high late error establishes that this retained
fit deteriorates later in that trajectory. It can motivate testing longer memory,
a different feedback response, or changed initialization, but does not identify
which explanation is correct. A predicted constant signal beside observed sampled
reversals similarly motivates an oscillatory-response hypothesis without proving
that the model structure cannot already represent that behavior.

Sampled turning points count reversals from a running extremum exceeding
`1e-8 + 0.01 * observed_range`. Accumulated change matters, so slow variation on a
dense time grid is not discarded just because each adjacent step is small.
Observed and predicted signals use the same threshold. This is neither a fitted frequency nor a statistical
oscillation test; sampling, noise and unresolved between-sample behavior matter.
Zero input describes sampled external drivers only, not the absence of latent
initial conditions, supplied auxiliaries or internal dynamics.

## What reaches the proposer

The packet has a strict schema and content hashes binding the exact lowered
candidate, full retained parameter vector, public training data and replay.
Numerical qualification includes the unresolved/converged status, budget flag,
cumulative residual calls, latest continuation seconds, and recent training costs.
It explicitly states that structural failure is not established. The wrapper
provides the matching fitted coefficients separately from scientific initial
guesses.

The compact packet includes at most 64 trajectory/target rows and six detailed
examples, selected deterministically from largest error, smallest error, the
largest zero-input error when available, then remaining high-error examples.
Each detail includes at most twelve exact aligned samples. Priorities are
endpoints, largest residual, observed extrema, first input/auxiliary change
brackets, sampled reversals, then evenly spaced fill. Counts of omitted rows and
samples are explicit. Evidence IDs support citations in the proposer response.
The full observed training arrays remain in the frozen input; complete predicted
training arrays remain in the replay artifacts.

No validation/test metrics, hidden-state trajectories, derivative labels,
structural diagnosis or mandatory repair recommendation enters the packet.
Observed mismatch persisting through two parameter vectors is still not proof of
structural failure. Continuing fitting and proposing a scientific alternative are
separate experimental branches.

## Checkpoints and failure handling

Both historical fits are verified and read without changes. New output must be
outside their experiment directories. Each fixed replay has a 300-second limit.
The current and preceding points have separate started markers and sealed results;
resume reads completed results and never grants a fresh clock to an interrupted
point. A completed current replay remains usable if the optional preceding replay
fails. Interrupted current evidence blocks proposing. Source, runtime, lineage,
result and packet changes are rejected.

The CPU smoke `scripts/smoke_residual_feedback.py` exercises real ODE replay,
learned latent initialization, parent immutability and deterministic resume. Its
original timeout history is explicitly synthetic and is not scientific evidence.
