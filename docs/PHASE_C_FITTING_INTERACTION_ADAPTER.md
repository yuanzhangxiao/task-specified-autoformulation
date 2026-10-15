# M37: training-only fitting evidence for interaction proposals

Implemented 15 October 2026. This is an offline adapter and a small synthetic
qualification. It does not launch proposals, change functions or run a benchmark
campaign. The public specification, benchmark data and fitting defaults are fixed.
The broader integration plan is in
[interaction fitting feedback](PHASE_C_INTERACTION_FITTING_FEEDBACK.md).
The subsequent [M38 discrepancy extractor and readable draft](PHASE_C_FITTING_FEEDBACK_DRAFTS.md)
consume this verified packet, adding conditional investigations and explicit
tests of their predicted improvement without executing a proposal.

## What the adapter provides

`fitting/interaction_feedback.py` consumes a complete, verified M32/M33 diagnostic
folder. Portable verification binds the candidate, fitted vector, initialization
rules, training observations, executed diagnostic stages and search evidence.
A detached summary or an unverified fitted parameter dictionary is insufficient.

The sealed `result.json` and `request.json` contain:

- Numerical availability, independent rollout agreement, search/optimization
  status, local parameter information and the scope of any optimum witness.
- Every target's training NMSE and signed bias. Up to two trajectories per target
  show the worst and a contrasting best response, selected deterministically.
  Their full sampled arrays determine excursion, peak, half-return and early,
  middle and late residual summaries. Omitted examples/windows are counted.
- The complete public executable model and retained parameters, causal
  initialization plan, and syntactic dependencies from equations/processes to
  outputs. Shared process consumers remain visible together.
- An advisory response schema asking for the measured mismatch, a conditional
  functional hypothesis, its predicted observable improvement and evidence IDs.
  Outcomes are function hypothesis, topology revision needed, or no change.

The response adapter reconstructs predictions from the saved normalized residuals
and observed targets at the retained vector. It does not integrate again. Flat,
tied or boundary peaks, censored recovery, repeated input events and overlapping
inputs retain their ambiguity. Sampled delays are associations, not causal
effects or fitted time constants. Unavailable or disagreeing rollouts produce
explicit unavailable response evidence, never a fabricated failure mechanism.

The request excludes validation/test arrays and scores, evaluator expectations,
reference parameters, raw backend output and sampled hidden states. Each target
is retained. The default 48,000-byte user-payload cap fails explicitly if exceeded;
it is not a token count and does not silently trim targets or equations.

## Important boundary: equation locations are not edit permissions

An `ode:`, `process:` or `output:` reference identifies a displayed law. A complete
RHS cannot recover the constructor's authoritative per-interaction allowed source
set, outer sign or mechanism obligation. Every packet therefore says
`edit_contract_status=requires_constructor_binding`.

This milestone records hypotheses only. Applying one later requires the actual
construction contract, restricted-expression validation and an atomic candidate
transaction. Observation/initialization changes and missing states/sources need
an explicit broader-stage request. Reference spelling errors are recorded as
unverified; even an existing reference does not establish scientific support.
No citation error itself is interpreted as a failed scientific hypothesis.

An accurate fit does not certify the equations. A poorly fitted model does not
prove the specification false. Local rank does not establish global parameter
identifiability. The current clean benchmark contract is preserved; sensor drift
and robust losses from M34–M36 are not enabled by this adapter.

## Astra's supplied report and the first real integration target

Inspected `output/mechanism-realizations-analysis-20261014/REPORT.html` and its
source proposal artifacts. The reported 40/40 completions are structural topology
completion, not function generation or fit success. All ten basin proposals say
`function_generation_performed=false` and `parameter_fitting_performed=false`.
Executable laws, parameter domains, observation maps and causal initialization
maps still need to be supplied before those models can enter fitting.

Two provisional handoff candidates are:

| Public case | Task | Reason to inspect first; unresolved work |
| --- | --- | --- |
| Coupled detention basin | `cell06_seed1` | Distinct upstream overflow and external downstream outlet. Define the shared transfer and area/unit conversions explicitly; topology alone does not verify conservation. |
| Independent detention basin | `cell07_seed3` | Downstream equation separates inflow and a height/crest-dependent loss. Define its function, scaling and initialization explicitly. |

These choices use structural coverage, not fit scores or private equations.
Neither is scientifically certified. In particular, the coupled seed-4 model
lacks a declared external outlet; several independent models use warning depth
where crest height needs inspection. The supplied report is an interpretation
artifact; the sealed source proposals are the handoff identities.

Verified source payload hashes:

```text
plan: f8b1d591c247e1b0ec573539cd5b56e77650146451e8df27aa6945580bfbe573
cell06_seed1: 9ab71c3c3f39cc495a6d42a837e0e519bb64484663ccc5a09c76cbd41794292f
cell07_seed3: 599870387c9115dfafbc9361e2433aa72dec74cd9fa1f26974b185e19778f7cd
```

Source proposals live under
`output/jetstream-mechanism-realizations-inspection-20261014/results/`.
Public basin training uses six trajectories per cell with 61 samples over
0–240 minutes. The initial downstream reading and public upstream-initial input
must be respected; no private per-trajectory hidden initial may be fitted.

## Local qualification and cost accounting

`interaction_feedback_qualification.py` freezes four small executable controls:
an initial-value path, a locally stationary poor fit, an actual
piecewise kink, and a parameter-bound case. It uses the existing fitter and
staged diagnostics. Expectations are in a separate evaluator artifact and never
enter the request. These controls qualify evidence translation, not function
discovery or feedback efficacy.

The tests additionally cover multiple targets, shared-process consumers, empty
windows, repeated/ambiguous events, unavailable diagnostics, tampered identities,
validation/test/reference injection, citation warnings, prompt caps and interrupted
resume. Reports retain every planned task, failures and unknown costs. Fitting,
diagnostics and adapter CPU/wall costs remain separate. Packet export adds zero
optimization, rollout or LLM calls. Repeating a completed qualification verifies
the checkpoints without refitting or regenerating packets.

Local standalone qualification on 15 October completed **4/4 controls, all
passing**, in `output/phase-c-interaction-feedback-m37-smoke`. The translated
diagnoses were:

| Control | Preserved numerical conclusion |
| --- | --- |
| Initial-value path | Accurate output with supported smooth local evidence |
| Stationary poor fit | Local stationarity and poor output fit; no global impossibility claim |
| Actual kink | Accurate output; stronger smooth parameter conclusion withheld |
| Bound-limited direction | Accurate output; an infeasible perturbation limits the conclusion |

Measured adapter export plus verification took 0.11–0.15 seconds per control on
the local machine. Those times exclude the separately recorded fitting and
diagnostic work and are not cluster performance estimates.

Verification: 151 focused fitting/feedback/response regressions passed, including
80 tests added for M37. Standalone `report` and completed `qualify` produced
identical reports and left every saved file unchanged. Ruff passes on all seven
new Python files; repository-wide Ruff still reports 37 pre-existing issues in
unrelated `analysis/claude` scripts.
The full `pytest -q` run exceeded a 300-second local verification allowance and
was stopped before completion; the full repository suite is not certified here.

Run from a pinned checkout with the existing Python environment:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/phase_c_interaction_feedback.py qualify --root output/m37-qualification
python scripts/phase_c_interaction_feedback.py report --root output/m37-qualification
```

This tiny control qualification is suitable for a local development machine;
there is no GPU, provider key or cluster submission requirement. On Delta/ACES
run within an ordinary CPU allocation if reproducing there. Use a new root for
changed code/policies; rerunning the same command resumes a matching experiment.
`prepare` freezes without fitting; `qualify --index 0` runs only the first task.

To export an existing verified diagnostic folder, replace the paths below with
the actual source and a new destination:

```bash
python scripts/phase_c_interaction_feedback.py export \
  --diagnostics /path/to/verified-diagnostics --output /path/to/interaction-packet
python scripts/phase_c_interaction_feedback.py verify \
  --diagnostics /path/to/verified-diagnostics --output /path/to/interaction-packet
```

`report` and `verify` are read-only. Keep the source diagnostic folder alongside
the export so verification remains possible.

## Next milestone

Obtain one complete executable candidate and its authoritative construction/edit
contract from the current pipeline. Qualify the public fitting request and this
packet on that candidate. Then bind one permitted interaction or shared-process
edit atomically, preserve the parent on failure, and run the small paired
continuation/response-only/full-feedback experiment in the design document.
Do not turn topology completion into a behavioral claim or silently synthesize
missing functions from the private benchmark model.
