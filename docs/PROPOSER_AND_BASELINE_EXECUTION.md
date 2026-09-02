# Proposer-first dual-cluster execution

## Scope

This protocol shortens proposer iteration without weakening the Phase-B
train/validation/test boundary. ACES is the primary public development platform;
Delta is the independent operating-point confirmation and sealed-evaluation
platform. A different accelerator is allowed to produce a different valid
candidate. Cross-cluster promotion depends on the same frozen requests and gates,
not byte-identical sampled text.

No job in this protocol reads the Phase-B test split or a private reference.
Candidate selection artifacts may move from ACES to Delta only through a
hash-recorded handoff.

## Proposer operating-point sequence

1. The failed 4,096/8,192/12,288-token Delta analysis remains the prerequisite
   for `configs/phase_b_proposer_transport_calibration_v2.json`.
2. ACES runs the unchanged six round-zero public requests at 16,384, 24,576, and
   30,000 output tokens on two H100s. The model is GPT-OSS-120B, reasoning is
   high, the context ceiling is 32,768, and every original gate is retained.
3. The smallest setting passing all gates is selected. No search, fit, judge,
   test data, or private reference participates in this calibration.
4. The passing ACES analysis is copied to Delta with its SHA-256. Delta derives
   a one-budget plan from that analysis and repeats the same six requests on four
   A40s.
5. The operating point is promoted only if the independent Delta condition also
   passes every gate. Candidate identity or exact text equality is not required.

The relevant entry points are:

- `scripts/hpc/submit_phase_b_proposer_transport_calibration_v2_aces.sh`
- `scripts/hpc/submit_phase_b_proposer_transport_confirmation_delta.sh`
- `scripts/verify_phase_b_proposer_transport_confirmation.py`

The handoff binds the source plan, source analysis, generated one-budget plan,
model, reasoning effort, token budget, public request count, and platform labels.

## Baseline pilot while proposer work continues

`configs/phase_b_public_baseline_pilot_v1.json` freezes two cells and three
seeds:

- `phase_b_dalla_man_t2_canonical_named_easy`, easy;
- `phase_b_anonymous_system_task_canonical_opaque_hard`, hard.

The initial adapters have a concrete role:

| Adapter | Role | ACES resource | LLM budget |
|---|---|---|---:|
| SINDy | classical partially observed control | CPU | 0 |
| PySR | symbolic-regression control | CPU | 0 |
| D3 native/no-tools | matched LLM discovery-agent control | 2 H100 | 5 proposals |

D3 receives the same public prompt and train/validation data as Autoformalism.
It uses GPT-OSS-120B at the selected high-reasoning proposer operating point.
Five proposal generations match the five-round Autoformalism pilot budget.
The adapter remains explicitly labeled: it models supplied observed dynamic
channels, uses the repository's restricted expression runtime, and uses its
native Adam/teacher-forced update rather than Autoformalism fitting or judging.

SINDy and PySR may run before proposer calibration finishes. D3 cannot be
submitted until the passing proposer analysis is frozen into
`d3_llm_operating_point.json`.

The first classical pilot is executed on Delta CPUs using
`configs/phase_b_public_baseline_pilot_delta_cpu_v1.json`. It preserves the
same two cells, three seeds, algorithms, and budgets as the ACES plan, but
records `delta_cpu` in the frozen task and resource ledgers. SINDy starts
independently. A small CPU prerequisite initializes one shared PySR/Julia depot
before the PySR array starts, preventing six array tasks from racing during
Julia package initialization. PySR performs a deterministic, bounded joint
validation-rollout search over per-target Pareto candidates, so multi-target
Phase-B cells are supported without opening test data.

The public SINDy control uses the finite-difference derivatives attached by the
public tidy-split loader. Its sequential-thresholded least-squares grid is
explicitly frozen in the pilot plan and selected only by causal validation
rollout. The development grid spans `1e-4` through `1e4` because normalized
library coefficients can be large when public channels have heterogeneous
units. This wider grid permits genuinely sparse or null supports; it does not
raise the restricted parser's 512-node safety limit.

All ACES baseline runs use `--development-only`. Their result schema contains
train and validation metrics but no test field. The handoff tool rejects a
selection unless `test_data_opened` is explicitly false and excludes data,
private references, and raw LLM responses.

The relevant entry points are:

- `scripts/hpc/submit_phase_b_public_baseline_pilot_aces_cpu.sh`
- `scripts/hpc/submit_phase_b_public_baseline_pilot_aces_d3.sh`
- `scripts/hpc/submit_phase_b_public_baseline_pilot_delta_cpu.sh`
- `scripts/summarize_phase_b_public_baseline_pilot.py`
- `scripts/freeze_phase_b_public_baseline_handoff.py`

### Freezing completed development results

Before any sealed-test or oracle-state evaluation, the completed public
train/validation pilot is frozen with
`scripts/freeze_phase_b_public_baseline_development_results.py`. The freeze
copies and hash-binds the frozen plan and task ledger, submission manifest, all
selected result and terminal-status files, and all development-summary/resource
files. It also records that derivatives came from `numpy.gradient`, every
target was represented as an observed dynamic state, validation used causal
one-step observed-state resetting, and neither oracle derivatives nor oracle
latent states were used. A repeated freeze fails if any copied source artifact
has changed.

### Interpretation of the glucose-disposal target

In the named Dalla Man tasks, `U` is an observed derived process rather than a
native simulator state. The private simulator computes insulin-independent
utilization `Uii`, insulin-dependent tissue utilization `Uid`, and reports
`U = Uii + Uid`. Delayed plasma-insulin action is carried by the dynamic state
`X`; in the canonical simulator, `Uid = (Vm0 + Vmx*X)*Gt/(Km0 + Gt)`.

The components enter different compartment balances: `Uii` is removed from
the plasma-glucose balance and `Uid` is removed from the tissue-glucose balance.
Consequently, the simulator does not integrate a separate equation for `U`.
A mechanistic reduced model can generate the public target algebraically as
`Uii + Uid`. The public SINDy/PySR control instead embeds every target,
including `U`, as an observed dynamic state and fits `dU/dt` estimated from its
sampled trajectory. That is a valid black-box trajectory representation, but it
does not by itself establish the required total-disposal decomposition or
delayed insulin-action mechanism.

## Resource accounting

The frozen plan records CPUs, GPU type/count, wall-time limits, and maximum LLM
calls per task. Realized summaries report process wall time, CPU core-hours, GPU
hours, logical LLM calls and tokens, and provider attempts. Queue time is kept
separate and is filled from scheduler accounting after completion. Local
open-weight inference is reported in hardware units; it is not assigned an
artificial zero monetary cost.

## Promotion and deferred adapters

This two-cell pilot is a plumbing and feasibility check, not the paper's final
baseline table. Full suite execution waits for:

1. a proposer operating point that passes on ACES and Delta;
2. complete, hash-verified pilot outcomes;
3. sealed Delta evaluation through the common endpoint vector;
4. confirmation that the D3 comparison label and compute budget remain accurate.

HDTwinGen and a controlled latent neural ODE require new partially observed
adapters and remain separate milestones. MEDA is reserved for a named biomedical
case study because its retrieval/constraint interface is domain-specific.
LLM-SR is most informative on fully observed or oracle-state subsets. Oracle
SINDy/PySR remain upper-bound controls and therefore run only on Delta with the
private-state capability. Deep delay autoencoder and SKANODE are not part of the
first pilot because their published problem structures are not yet matched to
the controlled-input, first-order Phase-B contract.
