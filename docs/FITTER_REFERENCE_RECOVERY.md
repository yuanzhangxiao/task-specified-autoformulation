# Reference-audit recovery and fitter readiness

This milestone changes the isolated diagnostic audit and reporting. It does not
change the optimizer, candidate equations, initial guesses, fitting budgets, public
data, proposer/judge inputs, or the original strict synthetic recovery threshold.
Reference equations, parameters, known hidden boundaries and reference-control
results remain excluded from proposer and judge feedback.

## Why the audit changed

The v1 Delta reference audit rejected a maximum absolute replay discrepancy of
2.2113e-5 against an absolute cutoff of 1e-5. The public training output SD is
approximately 3.826. This discrepancy is negligible for fitting, but the hard
cutoff prevented all eleven reference controls from starting. Independent local
replays at tighter tolerances also differed from the saved benchmark by more than
1e-5 while agreeing closely with each other. This is an engineering correction
informed by the opened diagnostics, not a new pristine numerical holdout.

The v2 audit uses only the training output scale, sigma, for both splits:

* Native-generator versus saved observations: maximum difference <=
  1e-8 + 1e-4 sigma.
* Each tight compiled replay versus saved observations: the same bound.
* Tight Radau versus tight BDF: maximum difference <= 1e-10 + 1e-6 sigma.
* Tight integration uses rtol=1e-10 and atol=1e-12.

The native generator retains its original LSODA settings. The two tight replays
use the restricted compiled reference equations and the original continuous
forcing laws, with explicit integration boundaries at input jumps. They are
independent solvers; the native generator is also independent of the compiler.
These are diagnostic numerical agreement tolerances, not proofs of a global
error bound. None is adjusted after reading this new campaign's fitting scores.
Every trajectory must pass. Nonfinite values, grid mismatches, forcing mismatches,
reference-translation mismatches and substantial prediction disagreement still
block generation. Reference coefficients and covariates are not fitted by the
audit. The separate sampled-input truth replay remains in the generation report.

The complete local generation check also exposed an inherited 60-second budget
for replaying an entire split. The larger reference system exceeded it despite
finite trajectories. V2 therefore grants reference generation rollouts 240 seconds
per split and reference verification 240 seconds per solver/split. Proposed-model
verification retains 60 seconds. No optimizer receives additional time or calls.
The reference-generation worker has a 45-minute supervised ceiling within its
50-minute Slurm allocation; the original v1 worker retains its 25-minute ceiling.

## Recovery without rerunning completed work

The source is the original `fitter-attainability-v1` output, not the earlier
feasibility output. Preparation reads and validates its immutable inputs and
copies them into a new output root. It re-derives the reference candidate and
truth from the exported specification and verifies the public forcing/initial
condition protocol. Numerical library versions and Python major/minor must match
the original environment; fitting configuration and generation budget must match.

All 28 completed proposed-model results and four completed generation records
are retained byte-for-byte under `retained/`, with their original identities and
new snapshot hashes. The original failed reference-generation record is retained
as provenance. The new summary combines those retained results with the eleven
new reference arms; it explicitly marks retained records. Old timings are not
presented as a new paired speed comparison.

Only reference tasks without a previous numerical fitting attempt are eligible.
A previous `fit_started.json`, saved fit or numerical terminal result blocks
this recovery path. An interrupted fit cannot acquire a fresh budget. Missing
results or `generation_unavailable` caused by the old gate are eligible. The old
output root is never modified. Repeating preparation or submission is idempotent.
Existing jobs are returned rather than resubmitted; partial submissions require
queue reconciliation. The original source must be inactive when taking its snapshot.

One smoke job precedes one reference-generation task, eleven fitting tasks, and
a summary job. The fitter still has 300 seconds for collocation and 600 seconds /
480 calls for refinement. Each array task requests one CPU, 16 GB, no GPU, and
at most two array tasks run concurrently. No numerical fitting or generation runs
on login nodes. Process supervision and deterministic checkpoint semantics remain
explicit. A failed generation prerequisite is reported as
`blocked_by_generation` for unstarted tasks, or `generation_unavailable` when a
worker recorded that result. The original worker status is retained separately.

## Delta commands

Use a clean checkout of `codex/fitter-reference-audit-v2`. The original run already
contains the reference export; no additional Mac/ACES export is necessary.

```bash
export AF_REPO_ROOT="$PWD"
export AF_PYTHON=/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python
export AF_CASADI_ROOT=/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps
export AF_SOURCE_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-attainability-v1
export AF_OUTPUT_ROOT=/work/hdd/bibo/yxiao2/phase_b/fitter-attainability-v2
export AF_ARRAY_CONCURRENCY=2
bash scripts/hpc/submit_reference_recovery_delta.sh
```

After completion:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-attainability-v2/summary.md
```

If the summary job is unavailable, reconstruct the reports using only Python's
standard library, from the same checkout:

```bash
"$AF_PYTHON" -S scripts/run_attainability_campaign.py summarize --output "$AF_OUTPUT_ROOT"
```

`summary.json` contains all results and the full reference audit.
`diagnostics.json` additionally extracts initializer messages, the last equation
defect, refinement stage sources and stopping messages, and whether a stage's
returned parameters equal the selected parameters. It does not infer optimizer
convergence from wrapper placeholder flags or from a finite rollout.

## Current fitter status and a bounded route to freezing it

The working research profile is:

1. Validate restricted equations and parameter/initialization domains. Observed
   initial states come from observations; fitted latent initial values or causal
   map parameters are shared and estimated using training only.
2. Attempt standard joint collocation. Mapped assembly and a bounded mesh reduce
   setup cost. Observed/constant node guesses allow optimization without a valid
   initial long rollout. Retain useful collocation parameter checkpoints.
3. Screen deterministic alternative points with state-only rollouts, preserving
   the best valid training point across every handoff.
4. Refine smooth models with forward sensitivities. Use directional polling for
   supported piecewise models and configured recovery cases. Budgets are shared
   across recovery stages; invalid augmented integrations supply no derivative.
5. Score the selected parameters using fresh production trajectories and inspect
   independent replay agreement. Separate usable finite results, optimizer
   convergence, strict synthetic recovery, practical fit quality and scientific
   adequacy. Never optimize against validation or test observations.

The earlier finite-difference stagnation, missed sampled-input changes, fixed-zero
latent-boundary restriction and loss of valid handoff points have targeted fixes
and regression evidence. Forward sensitivities reduced cost in controlled tests.
Standard collocation remains the preferred initializer; alternating and soft
joint alternatives did not justify further development. Supported continuous
piecewise primitives have working polling controls, not a guarantee for arbitrary
piecewise expressions. Shared-initialization limitations remain a public-protocol
question as well as a fitting issue.

The latest completed proposed-skeleton controls passed all eight fixed-parameter
node tests. Their joint pipeline reached the predeclared strict threshold in
14/16 arms; the two other validation NMSEs were approximately 0.00163 and 0.0983.
These are misses of a strict diagnostic target, not automatically unusable fits.
Joint collocation acceptance itself was reported for only 1/16 synthetic arms;
refinement and saved-start evidence are needed to explain the other recoveries.
Actual-public-data fits are finite but weak and do not establish convergence or
structural impossibility. The correct-skeleton fitting result is still pending.

The frozen contexts in this campaign omit lagged target inputs and measure full
rollouts. Its shared-initialization lower bound applies to that information
pattern; it should not be generalized to predictors using additional informative
measured history. The final integration check must confirm the intended target
history and observed-state reset policy in the exact pre-fitting runtime.

We can freeze the **algorithm choice** now for pre-fitting experiments and stop
opening new optimizer families. Claiming the fitter is **validated and ready to
freeze operationally** still requires:

* The eleven reference controls: successful ordinary-start recovery matters;
  near-truth or known-initial-condition success alone is weaker evidence. Report
  strict NMSE and practical quality separately. If only oracle starts work, the
  remaining optimization problem has not been isolated away.
* A resolved interpretation of actual-data fits with known versus shared latent
  initials and sampled versus continuous forcing. An impossible boundary/data
  protocol should not be reported as an optimizer or proposer defect.
* One focused integration pass with Sol's exact public contract and production
  fitting profile: representative smooth, supported-piecewise and invalid-start
  candidates; bounded completion, checkpoint resume, truthful status reporting,
  and preservation of the best valid result. Reuse existing controls where the
  code/configuration is identical; do not rerun a broad algorithm comparison.

If the reference controls succeed from ordinary starts, one reference campaign
plus that integration pass may be enough for a versioned operational freeze.
That is conditional, not a promised number of iterations. A successful freeze
means a specified, tested algorithm/profile with known scope and honest failure
reporting, not universal global convergence, unique latent recovery, or excellent
fit for every proposed skeleton. Remaining failures should trigger narrow fixes
supported by diagnostics. These controls do not yet certify the whole benchmark
suite or every equation the proposer could emit.
