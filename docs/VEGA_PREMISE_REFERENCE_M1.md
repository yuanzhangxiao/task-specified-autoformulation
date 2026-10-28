# Vega M1: finite reference for scientific premise revision

27 October 2026. This is an isolated synthetic correctness milestone for the
[premise-revision formulation](VEGA_PREMISE_REVISION_THEORY_HANDOFF.md).
The accompanying [reply to Lyra](VEGA_TO_LYRA_WAITING_PERIOD_REVIEW.md) records
agreement, independent criticism, and questions for the next research decision.

## Outcome

Vega reviewed Lyra's plan and both supplied programs, reproduced the finite
control, and implemented a durable reference in the repository. The 45,000
datasets reproduce every retained-set mask in the supplied packet. A run stopped
after two worlds and subsequently resumed produces byte-identical evidence to an
uninterrupted run. The implementation adds direct Gaussian event probabilities
and adversarial error-scoring checks.

The independent statistical finding matters more than the zero-error count:
under this particular design, only **0.0318488 datasets with a false premise claim
are expected among 45,000**. The probability of observing none is approximately
**96.8653%**. This is a correctness reference, not a demanding empirical test of
calibration, a new method, or evidence of biological realism.

## Scientific contract

The elementary laws stay fixed:

\[
L'=-\kappa L,\qquad R'=-\gamma R,\qquad x'=LR-x,
\qquad (L(0),R(0),x(0))=(1,1,0).
\]

The two premises are constant ligand, \(\kappa=0\), and constant response
capacity, \(\gamma=0\). They are named physical restrictions on one common
family. This fixture does not demonstrate automatic discovery of new premises.
The nine admitted worlds have \((\kappa,\gamma)\in\{0,0.1,0.2\}^2\).
They constitute the whole declared envelope; this catalog is not a conservative
discretization of continuous rates.

At time 5, eight independent output readings have standard deviation 0.05;
eight independent ligand readings have standard deviation 0.10. Stage 1 uses
the output mean; stage 2 intersects its surviving worlds with the ligand-mean
interval. With \(z=\Phi^{-1}(1-0.05/4)\), the radii are
0.03962277670148705 and 0.0792455534029741. Each channel interval has coverage
0.975. A union bound protects all claims across these two fixed stages with
probability at least 0.95; independence makes joint coverage 0.950625.
These thresholds do not support arbitrary repeated or adaptive testing.

Support, rejection, and joint conflicts are universal statements about a
**nonempty** retained set. All compatible violation patterns remain available;
there is no preferred-repair pruning. Empty sets suppress claims and remain in
every denominator. Empty compatibility may result from a sampling error and
does not itself prove that a physical assumption is false.

For example, the worlds \((0,0.2),(0.1,0.1),(0.2,0)\) have exactly identical
output responses. Their mask is 84 under zero-based catalog indexing. The
retained set supports neither premise and rejects neither individually, but
conflicts with their conjunction. The three possible failure patterns are
ligand alone, capacity alone, and both. More precise output-only fitting cannot
resolve this ambiguity.

## Implementation and relationship to Orion's evidence

| File | Responsibility |
| --- | --- |
| `src/autoformalism/research/premise_reference.py` | Fixed catalog, stable analytic response, predicate and diagnosis projections, truth-free classification, separate evaluator, analytic probabilities, sealed checkpoints and reports |
| `scripts/run_premise_reference.py` | Standalone CLI, bounded execution and read-only reporting |
| `tests/test_premise_reference.py` | Exhaustive semantics, adversarial observations, numerical limits, integrity and resume tests |
| `docs/VEGA_TO_LYRA_WAITING_PERIOD_REVIEW.md` | Scientific review and collaborator discussion |
| This document | Reproduction record and operational limits |

No production fitting or construction interface changed. Orion's fitting
diagnostics distinguish local information, sensitivity, numerical reliability,
and practical uncertainty. None supplies the global exclusion operation merely
by reporting a bad fit, a local optimum, or a rank result. This fixture implements
that operation by exhaustive enumeration and a prespecified observation model.
It is a small reference for a future evidence adapter, not an adapter claiming
that current nonlinear fits already yield calibrated world sets.

The matched diagnosis implementation derives conclusions from complete
violation patterns. Agreement over all 512 world subsets validates semantic
equivalence; both methods receive the same retained set. This is not a comparison
against the computational performance of published diagnosis algorithms.

Each world checkpoint stores every raw replicate and PCG64 states before and
after generation. The plan binds configuration, catalog, acquisition order,
implementation hash, package versions and platform. Writes are atomic and the
runner holds an exclusive lock. Resume validates seals and a contiguous state
chain; reporting recomputes conclusions from saved evidence and never generates
missing datasets. Completed runs are immutable. An interrupted unfinished world
is replayed from its previous world boundary. Raw replicates precede evaluation;
the inference function receives no generating-world ID.

Hashes detect accidental alteration, not deliberate rewriting and resealing of
an entire output root. Reports do not hold the writer lock; concurrent inspection
can fail validation transiently. Source or runtime changes require a fresh root,
including for report reconstruction. These are explicit operational limits.

## Recorded experiment

Configuration: 5,000 trials per world, seed 20261028, sequential NumPy PCG64,
all output replicates followed by all ligand replicates for each catalog world.

| Outcome | Count / 45,000 |
| --- | ---: |
| True world retained after stage 1 | 43,854 |
| Empty stage-1 set | 1,133 |
| Correct final singleton | 42,753 |
| Wrong final singleton | 7 |
| Empty final set | 2,240 |
| Any false support across either stage | 0 |
| Any false rejection across either stage | 0 |
| Any false joint conflict across either stage | 0 |

All seven wrong singletons exchange worlds \((0.2,0.1)\) and \((0.2,0.2)\),
which violate both premises. World recovery error therefore differs from
premise-claim error. Correct singleton, wrong singleton, and empty final set
partition all 45,000 datasets.

The analytic correct-singleton probability is 0.950625 for every world. The
largest per-world probability of a false premise claim across both stages is
\(3.1833348461\times10^{-6}\). The analytic calculation integrates the disjoint
Gaussian acceptance regions; it includes a false stage-1 claim followed by an
empty stage-2 set. See the collaborator reply for its formula. Evaluation uses
ordinary floating-point normal CDFs, not validated interval arithmetic.

Each world's separately reported 95% Clopper-Pearson interval for its zero
false-claim count is [0, 0.0007375038]. These intervals are not simultaneous over
worlds or metrics, and the pooled count does not prove zero population error.
No `upper CI < 0.06` screen is used as certification of the nominal 0.05 bound.

## Reproduction and provenance

The reviewed ZIP SHA-256 is
`8da2b8d5fcccebe72fed6c01272762e1f17e884e124d9ce0b01384ddd64c13ee`.
The standalone plan and ZIP plan are byte-identical. The supplied finite source
matches the source hash in its accompanying result. The original source was
replayed in an isolated ignored output directory after review.

Runtime for Vega's runs: Python 3.12.10, NumPy 2.3.5, SciPy 1.16.3,
macOS 26.5.2 arm64. The reference implementation SHA-256 is
`f456a7e3bdf09b6d3314e1717f0a03666b5487ad673e071a95f5db9ddc6fa7f1`.

All 45,000 sample-mean pairs and both-stage masks match the local replay of
Lyra's source exactly. Against the supplied packet, all masks and counts agree;
maximum absolute differences in output and ligand means are respectively
\(2.22\times10^{-16}\) and \(4.44\times10^{-16}\). The cross-runtime comparison
therefore establishes matching decisions with roundoff-level mean differences,
not byte-identical floating-point data across machines.

From the repository root, a fresh reproduction is:

```bash
.venv/bin/python scripts/run_premise_reference.py \
  --output-root output/premise-reference-reproduction \
  --trials-per-world 5000 --seed 20261028 --max-new-worlds 2

.venv/bin/python scripts/run_premise_reference.py \
  --output-root output/premise-reference-reproduction \
  --trials-per-world 5000 --seed 20261028

.venv/bin/python scripts/run_premise_reference.py \
  --output-root output/premise-reference-reproduction --report-only
```

The first command reports 10,000 recorded of 45,000 expected datasets and lists
seven missing worlds. The second completes the same sealed plan. Configuration
or source changes are rejected rather than silently resuming a different run.

Local evidence is under `output/vega-premise-reference-20261027/`:

- `supplied/`: verbatim packet contents.
- `lyra-replay/`: replay of the reviewed finite source.
- `milestone1-full/`: final uninterrupted run.
- `milestone1-resumed/`: final run stopped after two worlds, then resumed.
- Each final run has `plan.json`, nine `world-N.json` files and `summary.json`.
- `milestone1-verification.json`: row comparison, artifact hashes, and resume
  verification; reporting under `python -O` also reproduces the same summary.
- Earlier `development-smoke/` and `reference-*` runs preceded the final
  numerical edge-case fix and are not the final milestone evidence.

Generated evidence, raw data and attached source copies are ignored by Git.
Only implementation, tests and the authored documentation are committed.

## Verification and limits

The focused suite passed **578 tests**. It checks all 512 projection masks,
the retained-truth soundness implication, adverse observed means that force
false support/rejection/conflict, empty-set behavior, independent ODE integration,
equal/near-equal and extreme rate limits, corrupted evidence, source/runtime
drift, missing checkpoints, partial reports, and byte-identical resumed runs,
including an injected interruption after RNG draws but before checkpointing.
The review found and fixed a subnormal-rate underflow in the response kernel;
regression tests cover that edge case.

Repository-wide `pytest -q -n 4` completed with **10,875 passed, 8 skipped,
1 failed and 1 setup error** in 23 minutes. The setup error was the paired-fit
source-integrity guard: the broad run overlapped the final response-kernel edit,
and that guard fingerprints the entire Python source tree. The failure was a
30-second, budget-limited detention fit. Both affected tests subsequently passed
in a serial rerun with the source fixed (**2 passed in 35.04 seconds**). The
final 578-test reference suite was run after the last numerical and recovery
changes. The original broad invocation is not reported as a clean pass.

The eight skips require optional PyTorch, unavailable in this environment.
`ruff check .` reports 37 pre-existing errors confined to untracked
`analysis/claude/` files; the new Python files pass Ruff. These unrelated files
were preserved. Logs are `pytest-full.log`, `pytest-serial-recheck.log`,
`pytest-focused-final.log`, and `ruff-full.log` in the evidence directory.

The scientific milestone is deliberately bounded: fixed equations and known
scales, known independent Gaussian noise, a finite envelope, and two fixed
measurements. There is no continuous exclusion certificate, adaptive acquisition
policy, biological benchmark, natural-language premise generator, or automatic
repair here. The second-pulse extension was reviewed analytically; a continuous
experiment awaits agreement on the unresolved mathematical obstacle and the
scientific task. GEMOT collaborators should help decide which preparation,
delivery, measurement and tolerance assumptions are realistic.
