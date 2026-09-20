# Optional shared-process review: detention-basin pilot

The development qualification returned eight successful reference-skeleton fits:
coupled/independent basins, two observation-noise settings, two broad starts.
Those results establish that this dataset can be fitted when the structure and
rating-law shape are supplied. They do not establish successful model discovery.
This milestone tests discovery without those private references.

## Construction flow

1. The existing variable phase selects states and algebraic quantities from the
   public specification, optionally with training-trajectory summaries (`full`).
   `brief_only` omits those summaries throughout this initial construction.
2. The review-on arm makes **one optional scientific review call** on the complete
   inventory. It can propose algebraic processes with scientific roles, existing
   drivers, and at least two suggested consumers. It cannot change a state,
   supply an equation, or prescribe a latent initial value. `processes: []` is a
   successful response. An already-declared algebraic process can be referenced.
3. The ordinary topology phase chooses the actual dependencies and signs. The
   review's consumer/driver suggestions are advisory, not forced connections.
4. The ordinary function and initialization phases construct executable laws.
   A process has one equation and shared parameters; each consumer can use its
   scientifically appropriate sign and area conversion. The process has no
   independent initial condition.
5. The unchanged public fitter estimates parameters and causal initial maps.
   Training drives fitting; validation is descriptive in this single-visit pilot.

Invalid review delivery/content retains the original variable inventory. If new
identities cause topology/function/reconstruction failure, construction retries
from the original inventory, reusing cached replies where the full request
matches. Both routes consume the **same total** request/token budget; there is no
extra fitting attempt. An exhausted total budget can still stop construction.
Allocation interruption is resumable and is not treated as an invalid review.
All optional-review outcomes and fallback reasons are recorded.

The review is opt-in. Existing campaigns and default construction are unchanged.
No benchmark prompts or data are changed.

## Frozen comparison

| Factor | Values |
| --- | --- |
| Public case | Coupled basins; disconnected negative control |
| Initial prompt | `full`; `brief_only` |
| Seed | 0, 1 |
| Optional process review | off; on |
| Total models | 16, each constructed and fitted once |
| Observation noise | Existing development `noise1` release (1% SD) |
| Proposer | Pinned GPT-OSS-20B, low reasoning, temperature 0.2 |
| Shared-process guidance | On in both arms |
| Total call budget per model | 128 requests; 524,288 charged tokens |
| Per-call output ceiling | 8,192 tokens, including reasoning |
| Fitter | Frozen `collocation-single-target-v2` |
| Numerical budget | 120 s initializer + 180 s refinement; 240 residual-call cap |
| Revision, pruning, scientific LLM judge | None in this pilot |

Keeping the model/reasoning level fixed isolates the extra scientific review.
Testing high reasoning or GPT-OSS-120B on one H100 remains a separate serving and
construction calibration, not a change folded into this comparison.

The launcher reads only six sealed public assets (specification/train/validation
for each case) and checks their hashes against the qualified release manifest.
It does not read the diagnostic directory, clean hidden trajectories, generating
parameters, or test data. The new plan contains no private reference model.

## Read results

`SUMMARY.md` and `summary.json` preserve all 16 planned rows, including failures.
They report review acceptance/empty/invalid outcomes, fallback, output train and
validation NMSE, model size, actual process reuse, request/token costs, and
BDF/Radau numerical replay agreement. Token totals have explicit missing-usage
counts; raw provider usage remains cached for audit. Numerical agreement is not
accuracy or scientific certification.

`EQUATIONS.md` makes the discovered laws and initial maps reviewable. Full
candidates, initialization, public reconstruction evidence, stage failures, and
raw/cached calls are under `results/<task>/`. Review transfer laws and area
conversions in both arms, including algebraically equivalent inlined laws.

The disconnected control additionally rejects syntactic upstream dependencies
in target-generating equations **or initial maps**. This is a conservative
necessary check: canceled expressions can be false positives. Public graph
requirements are checked in construction, but threshold physics, positivity and
general water conservation are **not automatically certified**. Named sharing is
an implementation witness, not a mechanism-compliance score. Broader equation
review is required before claiming scientific benefit or promoting this dataset
as a released benchmark.

Interpret paired output errors together with valid-construction rates and
scientific equation evidence. A positive pilot should create useful reuse in the
coupled case without inventing transfers in the disconnected case or losing
construction robustness. With only two seeds, this is developmental evidence,
not a definitive method comparison. No automatic follow-up is submitted.

## Run on ACES

Use a pinned git archive in group scratch, then run:

```bash
export AF_REPO_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/repos/basin-review-COMMIT
export AF_PUBLIC_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-development-v1
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v1
bash "$AF_REPO_ROOT/scripts/hpc/submit_detention_process_pilot_aces.sh"
```

The wrapper supplies the existing Python, vLLM image and cache defaults. Override
`AF_PYTHON`, `AF_VLLM_IMAGE` or `AF_HF_HOME` only if their locations differ.
It submits a CPU preflight, one H100 proposal job, sixteen CPU fits (four at a
time, one CPU/16 GB each), and a summary job. Repeating a completed submission
returns its saved receipt. Partial/uncertain submission stops for inspection
instead of submitting duplicates. Source and launcher hashes bind all stages.

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v1
cat "$ROOT/submission_manifest.json"
cat "$ROOT/SUMMARY.md"
cat "$ROOT/EQUATIONS.md"
```

Do not delete fit-start markers to obtain a fresh optimization budget. Diagnose
interrupted jobs from saved stage artifacts before authorizing recovery.

## Local verification

```bash
pytest -q tests/test_process_review.py tests/test_detention_process_pilot.py tests/test_detention_process_submission.py
PYTHONPATH=src python scripts/smoke_detention_process_pilot.py --output /tmp/basin-process-smoke
```

The smoke uses a small temporary fixture and prescribed provider replies, then
runs the real frozen fitter and independent solvers and verifies unchanged
resume. It tests plumbing, not GPT-OSS scientific discovery. Existing benchmark
assets are never modified.
