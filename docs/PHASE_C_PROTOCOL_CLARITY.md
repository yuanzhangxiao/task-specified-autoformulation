# Protocol clarity and three topology diagnostics

Version: 2026-10-14. Prompt policy: `minimal-protocol-clarity-1`.
Fresh study: `protocol_clarity_confirmation`; eight Full cases, five seeds each.
Historical policies, artifacts, benchmark prompts and data remain unchanged.

## Prompt review decisions

Each API call is independent and includes a short system instruction. Its small
dictionary defines differential state, algebraic variable, process/shared process,
pathway and memory, and maps `sources`, `depends_on` and `uses.target` to these
terms. Stage-specific directions remain in the stage instruction. The dictionary
occurs once in the system message, not again in the user payload. Keeping a stable
prefix can permit serving-cache reuse; this does not make omitted instructions
available to a later request.

`mechanism_bindings` is the historical JSON name of **memory-state assignments**.
It does not record every mechanism's realization. A mechanism may involve several
variables, algebraic quantities, processes and pathways. For example,

\[
\dot m=f(m,u),\qquad z=g(m),\qquad y=h(z)
\]

stores history in differential state m and expresses it through algebraic z.
The memory assignment names m, while the full mechanism includes the pathway.
An instantaneous algebraic relation is not an independent memory coordinate.
No new universal self-loop or decay requirement is imposed.

All public mechanisms still require representation. Their coverage and deferred
checks appear in `requirements_checklist`. The prompt now labels a memory choice
as `required` or an `existing optional history claim`, instead of displaying an
unexplained false flag. The longer assignment instruction appears during relevant
variable work or when assignment problems need repair, rather than every call.
The underlying schema and path quantifiers are unchanged. We do not silently
reinterpret this narrow annotation as a general mechanism-coverage map.

`flow` was informal scientific language, not an additional schema type. The new
shared-process instruction uses pairwise internal `transfer` and the existing
wire value `influence` for other effects. Exchange across a model boundary has
only its actual modeled uses: no artificial second balance is required. This is
generic for mass, water, heat and other exchanges, not a glucose-specific rule.
Opposite contribution signs alone do not certify conservation.

## Reply controls and deletion ownership

The new policy rejects exact reply-control names (`stage_complete` and the six
`remove_*` operation names) as variable, equation or process declarations before
the transaction commits. General names such as `terms`, and ordinary names with
these strings as prefixes or suffixes, remain available. A conflicting public
name is reported as an explicit schema/public-namespace ambiguity; nothing is
silently renamed or deleted.

Wrong-category deletion feedback lists the actual retained owners and their
conditional removal keys. Rejected additions were never installed. For a
historical equation called `stage_complete`, the correction is
`remove_equations`, not `remove_variables`; no corrected subset of an invalid
transaction is committed automatically. The saved ownership archive has exactly
two parsed replies newly rejected by this guard: the known T1-hard metadata
process and Alien-device metadata equation. This is counterfactual replay, not
two newly repaired scientific models.

## More useful bounded repair

| Allowance | Previous | New |
| --- | ---: | ---: |
| Initial construction requests | 123 | 123 |
| Initial construction budget units | 393,216 | 393,216 |
| Global repair requests | 5 | 10 |
| Global repair budget units | 131,072 | 262,144 |
| Total requests | 128 | 133 |
| Total budget units | 524,288 | 655,360 |

Serving context and output limits are unchanged. Actual reported tokens replace
conservative reservations; these are ceilings, not predicted consumption.
Repair ends on success, budget/deadline exhaustion, or three consecutive repeated
semantic attempts without topology progress. Reordering declarations and changing
explanatory prose or the completion flag do not reset this repetition counter.
A different rejected structured alternative may be tried. No consumed budget is
reset on resume. The full edit history remains, with explicit repair-call-5 and
repair-call-10 checkpoints when reached. These are within-episode checkpoints,
not a randomized five-call versus ten-call comparison.

## Three diagnostics: offline qualification first

`search/topology_compatibility.py` implements only:

1. **Algebraic single-valuedness:** equal complete supplied arguments require
   one prediction, so differing observations cannot both be interpolated exactly.
2. **Identical closed subsystems:** equal complete initial conditioning, forcing
   histories, covariates and parameters imply identical predictions under a
   common deterministic law with a unique solution.
3. **Scalar autonomous reversal:** a directly observed, closed scalar state with
   constant forcing cannot rise and then fall under a unique autonomous ODE,
   absent resets or impulses.

All are conditional diagnostics, separate from structural acceptance and fitting
quality. A nonzero unavoidable exact-interpolation error is not automatically
unacceptable NMSE. Measurement uncertainty remains explicit. Generated quantities
are not replaced by measured histories; unknown latent initialization, undeclared
conditioning, incomplete forcing or unavailable assumptions cause deferral.
The implementation neither chooses a missing dependence nor reads reference
equations or held-out outcomes.

The evidence schema freezes all training target/input/auxiliary arrays, covariates,
times and explicit conditioning metadata with a digest. It deliberately rejects
old bounded prompt summaries as a substitute for complete evidence. The source
split must be training before its arrays are read. Interpolated forcing is checked
on complete defining knots, including union knots when grids differ.

The JSON audit preserves the full evidence. `REPORT.txt` is a shorter text
projection containing conclusions, hypotheses and cited witnesses; it is suitable
for review without a second verbose JSON rendering. No new initial training
highlights are added in this milestone. Compact observations can later occupy a
text-valued portion of the request while replies remain typed JSON. Abbreviations
or fewer characters do not automatically mean fewer model tokens; measure with
the deployed tokenizer before claiming token savings.

The diagnostics remain **offline**, not wired into live acceptance or feedback in
the new prompt pilot. Qualify the witnesses on all saved seeds first, then enable
candidate-specific questions under a separate frozen version. The broader
properties in `PHASE_C_TRAJECTORY_COMPATIBILITY.md` remain deferred.

## Saved-model qualification

All 40 ownership-feedback drafts were assessed against complete public training
evidence. Nineteen have at least one conditional exact-record conflict: T1-easy
1/5, T1-hard 5/5, T2-easy 5/5, T2-hard 5/5 and CSTR 3/5. Alien and coupled-basin
dynamic comparisons defer because necessary latent conditioning is unavailable;
the five independent-basin scalar comparisons found no conflict. These counts
are not a mechanism compliance score or a ranking of fitted models.

All ten T2 insulin subsystems share a large witness: identical permitted initial
conditioning and forcing, but observed insulin at time 95 is approximately
583.104 versus 23.667. The checks do not prescribe a missing scientific edge.
Some other witnesses are much smaller; exact-record incompatibility does not
establish unacceptable predictive error. Deterministic strongest-witness selection
avoids presenting the first numerical-rounding discrepancy as the main evidence.

All eight historical target/input source digests match the complete local public
training release. Recomputed bounded-packet digests differ only in 32 RMS values
at floating-point rounding scale; the exact differences are retained. Complete
auxiliary/covariate arrays are newly hashed, rather than incorrectly claimed to
have been certified by the narrower historical digest. Noise-free public data
are treated as the exact reproduction target, not as a bound on solver error.

See `output/topology-diagnostics-review-20261014/QUALIFICATION.md` and `audit.json`.
The audit retains all seeds, hypotheses, unavailable evidence and sample witnesses.
The prompt preview preserves all 160 displayed public briefs and output schemas.
Despite the added dictionary, total system-and-user content is 5,512,829 UTF-8
bytes versus 5,541,628 previously; these are not serving-token measurements.

## Commands and artifacts

Preview all actual historical predecessors under the new prompt policy:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/jetstream-ownership-inspection-20261013 \
  --output output/protocol-prompts-review-local \
  --policy minimal-protocol-clarity-1
```

Use an empty output directory. Historical replies do not answer these revised
prompts. The maintained review is
`output/protocol-prompts-review-20261014-final/PROMPTS.html`.

Offline diagnostic smoke demonstration (synthetic evidence only):

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_topology_compatibility.py \
  --example --output output/topology-compatibility-smoke
```

For real data, `--input` accepts the validated `{draft, evidence}` schema; the
conditioning assertions must have task/runtime provenance, not proposer prose.
Use the Python `build_evidence` API on a `DatasetSplit` with `name=TRAIN`.
The CLI preserves its source and rejects an output directory containing a
different saved input. It performs no LLM calls, fitting or campaign mutation.

The optional Jetstream launcher is
`scripts/jetstream/run_protocol_clarity_construction.sh` (`run`, `status`,
`inspect`). Use its newly pinned delivery bundle and a fresh output root; do not
resume historical campaigns under the new source. Review and tests do not launch
an experiment. All seeds' equations remain available in the existing reports.

## Verification

- Focused prompt, runtime, campaign, resume and diagnostic tests: 173 passed.
- Full `pytest -n 4`: 5,037 passed, eight skipped, five failures during concurrent
  fitting-code edits. Re-running all three affected files against stable sources
  passed 49/49. One failed test was renamed during those unrelated edits.
- `ruff check .`: 37 existing errors in `analysis/claude`; all changed Python
  files pass the targeted Ruff check. No unrelated files were rewritten.
- Synthetic diagnostic CLI, deterministic resume, launcher Bash syntax and
  whitespace checks pass. Both HTML reports have 162 pages with no broken local
  links. All 160 preview public briefs and response schemas are unchanged.
- Logs and rendered artifacts remain under `output/`; they are not committed.
