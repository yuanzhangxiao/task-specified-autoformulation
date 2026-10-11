# Readable topology and specific repair feedback

Policy: `minimal-readable-construction-1`.
Study: `readable_prompt_confirmation`.

This is a fresh opt-in construction experiment. Scientific acceptance, public
benchmark text/data, the response schema, model/sampling settings and per-episode
budgets are unchanged. Historical runs require their original pinned checkout;
never resume an old campaign using this source.

## Evidence motivating this revision

The ACES five-seed focused run completed 40 episodes: 19 structurally eligible
before global repair and 31 at the end. Variable-stage completion was 39/40;
shared-stage local completion was 36/39 reached stages. There were 229 requests,
2,344,688 observed tokens, no missing usage and no length-limited responses.
Per-episode tokens: median 49,160, unscaled MAD 19,775. These are declaration and
graph checks, not function-level mechanism compliance, NMSE or scientific recovery.

| Case | Before global repair | Final topology complete |
| --- | ---: | ---: |
| Dalla Man T1-easy | 1/5 | 4/5 |
| Dalla Man T1-hard | 3/5 | 5/5 |
| Dalla Man T2-easy | 4/5 | 4/5 |
| Dalla Man T2-hard | 0/5 | 2/5 |
| CSTR | 3/5 | 4/5 |
| Alien device | 3/5 | 5/5 |
| Coupled basins | 1/5 | 2/5 |
| Independent basins | 4/5 | 5/5 |

Final completions by seed 0–4 were 4, 7, 7, 7, 6 out of eight. The separate
Jetstream seed-0 pilot reached 3/8. Neither this comparison nor a changed prompt
run establishes a hardware/provider effect; use repeated seeds on a fixed host.

All 229 ACES request/event chains were checked. Recurrent issues included
shared-process references inside joint local functions, coefficients/function
labels copied into fixed conversions, supplied inputs declared as generated,
and a retained memory assignment without its required target path. A coupled
basin reply also used `phi_upstream_depth_local_0` as a local-process dependency;
display-label confusion was not restricted to shared processes.

## What changes

Use ODE, algebraic equation, variable, local process, shared process,
contribution, parameter and required mechanism consistently. A mechanism can
involve several variables/processes and their pathway. Existing JSON field names
remain stable; `depends_on` and `sources` list dependencies, and `uses.target`
is the generated quantity whose equation receives the contribution.

A shared process already compiles to one algebraic quantity. Its declaration
owns its definition and signed uses, so declaring a second variable/equation is
unnecessary. This change does not redesign the model schema or infer which
quantities should be shared.

The model view shows `P depends on (x, z)` and its signed appearances in the
equations. Local contributions appear as independent `f(...)` occurrences with
meanings in order. There are no synthetic function names, gain names or occurrence
IDs to copy into declarations. Fixed conversions remain explicit. Hidden fitted
magnitudes are not fixed to one; missing conversions are not claims of unity.
Function and parameter identities remain runtime-owned later-stage work.

The `transfer` tag still denotes a pairwise internal transfer with two distinct
equations and opposite signs. Other reusable quantities use the existing
`influence` schema value. Opposite signs alone do not prove conservation; matching
scaling/conversions and physical interpretation still require later assessment.

Conditional feedback now:

- Explains an algebraic cycle as a relationship between equations. If a declared
  algebraic quantity is scientifically a state and no explicit type rule forbids
  that choice, shows a type-only edit that preserves shared dependencies/uses.
- Asks whether an unknown symbol is a quantity or a later coefficient. The
  proposer confirms through the corresponding edits; there is no new free-text
  response field or automatic inference from a `k` prefix. Self-described
  parameters also receive this question, without changing their type silently.
- Explains rejected fixed conversions as a role decision and retains attempted
  values. Only the proposer can replace an unknown coefficient with `null`.
- Distinguishes an already inserted shared contribution from a local composite
  function of it. The runtime does not split a nonlinear joint function into a sum.
- Shows all forbidden supplied variable declarations in a rejected reply and a
  conditional retry preserving the other declarations. No partial transaction
  is committed automatically.
- Explains pending shared assembly when receiving topology entries are absent;
  requesting `terms=[]` inserts existing uses, not duplicate local terms.
- Keeps required paths and binding alternatives visible alongside structural
  errors. Shows recently changed process dependencies/uses and the remaining
  global repair allowance. No deleted effect is restored automatically.

The original three local delivery attempts, two extra variable recovery calls,
at most five global repairs, two identical no-progress stop and overall token/
request limits remain. A repair can edit multiple variables/processes at once.
The agenda orders work but never relaxes a public requirement or changes final
eligibility. A fresh live run is required to measure recovery improvement.

## Review and test

Offline, on a saved public construction inspection:

```bash
PYTHONPATH=src .venv/bin/python scripts/review_focused_construction_prompts.py \
  --source output/aces-focused-seeds5-inspection-20261011 \
  --output output/readable-prompts-review \
  --policy minimal-readable-construction-1
```

The HTML shows complete requests, model views, feedback and marked differences.
It uses each actual historical predecessor, including the previous rejected reply;
it does not pretend old replies answered new prompts. Public briefs stay verbatim.

For the next Jetstream run, use the delivered bundle on the existing H100 VM:

```bash
bash scripts/jetstream/run_readable_construction.sh run
bash scripts/jetstream/run_readable_construction.sh status
```

Default: eight cases × seeds 0–4, Full specification, fresh root
`~/af-construction/readable-seeds5-v1`. The existing pinned server and compatibility
evidence are reused, with weights/model/serving provenance checked. No packages,
weights or additional servers are installed. Each construction retains the prior
128-request/524,288-token limit. One worker has a three-hour window; rerunning
`run` resumes pending work without resetting any episode's budget.

After the worker stops:

```bash
bash scripts/jetstream/run_readable_construction.sh inspect
```

Download from a local terminal:

```bash
scp exouser@149.165.152.222:/home/exouser/af-construction/readable-seeds5-v1/inspection.tar.gz \
  ~/Downloads/jetstream-readable-inspection.tar.gz
```

Optional ACES run from the same delivered checkout, after inode headroom is
restored and with a separate root:

```bash
export AF_COMPARISON_STUDY=readable_prompt_confirmation
export AF_COMPARISON_SEEDS='0 1 2 3 4'
export AF_COMPARISON_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-readable-prompts-seeds5-v1
export AF_COMPARISON_SOURCE="$PWD/inputs/topology-source"
bash scripts/hpc/start_phase_c_construction_comparison.sh run
```

Use the same exports and `inspect` to package results. No interaction, fitting,
critic, test-data access or automatic follow-up is part of either run. Do not
pool hosts or select only successful seeds when comparing prompt versions.
