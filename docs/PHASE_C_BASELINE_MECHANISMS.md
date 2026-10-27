# Phase C external baselines: public mechanism tests

Protocol: `phase-c-baseline-mechanisms-1`. Status: **run on js2 on 2026-10-22
at the user's request, before review** (results below). LLM-SR's expression
form is added in a second root, `phase-c-baseline-mechanisms-v2`, which
assesses every other method's picks again; not yet run. The LLM-SR
transcriptions are still `proposed_pending_review`; a change after review is a
new protocol version in a new root, reported beside this one. Since the user's
decision of 2026-10-22, the transcriptions also set the test score of the
eleven LLM-SR picks that read the history, so their review gates LLM-SR's test
scoring as well.

This applies to every external baseline's sealed Phase C pick the same finite
public mechanism tests our own Phase C construction baseline gets
([PHASE_C_CONSTRUCTION_BASELINE.md](PHASE_C_CONSTRUCTION_BASELINE.md)):

- the six named cells (T1 easy/hard, T2 easy/hard, CSTR, alien device): the
  fitted public mechanism tests, `fitted-public-mechanism-tests-1`
  ([FUNCTIONAL_MECHANISM_ASSESSMENT.md](FUNCTIONAL_MECHANISM_ASSESSMENT.md));
- the two detention cells: the finite basin predicates,
  `phase-c-basin-finite-predicates-1`.

The rules, operating points (the first three training trajectories, at their
first, middle and last samples), solvers (Radau and BDF must agree) and
tolerances are the construction baseline's, read from the same qualified
public release through `research/phase_c_inputs.py`. Nothing is refit or
reselected, no model is called, and no test or private file is opened.

Code: `src/autoformalism/rebuttal/phase_c_baseline_mechanisms.py` (adapters,
plan, run, report), `src/autoformalism/rebuttal/llm_sr_transcriptions.py`,
`scripts/assess_phase_c_baseline_mechanisms.py`,
`scripts/phase_c_llm_sr_transcription_check.py`. Transcriptions:
`configs/phase_c_llm_sr_transcriptions_v1.json`.

## Which model is assessed

Each method's development pick, the model the frozen test evaluation scores:

| Method | Runs | Model assessed |
|---|---|---|
| SINDy, PySR | `phase-c-classical-v1` | the selected training equations, one per target, as an ODE (not the train+validation refit) |
| D3 | `phase-c-d3-hosted-120b-v2` | its native one-step map `x[k+1] = x[k] + increment` |
| LLM-ODE (description prompt) | `phase-c-llm-ode-hosted-120b-v1` | the sealed system, as an ODE |
| LLM-ODE (published prompt) | `phase-c-llm-ode-published-prompt-hosted-120b-v1` | the same |
| LLM-SR | `phase-c-llm-sr-hosted-120b-v1` | a reviewed ODE transcription of its program (below) |
| LLM-SR, expression form | `phase-c-llm-sr-expression-hosted-120b-v1-reseal` | the sealed system, as an ODE: one expression of the current values per target, read as written ([PHASE_C_LLM_SR.md](PHASE_C_LLM_SR.md)) |

**Grammar.** Each baseline's model is read with the external-baseline grammar
it was selected under (`expressions.parser.baseline_parser`): numpy and SymPy
function names, real and symbolic exponents, any finite constant. Of the 80 D3
and LLM-ODE picks, 25 use a function or exponent our own grammar refuses; every
SINDy and PySR pick parses in both. The mechanism code now takes the grammar
as an argument (`grammar="baseline"`); our method's models are still read with
the default grammar, and its behaviour is unchanged (the existing mechanism and
basin tests pass unchanged). Separately, the basin algebra now keeps a power
with a real or symbolic exponent as one opaque factor; before, it truncated a
real exponent to an integer, which only the baseline grammar can reach.

**Failures.** A run that failed has no model. Its predicates stay unresolved:
it confirms nothing (confirmed 0; possible 1), as an unavailable model of ours
does, and it stays in every denominator. A run stopped by the service is not a
failure of the method: preparation refuses `endpoint_unavailable` until it is
rerun. LLM-SR's look-ahead rule applies: a pick flagged as reading later rows
is LLM-SR's failure (decided 2026-10-15). Current failures: SINDy CSTR (both
seeds), PySR T2 easy seed 1, T2 hard seed 1 and coupled detention seed 0
(timeouts), LLM-SR task 5 (T2 easy seed 1, look-ahead). D3 and both LLM-ODE
campaigns completed every task. LLM-SR's expression form has five
(`rollout_failed`): T1 easy seed 1, T1 hard seed 0, T2 easy both seeds and
T2 hard seed 1.

## Which baselines could introduce a latent state

The public prompt allows latent states, but not every baseline received that
sentence, and only one could act on it. Checked against the logged prompts:

| Baseline | Task text received | Can its model hold a state it was not given? |
|---|---|---|
| D3 | the whole public prompt, including section C's "Latent states may be introduced when scientifically useful" | No. Its own format asks for one equation per listed state and says "Do not add other states", as upstream's code skeleton fixes the state list. |
| LLM-SR | sections A and B, ending "Latent states may be introduced subject to Section C" (section C withheld) | Yes, in effect: its function receives whole columns, so a program can carry state through a row recursion. 11 of 15 picks do. |
| LLM-SR, expression form | the same sections A and B | No: a sample is evaluated only if its body is one expression of the current values. |
| LLM-ODE, description prompt | the same sections A and B | No: a reply is one right-hand side per observed variable, `x_0 … x_n`. |
| LLM-ODE, published prompt | no task text (as released) | No. |
| SINDy, PySR | none | No: libraries over the observed channels. |

Sections A and B are the six named cells' layout. In the detention cells the
LLM-SR and LLM-ODE description prompts carry sections A to D instead, which
state that upstream depth is unobserved after the first instant and ask for
"causal upstream memory".

## D3: a one-step map

As in Phase B, D3's recurrence is assessed as it runs: `A = I + Jacobian(increment)`
on its states, one unit step, auxiliaries clamped to their measured values, and
its saved auxiliary laws (which its rollout never executes) set aside. D3
writes one equation per observed state and is told not to add others, so it
has no hidden state.

**Discrete input conversion (decided by the user on 2026-10-22).** The basin
predicate for an ODE requires the volumetric input's depth conversion
`area * coefficient(inflow) = 1`. D3's increment covers one sample interval,
4 minutes in the detention cells, so the same conversion reads
`area * coefficient(inflow) = 4` there: the explicit-Euler image of the ODE's
identity, with the increment never divided by the interval. The other basin
probes are unchanged: zero discharge at or below the crest and recession above
it, and area-weighted transfer cancellation, are statements about signs and
zeros, which one step's scaling preserves. `basin_equation_checks.assess`
takes the interval as `increment_step`; a native row must carry it, and only a
native row may.

## LLM-SR: reviewed ODE transcriptions

LLM-SR's picks are numpy programs. Eleven of the fifteen read the history of
their inputs: low-pass filters of the meal or insulin series, one-row delays,
and in the coupled detention cell a simulation of the hidden upstream basin,
which both picks name `h_up`. The tests probe the states of an ODE, so each
pick was transcribed by hand into one (decided by the user on 2026-10-22).
Each recursion becomes a hidden (latent) state, under fixed rules:

1. **Filter.** `z[k] = a*z[k-1] + (1-a)*u` with `0 <= a < 1` becomes
   `dz/dt = (u - z)/T`, where `T` is the recursion's mean delay (the centroid
   of its impulse response): `T = D*a/(1-a)` when `u` is the current row's
   value and `T = D/(1-a)` when it is the previous row's. The ODE keeps the
   filter's gain and its lag behind a ramp. An Euler filter written with a time
   constant `tau` gets `T = tau`; a copy of the previous row (`a = 0`) gets `T = D`.
2. **Euler.** Any other recursion `z[k] = z[k-1] + s*g` (including a filter
   weight outside [0, 1)) is one explicit-Euler step: `dz/dt = s*g/D`.
3. **Lag.** A previous-row value `x[k-1]` used beside the current one is the
   filter rule's copy of the previous row: `d(x_prev)/dt = (x - x_prev)/D`; a
   backward difference becomes `x - x_prev`.
4. **Initial.** A hidden state starts at the program's own first-row value.
5. **Guard.** Guards that cannot change a value on the public domain are
   dropped (`max(area, eps)` on surveyed areas of 800 and 1200 m², NaN
   replacement of finite values, a clamp on a state its ODE keeps in range);
   every other guard is kept with `max`, `min`, `clip` or `heaviside`.
6. **Coefficients.** The sealed values after the program's own handling
   (defaults, clamps); coefficient branches resolved with them; the program's
   names, prefixed by the target in multi-target picks.
7. **Direct.** Code of the current values is rewritten exactly.
8. **Time unit.** Rows are converted through `D`: 1 min (Dalla Man), 0.1
   (CSTR, alien device), 4 min (detention). The program's output is already a
   derivative in the cell's unit.

Each transcription names the SHA-256 of the exact sealed program text and its
exact coefficients; the adapter refuses one that does not match. Hidden states
may start from a first observation of a target (`Gp[0]`), so their rows use
the context's `fitted_initialization`.

Two transcriptions show what the programs did in physical units:
- Task 12 (coupled, seed 0) steps its upstream basin with a fitted `dt` of
  8.86 min per 4-min row, so in minutes `dh_up/dt = 2.2*(inflow_up - Q_over)/area_up`.
- Task 13 (coupled, seed 1) steps it once per row with no `dt`, so
  `dh_up/dt = (inflow_up - Q_ov)/area_up/4`.
The basin tests read these as written.

### Fidelity of the transcriptions

`scripts/phase_c_llm_sr_transcription_check.py` rolls each program out on the
grid exactly as its sealed validation score did, and its transcription with
Radau at the mechanism tests' tolerances, on training and validation data. It
reports the normalized difference between the two (mean squared difference in
training standard deviations, as the shared NMSE scales) and, for the four
current-value picks, the largest relative difference of the right-hand sides
on the training rows.

| Task | Cell, seed | Hidden states | Validation: program vs data | ODE vs data | ODE vs program | Training: ODE vs program | Failed rollouts (train, val; program/ODE) |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | T1 easy, 0 | 3 | 0.0012 | 0.0017 | 0.00041 | 0.00079 | 0/0, 0/0 |
| 1 | T1 easy, 1 | 2 | 0.0017 | 0.0019 | 0.00025 | 0.00031 | 0/0, 0/0 |
| 2 | T1 hard, 0 | 3 | 0.0011 | 0.0013 | 0.00023 | 0.00025 | 0/0, 0/0 |
| 3 | T1 hard, 1 | 4 | 0.0013 | 0.002 | 0.00036 | 0.0003 | 0/1, 0/0 |
| 4 | T2 easy, 0 | 8 | 0.33 | 0.51 | 0.033 | 3.0e+12 | 0/1, 0/0 |
| 6 | T2 hard, 0 | 6 | 0.087 | 0.49 | 0.33 | 0.33 | 0/0, 0/0 |
| 7 | T2 hard, 1 | 10 | 0.11 | 0.13 | 0.019 | 0.021 | 0/0, 0/0 |
| 8 | CSTR, 0 | 0 | 1.3e+21 | 1.3e+21 | 2.5e+19 | 8.4e+18 | 0/0, 0/0; rhs differs by 1.4e-17 |
| 9 | CSTR, 1 | 0 | 0.62 | 2.9e+20 | 2.9e+20 | 5.5e+18 | 0/0, 0/0; rhs differs by 0.0e+00 |
| 10 | Alien device, 0 | 2 | 2 | 2.5 | 0.73 | 1.6 | 0/0, 0/0 |
| 11 | Alien device, 1 | 4 | 0.15 | 0.29 | 0.054 | 0.05 | 0/0, 0/0 |
| 12 | Detention coupled, 0 | 1 | 0.003 | 0.0045 | 0.00096 | 0.00076 | 0/0, 0/0 |
| 13 | Detention coupled, 1 | 1 | 0.033 | 0.046 | 0.0025 | 0.0092 | 0/0, 0/0 |
| 14 | Detention independent, 0 | 0 | 0.00013 | 9.2e-05 | 1.5e-05 | 2e-05 | 0/0, 0/0; rhs differs by 0.0e+00 |
| 15 | Detention independent, 1 | 0 | 0.00027 | 0.00027 | 1.5e-05 | 1.8e-05 | 0/0, 0/0; rhs differs by 1.1e-16 |

Run on 2026-10-22 against the frozen release (receipt as in the plan). Each
program's grid rollout reproduced its sealed validation error exactly.

- **Exact rewrites.** The four current-value picks (8, 9, 14, 15) match their
  programs' right-hand sides to rounding. Their rollouts differ only because the
  program is stepped on the grid and the ODE is solved adaptively; these are the
  2026-10-15 solver-check numbers (CSTR seed 1: 0.62 on the grid, 2.9e20 solved
  accurately).
- **Close.** T1 and coupled detention: 0.00023 to 0.0025 of the training
  variance. T2 hard seed 1, T2 easy seed 0 and alien seed 1: 0.019 to 0.054.
  T2 easy seed 0's program itself diverges on some training trajectories
  (training error 3.5e38); its ODE does not.
- **Approximate.** T2 hard seed 0 (0.33) and alien seed 0 (0.73). Each program
  cancels two large terms so that a one-row difference feeds back on its own
  derivative, with gain 0.98 (insulin: 17.38875 against 17.38844) and 1.06
  (output: 10.59 against 0.792 x 13.38). That is a discrete-time device; no ODE
  reproduces it exactly. Structure and fitted coefficients are transcribed
  exactly, so the pathways and signs the tests probe carry over, but the
  operating points differ.
- The ODEs of tasks 3 and 4 fail on one training trajectory each (train_005),
  at a pole of the fitted program that the grid steps over. Task 3's insulin
  term has a pole at |Gp| = 127 (fitted `Km_ins = -127`). Task 4 runs away, as
  its program does, to the pole of its U term at Gp = -1040. Neither fails on
  validation, and the tests probe train_000 to train_002 only.

**The filter rule was revised once, before any assessment.** The first version
matched each filter's decay per step (`T = -D/ln a`). It broke two picks. Task
6's insulin filter has a 0.057-min time constant inside the near-cancellation
above; the decay-matched `T` was six times too long, flipped the feedback, and
the transcription's glucose went negative. Task 4's program copies the previous
row through a filter of weight 1e-11; the decay-matched `T` of 0.04 min dropped
the one-row delay, and the transcription diverged (1e12). Matching the mean
delay fixed both. For the slow T1 filters the two rules differ by a few 1e-4.
Neither rule is exact here. The decay-matched lag reproduces a filter at the
rows only when its input is held constant between rows. The shared evaluator
instead interpolates supplied inputs linearly, and many of these filters act on
simulated states (glucose terms, insulin, the device's output). Each rule keeps
part of the filter's response: decay matching keeps its pole, mean-delay
matching its gain and delay.

## Results (js2, 2026-10-22)

Plan `c90cb0ab2f6caa956847946a9d6eb87e1f0f82d5219249f4a3beab0cf26dcabc`, source
`951e793`, root `~/af/runs/phase-c-baseline-mechanisms-v1`; all 96 rows assessed
in under a minute. Confirmed = the fraction of a cell's public mechanism
requirements confirmed; per method, case means over the two runs, then the
mean and sample SD over the eight cells. A failed run confirms nothing.

| Method | Runs completed | Runs confirming every requirement | Confirmed, mean (SD) | Possible, mean |
|---|---:|---:|---:|---:|
| SINDy | 14/16 | 0/16 | 0.042 (0.118) | 0.242 |
| PySR | 13/16 | 0/16 | 0.073 (0.137) | 0.444 |
| D3 | 16/16 | 5/16 | 0.460 (0.264) | 0.656 |
| LLM-ODE, description prompt | 16/16 | 4/16 | 0.271 (0.454) | 0.544 |
| LLM-ODE, published prompt | 16/16 | 0/16 | 0.125 (0.231) | 0.325 |
| LLM-SR (through the transcriptions) | 15/16 | 5/16 | 0.429 (0.405) | 0.633 |

Confirmed per cell (seed 0 / seed 1; "fail" is a failed run):

| Method | T1 easy | T1 hard | T2 easy | T2 hard | CSTR | Alien | Det. coupled | Det. independent |
|---|---|---|---|---|---|---|---|---|
| SINDy | 0/0 | 0/0 | 0/0 | 0/0 | fail/fail | 0/0 | 0/0 | 0.33/0.33 |
| PySR | 0/0 | 0/0 | 0.5/fail | 0/fail | 0/0 | 0/0 | fail/0 | 0.33/0.33 |
| D3 | 1/0 | 1/0 | 0.5/0 | 0.5/0.5 | 1/0 | 1/1 | 0/0.2 | 0.33/0.33 |
| LLM-ODE, description | 0/0 | 0/0 | 1/1 | 0/0 | 0/0 | 1/1 | 0/0 | 0.33/0 |
| LLM-ODE, published | 0/0 | 0/0 | 0.5/0.5 | 0.5/0.5 | 0/0 | 0/0 | 0/0 | 0/0 |
| LLM-SR | 1/1 | 0/0 | 1/fail | 0.5/0.5 | 0/0 | 1/1 | 0.2/0 | 0.33/0.33 |

T2 hard's delayed insulin action is capped at unresolved for every method (its
identification limit), so 0.5 is that cell's maximum. No baseline confirms
every basin predicate.

## Running it

Every input is development data or a sealed pick; the plan records each
file's SHA-256. From a checkout of this branch:

```bash
python scripts/assess_phase_c_baseline_mechanisms.py prepare \
  --release RELEASE --construction-config configs/phase_c_construction_v1.json \
  --root ROOT --classical CLASSICAL_RUN --d3 D3_RUN --llm-ode LLM_ODE_RUN \
  --llm-ode-published-prompt LLM_ODE_PUBLISHED_RUN \
  --llm-sr LLM_SR_RUN --llm-sr-transcriptions configs/phase_c_llm_sr_transcriptions_v1.json \
  --llm-sr-expression LLM_SR_EXPRESSION_RUN
python scripts/assess_phase_c_baseline_mechanisms.py run --root ROOT --workers 8
python scripts/assess_phase_c_baseline_mechanisms.py report --root ROOT
```

A saved row is reused; an interrupted rollout stays unresolved rather than
getting a fresh budget, as in the construction baseline.

On js2 (decided by the user on 2026-10-22), from a clean checkout of this
branch, after copying the SINDy/PySR run to `~/af/runs/phase-c-classical-v1`:

```bash
bash scripts/jetstream/run_phase_c_baseline_mechanisms.sh 12
```

It prepares in the foreground, then assesses detached, 12 rows at a time, into
`~/af/runs/phase-c-baseline-mechanisms-v2` (`runner.log`, `SUMMARY.md`,
`summary.json`), reading LLM-SR's expression form from its reseal. The first
root, `phase-c-baseline-mechanisms-v1`, ran from `951e793` without the
expression form. A dry prepare on js2 on 2026-10-27, in a temporary root,
froze 112 rows, the first root's 96 identically, so their assessments should
repeat. It refuses to start beside another live run, as the rollouts'
wall-clock limits assume their own CPUs. On a laptop, 12 rows took 11 s.

## Aggregation

Per run: confirmed = passes / requirements; possible = (passes + unresolved) /
requirements. Per method: case means over runs, then the macro mean and the
sample SD across cases, separately for all eight cells, the six named cells
and the two basin cells; plus the number of runs that confirm every
requirement, and the number completed.

## Limitations

- **The transcriptions are readings, not the programs.** They are exact for
  the four current-value picks and for the structure and coefficients of every
  pick. But a filter keeps only its gain and mean delay, a one-row delay
  becomes a first-order lag, and an explicit-Euler recursion's use of the
  previous row is not represented. The fidelity table states how close each
  one is. Since they now also set the score of the picks that read the
  history, this limits LLM-SR's score too.
- **Finite predicates.** As for our method: necessary local tests at fixed
  points, not proof of a mechanism. Coverage is the same for every method.
- **Guards hide identities.** A nonnegative-storage guard puts the input
  inside a piecewise function, so the inflow conversion of such a model is
  unresolved rather than passed or failed. Our method's models are read the
  same way.
- **D3's discrete identity is the Euler image.** It matches an exact
  discretization only while the outlet is closed.

## Decisions for test scoring (user, 2026-10-22)

Recorded here because they govern the same picks:

1. The comparison metric is the free-rollout NMSE on test, for every method.
   (The draft's "causal one-step prediction" headline is outdated.)
2. Failed, timed-out and invalid runs stay failures. They count against a
   completion rate, and their NMSE is +infinity: the worst value in a median.
3. Every score comes from an accurate solve: the shared evaluator's adaptive
   solver for every ODE, and the exact iteration of D3's one-step map.
   Re-solving every SINDy, PySR and LLM-ODE pick on validation with Radau at
   rtol 1e-10 changed no NMSE by more than 0.1%.
4. LLM-SR's picks are scored as ODEs: the four current-value picks by their
   exact rewrite, the eleven that read the history by these transcriptions.
   The programs' grid rollouts stay in the sealed record and score nothing.
   (This replaces the first decision, the rollout closest to the paper; see
   [PHASE_C_LLM_SR.md](PHASE_C_LLM_SR.md).)
5. Each method's test path is first rehearsed on validation data and must
   reproduce its validation numbers before test data open: the sealed ones,
   and for LLM-SR its ODEs' numbers solved with the shared solver.

Decided by the user on 2026-10-23:

6. **An implicit retry, for every method including ours.** Integrate with
   RK45 as now; if a trajectory fails numerically (time limit or step-size
   failure), integrate it again with Radau at the same tolerances, and fail it
   only if both fail. Why: on validation one model needs it, LLM-SR alien
   seed 0's transcription, which RK45 cannot finish in the time limit and
   Radau solves in under 2 s ([PHASE_C_LLM_SR.md](PHASE_C_LLM_SR.md)). The
   reverse also happened: Radau at rtol 1e-10 timed out on one LLM-ODE pick
   that RK45 solved. Where both finished, they agreed within 0.1% on every
   validation pick. A real singularity, such as the poles of LLM-SR tasks 3
   and 4, still fails under both.
7. **One test path for every method,** built on Phase B's method-neutral
   post-freeze evaluator (`rebuttal/postfreeze_evaluation.py`), as the
   planning agent proposed. It needs the baseline grammar for the external
   baselines' models (as it stands it reads every model with our narrow
   grammar and would fail 25 baseline picks), LLM-SR's ODEs, the retry in 6,
   and the validation rehearsal in 5.
8. **NMSE aggregation as for every other method:** each case's median over its
   seeds, then the median and unscaled MAD of the case medians, with failures
   at +infinity before either median. With two seeds a case median is the
   midpoint of its two runs, so one failed run makes its case +infinity.
