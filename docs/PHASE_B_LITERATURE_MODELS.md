# Literature models on the nine-case Phase B comparison

**Status: approved on 2026-11-10 and implemented in
`src/autoformalism/rebuttal/phase_b_literature.py`, run by
`scripts/phase_b_literature_models.py`. The implementation section at the end
records the choices this sheet left open.**

A literature model is a model an expert would write from the published
literature for a case, using only that case's public channels, fitted to its
training trajectories and selected on validation, then scored by the same
free-rollout validation NMSE as every compared method. Equations and starting
values come from public sources only: Dalla Man, Rizza and Cobelli (2007),
*IEEE TBME* 54:1740, as encoded in BioModels BIOMD0000000379; Bergman et al.
(1979) and the extended minimal model; and the textbook CSTR energy balance.
The benchmark generators' code and private files are not read.

Each row is labeled by what it measures. Where a case's generator *is* the
published model, the fitted literature model is the generator's own structure,
a privileged upper bound, not an independent expert result.

## Cases

| Case | Model | Label |
|---|---|---|
| T1-easy canonical, named | DM07 plasma balance + gut | generator's published structure (upper bound) |
| T1-easy perturbed, named | same model | published structure under changed exchange laws |
| T1-hard canonical, named | DM07 glucose, gut, insulin and secretion subsystems (Gt supplied) | generator's published structure (upper bound) |
| T2-easy canonical, named | DM07 plasma balance, gut, insulin, secretion, remote action, U | generator's published structure (upper bound) |
| T2-hard canonical, named | full DM07 (Uii supplied) | generator's published structure (upper bound) |
| T2-hard canonical, named | extended Bergman minimal model + two-compartment gut | classic expert model |
| CSTR easy, named | textbook reactor-temperature balance (C, Tj supplied) | textbook structure on the affected historical export; not an upper bound there (see below) |
| T1-easy canonical and perturbed, obfuscated | none | not applicable: no names, so no literature can be chosen |
| Alien device easy | none | not applicable: no literature exists |

## Equations

Public channel names; `meal(t)` is the public `meal_event_g` channel as the
compared methods received it (an amount in grams on its minute, linearly
interpolated), `uI` the public insulin forcing.

**DM07 (Dalla Man et al. 2007).** Glucose:
`dGp/dt = EGP + Ra − Uii − E − k1·Gp + k2·Gt`,
`dGt/dt = −Uid + k1·Gp − k2·Gt`, `G = Gp/VG`.
Gut: `dQsto1/dt = −kgri·Qsto1 + 1000·meal(t)`,
`dQsto2/dt = kgri·Qsto1 − kempt·Qsto2`,
`dQgut/dt = kempt·Qsto2 − kabs·Qgut`, `Ra = f·kabs·Qgut/BW`,
`kempt = kmin + (kmax − kmin)/2·[tanh(a(Qsto − b·D)) − tanh(c(Qsto − d·D)) + 2]`,
`a = 5/(2D(1 − b))`, `c = 5/(2D·d)`, `D` = the last meal's amount in mg,
read from `meal(t)`. Insulin: `dIl/dt = −(m1 + m3)·Il + m2·Ip + S`,
`dIp/dt = −(m2 + m4)·Ip + m1·Il + uI`, `I = Ip/VI`,
`m3 = HE·m1/(1 − HE)`, `HE = −m5·S + m6`. Production:
`EGP = [kp1 − kp2·Gp − kp3·Id − kp4·Ipo]+`, `dI1/dt = −ki(I1 − I)`,
`dId/dt = −ki(Id − I1)`. Utilization: `Uii = Fcns`,
`Uid = (Vm0 + Vmx·X)·Gt/(Km0 + Gt)`, `dX/dt = −p2U·X + p2U·(I − Ib)`,
`U = Uii + Uid`. Secretion: `S = γ·Ipo`, `dIpo/dt = −γ·Ipo + Spo`,
`Spo = Y + K·[dG/dt]+ + Sb`, `dY/dt = −α(Y − β(G − Gb))`. Excretion:
`E = ke1·[Gp − ke2]+`. Published values (normal subject): VG 1.88, k1 0.065,
k2 0.079, VI 0.05, m1 0.190, m2 0.484, m4 0.194, m5 0.0304, m6 0.6471,
kmax 0.0558, kmin 0.0080, kabs 0.057, kgri 0.0558, f 0.90, b 0.82, d 0.010,
BW 78, kp1 2.70, kp2 0.0021, kp3 0.009, kp4 0.0618, ki 0.0079, Fcns 1,
Vm0 2.50, Vmx 0.047, Km0 225.59, p2U 0.0331, K 2.30, α 0.050, β 0.11, γ 0.5,
ke1 0.0005, ke2 339, Gb 95, Ib 25, Sb 1.8.

Each case keeps only what generates its targets from its public channels:
a supplied auxiliary enters as given and its own equation is dropped
(T1-easy and T2-easy take EGP, Uii, E and Gt as supplied; T1-hard takes Gt;
T2-hard takes Uii).

**Extended Bergman minimal model (T2-hard).**
`dGp/dt = −(SG + X)·Gp + SG·Gpb + Ra`, `dX/dt = −p2·X + p3·(I − Ib)`,
`dI/dt = −n·(I − Ib) + γB·[Gp/VG − h]+ + uI/VI`, with two-compartment
absorption `dQ1/dt = −Q1/tmax + 1000·meal(t)`, `dQ2/dt = (Q1 − Q2)/tmax`,
`Ra = f·Q2/(tmax·BW)`. Fitted: SG, Gpb, p2, p3, Ib, n, γB, h, tmax, f; VG and
BW at the published values. Uii is not used: the minimal model lumps
insulin-independent uptake into SG.

**CSTR temperature balance.**
`dT/dt = q·(Tf − T) + r_ref·exp(−ER·(1/T − 1/350))·C − h·(T − Tj)`, with C
and Tj supplied. Fitted: q, r_ref, ER, h. The reaction term is written about
350 K because the prefactor and activation energy are otherwise nearly
collinear. Starts are generic (q 1, h 1, ER 10,000, r_ref from the training
data's scale), not the generator's values.

## Fitting and selection

- Free rollout of the training trajectories from causal initial values: the
  observed targets' initial readings, supplied auxiliaries throughout, and
  hidden states at the published basal steady state, shifted by a fitted
  linear map of the initial readings' deviation from basal (one map shared by
  all trajectories). Nothing is fitted per trajectory.
- Residuals are each target's error divided by its training standard
  deviation over every training sample; `scipy.optimize.least_squares` in log
  parameters, each bounded within a factor of 10 of its start (shape
  fractions kept in (0, 1)); RK45 with a Radau retry, steps limited to one
  sample.
- Three starts: start 0 at the published (or generic) values, starts 1 and 2
  at log-uniform perturbations of ±0.3 under seeds 1 and 2. The start with the
  lowest validation NMSE is kept, as in the Phase C reference fits; every
  start is reported.
- For the DM07 rows, the published values without fitting are also scored, as
  "published values, unfitted".
- Scoring: validation free-rollout NMSE per target with the training standard
  deviation, averaged over targets, as for the compared methods. No test data.

## Cost

CPU only: seven model/case pairs, three starts each, plus five unfitted
evaluations. On an eight-worker laptop a DM07 start takes about 4 to 15
minutes.

## Implementation

- **One encoding, two evaluators.** Each model is written once in the
  restricted grammar and compiled by the shared validator. The fit evaluates
  the validated expressions over all of a split's trajectories at once; the
  shared simulator scores the fitted candidate one trajectory at a time, RK45
  with a Radau retry and steps of at most one sample, exactly as it scores the
  compared methods. A start counts only if the two training NMSEs agree
  (relative 1e-4, or 1e-9 absolute where both are at round-off).
- **The meal dose D.** `meal_dose_mg`, the last meal's amount in mg, is
  derived causally from the public meal channel and read as an input. Before
  the first meal D is 1 mg: the stomach is empty and k_empt is k_max.
- **Held constants.** VG 1.88, VI 0.05 and BW 78 stay at the published
  values, as in the Bergman row: they only rescale other parameters. Insulin
  is written as plasma concentration, I = Ip/VI, the same model in other units.
- **Latent starting values.** The basal state is the published model's
  fasting steady state, solved numerically (Gp 173.85 mg/kg, I 25.06 pmol/L,
  Gt 131.38 mg/kg, EGP 1.92 mg/kg/min). The map reads every public target,
  auxiliary or input channel whose first value varies over the training
  trajectories, the meal channel included, each divided by its spread over
  those first values; each latent state's shift is scaled by the largest value
  it reaches under the published model's reference meal (78 g). Because the
  meal channel is interpolated linearly, a meal on the first minute delivers
  half its grams through the inflow term; the map's meal weight can put the
  rest in the stomach. The unfitted rows have no map.
- **Starts that are not published.** Extended minimal model: SG 0.025,
  Gpb 178.6, p2 0.025, p3 2e-6, Ib 25, n 0.14, γB 0.5, h 95, tmax 40 and
  f 0.9, typical normal-subject magnitudes, not fitted to these data. CSTR:
  r_ref starts at the least-squares rate given the other generic starts and
  finite-difference slopes of the training temperatures.
- **Least squares.** Trust-region reflective, Jacobian scaling, relative
  finite-difference step 1e-3 (well above the solver tolerance 1e-7),
  ftol and xtol 1e-6, at most 200 evaluations and three hours per start; at
  the time limit the best point seen is kept.
- **The CSTR row.** With C read from data, the temperature balance is
  open-loop unstable at these operating points: its local growth rate,
  -q - h + r_ref exp(-ER(1/T - 1/350)) C ER/T^2, is positive along nearly every
  trajectory (about 0.4 per time unit at the base point, up to 2), so a
  30-unit free rollout amplifies small errors, such as the linear
  interpolation of C between samples, by e^10 or more. At the generator's
  documented parameters (q 1, h 2.5, ER 8750, r_ref about 80, beta 80) the
  balance reproduces 7 of the 16 historical training trajectories and runs
  away on 5; five of the historical trajectories also miss the balance by 4
  to 8 K per time unit, the export's known generation defects
  (`docs/PHASE_C_FITTING_STATIONARY.md`). On the corrected Phase C cell the
  data follow the balance closely, yet it still runs away on 6 of 16. So
  the temperature balance alone is not an upper bound on either cell.
- **CSTR rows beside the table.** Agreed on 2026-11-10: report both the
  approved temperature balance and the textbook CSTR with its reactant
  balance, `dCr/dt = q (Cf - Cr) - k(T) Cr` and
  `dT/dt = q (Tf - T) + beta k(T) Cr - h (T - Tj)` with
  `k(T) = r_ref/beta exp(-ER (1/T - 1/350))`, Tj supplied, Cr started at C's
  first reading and never reset to it. beta starts at the least-squares heat
  per unit of reactant consumed, from finite-difference slopes of the
  training data at the generic q and h; the other starts are as above. At
  the documented parameters this closed loop reproduces the corrected Phase C
  trajectories to NMSE about 1e-7 (training) and 2e-8 (validation), and the
  historical ones to 0.87 and 0.54. Three pairs run beside the approved
  campaign (`src/autoformalism/rebuttal/cstr_literature.py`), by the same
  protocol and each also scored at the documented parameters as a diagnostic:
  the closed loop on the Phase B cell (`cstr_closed_loop`), and both models
  on the corrected Phase C counterpart,
  `phase_c_cstr_controlled_reactor_mechanism_canonical_named_easy_reset_v1`
  (`cstr_balance_phase_c`, `cstr_closed_loop_phase_c`), which has the same
  roles and 16 training and 4 validation trajectories; only its public
  training and validation tables are read. Phase C rows are reported beside
  the Phase B table, not in it.

```bash
python scripts/phase_b_literature_models.py run \
  --inputs /path/to/phase-b-nine-case-v1 --output /path/to/output --workers 8
python scripts/phase_b_literature_models.py report --output /path/to/output
python scripts/phase_b_literature_models.py cstr --pair cstr_closed_loop \
  --source /path/to/phase-b-nine-case-v1 --output /path/to/cstr_closed_loop
python scripts/phase_b_literature_models.py cstr --pair cstr_closed_loop_phase_c \
  --source /path/to/phase-c-public-v2 --output /path/to/cstr_closed_loop_phase_c
```
