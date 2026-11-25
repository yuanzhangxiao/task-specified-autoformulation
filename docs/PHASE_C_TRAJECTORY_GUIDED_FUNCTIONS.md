# Using training trajectories to construct functions

The [complete discharge law experiment](PHASE_C_COMPLETE_LAW_NOMINATION.md)
compares four entire rate equations, fits them from interval water balances,
and predicts each of six omitted basin training events. Synthetic controls
measure recovery of the equation separately from predictive error.

**Design for discussion.** Phase C already passes training evidence to the function stage. Given an admitted topology, the proposer writes expressions for its processes and equations. I propose that the runtime first compare a few function families against measurements conditional on the sources of that particular equation. The proposer can use scientific knowledge to choose among them or offer another expression.

The new topology export gives each variable’s type, RHS sources and initialization. It leaves open how to group the sources into terms. For example, the same edges from glucose and insulin can represent an additive production and loss law or a joint response. First assemble candidate balances under the public requirements. In the basin task one volumetric transfer contributes $-Q/A_{\rm up}$ upstream and $+Q/A_{\rm down}$ downstream. Then choose forms for the unknown processes.

## What a trajectory tells us

| Equation | Evidence the runtime can use | What the evidence can distinguish |
|---|---|---|
| Observed algebraic variable, e.g. \(U=H(x)\) | Measured \(U\) against observed argument values at the same time, across training trajectories | An offset, a slope, or a change of slope when the other arguments are comparable. A flattened time trace alone gives little evidence for saturation in \(H\). |
| Observed differential state, e.g. \(\dot I=F(x)\) | For each interval, compare \(I(b)-I(a)\) with \(\int_a^b F(x(t))\,dt\). Use several interval widths and respect input discontinuities. | The observations constrain the sum of all rates in the insulin balance. An insulin peak says that production and loss balance at that time. It does not identify either function individually. |
| Equation with latent arguments or a shared process | Simulate the coupled equations with their declared initializations, using one parameter vector and the same process law in every equation that uses it. | Different hidden mechanisms can give the same observable trajectory. Preserve alternatives where the available observations cannot distinguish them. |

## When the balance isolates one function

The coupled-basin public specification gives the basin areas, the overflow crests, and the two measured inflows. In training event `train_003` the upstream water starts at its crest and upstream inflow is zero throughout. The upstream basin therefore supplies zero transfer. The downstream balance directly identifies the average outlet discharge on each interval.

\[
\overline Q_{\rm out}=\overline Q_{\rm in,down}-A_{\rm down}\frac{h(b)-h(a)}{b-a}.
\]

Compare three one-parameter laws, \(Q=k[\max(h-h_{\rm crest},0)]^p\), with \(p=1,1.5,2\). Fit \(k\) separately for each power by least squares over the 60 four-minute intervals of `train_003`. Average the inlet at the two endpoints using its public linear interpolation. Approximate the interval mean of the powered head by averaging its two endpoint values.

A second training event supplies a check at lower water levels. In `train_000`, the upstream basin needs \(800(0.35-0.20)=120\) m³ of water to reach its crest. Integrated upstream inflow first reaches that volume at minute 16. The four complete intervals up to that minute also have zero upstream transfer.

| Trial power | Outlet-flow RMSE on the four intervals of the second training event (m³/min) |
|---:|---:|
| 1 | 0.970 |
| 1.5 | 0.083 |
| 2 | 0.289 |

This favors the three-halves law for the outlet among the three candidates. The four check intervals cover a narrow range of downstream depths. Whole-system rollout across other events remains the relevant validation. The reusable algorithmic operation is to isolate an unknown process when the public balance and measured inputs permit it. When several rates remain unknown, fit them together.

## When two different histories give similar arguments

Take the saved T2 inventory-0 graph `g_9885c80dc2db2b32`, which admits \(\dot I=F(G_p)\). On the public training trajectory `train_001`, the administered insulin is zero throughout. The two-minute average rate is \([I(t+1)-I(t-1)]/2\) on this one-minute grid.

| Minute | Plasma glucose mass (mg/kg) | Plasma insulin (pmol/L) | Average insulin rate (pmol/L/min) |
|---:|---:|---:|---:|
| 20 | 184.966 | 50.454 | +3.550 |
| 193 | 184.923 | 55.396 | −0.596 |

At nearly the same glucose value, insulin is rising in one interval and falling in the other. Four- and six-minute windows give the same sign pattern. This is evidence to *compare* models. The glucose values differ slightly, the rates are interval averages, and the two times also differ in history. Calling this a deterministic contradiction would require stronger evidence or explicit bounds on the function.

First test a simple glucose-dependent production law on its own terms. On graphs that also admit insulin as a source, compare a production term and an insulin-dependent loss, e.g. \(\dot I=P(G_p)-kI\). A graph with a suitable hidden state permits another explanation involving delayed production. Compare these candidates on multiple training trajectories and report which uncertainty remains. If the evidence instead points to a source missing from a graph, return the observation to topology repair. An expression search should respect the graph it is evaluating.

## A first experiment

1. **Hold the graphs fixed.** Take a small, structurally diverse sample spanning the corrected algorithm-generated and proposer-repaired graphs. Include T2-easy and the coupled basin. Compare the same graphs across arms and record inventory, provenance, initialization and shared processes.
2. **Build a short list per equation.** Start with the simplest permissible form. Add a saturating family for a repeatedly flattening *conditional* response, or a two-source interaction when the slope with respect to one measured source changes with the other. Record how many trajectories and how much of the argument range support the comparison. When evidence is weak, keep competing forms. Maintain any public constraints on units, signs and parameter sharing. A peak time can guide a diagnostic without prescribing a delay or an equation.
3. **Compare three ways to choose expressions.** Use the current proposer, the algorithm's short list alone, and a hybrid in which the proposer sees the evidence and may substitute its own expression. Give each arm the same number of full-model fits, optimizer starts and compute cap per graph. Give the two proposer arms the same model and call budget. Include initializer formulas in the fitting contract. Score models with latent sources by simulating the whole system from the permitted initial conditions.
4. **Evaluate the decision.** During construction, use only training trajectories. Hold out entire trajectories *within training* to compare families where the number of trajectories permits it. Freeze each arm's candidates before applying the independent validation trajectories. Compare the fraction of graphs that yield a stable rollout, paired validation NMSE by target, public mechanism checks and compute used. Track whether the hybrid's winning expression came from its own proposal or the supplied short list. Reserve test trajectories for final evaluation.

Kacprzyk, Piskorz and van der Schaar model trajectory shapes directly. Their formulation motivates recording shape as evidence. Inferring a governing function under a fixed graph is a separate inverse problem. Messenger and Bortz give a weak-form approach to learning dynamics without pointwise numerical differentiation, relevant to the interval comparison above. [Skip the Equations](https://proceedings.mlr.press/v267/kacprzyk25a.html) · [Weak SINDy](https://doi.org/10.1137/20M1343166)

## Implemented pilot on the ACES graph replay

The frozen cohort contains six of the 143 admitted graph alternatives from job 2403807. It includes two algorithm-generated T2-easy graphs, two algorithm-repaired T2-easy graphs originally proposed by 20B, and both admitted coupled-basin graphs. Within each group we choose the sparsest graph, then prioritize a different variable inventory and maximize differences in edges, variable types and initialization rules. All three arms use the same six graphs.

| Arm | How it obtains its two complete models for each graph |
|---|---|
| Proposer | Two independent whole-model requests to GPT-OSS-20B. Each sees the public task, fixed graph, and summaries of every training trajectory. |
| Algorithm | Two combinations of equation families. For eligible equations, two-fold cross-validation splits by whole training trajectory. It compares predictions of algebraic values or integrated differential-state changes. Candidates involving hidden quantities need full-model fitting. |
| Hybrid | Two independent 20B requests with the same task and summaries, plus the algorithm’s two complete models and the evidence for them. The proposer can select one or write its own. |

The proposer control uses a single request per model. Historical staged requests used a different interface. The algorithmic library includes affine, saturating and interacting terms. In both basin graphs the downstream derivative omits downstream depth and downstream inflow, precluding the direct downstream outlet balance in the worked example.

The comparison reserves 24 physical generation requests and 36 training fit slots. Each eligible formula receives up to 1,000 residual evaluations total over two starts, with a 180-second optimization wall budget, using the same fixed-step integration and least-squares fitter. Rank 1 and rank 2 use seeds 100 and 101 respectively in every arm. The fitter replays the result with two and four integration substeps per sample interval and checks the predictions for numerical agreement. A failed or invalid candidate consumes its slot.

For each of the 18 graph/arm combinations, the lowest verified training NMSE selects a winner. The selection freezes before validation runs. Each winner runs open-loop on the separate validation trajectories with fixed parameters. The report shows paired errors, fitted equations, fitted dependency paths, nonnegativity and basin volume diagnostics, rollout curves and compute counts. The pilot never uses test trajectories. Its selected graphs and two draws per graph support a paired exploratory comparison.

On ACES, run the launcher from the extracted phase-c-trajectory-functions-20b-v1 directory. The first command exercises a training rollout with the installed solver. The second schedules generation, the CPU fits, and validation. The remaining commands show progress and package the report.

    bash scripts/hpc/start_trajectory_functions_aces.sh preflight
    bash scripts/hpc/start_trajectory_functions_aces.sh run
    bash scripts/hpc/start_trajectory_functions_aces.sh status
    bash scripts/hpc/start_trajectory_functions_aces.sh inspect

## Source changes proposed during function construction

The proposer now chooses a route. On `fixed_graph` it writes equations with the existing sources or, in the hybrid arm, selects a complete shortlisted model. On `request_topology_change` it gives the scientific reason and the complete equations and parameter starts. The runtime derives added and removed sources from the expressions. Receivers, variable types, and permitted initializer dependencies stay fixed during this experiment. A malformed fixed-graph reply receives an error in its selected route.

For a requested change the runtime screens the actual child graph against the public requirements, assigned memory and all observed generated variables in training. The fitter, validation scorer and report use that same child graph. Both routes in the new experiment check for reserved names and, where applicable, an algebraically identifiable defect in the internal basin transfer. For named basin depths, multiply each derivative by its surveyed area before checking whether upstream-dependent transfer terms cancel. Some forms remain undecidable. A passing transfer check establishes only this identity on the expression domain; the rollout report also measures downstream depth, water gain and predictive error.

### What the archived basin replies contained

Eight 20B replies and four algorithm controls from ACES job 2404388 were replayed. Seven original replies had failed the fixed-source check. In the replay the auditor submitted each of those seven unchanged formula bundles as a hypothetical source change. No earlier answer contained this route choice or its scientific reason.

- Slots 024 and 025 proposed the same equations from different coefficient starts. With $[x]_+=\max(x,0)$ the proposed transfer was $Q=k_u[h_u-c_u]_+$ and the outlet was $D=k_d[h_d-c_d]_+$. The volume balances were $A_u\dot h_u=q_u-Q$ and $A_d\dot h_d=q_d+Q-D$. Both starts converged to training NMSE 0.06050. The training-selected fit obtained validation NMSE 0.07528 on three events, versus 4.81232 for the selected fixed model on the same parent. The minimum predicted validation depth rose from −1.421 m to 0.128 m.
- Slot 031 passed the graph screen. Its upstream balance lost one copy of $Q$ while downstream received $A_u Q$. In a public training event with $A_u=800$ and upstream depth above crest, total storage gained an extra $799Q$. Setting the common coefficient to zero would remove the required transfer. The runtime rejected this formula before fitting.
- Three replies used undeclared flows; one omitted the runoff sources and declared unused coefficients. The one previously eligible hybrid reply remained eligible. Fitting used the original budget and training split, then froze the winners before validating. The comparison uses a previously inspected validation set.

Run the offline replay from the project root. Each command verifies saved records before resuming.

    study=output/basin-function-revision-20261123/replay
    .venv/bin/python -m scripts.replay_trajectory_function_revisions --archive output/trajectory-functions-20261123/returned --root "$study"
    .venv/bin/python -m scripts.trajectory_functions validate --root "$study" --validation output/trajectory-functions-20261123/inputs/validation.json
    .venv/bin/python -m scripts.trajectory_functions inspect --root "$study"

### Prospective ACES test

The `basin-review` profile uses the same two basin parents, two seeds and three arms, with a ceiling of eight new 20B requests and twelve CPU fit slots. Each reply records its actual route and reason. The `source-review` profile enables the same interface on all six registered graphs. The published fixed-graph run remains reproducible with its original bundle at commit `55f1731`.

On ACES, from the new extracted directory:

    bash scripts/hpc/start_trajectory_functions_aces.sh preflight basin-review
    bash scripts/hpc/start_trajectory_functions_aces.sh run basin-review
    bash scripts/hpc/start_trajectory_functions_aces.sh status basin-review
    bash scripts/hpc/start_trajectory_functions_aces.sh inspect basin-review

## Checks on the functions selected from training trajectories

Three controlled experiments on the Phase C public development data examine different decisions in function construction. Each writes a training plan and intermediate receipts, locks its selection, and then scores the frozen parameters on validation. The public benchmark prompts and data remain the original release.

| Question | Experiment entry points | Scope |
|---|---|---|
| Can the data distinguish two discharge functions under a shared water balance? | `scripts/basin_function_family.py`, `scripts/basin_family_stability.py`, `scripts/basin_balance_inference.py` | The declared balances, areas and crests supply the structure. Compare nine discrete power pairs, then estimate continuous powers with training intervals where each flux can be distinguished. Check sensitivity to event, starting vector and sampling interval. |
| Does showing outlet measurements change the proposed discharge functions? | `scripts/basin_process_proposals.py`, `scripts/replay_basin_process_proposals.py`, `scripts/basin_process_start_control.py` | Two matched 20B prompts ask for the two unknown functions and differ in five numerical rows. Normalize scoped head aliases, fit from common starts and exchange fitted vectors between equivalent formulas. |
| Can the data rule out a family before fitting and motivate a revision? | `scripts/audit_affine_direction.py`, `scripts/trajectory_boundary_trial.py`, `scripts/t2_equilibrium_repair.py`, `scripts/t2_challenge_folds.py` | Use open-loop training intervals and declared initialization to bound the forcing at a reference trajectory. Test opposing changes on the same side of the reference driver. Where applicable, compare a boundary-preserving rate, a fasting equilibrium and an added restoring term with matched fitting budgets. |
| Does a local fit to temperature changes predict a whole trajectory? | `scripts/thermal_balance_trial.py`, `scripts/thermal_forecast_continuation.py` | Hold out complete training events to compare four public temperature-law families, then integrate the selected functions. A second experiment continues two families from short training windows to whole-event simulation and verifies resolution before freezing. CSTR-easy supplies the required forcing channels. The control abstains on CSTR-hard. |

The reusable `event_horizons` function in `scripts/trajectory_horizon_audit.py` measures training forecast error at one eighth, one quarter, one half and the full event from its permitted initial condition. It uses elapsed time for irregular grids and records resolution failures separately. Short-horizon warnings identify problematic starts, while complete forecasts qualify retained parameter vectors.

The local comparison, whole-event fit and parameter-transfer check answer separate questions. In the basin, two formulas proposed under different prompts gave identical predictions after the same coefficient starts were supplied. In the reactor, the function with the smallest local error in all 16 training folds diverged in an open-loop forecast. In T2, a change that eliminated negative insulin still left the model too close to zero late in several events. Inspect per-target error and representative curves alongside the pooled NMSE.

The balances and the small families in these controls were supplied by the researchers. The training evidence chooses among or rejects candidates under those assumptions. The controls do not establish an end-to-end autonomous constructor for these benchmarks.

### Enforcing a declared pointwise error before fitting

A campaign can opt into a maximum absolute error for an observed differential target. The JSON policy maps each exact dataset ID to its target names and absolute tolerances, expressed in that target's units. Supply the JSON file with `prepare --pointwise-tolerances <path>` or set `AF_POINTWISE_TOLERANCES` when preparing a new ACES result root. The policy is part of the sealed intake. A fitting objective stated only in terms of NMSE carries no additional pointwise requirement.

All three arms apply the check to the admitted graph and its actual equations. A proven violation records the reference trajectory, opposing measured changes, and edit options allowed by the selected interface. The fitting worker writes an ineligible receipt before reserving an optimizer call. An unverified case retains its previous eligibility. The original run bundles preserve their own source identities.

To exercise this path with the 36 saved replies, use the frozen training archive and a separate output directory:

    printf '%s\n' '{"phase_c_dalla_man_t2_canonical_named_easy_rates_v1":{"I":1}}' > insulin-error.json
    python -m scripts.replay_trajectory_function_pointwise \
        --archive output/trajectory-functions-20261123/returned \
        --root output/trajectory-pointwise-replay \
        --pointwise-tolerances insulin-error.json

### Testing whether measured evidence helps 20B propose a repair

The paired experiment takes the three original T2 equations with a certified 1 pmol/L conflict. Each parent receives three new seeds, 211–213. For every parent and seed, the control prompt reports the failed requirement. The paired prompt adds the measured constant-forcing reference and the two opposing changes. Both prompts expose the same scientific task, parent equations, allowed sources, and edit schema. They give no revised formula or later fitting results. One H100 serves all 18 calls. Each call can replace only the insulin RHS and introduce coefficients used in that expression. The runtime retains the rest of the model and checks the dependencies of the edited equations.

`scripts/pointwise_repair_fitting.py` compares each admitted revision with a matched refit of the original parent using training trajectories. Record separately whether the output parses, whether it changes the equation, whether a contradiction can still be proved, whether simulation succeeds, and the maximum observed insulin error at the fitted vector. Removing the proof only establishes that the earlier argument no longer excludes the candidate. The fitted predictions show whether that particular vector satisfies the requested tolerance.

The standalone ACES entry point is `scripts/hpc/start_pointwise_repair.sh`. Its `preflight`, `run`, `status`, `resume` and `inspect` actions use the sealed three-parent training intake in the committed package. The fitting CLI also needs the original training campaign containing the three parent fit receipts. Both stages leave formal validation and test closed.

### Checking whether a scalar repair can reproduce the order of training trajectories

For a rate of the form `y' = a*y + b*x + q(w)`, consider two training events with equal observed initial `y` and identical supplied histories `w`. Their predicted difference at time `t` is the integral of `b*(x1(s)−x2(s))*exp(a*(t−s))` over the interval. If one history of `x` stays above the other, the sign of that difference is fixed by the shared coefficient `b`. One contrary pair rules out a nonnegative coefficient and a second contrary pair rules out a nonpositive coefficient. Half the smaller of the two observed separations gives a lower bound on the largest absolute error at the observed times.

For new function proposals with a declared per-target absolute error tolerance, the pre-fit gate now runs the interval proof and the ordered-pair proof together after graph admission. The graph admission step already checks whether equal initial conditions and equal forcing histories require identical trajectories. The same function gate applies to proposer-generated, algorithm-generated and hybrid candidates. Each proof records its own outcome. Either can exclude a candidate if its lower bound strictly exceeds the target tolerance. The ordering proof uses exact forcing, observed initial values, and complete linear or previous-value interpolation on matching grids. It abstains when these assumptions or the scalar equation form cannot be verified. A pooled NMSE objective can still improve in the presence of a pointwise error.

To replay the original 18 replies, use the directory containing the archived generation plan, intake, candidate and call receipts:

    python -m scripts.audit_ordered_insulin_repairs \
        --generation path/to/generation --output path/to/ordered-training-audit.json

The historical replay preserves each original decision and audits the admitted equations. For each newly certified conflict it tries one linear term per additional permitted supplied history and reruns the ordering proof.

To submit all original replies and each of those single-source edits to the current, complete repair gate offline, use

    python -m scripts.screen_ordered_repair_options \
        --generation path/to/generation --output path/to/screened-options.json

Of the 18 original replies, 15 failed the equal-history graph check and three failed the ordering proof. Of the 13 algorithm-generated edits to those three equations, ten had a proven conflict and three reached the shortlist. All three added supplied EGP. Two extended replies for parent 002 and one extended a reply for parent 015. All replies for parent 014 had already failed the graph check.

To bypass the proposer and generate options directly from the three original parents, use

    python -m scripts.screen_ordered_repair_options \
        --intake path/to/generation/intake.json --output path/to/parent-options.json

This generates a separate set of 13 edits, five for parent 002 and four each for 014 and 015. One edit per parent passed the current gate, each adding EGP to the original insulin rate. Both modes record the full shortlisted models, their identifiers, and the individual check results. Selection by a proposer, parameter fitting, and empirical accuracy are separate experiments.

### Following a source change through a complete model

The three public training folds each withhold five challenge trajectories. The fasting reference remains available in each fit. The scalar comparison uses observed initial insulin and benchmarks four rate laws under the same timescale grid. A second comparison uses identical piecewise-linear production functions with signed or nonnegative coefficients. The latter constraint keeps insulin nonnegative between observations when its initial value is nonnegative. A saved coefficient identifies both its supplied channel and its basis knot, which allows forecasts to be independently checked after serialization.

Run these commands from the checkout with the archived paired generation and cold-fold receipts in `$AF_EVIDENCE`. Each command verifies an existing matching checkpoint on resume.

    python -m scripts.compare_insulin_order_repairs run --paired "$AF_EVIDENCE/paired" --root "$AF_EVIDENCE/ordering-fit"
    python -m scripts.compare_positive_insulin_rates run --paired "$AF_EVIDENCE/paired" --root "$AF_EVIDENCE/positive-rates-final"
    python -m scripts.transfer_insulin_order_law run --paired "$AF_EVIDENCE/paired" --single "$AF_EVIDENCE/ordering-fit" --cold "$AF_EVIDENCE/aces-cold-folds/results" --root "$AF_EVIDENCE/order-transfer"
    python -m scripts.transfer_positive_insulin_law verify --evidence-root "$AF_EVIDENCE" --root "$AF_EVIDENCE/positive-transfer"

The coupled comparisons replace the insulin rate in each saved model and keep the other fold-specific fitted equations and initializers. The outputs report glucose, insulin, disposal, and pooled NMSE on the same events. EGP is a supplied endogenous glucose production trajectory in this easy task. Its value as a predictor and its causal role are separate questions. The model bundles contain fitting-fold coefficients and do not incorporate held-trajectory initial estimates or held-trajectory parameter fits.

For a separate check on the original four development-validation events, prepare the two restoring parents and their two revised models from full-training fits. Preparation accepts only the experiment evidence directory. The second command verifies that freeze before opening the registered validation intake.

    python -m scripts.validate_positive_insulin_transfer prepare --evidence "$AF_EVIDENCE" --root "$AF_EVIDENCE/positive-validation"
    python -m scripts.validate_positive_insulin_transfer validate --root "$AF_EVIDENCE/positive-validation" --archive "$AF_EVIDENCE/aces-equilibrium/results" --validation output/trajectory-functions-20261123/inputs/validation.json

The revised insulin equation and all other model coefficients remain fixed throughout this comparison. The evaluator reproduces the archived parent forecasts, independently integrates the revised insulin equation and scores all three targets. The development results identify changes in predictive accuracy on these four previously studied events.

### Checking initial algebraic outputs before and after fitting

The trajectory-function fitter now compares the order of observed initial values with the order predicted by each supplied and selected parameter vector. It evaluates the declared initializers and algebraic equations directly. Two measured values are distinguishable when their difference exceeds twice the public per-measurement error bound. The fit receipt records the worst reversal, ties, counts and errors. A missing measurement bound or unavailable initial comparison receives an explicit unresolved status. This diagnostic supplies feedback about the particular vector. To exclude an equation family would require a bound covering every allowed vector and an explicit accuracy requirement.

For the T2-easy disposal study, the training-only replay and matched continuation run locally from the checkout containing Orion's original training freeze:

    PYTHONPATH=src:. .venv/bin/python -m scripts.audit_t2_initial_disposal_order
    PYTHONPATH=src:. .venv/bin/python -m scripts.fit_t2_initial_disposal_order

The replay checks the computed initial observations against all six saved parameter vectors. The family analysis bounds the smallest attainable maximum initial error for the two affine disposal equations. For three training events, the middle event's initial inputs are exactly the midpoint of the two endpoints. Its measured output differs from their mean by 0.003732 mg/kg/min, forcing a maximum error of at least half that value. The exported shared parameter vector attains that bound to numerical precision on all 16 initial observations.

The continuation uses the same delayed equation, frozen insulin law, physical constraints, full-trajectory training objective, two starts and budget for each fit. The control fits all training observations. Two additional fits require predicted initial differences to retain at least 1% or 50% of each distinguishable measured gap. These fractions are experimental settings. There are 29 comparisons across three distinct measured initial disposal levels. Each arm saves the selected vector, both starts' outcomes, and an independent full-trajectory replay.

To test one change to construction under the same 50% condition, run

    PYTHONPATH=src:. .venv/bin/python -m scripts.probe_t2_initial_tissue_map

The nested model adds a shared gain on the measured initial tissue glucose relative to the fasting reference in the initializer for insulin action. At zero gain it reproduces the parent. The two fits start from the same values of every common coefficient as the parent comparison. Both starts preserved the disposal ordering. The selected fit improved pooled training NMSE from 0.16076 to 0.10768 relative to the parent under the same constraint, while its worst initial level error grew from 0.0510 to 0.2309 mg/kg/min. The unconstrained parent reached 0.09433 pooled training NMSE and reversed all 29 distinguishable comparisons. Across these training events, the sum of the initial tissue and plasma glucose masses is constant, so replacing centered initial tissue glucose with centered initial plasma glucose and reversing the coefficient sign gives identical initial predictions.

The action derivative depends only on insulin. For the same insulin trajectory, a change in initial action therefore persists throughout the event. Compared with the parent fitted under the same 50% constraint, the change improved pooled NMSE on all 14 events with the common initial glucose but worsened both events with a different initial glucose. Any event-omission evaluation should identify coefficients, starts and normalization constants learned using the omitted event. A fresh forecast evaluation needs training-fold-only preparation.
