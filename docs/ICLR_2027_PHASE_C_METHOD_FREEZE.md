# Frozen post-submission experiments

The completed CPU forecast evaluation, the fitted-coupling ablation and the earlier matched intervention controls are documented in the [experimental results for the author team](ICLR_2027_FROZEN_PHASE_C_RESULTS.md).

The revised method with both real proposer returns and independently replayed numerical fits is the November T2-easy experiment. Its construction commit is `ec485a5`, including authenticated calls inherited from `be80173`. Its matched fitting commit is `1ea3499`. The frozen [experiment manifest](../configs/iclr_2027_phase_c_method_freeze_v1.json) binds both Git revisions, source-tree digests, construction and fitting plans, model weights and the exact six-model fit export. Both original deployment bundles identify their respective commits in `SOURCE_COMMIT`. This composite freeze reproduces the run while later research can continue on the development branch.

## What was executed

The 120B proposer received two fitted parents and public training evidence. Each parent produced a batch of five proposed alternatives. Five additional seeds attempted construction from an empty topology. The runtime applied deterministic equation, initialization, training-consistency and public-path checks, allowed up to three corrections per unresolved branch, and retained deferred checks explicitly. This stage used deterministic feedback, with no separate judge call. Thirty-nine physical requests produced four distinct eligible Candidate 15 descendants and one Candidate 21 descendant. None of the five fresh seeds reached function construction. All five retained children went to fitting.

The fitter optimized shared coefficients and two hidden initial values per training trajectory on the sixteen public training records. Each of two stages had at most 1,000 whole-training evaluations and 900 numerical seconds. The six tasks are the five children and a zero-amplitude restriction of `c15-4` that recovers the fitted Candidate 15 parent equations. We use each task's lowest training error passing the sampled-domain check in the matched run. The freeze does not substitute any supplementary optimizer start, sign-flipped control or unreturned new proposer alternative.

## Eight-minute forecast experiment

The primary comparison is the training-selected meal-memory model `c15-3` against the parent-equivalent zero-basal control on corrected public T2-easy validation. The roster, coefficients, public file digest, root policy, solver tolerances and scoring rules are sealed before reading validation trajectories. Validation in this project is development data and has been used by earlier iterations. The official test set is outside the experiment.

For a new trajectory, fit only its two independent initial values on observations at times up to eight minutes from the start. Two preset starts use the declaration defaults and the per-coordinate median from the training fits. A bounded value-only least-squares fit applies equally to smooth and clipped equations. The retained start minimizes independently verified prefix error. Our forward routine accepts only initial observations and a typed forcing schedule. Every candidate's RHS uses generated quantities and external forcing only. The declared T2-easy policy permits measured auxiliaries, including their permitted initial readings. Future target samples first enter the evaluator after the predictions are fixed. An independent DOP853/Radau agreement check precedes scoring.

The primary aggregate pools post-prefix samples per target with the training-only target SDs and then averages Gp, I and U equally. We also retain each trajectory's error, its median and unscaled MAD, paired differences, per-target errors, sample minima and last-prefix-observation persistence. Numerically verified forecasts are included even when a predicted concentration becomes negative. A missing forecast leaves the complete-roster aggregate unavailable. Stored row predictions and reservations make the analysis resumable without spending a second root-fitting budget.

Run on CPU from the project checkout and its scientific environment:

~~~bash
python -m scripts.iclr_frozen_t2_forecast freeze --construction output/t2-breadth-results-20261108/returned --fits output/t2-breadth-fitting-20261108/verified-campaign --root output/iclr-method-freeze-20261109/frozen-run
python -m scripts.iclr_frozen_t2_forecast score --validation-file artifacts/phase-c-development-final-v2/public/phase_c_dalla_man_t2_canonical_named_easy_rates_v1/val.json --root output/iclr-method-freeze-20261109/frozen-run
~~~

## What can run across benchmarks

The general two-hard-anchor control uses the controlled-search implementation at `6a639ae`. It constructs from the public task with the staged 20B proposer, the advisory 120B critic, the multi-target collocation fitter, and pruning. Full and fixed-state share the exact fitted initial proposal and critic receipt. Each arm has 15 visits for each of three paired seeds on T2-hard and corrected CSTR-hard, giving twelve endpoints. Retain all dispositions before confirmation. The fixed-state arm can change equations. It cannot change the state representation. This comparison tests the value of representation revision for the general constructor. [Treatment definitions](ICLR_2027_CONTROLLED_SEARCH.md)

The November breadth constructor and fitter depend on T2-easy's Gp, I and U targets, fitted training parents and specific meal/disposal witnesses. The hard glucose task instead generates Gp and I, and the hard reactor generates T. Running the general control does not measure the 120B fitted-parent experiment on either hard case. To qualify future use of the improved revision policy across tasks, freeze a public-input adapter with matching controls, obtain one real fresh start → revision → fit → forecast on each hard anchor, and bind that protocol before opening independent confirmation. The 120B peak-timing and late-recovery campaign at `4341150` is a separate experiment.

A low-cost first step is to use the six Full and six fixed-state endpoints of an existing larger campaign if it already runs the identical scientific design. When a separate two-arm run is needed, the standard preparation entry point accepts the smaller roster. Use a separate output directory. The original ACES bundle recorded in the manifest accepts `AF_JUDGE_GATE_ROOT` pointing to the verified `judge-jetstream-known-cases-v1` gate. The deployment correction at `ab00428` updates this path without changing the scientific method. [Cluster setup and serving requirements](ICLR_2027_ACES_EXECUTION.md)

~~~bash
python -m scripts.iclr_controlled_search prepare --public "$AF_REBUTTAL_PUBLIC" --root "$AF_TWO_ARM_ROOT" --cells acquisition_00 sensor_00 --arms full fixed_gamma --seeds 0 1 2 --visits 15
~~~
