# How the discovered equations change during search

This is an experimental handoff for the author team. The examples use saved model declarations, critic output, parent identifiers, fits and selection records. The submitted six-visit cohort, a pre-submission one-step pilot, and the corrected Phase C experiments use different protocols. The examples were chosen for traceability. Population counts below use every planned matched pair, including those missing an executable model. [Figures and evidence files](../output/iclr-structure-examples-20261110/REPORT.html)

## Recorded changes during search

### Submitted method, named T2-hard, Full, seed 0

At the first visit the proposer already had a latent state $X$ driven by insulin. After expanding the instantaneous alias for delayed action, the relevant equations had the form $\dot X=aI-bX$ and a glucose-balance contribution proportional to $I X$. Visits are numbered 1–6 here; the archive indexes them as rounds 0–5.

| Visit | Change to the executable model | Trial validation NMSE | Retained validation NMSE |
|---|---|---:|---:|
| 1 | Glucose uses the original latent $X$. Fitted coefficient on $IX$ is −0.29672. | 1.235 | 1.235 |
| 2 | A new differential state $Z$ replaces the instantaneous delayed-action alias: $\dot Z=cI-dZ$. Glucose now uses $IZ$ with coefficient −0.09535. $X$ still evolves but no path leads from it to either output. | 1.132 | 1.132 |
| 3 | The inner utilization law changes sign while glucose continues to subtract it. The assembled coefficient on $IZ$ becomes +0.0000773. | 1.011 | 1.011 |
| 4 | Refitting gives an assembled coefficient of +0.0000875. | 0.989 | 0.989 |
| 5–6 | Two alternatives are tried. The visit-4 model remains the incumbent. | 1.095, 1.096 | 0.989, 0.989 |

This is the clearest example from the submitted implementation of state addition, rerouting an interaction, fitting parameters and rejecting later trials. It also illustrates why lower error alone is inadequate: the fitted term named *disposal* adds to the glucose derivative. Both verifier and critic were enabled. The historical export preserves the trial and incumbent equations at each visit, but lacks the critic's actual words. It cannot identify which particular advice motivated visit 2 or 3. [All trial and retained equations and coefficients](../output/iclr-structure-examples-20261110/evidence/submitted-t2-hard.json) · [Independent signed-term audit](ICLR_2027_HISTORICAL_FOLLOWUP.md)

### Alien device, earlier development pilot, seed 3

This is the closest archive to a single worked example with a proposer, deterministic rejection, scientific critic, numerical fitter, pruning and final selection. It used Gemini for both proposer and critic and evaluated **one-step-ahead** predictions. Use it to explain the sequence of decisions. Its numerical scores belong to that earlier pilot.

1. **Initial proposal.** For the observed response $y$ and command $u$, the proposer gave $\dot y=-a_y y+b_y u+c_x x$, $\dot x=-a_x x+b_x u$, with $x(0)=0$. After fitting, pruning deleted the $x$ contribution to $\dot y$ and the self-decay term in $\dot x$. The retained parent had one-step validation NMSE 0.000482. The fit before pruning had several zero gains and very poor error. Pruning also refitted the remaining parameters.
2. **Recorded feedback.** The post-pruning critic identified that $x$ could no longer affect the target and advised either deleting $x$ or connecting it to the equation for $y$. The next attempted child reproduced the original structure. The deterministic runtime rejected it as a structural duplicate. A later child that drove $x$ with $y$ obtained validation NMSE 382.0.
3. **Selected revision.** Another child of the same reviewed parent introduced the algebraic process $P=u^2$, drove $x$ with $P$, and fed $x$ back into $y$. A final pruning/refit removed the near-zero $-a_y y$ contribution. The selected fitted equations, rounded, were $\dot x=-0.16660x+0.51171u^2$ and $\dot y=-0.67068u+0.06751x$. Its validation NMSE was 0.000468 (0.000487 before the last pruning/refit). Both links in the new memory path have nonzero fitted coefficients.

The critic identified the disconnection. The proposer supplied the squared-input law. The pruning and fitting records identify what survived. The actual critic event, exact parent links, duplicate rejection, rejected numerical alternative and selected model ID all agree. These observations cannot measure what the proposer would have done in a critic-off run. [Critic instruction, proposals, fits and parent links](../output/iclr-structure-examples-20261110/evidence/early-critic-and-pruning.json)

### Coupled detention basins, corrected Phase C, brief-only, seed 1

The first variable reply proposed upstream depth $h_u$ and downstream depth $h_d$ as differential states. An empty follow-up twice left the verifier's upstream-memory requirement unresolved. Its repair message listed that exact obligation. The next reply added a distinct differential state, `mem_up`. Upstream depth itself can carry storage history; the added state was a response to the bookkeeping requirement and does not establish that physics needed a third state.

The optional process proposal supplied one overflow $Q$ with opposite signs in the depth balances, giving $\dot h_u=(q_u-kQ)/A_u$ and $\dot h_d=(q_d+kQ)/A_d$. The proposed threshold law was $Q=A_u\max(0,h_u-h_{crest})$. The same reply proposed routing downstream discharge back into the supplied inflow. The verifier rejected that use because an external input cannot receive a model contribution. A separate outlet was then absent from the fitted model. The fitter returned **$k=0$**. The model passed both inflow-conversion checks and transfer cancellation, failed the outlet and upstream-memory tests, and scored validation NMSE **6.055**. Its seed-0 partner also passed 3/5 checks but passed upstream memory and failed transfer cancellation, with NMSE **1.796**.

This one-visit example traces literal repair feedback, a new state, a proposed shared law, and a numerical fit that deactivates that law. The optional process proposal was part of construction; there was no separate scientific critic or final pruning phase in this pilot. [Submitted replies and fitted balances](../output/iclr-structure-examples-20261110/evidence/phase-c-basin.json)

### Numerical feedback with an independently refitted control

For a positive free-forecast example, the frozen November T2-easy run supplied the 120B proposer with the fitted parent's depletion problem and three shifted 90 g meals. The measured insulin peaks occurred at minutes 101, 161 and 221. The proposer introduced $M(0)=0$, $\dot M=k_r u_M-M/\tau$, and added $g_M M$ to insulin production. It passed the structural checks on its first attempt; the boundary sign remained deferred. The fitted time constant was 62.79 min. Against the independently refitted parent family, the child's training NMSE fell from 0.16556 to 0.11673. Validation insulin NMSE improved from 0.1575 to 0.0841, with improvement on all four schedules. Pooled three-output NMSE improved from 1.605 to 1.536; fasting glucose worsened. Silencing just $g_M$ at the child's fitted coefficients and inferred initials reproduced the fasting forecast exactly and worsened every meal forecast. This run documents numerical feedback and a functionally active revision, with no separate LLM critic. [Request evidence and actual alternative](../output/iclr-structure-examples-20261110/evidence/phase-c-meal.json) · [Matched fitting and deletion experiment](ICLR_2027_FROZEN_PHASE_C_RESULTS.md)

## Structural stability across seeds

The submitted proposed-method campaign used **two algorithm seeds (0, 1)** per condition. The submitted external-baseline comparison used three repetitions. At the common six-visit cutoff there are 80 inter-seed pairs across the nine cases, two prompt variants and the assigned ablations. Both executable models and their fitted mechanism tests were available in 38 pairs (four named glucose cases). Both validation scores were available in 78. For the **fully enabled Full-prompt arm** specifically, the corresponding counts are 3/9 and 8/9. Missing exports and missing fits remain in the roster. [Submitted experiment configuration](../configs/final_component_campaign_v1.json) · [Every pair and its availability](../output/iclr-structure-examples-20261110/results/seed_pairs.csv)

We compare the structures at four levels.

1. **Exact equation structure.** The existing fingerprint preserves public channel names, signed additive terms and declaration metadata, and is invariant to consistent renaming of internal symbols. The stricter function fingerprint additionally records expression operators and fixed constants.
2. **Direct equation dependencies.** Substitute each stored fitted coefficient, simplify exact zero contributions with the repository's restricted expression engine, and inline named algebraic processes along dependency paths. Keep only differential states capable of reaching a generated output. Make a directed edge for each source in a state derivative or output readout. Public input, auxiliary, fixed-covariate, explicit-time and target identities stay anchored. Match internal observed states to observed states and latent states to latent states. Enumerate every one-to-one relabeling, including unequal state counts, and report the largest edge Jaccard $|E_0\cap E_1|/|E_0\cup E_1|$. A two-empty comparison is unavailable. This score ignores sign, multiplicity, functional form and initial-value formulas. It does not establish equivalence under general changes of latent coordinates or prove a tiny coefficient has a material effect.
3. **Public input-to-output pathways.** Compare the set of time-varying supplied input or auxiliary channels that reach each target, allowing any number of intermediate states. This describes externally visible dependencies while allowing different internal realizations.
4. **Functional stability.** Keep each public mechanism's pass, fail or unresolved status, the validation NMSE of each seed and their absolute difference. Examine paired intervention responses where physical reference trajectories exist.

| Matched pairs | Submitted Phase B, 6 visits | Corrected Phase C pilot, 1 visit |
|---|---:|---:|
| Both fitted models available / planned | 38/80 | 10/16 |
| Identical signed structure fingerprint | 0/38 | 0/10 |
| Identical function fingerprint | 0/38 | 0/10 |
| Equal number of active latent states | 15/38 | 8/10 |
| Direct-equation edge Jaccard, median [MAD] | 0.679 [0.114] | 0.500 [0.183] |
| Public-path Jaccard, median [MAD] | 0.857 [0.143] | 0.500 [0.250] |
| Same set of public paths | 17/38 | 4/10 |
| Same result on every public mechanism test | 23/38 | 5/10 |

The September Phase C pilot included eight cells across four domains, two seeds and two prompts. Its scores use its own corrected data and checks. Among its ten comparable pairs, two had identical unsigned direct wiring despite different signed-structure fingerprints. The two cohorts are each summarized on their own denominator.

In the submitted cohort, the absolute inter-seed validation NMSE gap had median **0.160 [MAD 0.110]** among the 17 pairs with the same public paths, and **0.465 [0.338]** among the 21 with different paths. Per-mechanism verdicts agreed exactly in 12/17 and 11/21, respectively. Conditions share benchmarks and seeds, and each seed refits its own coefficients. This is a descriptive association. Among all 78 pairs with two validation scores, including missing model exports, the median gap was 0.240 [0.199]. [Matched-pair analysis](../output/iclr-structure-examples-20261110/results/seed_summary.json)

| Example | Active structure in seeds 0 and 1 | Validation NMSE, 0 / 1 | Fitted mechanism outcome |
|---|---|---:|---|
| T1-easy, submitted Full | Both learned a leaky meal state $\dot M=au-bM$. Seed 0 added an explicit time term in the glucose balance; seed 1 added a direct meal term. Identical public input paths; direct-wiring Jaccard 0.80. | 0.246 / 0.397 | Meal response passed in both. |
| T1-hard, submitted Full | Seed 0 used two target-reachable meal filters. Seed 1 used a direct meal term and measured tissue glucose with no active latent state. Wiring Jaccard 0.20. | 0.243 / 0.177 | Meal response passed in seed 0 and was unresolved in seed 1, which had lower NMSE. |
| T2-hard, submitted Full | Seed 0 used one active insulin-driven state in a bilinear glucose term. Seed 1 used an insulin filter and a separate meal filter, and also a direct meal-to-insulin term. Wiring Jaccard 0.571. | 0.989 / 0.398 | Both passed meal and were unresolved on delayed insulin action. Under a short fasting insulin pulse the two glucose responses peaked at +35.59 and −3.86 mg/kg; the reference peak change was −2.84 mg/kg. |
| Alien device, corrected Phase C, brief-only | Both declared a state called $m$. Seed 0 integrated the measured output from $m$. Seed 1 read the output algebraically from $m$ and used a telemetry channel in $\dot m$. Wiring Jaccard 0.333. | 0.998 / 1.434 | Input memory was unresolved in seed 0 and passed in seed 1. |
| CSTR-easy, corrected Phase C, brief-only | Both had identical unsigned direct dependencies. The process functions and fitted coefficients differed. | 0.0097 / 0.4702 | Both scored 0 confirmed, with a fail in seed 0 and unresolved in seed 1. |

There is also a **five-seed topology-only check** on the corrected coupled basin. All five represented the same upstream loss/downstream gain orientation. Four included a distinct downstream outlet. Seed 4 lacked it, and seed 2 labeled the correctly oriented internal process an influence. That experiment stopped before function construction and fitting, so it has no NMSE or fitted compliance score. [All five declared topologies](../output/iclr-structure-examples-20261110/evidence/five-basin-topologies.json)

## Reproduction and missing historical feedback

Run from the project checkout against the original archived model inventory and public pilot. The analysis verifies source identities, retains failed rows and performs no selection, fitting, LLM requests or test-trajectory reads.

```bash
.venv/bin/python -m scripts.iclr_seed_wiring --history output/iclr-local-completion/historical --inventory artifacts/dalla-all-models-review-2026-09-25/source/inventory.json --pilot output/phase-c-inspection-2026-09-30 --output output/iclr-structure-examples-20261110/results
```

For verbatim critic attribution in the **submitted** example, the source ACES tree can supply the critic receipt from the retained parent's originating round plus the next proposal and result. The evidence folder includes a read-only collector for the first six visits of named T1-easy, CSTR-easy, alien-easy and T2-hard, Full seed 0. It also collects any available pruning records separately. They are later endpoints outside the submitted six-visit cutoff. No scheduler allocation is required. [ACES collection script](../scripts/hpc/collect_iclr_example_traces.sh)

On the ACES login node, run `bash scripts/hpc/collect_iclr_example_traces.sh` from this revision of the checkout, or upload the standalone script to group scratch and run it there. It prints the exact archive path and the number of missing receipts.
