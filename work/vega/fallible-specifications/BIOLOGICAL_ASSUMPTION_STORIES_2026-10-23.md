# Biological stories for diagnosing missing or mistaken model assumptions

Prepared by Vega, 23 October 2026, for discussion with the GEMOT collaborators.

These five proposed cases ask an agent to identify and test a questionable premise
about a biological experiment. The biological phenomena have published precedents;
the proposed Autoformalism tasks, fitting signatures and evaluation protocols are
research designs, not experiments already performed by us. No case is assumed to
be available in GEMOT's current data.

I recommend beginning the discussion with **TGF-beta stimulus depletion**, then
**downstream DNA loading of a regulator** and **rapid-equilibrium assumptions in
drug binding**. Population averaging and culture-history-dependent initial states
provide two different, useful tests of the same methodological idea.

## What counts as assumption revision

The desired output identifies a premise, such as “the bioactive stimulus remains
constant throughout the experiment,” explains why it is questionable, and proposes
evidence that could distinguish it from alternatives. The output should not merely
request a more flexible response curve.

Changing an assumption can change the compiled equations. Removing a reservoir,
homogeneity or rapid-equilibrium approximation may introduce a state or a population
average. The distinction is that the biological premise drives the reconstruction,
with ordinary reaction laws fixed in advance. It is not a search over arbitrary
exponents, Hill coefficients or new functional forms. For especially strict
controls, keep a sufficiently general model already present and revise only its
input, population or initial-state constraints.

Also distinguish three origins of a premise: a scientist's explicit hypothesis,
missing experimental information, and an approximation introduced by the model
builder. An omitted protocol detail is not evidence that the scientist asserted
something false. Each benchmark should record the premise's provenance and allow
the agent to ask for missing information. Explicitly reported experimental facts
and raw observations remain fixed.

| Case | Assumption to investigate | What can remain fixed |
| --- | --- | --- |
| TGF-beta stimulation | A single addition maintains a constant bioactive concentration | Intracellular receptor/Smad equations; change the independently constrained input |
| NF-kappaB dynamics | The population average behaves like one representative cell | Intracellular network and local rate laws; change population/observation aggregation |
| Regulatory DNA load | Downstream binding sites do not affect the upstream regulator | Production, degradation and DNA-binding chemistry; account for the finite regulator pool |
| Drug washout | Drug-target binding equilibrates much faster than the protocol changes | Binding and receptor chemistry; relax the timescale approximation |
| Bacterial preparation | Fresh medium and total cell count specify the initial physiological state | Predeclared state-dependent kinetics; revise the initial-state constraint |

## Current fitting foundation

The [updated Vega handoff](../../../docs/VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md)
includes the measurement and uncertainty qualifications through M42. The newer
[Phase C entry point](../../../docs/PHASE_C_START_HERE.md) and linked result reviews
extend the relevant evidence through M47.

- [M37/M38 feedback](../../../docs/PHASE_C_FITTING_FEEDBACK_DRAFTS.md) can describe
  verified training-response discrepancies, conditional investigations, alternatives
  and testable predictions. This is an offline qualified interface; the documented
  current-constructor revision/refit integration remains pending.
- [M39-M42 uncertainty](../../../docs/PHASE_C_FITTING_NONLINEAR_PROFILE_SUPPLEMENT.md)
  shows why nuisance parameters must be reoptimized and incomplete diagnostic
  searches must remain unresolved. The supplement completes all 36 selected
  nonlinear profiles without changing the fitted endpoints. Full local rank and
  accurate sensor predictions can coexist with broad physical-parameter support.
- [M46 hidden initial states](../../../docs/PHASE_C_FITTING_MULTIPLE_ROOTS.md) and
  the [M47 returned results](../../../docs/PHASE_C_FITTING_PREFIX_NOISE_RELIABILITY.md)
  distinguish prediction from individual-state recovery. M47's October 21 review
  records 35 complete tasks and one unavailable numerical forecast verification.
  Its support-based rule improves the reported paired common-horizon forecasts,
  while exact hidden-state ambiguity remains. The review checks summary consistency;
  the full operation archive was not supplied with that return.

These results support richer questions than “which parameter vector fits best?”
They do not supply calibrated confidence percentages or an automatic verdict
that a biological assumption is wrong. Profile support is conditional on the
admitted model, nuisance variables and parameter bounds. New cross-condition,
single-cell or biological-input analyses below still require qualification.

## 1 A signaling pathway seems to adapt because its stimulus is disappearing

**Story to discuss.** A biologist adds TGF-beta to cultured epithelial cells and
records a Smad signaling response. A model treats the supplied dose as a persistent
input. It captures the early response but has difficulty explaining its later
decline across several doses and culture conditions using shared intracellular
parameters. The initial task description gave the dose, but did not establish
that bioactive ligand remained available at that concentration.

**The assumption revision.** “Added once and never deliberately washed out” does
not establish “maintained at a constant extracellular concentration.” Investigate
a finite, depleted ligand pool before changing intracellular response functions.

**Why this is concrete.** Clarke et al. explicitly examined the constant-ligand
assumption in earlier models. In their studied cultures, ligand depletion was a
major determinant of Smad signal duration. Zi et al. subsequently investigated
how ligand dynamics and medium volume affect responses. These are scoped findings,
not a claim that intracellular feedback is irrelevant in every TGF-beta system.
[Clarke et al., 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2668365/);
[Zi et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3130555/).

**What would make the agent suspicious.** Late residuals, apparent deactivation
rates that change with medium volume or cell number, and a tradeoff between input
persistence and intracellular deactivation would motivate checking the input.
These are proposed diagnostic patterns, not unique signatures of depletion.
One flexible fit to one trajectory could conceal the error entirely.

**A discriminating check.** Measure bioactive ligand in the medium independently
of the Smad response. Combine that with controlled replenishment or a suitable
medium-transfer assay. Compare conditions that change the available ligand pool
while monitoring changes in cell state. Replenishment alone does not isolate every
alternative; receptor downregulation, feedback, ligand inactivation and changed
readout behavior remain possible.

**A bounded modeling test.** Hold the intracellular model fixed and compare a
constant input with an independently constrained ligand trajectory. A freely
fitted input at every time point would make the test meaningless.

**For the biologists to debug:** Which cell system permits a reliable bioactivity
measurement? Can medium volume or replenishment be varied without introducing a
larger biological change? Under which conditions is depletion negligible?

## 2 The average cell appears to stop oscillating although many cells continue

**Story to discuss.** TNF stimulation produces a large first peak and diminishing
later peaks in mean nuclear NF-kappaB. A model with sustained intracellular
oscillations misses the mean; changing feedback parameters can fit that mean but
then conflicts with available single-cell behavior. The task silently equated
an average over cells with the trajectory of one typical cell.

**The assumption revision.** Cells need not respond in synchrony or have identical
initial states and timings. An average can lose oscillatory amplitude as individual
responses drift out of phase. Retain the intracellular network and investigate
the population representation.

**Published grounding.** Zambrano et al. show heterogeneous single-cell NF-kappaB
responses alongside a strongly damped average. Paszek et al. examine the role of
heterogeneous timing in a mechanistic framework. The populations also contain
different response types; replacing “one representative cell” with “every cell
is an identical undamped oscillator” would create another questionable premise.
[Zambrano et al., 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0090104);
[Paszek et al., 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2895068/).

**What would make the agent suspicious.** Different intracellular parameters may
be required for early versus late mean behavior, or for the mean versus individual
traces. Once a reviewed ensemble is admitted, intrinsic damping and timing
dispersion may trade off. A precise estimate conditional on a homogeneous model
does not validate homogeneity. Bulk data alone may fit well and remain ambiguous.

**A discriminating check.** Track the same cells over time, quantify individual
amplitudes and peak timing, and reconstruct the observed average from those traces.
Preserved individual oscillations with growing timing dispersion support this
explanation. Intrinsic damping, nonresponders, stimulus drift and imaging effects
must also be assessed; peak alignment alone can manufacture an apparent pattern.

**A bounded modeling test.** Use a fixed intracellular model and a predeclared,
low-dimensional distribution of initial states or response timings. Compare zero
heterogeneity with independently constrained heterogeneity. Avoid arbitrary
per-cell trajectories chosen merely to fit the average.

**For the biologists to debug:** Is there a suitable dataset with both longitudinal
single-cell measurements and an aggregate? Does the actual aggregate use equal
cell weights, or do cell size, expression and image processing change the weights?

## 3 An apparently irrelevant DNA construct changes the regulator being modeled

**Story to discuss.** A synthetic circuit expresses LacI, and GFP reports its
activity. A second plasmid is treated as background experimental context. A shared
upstream model no longer explains both constructs and both induction and withdrawal
responses. The temptation is to alter the production curve or add an unexplained
delay. The missing detail is that the second plasmid carries LacI-binding sites.

**The assumption revision.** Downstream DNA can bind enough regulator to alter
its availability and dynamics. “This construct is downstream of the regulator”
does not establish “it cannot affect the regulator.” Include the load on the
finite regulator pool using ordinary binding chemistry.

**Published grounding.** Jayanthi et al. compared an operator-bearing client
plasmid with a matched blank plasmid and found delayed induction and faster
de-induction. The GFP reporter was present in both configurations; the manipulated
load was the additional operator-bearing construct. Separate reporter experiments
show that binding load and competition for degradation resources can both matter.
[Jayanthi et al., 2013](https://web.mit.edu/ddv/www/papers/ACS2013.pdf);
[Moriya et al., 2019](https://pubmed.ncbi.nlm.nih.gov/30917535/).

**What would make the agent suspicious.** Apparent production/degradation parameters
may depend on the extra construct, and a change that improves induction may worsen
withdrawal. This directs attention to construct context. Neither the residual
shape nor sensitivity to a proposed load parameter proves sequestration.

**A discriminating check.** Compare otherwise matched constructs with intact
versus binding-defective operator sites; measure copy number and growth, and
regulator abundance where feasible. Account for expression burden, dilution,
reporter maturation and shared degradation machinery. Operator mutation is useful
only if it does not introduce uncontrolled changes in those quantities.

**A bounded modeling test.** Fix production, degradation and regulator-DNA binding
laws. Revise the assumption that the extra binding-site pool is negligible.
This may require additional bound-state bookkeeping; that is a consequence of
the biological revision, not a discovery of a novel response function.

**For the biologists to debug:** Is the relevant load regulator binding or general
resource competition in their system? Can an independent upstream readout separate
an actual dynamical change from a changed reporter response?

## 4 Washing out a drug does not instantly reset its target occupancy

**Story to discuss.** A biologist exposes EGFR-expressing cells to lapatinib and
then removes the drug from the medium. A model using instantaneous equilibrium
occupancy predicts recovery too quickly. It could compensate by making downstream
recovery artificially slow. The premise to investigate is the timescale separation
used to eliminate binding dynamics.

**The assumption revision.** The current free-drug concentration may not uniquely
specify current target occupancy. Bound drug and receptor conformations can retain
history. Relax rapid equilibrium while keeping a reviewed receptor-binding and
turnover mechanism.

**Published grounding.** Wood et al. measured slow lapatinib dissociation in
purified systems and prolonged suppression of receptor phosphorylation after
cellular washout. They also note that cellular recovery involves other processes.
Kleiman et al. modeled receptor conformations and phospho-turnover, illustrating
why cellular drug response cannot be reduced to equilibrium affinity alone.
[Wood et al., 2004](https://pubmed.ncbi.nlm.nih.gov/15374980/);
[Kleiman et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3222462/).

**What would make the agent suspicious.** A single parameter set fits sustained
exposure but misses short pulses or washout. Responses depend on exposure history
even at comparable current free concentration. Under a finite-kinetics model,
binding, recovery and residual exposure may be poorly separated. History dependence
challenges the shortcut but does not identify which hidden state carries it.

**A discriminating check.** Combine controlled exposure histories with independent
binding/occupancy measurements and checks of free drug remaining near and inside
cells. Phosphorylation is a downstream readout, not direct occupancy. Retained
intracellular drug, rebinding, receptor turnover and downstream adaptation can
coexist with slow dissociation.

**A bounded modeling test.** Predeclare the chemistry and compare the rapid-
equilibrium approximation with its finite-timescale realization. A minimal
one-step binding control can test the methodology, but should not be advertised
as a complete lapatinib mechanism.

**For the biologists to debug:** Which target and compound offer the clearest
separation between binding kinetics, transport and downstream recovery? Is
lapatinib too complex for the first case, despite its strong biological precedent?

## 5 Fresh medium does not guarantee a fresh physiological initial state

**Story to discuss.** Two preparations of the same bacterial strain are moved
from stationary phase into fresh medium. At the same nominal cell count, a
biologist starts an ampicillin survival experiment. The model captures early
killing but misses different surviving tails. The task specified medium and
cell count, but not enough preparation history to establish which cells had
resumed growth.

**The assumption revision.** Transfer to fresh medium does not necessarily put
every cell into the same active or equilibrium state. Initial physiological
composition can retain preparation history even when bulk counts are similar.

**Published grounding.** Jõers et al. followed heterogeneous growth resumption
and antibiotic sensitivity after transfer from stationary phase. Harms et al.
show how inoculum and growth phase can confound persistence assays. This is not
a universal explanation of persistence: Umetani et al. found that many persisters
in their exponential-culture experiments were growing before treatment.
[Jõers et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20435730/);
[Harms et al., 2017](https://journals.asm.org/doi/10.1128/mbio.01964-17);
[Umetani et al., 2025](https://elifesciences.org/articles/79517).

**What would make the agent suspicious.** Shared within-state kinetics may fit
early observations but fail on history-dependent tails. Allowing an initial
non-growing fraction can expose a tradeoff with awakening or killing parameters.
An accurate survival curve alone may identify neither that fraction nor its cause.
This closely matches Orion's distinction between good predictions and ambiguous
hidden initial states, although the bacterial example itself is unqualified.

**A discriminating check.** Record preparation history and follow individual-cell
growth before treatment. Compare matched preparations with independent information
about initial state. Check actual antibiotic exposure and survivor regrowth;
resistant variants, clumps, drug degradation and counting artifacts remain
alternatives. Restrict the story to an experimentally supported strain, preparation
and drug combination.

**A bounded modeling test.** Start with fixed active/non-growing compartments and
fixed state-dependent kinetics. Compare an all-active initial constraint with an
independently constrained initial mixture. If the original candidate has only one
compartment, adding another also changes its state construction; do not hide that
fact by calling it merely a parameter update.

**For the biologists to debug:** Can they nominate a preparation in which the
initial-state explanation is independently testable? Does a two-state approximation
preserve the biology well enough, or would it introduce a misleading dichotomy?

## Two additional assay-assumption cases

These are useful if the collaborators want observation/input controls alongside
the biological premises above.

**The device changes the delivered drug.** A model predicts response from the
programmed inlet dose, but a PDMS microfluidic device absorbs some compounds.
The premise is that nominal concentration equals cellular exposure. Keep the
biological equations fixed and independently characterize delivery and washout.
Cell-free measurements and a controlled material comparison can help; outlet
concentration alone need not equal exposure at the cells. This is an experimentally
grounded input-contract case, with chemical- and device-specific effects.
[Gómez-Sjöberg et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20936785/);
[Auner et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6512955/).

**The reporter becomes visible after the biology has already happened.** An
anaerobic bacterial conjugation assay counts fluorescent transconjugants. Counts
change with subsequent oxygen exposure and reporter maturation, potentially
suggesting altered transfer kinetics. The premise is that fluorescence immediately
and comparably reports the latent biological event. Pinilla-Redondo et al. directly
studied aerobic fluorescence recovery after anaerobic mating, including blocked
protein synthesis. Split-sample maturation controls and an independent transfer
readout can constrain the observation process without changing transfer laws.
Reporter, organism and handling details matter.
[Pinilla-Redondo et al., 2018](https://journals.asm.org/doi/10.1128/aem.02507-17).

## What the agent should return in a first demonstration

For a selected case, the concrete output should contain:

1. The exact disputed premise, where it came from, and which contexts it covers.
2. The measured mismatch and the qualified numerical evidence available so far.
3. A small set of competing explanations, including computation or missing
   experimental information when relevant.
4. A specific check whose possible outcomes distinguish those explanations.
5. A conditional revision and a fresh prediction that could fail.

For example: “The current model assumes constant bioactive TGF-beta after a
single addition. The late response mismatch motivates checking that premise.
Ligand depletion and intracellular regulation remain competing explanations.
Measure medium bioactivity and compare controlled replenishment. Then test whether
the same intracellular parameters predict the new condition.” This is a proposed
response format, not an existing fitted result or a demonstrated automatic diagnosis.

The evaluation should reward identifying a falsifiable assumption and knowing
when evidence is insufficient. It should penalize confidently naming the hidden
cause from a residual alone. A candidate that preserves the original premise and
explains the same qualified evidence remains a counterexample to declaring that
premise incompatible within the examined scope.

For each accepted biological case, use both a condition where the approximation
is adequate and one where it fails, plus an ambiguous condition where the agent
should request evidence. A sufficient-reservoir control for case 1, for example,
prevents the system from learning that “constant ligand” is always wrong. Keep
the kinetic vocabulary, hypothesis access and total budgets matched across
baseline methods. Independent outcome data should assess predictions rather than
be disclosed retrospectively to the proposing agent.

No simulation, implementation, laboratory experiment or communication with the
GEMOT authors is initiated by this discussion document. The next scientific step
is their review of case realism and the evidence needed to adjudicate each premise.
