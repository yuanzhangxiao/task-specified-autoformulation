# Biological discussion: what decision should an assumption revision support?

Prepared by Vega, 28 October 2026; revised after Lyra's task-selection response.
Discussion draft for the prospective GEMOT collaboration. It has not been sent
to the collaborators. The illustrative case below is synthetic; no biological
tolerance, assay accuracy or acquisition cost has been established.

We would like to begin with a modeling decision. What did the scientist
need to predict, choose or explain, and what changed after an assumption was
revised? An independent assay, intervention or protocol record that could judge
the revision would be useful. Better fit to the same outcome alone
may not distinguish competing explanations.

Our methodological aim is to connect an assumption change to the evidence that
warrants it. The required conclusion should follow the scientific decision;
our current mathematical examples should not determine which biology we study.

## Three potentially useful outcomes

| Outcome | What the scientist receives | When it could complete the task |
| --- | --- | --- |
| Useful prediction | A response function with uncertainty over the relevant protocols, while causes may remain unresolved. | The decision can be made reliably without choosing among the surviving explanations. |
| Joint premise revision | Evidence that a specified combination of assumptions cannot all be retained, together with the unresolved alternatives. | Withdrawing that combination changes the model's use, experimental design or interpretation sufficiently. |
| Constituent attribution | Evidence challenging a particular scoped assumption, such as practically constant ligand availability. | Knowing which assumption fails changes what the scientist measures, controls or concludes. |

These requirements are not a ranking of scientific value. A joint revision can
complete the decision; an individually identified failure can be irrelevant.
Complete mechanism recovery is a further requirement to request only when useful.

Fix acceptable claim content, scope and usefulness before comparing methods.
Naming the conjunction “ligand availability and responsiveness are both constant”
does not make its rejection an identification of which constituent failed.
Relabeling must not change the score. Report what was withdrawn or qualified and
which alternatives remain; simply allowing everything does not explain the
evidence-supported change.

## What kind of returned model is acceptable?

One possible requirement is a model that remains statistically compatible with
the evidence used in the analysis. This is appropriate when the model is offered
as a surviving explanation. Compatibility is conditional on the observation and
uncertainty assumptions; it does not establish biological truth or identify the
mechanism. New evidence can remove a previously useful representative model,
so its eligibility must be checked again.

An explicitly labeled predictive approximation can remain useful for a target
without surviving as an explanation of all observations. Report its predictive
guarantee and compatibility status separately. Agree on this choice before
evaluation. Either output should use a consistent model across its declared
protocols; experimental inputs remain supplied inputs.

## Details that would determine the mathematical problem

| Question for the collaborators | Why the answer matters |
| --- | --- |
| What changes after the conclusion: a dose, protocol, measurement, interpretation or decision to stop? Would a joint revision suffice? | This identifies useful claim content and prevents unnecessary attribution. |
| Is the target a biological concentration, bioactivity, measured signal, population mean or individual observation? | These require different observation maps and error statements. |
| What prediction domain and error are scientifically meaningful? | One endpoint, a trajectory, protocol rankings and dose choice impose different requirements. Approximation tolerances and operating windows must be explicit. |
| Which initial conditions, kinetic quantities, observation gains and preparation factors are shared across runs, and what supports that sharing? | Cross-run inference depends on these facts; outcome fit alone cannot supply them. |
| Does a logged intervention describe the command or verified delivery? Are runs independent, matched, or sequential within one preparation? | Delivery uncertainty and accumulated history may change the possible explanations. |
| Which independent assay, calibration, intervention or existing record could test the disputed premise? | Adjudication should distinguish explanations, not merely repeat their common outcome. |
| What are the achievable accuracy, sample dependence and total costs of those actions? | Setup, repeated samples, time and shared batches belong in the acquisition comparison. These values remain unspecified. |

If a response increment is estimated by subtracting a baseline measurement,
we need the raw pairing and uncertainty: shared baselines can induce correlated
errors. Subtracting an estimated mean does not create a noiseless measurement.

## Optional illustration to correct or replace

A scientist adds ligand and observes a downstream signal. Declining ligand
availability and declining cellular responsiveness could produce the same
signal in our simplified control. Repeating a stimulus at another time may help
separate them, but only under justified preparation and readout sharing. A
concentration assay may also differ from the required bioactivity assay.

This example should be replaced if the consequential failure instead concerns
population averaging, initialization, observation mapping or another premise.
The immediate need is a realistic decision and independent evidence for judging
its revision, rather than a case engineered to require a particular algorithm.
