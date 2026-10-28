# Two tasks for biological review: prediction and warranted premise revision

Prepared by Vega, 28 October 2026. Discussion draft for the prospective GEMOT
collaboration; it has not been sent to the collaborators. The stories below are
synthetic. Their biological interpretation, useful tolerances and feasible
measurements remain open to correction.

We want to revise assumptions about an experiment while keeping a reviewed
mechanistic model family fixed. Our current calculations show that obtaining a
useful prediction need not identify which assumption failed. The purpose of this
discussion is to establish when that distinction matters experimentally.

## Two versions of the same scientific question

**Story P: predict the response to another stimulus.** A scientist adds a ligand
and records a downstream signal. A model assuming both sustained ligand
availability and sustained cellular responsiveness fails. Ligand loss and loss
of response capacity are competing explanations. The immediate decision is which
specified dose and timing protocol will produce the required mean response.
Could a model support that decision while explicitly leaving the cause unresolved?

The returned object would be a response function over the agreed dose/timing
protocols, with one consistent set of model choices across those protocols and
uncertainty from the surviving explanations. Representative parameter values
could support prediction without becoming claims about the true biological cause.
Predictions concern a defined measured mean; an individual noisy replicate would
require a different uncertainty statement.

**Story J: revise a named assumption with evidence.** The observations and future
decision are the same, but the scientist also needs at least one named assumption
that can be challenged, such as the original constant-ligand premise. This might
change which input is measured or controlled in later experiments. A useful response
curve alone is insufficient: compatible explanations preserving constant ligand
must be addressed before calling that premise false.

Here the output adds a scoped claim, such as “bioactive ligand changes by more
than the agreed tolerance over this window,” supported by the admitted evidence.
If ligand loss and reduced responsiveness remain indistinguishable, the output
must retain that ambiguity. A joint statement that the two premises cannot both
hold is useful, but does not complete a request to identify one failed premise.
Both changes can also occur together. Removing a modeling restriction permits
alternatives; it does not establish that its opposite is true.

## A small mathematical explanation to challenge

Our illustrative family is

\[
\dot L=-\kappa L,\qquad \dot R=-\gamma R,\qquad
\dot x=aLR-x.
\]

Here \(L\) represents available ligand and \(R\) effective response capacity.
Neither variable is yet tied to a validated molecular measurement. A baseline
response identifies combinations such as amplitude \(A\) and summed decay
\(s=\kappa+\gamma\), rather than necessarily identifying the two decay rates.

For a separate matched run receiving one added pulse at time \(\tau\), the
illustrative mean increment at delay \(\Delta\) is

\[
\Delta m=d\eta e^{-\gamma\tau}F_s(\Delta),\qquad
F_s(t)=\frac{e^{-st}-e^{-t}}{1-s}.
\]

The formula has its continuous limit at \(s=1\). The quantity \(d\) is an
actually delivered dose multiplier; \(\eta\) combines preparation and readout
scale. At one pulse time, observations can constrain the product
\(\eta e^{-\gamma\tau}\) without separating its factors. Two pulse times can
separate them in the idealized noiseless model only if the relevant scale is
shared. Allowing an independent scale for every run can restore the ambiguity.
This is a conditional explanation, not evidence that experimental scales are
shared or that two noisy measurements suffice in practice.

## Answers that would change the problem

| Question for the collaborators | Why the answer matters |
| --- | --- |
| What actual decision needs the model: one endpoint, several protocols, a ranking, or a dose choice? | This fixes the output and error criterion. Several endpoints should be required by the science, rather than added to create difficulty. |
| Is the target ligand concentration, ligand bioactivity, downstream signal, or target occupancy? | These quantities need different observation maps. A concentration assay may not establish bioactive exposure. |
| Which preparation, gain and kinetic quantities are shared across runs, and what independently supports that sharing? | Sharing enables inference from the second protocol; fit quality alone cannot supply it. |
| Does a logged dose describe the command or verified delivery? | Delivery uncertainty belongs in the model unless independently constrained. |
| Can pulses be compared in separate matched preparations, or only sequentially in one preparation? | The toy calculation uses separate runs with reset clocks. Multiple pulses within one run require their accumulated history. |
| What changes count as practically nonconstant, over which window? | Exact zero decay is usually a different question from acceptable approximation error. |
| Which assay, calibration or protocol record is available, with what accuracy and total cost? | Evidence choices must be compared using actual acquisition constraints. |

If a pulse increment is estimated by subtracting a baseline measurement, we also
need the raw measurement pairing: shared baselines induce correlated errors.
Subtracting estimated means does not create a noiseless new measurement.

The most useful next contribution would be one real example with a consequential
decision, an initially plausible assumption, and independent evidence that can
judge its revision. Either story may prove unimportant or unrealistic. That
would be valuable feedback before selecting a new experiment or theoretical
claim.
