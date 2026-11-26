# Determining function forms when some states are hidden

This CPU experiment takes a candidate graph, its initialization rules, the input interpolation contract and measured training events. It returns comparisons of function families, with the observations and assumptions behind each comparison. The construction pipeline can use a qualified exclusion to narrow a choice, or keep multiple forms for whole-event fitting. A predictive preference alone leaves the scientific interpretation open.

## Algorithm

| Step | What the constructor can use | What it returns |
|---|---|---|
| 1. Compare matched starts | For a measured target, group training events by start time and every argument of its equation except one. Infer equality of hidden initial values from the declared initializer. An unknown shared map with equal arguments gives equal hidden values. A freely estimated trajectory-specific initial value gives no such guarantee. | Compare algebraic output values or independently bounded instantaneous right-hand rates at the remaining argument's distinct levels. Three or more levels can exclude an affine response on that slice. Record the number of usable levels even when the evidence stops there. |
| 2. Contrast an isolated input switch | Under \(\dot y=A+g(u)\), assume \(A\) is continuous across a recorded step from \(l\) to \(h\). Hidden states may differ between events. | The one-sided rate change equals \(g(h)-g(l)\). With bounded sample error and within-interval curvature, solve the resulting interval inequalities around all switches. Report sampled values relative to one offset per connected group of levels, or a contradictory cycle. |
| 3. Eliminate one hidden state | Under the proposed pathway \(\dot z=-bz+g(u),\ \dot y=-ay+kz\), assume shared positive decay rates and known input interpolation. | Solve the integrated identity \(y''+(a+b)y'+ab\,y=kg(u)\) for the rate sum, product and effective forcing at supported levels. The window weights and their first derivatives vanish at both ends, so the calculation only needs sampled \(y,u\). Check rank, conditioning, physical rates and, when supplied, continuous measurement/interpolation bounds. |
| 4. Compare remaining choices | Fit competing functions on the same events and forecast separate whole training-development events using the same initialization and budget. | Report paired forecast errors. For this synthetic study a winner must beat each rival by more than 0.015 output units RMSE on a majority of events and lose by that margin on none. Keep a tie when the observations do not separate the choices. |

The checks at steps 1–3 apply independently. An absent input switch, an unconstrained hidden initial value or an incompatible graph can each leave a check unavailable. The strict two-state pathway requires a hypothesis about functional structure in addition to its edges. The elimination determines \(kg\), and the decay rates up to interchange. Distinct \(k\) and \(g\) can produce the same measurements.

## Results

- **T2-easy.** The six existing graphs contain 18 measured equations, of which 11 take a hidden state as an argument. In the original two fitting folds this gives 22 hidden-source equation–fold cases. Eight isolated one measured argument at two levels. None isolated three levels. Pooling the 16 training events separately yielded three levels for four of the 11 equations. All four are differential, so their first-minute output changes give average rates. The public observations supply no certified error bound relating those averages to the instantaneous rates. The T2 screen established zero affine exclusions. None of these graphs and input schedules qualified for the isolated-switch or single-hidden-state cascade check.
- **Recovery at input switches.** Four events with different, unobserved initial values gave relative responses \(0,1,4,9\) at inputs \(0,1,2,3\). With \(g(0)\) anchored to zero and polynomial degree at most three, the unique interpolant is \(g(u)=u^2\). Without a degree constraint, adding \(c\,u(u-1)(u-2)(u-3)\) gives the same observations for any \(c\). With bounded measurement error and hidden-state drift, all true values remained within the reported intervals, and these still ruled out an affine response. A separate multiplicative hidden interaction produced a cycle inconsistent with an additive response.
- **Prediction with a hidden pathway.** There were 64 controlled trials spanning two generating laws, two training-input designs, two noise levels and eight seeds. Neither the estimator nor the forecasts received the simulated hidden trajectory. With only inputs \(0,1\), the families \(c_0+c_1u\) and \(c_0+c_1u^2\) coincide there. All 32 trials remained undecided on new events repeating those levels. New development events at intermediate inputs selected the generating family in all 32. With four training levels, new events repeating those levels selected it in all 32. For the 16 four-level quadratic trials, the bounded moment test also excluded affine forcing in every case. These are conditional recovery rates under the simulated cascade and a two-candidate forecast comparison.
- **An ambiguity that forecasting cannot resolve.** For \(\dot y=z^2,\ \dot z=-bz,\ z>0\), put \(w=z^2\). Then \(\dot y=w,\ \dot w=-2bw\) produces exactly the same \(y(t)\) for every corresponding hidden initial value. Measuring \(y\) more often cannot separate those hidden equations. A measurement of the hidden state or a public assumption fixing its meaning can.

All development and evaluation events in the pathway controls are synthetic. A fixed prefix of one time unit estimates the initial output slope for each synthetic forecast. Scoring starts afterward. The T2 audit only reads official training packets and carries out no fitting or model selection on the pooled events.

### A T2 example where curvature and a missing argument are different explanations

Graph04 proposes \(\dot I=F(EGP,X_{\rm meal},u_{\rm insulin})\). The graph gives all events the same hidden \(X_{\rm meal}(0)\). In three events with zero administered insulin, the initial EGP levels were equally spaced at 1.87954, 1.91579 and 1.95203. The observed first-minute insulin slopes were respectively \(+0.00324,-0.30591,+0.71478\) pmol/L/min. For an affine dependence on EGP with all other arguments fixed, the middle **instantaneous** rate must equal the average of the two endpoints. Applying that calculation to the measured slopes gives \(+0.35901\), a departure of \(0.66492\). If every slope differs from its instantaneous rate by at most \(\eta\), the departure attributable to this uncertainty is at most \(2\eta\). A justified \(\eta<0.33246\) pmol/L/min would exclude the affine form on this slice.

There is a second check on the same graph. At the **same** middle EGP level and the same other initial arguments, events train_013 and train_014 have first-minute slopes \(+0.30591\) and \(-0.30591\). If both rate errors were strictly below half that separation (about \(0.30591\) pmol/L/min), any single-valued \(F\) of those arguments would fail. Adding a quadratic term would then leave the contradiction in place. The available data establish neither error bound. Both calculations use selected events to show the sensitivity of the inference.

## Reproduction

From the repository root, use the project Python environment. The T2 commands require the sealed packets from the matched equation-constructor experiment already present in its default output directory.

~~~sh
.venv/bin/python -m scripts.check_hidden_jump_controls
.venv/bin/python -m scripts.check_hidden_pathway_controls
.venv/bin/python -m scripts.audit_hidden_shape_t2
.venv/bin/python -m scripts.audit_pooled_hidden_shape_t2
.venv/bin/python -m scripts.report_partial_shape_experiment
~~~

Each numerical command verifies its existing checkpoints on resume. The report is written to `output/partial-shape-20261126/REPORT.html`. The three checks are available as research functions. They use graph and measurement contracts supplied by the caller. On the evidence from these six graphs, the next construction step for T2 is to carry competing forms into matched whole-event forecasts. This experiment provides no forecast gain estimate on T2.
