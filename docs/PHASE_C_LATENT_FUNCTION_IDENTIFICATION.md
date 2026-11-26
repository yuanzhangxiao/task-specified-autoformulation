# Recovering functions that involve hidden states

An unmeasured state can sometimes be reconstructed in its scientific units. An observed total, a calibrated readout or an independently specified balance can supply that information. The constructor should first ask which unknown quantities the measurements determine, then whether the training events distinguish functions of those quantities. These are separate questions.

## Which equation does the data constrain?

Suppose the public specification gives an observed quantity \(r=A(x,u)+cz\), with \(A\) and the nonzero gain \(c\) known. Then \(z=(r-A)/c\) is available during training. The roles of \(x,u\) here are measured sources. In forecasting, the candidate must generate its own states from permitted inputs and initial values.

| Equation to construct | Data comparison | Conditions to check |
|---|---|---|
| Differential latent \(\dot z=f(z,x,u)\) | Compare \(z(t_1)-z(t_0)\) with the integral of the proposed \(f\) along each reconstructed training trajectory. | Calibrated recovery of \(z\), sufficient independent variation of the arguments, and a justified interpolation or integration error bound. |
| Algebraic latent \(z=h(x,u)\) | Compare the reconstructed values of \(z\) directly with \(h(x,u)\). | The readout is invertible in \(z\) on the feasible range. Training samples must discriminate members of the proposed class. |
| Latent-dependent contribution \(\dot y=B(x,u)+H(z,x,u)\) | Compare measured increments of \(y\) minus the integral of the known contribution \(B\) with the integral of \(H\). | An independent reconstruction of \(z\) and a way to separate \(H\) from other unknown terms. Two free functions of the same arguments can absorb one another. |
| Measured target with measured arguments \(\dot y=F(x,u)\) | Compare local changes in the estimated rate when varying one argument, holding the rest fixed. A four-corner contrast \(F(x+h,u+k)-F(x+h,u)-F(x,u+k)+F(x,u)\) isolates a mixed contribution. | Independent support around a point, estimates of the **rate** with uncertainty, and comparable values of the other arguments. \(d^2y/dt^2\) along one trajectory blends all changing arguments. |

For example, if the law belongs to the space of polynomials of total degree at most two, a full-rank design in the six columns \(1,x,u,x^2,xu,u^2\) uniquely determines those coefficients with exact consistent observations. Without the family restriction, other smooth laws agree at finitely many sampled points. Sample error and poor conditioning can leave wide coefficient intervals even at full rank.

A known differential readout can also fix a latent coordinate. For example, \(\dot r=A(x,u)+cz\) gives \(z=(\dot r-A)/c\) in the ideal continuous record. Estimating \(\dot r\) from observations introduces a second source of uncertainty. A state estimated by first fitting the candidate's own unknown \(A\) or \(c\) cannot then independently verify that same assumption. Observability and structural identifiability tests can assess specified candidate models under their assumptions on outputs, parameters, and inputs. [Villaverde, Tsiantis and Banga, 2019](https://doi.org/10.1098/rsif.2019.0043)

### Can the hidden dynamics and the downstream function *both* be unknown?

Yes, under a stronger assumption about how the input enters. Consider the scalar model \(\dot z=g(z)+u,\ \dot y=h(z)\). Both \(g\) and \(h\) are unknown. Assume a calibrated unit gain on the measured input, \(h\) strictly monotone with nonzero derivative on the accessible interval, and one known reference pair \(z_*,v_*\) where \(v=h(z)\). In the ideal differentiable-output record, write \(v=\dot y\). For an algebraic readout \(y=h(z)\) use \(v=y\) instead.

1. At a given value of \(v\) the hidden \(z\) must be the same across events. The chain rule gives \(\dot v=A(v)+B(v)u\), where \(A(v)=h'(z)g(z)\) and \(B(v)=h'(z)\). Two distinct values of \(u\) at the same \(v\) identify the slope \(B(v)\) and intercept \(A(v)\). A third level tests whether the assumed affine dependence on \(u\) holds.
2. If the experiment supplies these comparisons throughout an interval, reconstruct the hidden coordinate by integrating \(z(v)=z_*+\int_{v_*}^v 1/B(s)\,ds\). Now \(h(z(v))=v\) and \(g(z(v))=A(v)/B(v)\). For example, if the measured relation on \(v>0\) were \(\dot v=-2v+2\sqrt v\,u\) and \(z(1)=1\), these steps give \(z=\sqrt v,\ h(z)=z^2,\ g(z)=-z\). This determines the hidden RHS and the downstream term in the specified coordinate.
3. Without the reference, a translation of \(z\) survives. An unknown forcing gain also permits a change of scale. Finite levels of \(v\) constrain \(A,B\) only there, so the integrals require error bounds, interpolation assumptions or competing families. For a differential target, obtaining \(\dot v=\ddot y\) is especially sensitive to noise. A first synthetic test of this proposal should use the algebraic readout \(v=y\), step or crossed inputs that revisit the same values of \(v\), and an otherwise identical trial where input is determined by \(v\). Only the first design supplies independent contrasts. This example is a derivation for the next experiment.

### An ambiguity that survives infinitely many forecasts

Take \(\dot z=f(z,u)\) and \(\dot y=H(z,u)\). If the model permits an invertible differentiable change of hidden coordinate \(w=\phi(z)\), then \(\dot w=\phi'(\phi^{-1}(w))f(\phi^{-1}(w),u)\) and \(\dot y=H(\phi^{-1}(w),u)\) reproduce the same observations, using \(w(0)=\phi(z(0))\). The analogous algebraic factorization is \(w=\phi(h(x,u))\), followed by the inverse transformation in each downstream use. The transformed domain, initializers, public units or calibrations, and permitted function families all have to be admissible. A hidden-state name by itself does not supply this calibration. The coordinate ambiguity has a classical counterpart in uniqueness up to isomorphism for a class of minimal analytic realizations. [Sussmann, 1976](https://link.springer.com/article/10.1007/BF01683278)

## Exact synthetic control

The experiment gives the constructor the observed readouts \(r=x+z\) and \(s=u+w\), and the known contribution \(1\) in \(\dot y=1+H(z,x)\). The unknown functions are \(f(x,u)=\dot z,\ h(x,u)=w,\ H(z,x)\). Each belongs to the same six-dimensional polynomial class above. The constructor receives input values and the two endpoint observations of \(r,y\) over each unit-time event, plus \(s\). It receives neither hidden-state labels nor derivative measurements.

Held inputs make \(f(x,u)\) constant during each event, so \(z\) is linear between its reconstructed endpoints. This permits exact integration of each quadratic basis term for \(H\). For example the mean of \(z^2\) is \((z_0^2+z_0z_1+z_1^2)/3\). Exact rational elimination checks rank and consistency before returning any coefficient vector.

| Control | Result |
|---|---|
| Eighteen events crossing three input levels for each argument and two hidden starts | All three matrices had rank six. The unique laws in the declared class were \(f=1+xu,\ h=2+x^2-u,\ H=z^2+2zx\). |
| Six events with \(x=u\), including both hidden starts | The \(f\) and \(h\) matrices had rank three. For example \(xu\) and \(x^2\) agree everywhere on this design. The independent variation of \(z\) kept the \(H\) matrix at rank six. |
| Unknown gain in \(r=x+cz\), with all eighteen events | All measured samples remained exactly the same after doubling \(z,\dot z\) and its initial value, halving \(c\), and replacing \(H(w,x)\) by \(H(w/2,x)\) in the new coordinate \(w=2z\). The separate algebraic readout still fixed \(h\). |

Run the resumable CPU control from the repository root with the command below. Its summary is written to `output/anchored-latent-functions-20261126/summary.json`.

~~~sh
.venv/bin/python -m scripts.check_anchored_latent_functions
~~~

## Constructing candidates

1. Derive the measurable state, flux, or combination from public identities. Record whether the identity is part of the task or a new, testable model assumption. For T2, the available total \(U\) and insulin-independent contribution \(U_{ii}\) give the effective remainder \(D=U-U_{ii}\); interpreting \(D\) as a single insulin-mediated flux assumes the two-contribution balance. This subtraction does not recover a named hidden insulin-action state or factor \(D\) into separate responses. A forecast must generate \(U\) and cannot read its future observations.
2. Where the readout fixes the coordinate, recover the form within a declared class if the training design supports it. Use intervals or candidate-specific weak-form checks for noisy or sparsely sampled records. Where only a combination is fixed, generate candidates that satisfy that identity by construction, for example \(U=U_{ii}+H(z,G_t)\) under the two-contribution assumption. This preserves the measured meaning of the flux while leaving its dynamics and arguments open to examination.
3. Where the evidence only excludes a family, use that fact to constrain a small expression grammar or screen a small set of scientifically proposed functions. A mixed four-corner contrast incompatible with zero, under controlled arguments and error bounds, excludes a purely additive \(A(x)+B(u)\) on that rectangle. It admits many interactions, including products and joint saturation. Group hypotheses known to be observationally equivalent before spending fitting budget on them. Forecast comparisons can rank the remaining families when their predictions differ.

The synthetic result uses exact balances and a finite degree bound; it is a positive control for what information *would* suffice. Applying the same reasoning to benchmark models first requires an audit of which latent coordinates or effective fluxes their public measurements actually fix.
