# Independent stormwater basins — negative control v1

Predict downstream basin depth `h_down` (metres) from measured local runoff
`inflow_down` (cubic metres per minute). A second, hydraulically disconnected
basin receives `inflow_up`; its runoff and initial depth are also supplied.
There is no pipe, spillway, overflow route, or common unmeasured inflow linking
these two basins. Their releases go to separate receiving channels.

Represent downstream accumulation, a threshold-dependent outlet, nonnegative
storage and its local water balance. Do not introduce a physical transfer from
the disconnected upstream basin. It need not be modeled to predict the target.
No shared process is required merely because the outlet laws have similar forms.

Time is in minutes; forcing uses linear interpolation between samples. Fixed
covariates `area_up`, `area_down` (square metres), `crest_up`, `crest_down`,
`initial_up`, and `warning_depth` (metres) are surveyed geometry and the initial
upstream gauge reading. Unknown outlet coefficients are shared across events.
Both outlets operate freely, without tailwater submergence; unmeasured gains
and losses are negligible over each event. The target's initial reading is
available; later target observations cannot drive or reset open-loop rollouts.

Fit using training only. Validation-specific initial-state fitting is forbidden.
No hidden trajectory or exact generating law is proposer input. Equivalent
equation representations receive equal credit. This is a development negative
control, not an additional independent benchmark family.
