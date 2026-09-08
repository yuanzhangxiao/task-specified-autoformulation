# Staged outer-weight sign contract

New equation-topology replies must classify each complete contribution as
`positive`, `negative`, or `unrestricted`. There is no missing-value default.
The choice describes the sign of an identifiable outer scalar weight, not the
range or monotonicity of the complete interaction function.

- `positive`: the runtime adds the contribution and derives a nonnegative
  domain for one identifiable direct signed gain.
- `negative`: the runtime subtracts the contribution once and derives the same
  nonnegative magnitude domain.
- `unrestricted`: the runtime adds a signed function; direct coefficients and
  offsets remain real-valued.

Rates, time constants, scales, and positive-shape parameters remain positive
because of their semantic role. Signed thresholds and parameters inside sums,
differences, nonlinear calls, and other internal laws are not changed. A clear
whole-expression negative factor under a fixed sign triggers localized repair;
the runtime does not strip arbitrary negative syntax.

Frozen `add`/`subtract` equation terms retain their original schema,
serialization, lowering, and domain checks. Thus replay does not silently
change the admissible parameter set.

The ACES probe uses the current variable/equation topology and atomic function
reply path on one reviewed toy topology. It compares an unrestricted signed
baseline reply with an offline fixed-positive replay that changes only the
topology sign. It does not fit parameters, call a scientific judge, open test
data, or use a private reference. The signs are prespecified, so this probe does
not evaluate the LLM's topology-stage sign selection. Integration with the later
hybrid batched function and prefit-feedback branch is a follow-on after this
isolated contract gate passes.
