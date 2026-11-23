"""A shared affine flow can have only one direction on a stipulated side."""

from fractions import Fraction
from math import exp

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.research.affine_direction_certificate import certify
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from scripts.trajectory_boundary_trial import boundary
from tests.test_trajectory_function_families import event, packet


def evidence(
    *,
    reference=(10, 10, 10, 10),
    x=(2, 2, 2, 2),
    w=(0, 0, 0, 0),
    y=(10, 40, 26, 10),
    tolerance=0.0,
    interp="linear_complete_knots",
    reset_free=True,
    cov=False,
):
    vals = []
    ctx = ValidationContext(
        targets=("y",), external_inputs=("x", "w"), fixed_covariates=("c",)
    )
    for i, (xx, ww, yy) in enumerate([((0,) * 4, (0,) * 4, reference), (x, w, y)]):
        vals.append(
            event(
                f"train_{i}",
                np.array([0, 4, 8, 12.0]),
                {"y": np.array(yy)},
                {"x": np.array(xx), "w": np.array(ww)},
                {"c": 2.0 if cov and i else 1.0},
            )
        )
    payload = packet(
        vals,
        ctx,
        dependencies_exhaustive=True,
        shared_parameters=True,
        open_loop=True,
        reset_impulse_free=reset_free,
        uniqueness_assumed=True,
        interpolation={"x": interp, "w": "linear_complete_knots"},
        absolute_error_bounds={"x": 0, "w": 0, "y": tolerance, "c": 0},
        initial_rules={"y": {"mode": "observed_initial"}},
    )
    return TrainingCompatibilityEvidence.model_validate(payload["evidence"])


def family(expression="a+b*x", sources=("x",), *, gate=None, extra=False):
    f = {
        "equations": {"y": expression},
        "parameters": {"a": 3.0, "b": -2.0},
        "initializers": {},
    }
    g = {
        "variable_types": {"y": "differential"},
        "sources": {"y": list(sources)},
        "initializers": {"y": {"mode": "observed_initial", "channel": "y"}},
    }
    if extra:
        g["variable_types"]["z"] = "algebraic"
        g["sources"]["z"] = ["x"]
        f["equations"]["z"] = "x"
    if gate is not None:
        f = boundary(f, "y", gate)
        g["sources"]["y"].append("y")
    return g, f


@pytest.mark.parametrize("gate", [None, 5])
@pytest.mark.parametrize("side", [1, -1])
def test_parameter_independent_conflict_and_numeric_inequalities(gate, side):
    e = evidence(x=(side * 2,) * 4, tolerance=1.0)
    g, f = family(gate=gate)
    r = certify(g, f, "y", e, tolerance=1.0)
    assert r["status"] == "conflict"
    assert r["reference"]["derivative_bound"] == [-0.25, 0.25]
    assert r["reference"]["forcing_values"] == {"x": 0}
    assert r["side"] == ("at_or_above" if side == 1 else "at_or_below")
    assert [c["relation"] for c in r["checks"]] == (
        ([">=", ">="] if gate else []) + [">", ">"]
    )
    assert (
        r["rise"]["required_change"] == 26
        and r["rise"]["maximum_permitted_change"] == 1
    )
    assert (
        r["fall"]["required_change"] == 26
        and r["fall"]["maximum_permitted_change"] == 2
    )


@pytest.mark.parametrize(
    "expression,sources,kwargs",
    [
        ("a+b*x + z", ("x", "z"), {"extra": True}),
        ("a+b*x - y", ("x", "y"), {}),
        ("a+b*t", ("t",), {}),
        ("a+b*tanh(x)", ("x",), {}),
        ("a+b*x**2", ("x",), {}),
        ("a+b*x/(x+1)", ("x",), {}),
    ],
)
def test_unsupported_state_time_or_nonlinearity_never_certified(
    expression, sources, kwargs
):
    g, f = family(expression, sources, **kwargs)
    assert certify(g, f, "y", evidence())["status"] == "unverified"


@pytest.mark.parametrize(
    "kwargs,tol",
    [
        (
            {"x": (1, -1, 1, -1)},
            1.0,
        ),  # forcing crosses reference despite outer endpoints
        ({"y": (10, 10.3, 9.8, 10)}, 1.0),
        ({"interp": "unknown"}, 0.0),
        ({"reset_free": False}, 1.0),
    ],
)
def test_uncertain_contract_small_changes_and_intersample_crossings(kwargs, tol):
    assert (
        certify(*family(), "y", evidence(**kwargs), tolerance=tol)["status"]
        == "unverified"
    )


def test_constant_other_forcing_is_used_only_at_the_reference_value():
    g, f = family("a+b*x+tanh(w)", ("x", "w"), gate=5)
    assert certify(g, f, "y", evidence())["status"] == "conflict"
    assert certify(g, f, "y", evidence(w=(0, 1, 0, 1)))["status"] == "unverified"
    g, f = family("a+b*x*c", ("x", "c"))
    assert (
        certify(g, f, "y", evidence(cov=True, x=(1, 2, 3, 2)))["status"] == "unverified"
    )


def test_scope_confirmation_and_error_checks():
    g, f = family()
    g["initializers"]["y"] = {"mode": "value", "scope": "shared_training"}
    assert certify(g, f, "y", evidence())["status"] == "unverified"
    with pytest.raises(ValueError, match="finite"):
        certify(g, f, "y", evidence(), tolerance=float("inf"))
    e = evidence().model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="digest"):
        certify(g, f, "y", e)


@pytest.mark.parametrize("anchor", [0, -8])
def test_signed_ungated_target_needs_no_positive_anchor(anchor):
    e = evidence(reference=(anchor,) * 4, x=(1, 2, 1, 2), y=(-8, 22, 8, -8))
    result = certify(*family(), "y", e, tolerance=1)
    assert result["status"] == "conflict"
    ref = result["reference"]
    assert ref["gate_threshold"] is None
    assert ref["minimum_permitted_endpoint"] == anchor - 1
    assert ref["exact_bounds"]["raw_derivative"] == ["-1/12", "1/12"]


def test_unsaturated_gate_uses_both_endpoint_uncertainties_for_raw_rate():
    e = evidence(reference=(2,) * 4, x=(1, 2, 1, 2), tolerance=0.25)
    r = certify(*family(gate=5), "y", e, tolerance=0.5)
    assert r["status"] == "conflict"
    ref = r["reference"]
    assert ref["gate_threshold"] == 5
    assert ref["exact_bounds"] == {
        "endpoint_lower": ["7/4", "5/4"],
        "minimum_gate_multiplier": "1/4",
        "mean_derivative": ["-1/12", "1/12"],
        "raw_derivative": ["-1/3", "1/12"],
    }
    lo, hi = map(Fraction, ref["derivative_bound"])
    assert lo <= Fraction(-1, 3) and hi >= Fraction(1, 12)
    assert all(c["value"] >= c["bound"] for c in r["checks"][:2])


def test_unsigned_rate_passes_through_taper_and_zero_lower_gate_abstains():
    e = evidence(reference=(2, 3, 4, 5), x=(1, 2, 1, 2), tolerance=0)
    r = certify(*family(gate=10), "y", e, tolerance=0)
    assert r["status"] == "conflict"
    assert r["reference"]["exact_bounds"]["raw_derivative"] == ["1/4", "1/4"]
    unbounded = evidence(reference=(0.5,) * 4, x=(1, 2, 1, 2), tolerance=0.25)
    assert (
        certify(*family(gate=10), "y", unbounded, tolerance=0.25)["status"]
        == "unverified"
    )


def test_exact_feasible_signed_rates_under_partial_gate_do_not_conflict():
    # F = x - 2. With x=0, a positive depth below 40 follows y'= -2*y/40.
    # In the comparison event x rises linearly from 0 to 4 between t=4 and 8.
    # F crosses zero at t=6, after which y'=F. Endpoints straddle a fall and rise.
    times = [0, 4, 8, 12]
    reference = [8 * exp(-2 * t / 40) for t in times]
    y4 = 28 * exp(-8 / 40)
    y8 = y4 * exp(-2 / 40) + 2
    e = evidence(reference=reference, x=(0, 0, 4, 4), y=(28, y4, y8, y8 + 8))
    assert certify(*family(gate=40), "y", e, tolerance=1e-7)["status"] == "unverified"


@pytest.mark.parametrize("scale", [1e-5, 1e-2, 1.0, 20.0, 50.0])
@pytest.mark.parametrize("raw", [-2.0, -0.4, 0.0, 0.8])
def test_exact_constant_forcing_is_inside_reference_bound(scale, raw):
    from autoformalism.research.affine_direction_certificate import _reference

    y0 = scale / 2
    ts = [0, 4, 8, 12]
    ys = [y0 * exp(raw * t / scale) if raw < 0 else y0 + raw * t for t in ts]
    e = evidence(reference=ys)
    witness = _reference(e.trajectories[0], "y", ["x"], Fraction(1, 10**9), 0, scale)
    if min(ys) <= 1e-9:
        assert witness is None
    else:
        assert witness is not None
        _, lo, hi = witness
        assert lo <= Fraction(raw) <= hi
