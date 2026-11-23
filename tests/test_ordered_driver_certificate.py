"""A scalar affine model with ordered forcing must preserve its gain's sign."""

from fractions import Fraction

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.research.ordered_driver_certificate import certify
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from tests.test_trajectory_function_families import event, packet


def observations(rows=None, **overrides):
    # ID, time, target, x, other supplied input w
    rows = rows or [
        ("train_a", [0, 1, 2], [0, 0, 0], [0, 0, 0], [2, 3, 4]),
        ("train_b", [0, 1, 2], [0, 4, -8], [1, 1, 1], [2, 3, 4]),
    ]
    ctx = ValidationContext(targets=("y",), external_inputs=("x", "w"))
    data = [
        event(
            name,
            t,
            {"y": np.array(y, dtype=float)},
            {"x": np.array(x, dtype=float), "w": np.array(w, dtype=float)},
        )
        for name, t, y, x, w in rows
    ]
    contract = {
        "initial_rules": {"y": {"mode": "observed_initial"}},
        "dependencies_exhaustive": True,
        "shared_parameters": True,
        "open_loop": True,
        "reset_impulse_free": True,
        "interpolation": {"x": "linear_complete_knots", "w": "linear_complete_knots"},
        "absolute_error_bounds": {"x": 0, "w": 0, "y": 0},
    }
    return TrainingCompatibilityEvidence.model_validate(
        packet(data, ctx, **(contract | overrides))["evidence"]
    )


def model(text="a*y+b*x+tanh(w)"):
    used = RestrictedParser().parse(text, location="y").symbols
    parameters = {n: 0.0 for n in used if n in {"a", "b"}}
    g = {
        "variable_types": {"y": "differential"},
        "sources": {"y": sorted(used - parameters.keys())},
        "initializers": {"y": {"mode": "observed_initial", "channel": "y"}},
    }
    return g, {"equations": {"y": text}, "initializers": {}, "parameters": parameters}


def test_same_pair_can_reverse_with_state_feedback_and_nonlinear_other_forcing():
    e = observations()
    g, f = model("a*y+b*x/(1+a**2)+tanh(w)")
    result = certify(g, f, "y", e, tolerance=1)
    assert result["status"] == "conflict" and result["driver"] == "x"
    assert result["exact_lower_bound"] == "2"
    assert result["worst_sampled_absolute_error_lower_bound"] == 2
    a, b = (
        result["witnesses"]["reject_nonnegative"],
        result["witnesses"]["reject_nonpositive"],
    )
    assert (a["high_event"], a["low_event"], a["index"], a["time"]) == (
        "train_b",
        "train_a",
        2,
        2,
    )
    assert (
        a["observed_high"] == -8 and a["observed_low"] == 0 and b["observed_high"] == 4
    )
    assert a["exact_observed_gap"] == "8" and b["exact_observed_gap"] == "4"
    assert a["matching_other_channels"] == ["w"]
    assert result["evidence_sha256"] == e.sha256
    # The search does not substitute supplied starting coefficients.
    f["parameters"] = {"a": 3.8, "b": -100.0}
    assert certify(g, f, "y", e, tolerance=1) == result


def test_exact_threshold_and_outward_rounding():
    e = observations()
    for tol, expected in [
        (2, "unverified"),
        (np.nextafter(2.0, 0), "conflict"),
        (3, "unverified"),
    ]:
        assert certify(*model(), "y", e, tolerance=tol)["status"] == expected
    e = observations(
        [
            ("train_a", [0, 1, 2], [0, 2**-56, 0], [0, 0, 0], [0, 0, 0]),
            ("train_b", [0, 1, 2], [0, 1, -2], [1, 1, 1], [0, 0, 0]),
        ]
    )
    r = certify(*model(), "y", e, tolerance=0)
    exact = (1 - Fraction(1, 2**56)) / 2
    assert Fraction(r["exact_lower_bound"]) == exact
    assert Fraction(r["worst_sampled_absolute_error_lower_bound"]) <= exact


@pytest.mark.parametrize("a,b", [(-0.8, 2), (0.5, -2), (0, 0), (-1e-4, -1e-2)])
def test_exact_feasible_ode_cannot_generate_both_sign_witnesses(a, b):
    # Constant x and w. Exact solution with shared initial y(0)=1.
    rows = []
    t = np.array([0.0, 0.4, 1.0, 5.0])
    for i, x in enumerate([0, 1, 5]):
        y = np.exp(a * t) + (b * x + 0.5) * (np.expm1(a * t) / a if a else t)
        rows.append((f"train_{i}", t, y, [x] * 4, [0.5] * 4))
    assert (
        certify(*model("a*y+b*x+w"), "y", observations(rows), tolerance=1e-9)["status"]
        == "unverified"
    )


@pytest.mark.parametrize(
    "text",
    [
        "a*y+b*x*w",
        "a*y*x+b+x",
        "a*y+b*tanh(x)",
        "a*y**2+b*x",
        "a*y+b*x/(1+w)",
        "b*x+abs(y)",
        "a*y+b*x+t",
    ],
)
def test_varying_gains_nonaffine_target_and_explicit_time_abstain(text):
    assert (
        certify(*model(text), "y", observations(), tolerance=0)["status"]
        == "unverified"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"reset_impulse_free": None},
        {"shared_parameters": False},
        {"dependencies_exhaustive": None},
        {"open_loop": False},
        {"initial_rules": {"y": {"mode": "common_map"}}},
        {"absolute_error_bounds": {"y": 0.01, "x": 0, "w": 0}},
        {"absolute_error_bounds": {"y": 0, "x": None, "w": 0}},
        {"interpolation": {"x": "unknown", "w": "linear_complete_knots"}},
    ],
)
def test_missing_and_uncertain_contracts_abstain(change):
    assert (
        certify(*model(), "y", observations(**change), tolerance=0)["status"]
        == "unverified"
    )


def test_unobserved_initializer_mismatched_grids_and_initial_values():
    g, f = model()
    g["initializers"]["y"] = {"mode": "shared_parameter"}
    assert certify(g, f, "y", observations())["status"] == "unverified"
    rows = [
        ("train_a", [0, 1, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]),
        ("train_b", [0, 1, 3], [0, 4, -8], [1, 1, 1], [0, 0, 0]),
        ("train_c", [0, 1, 2], [0.01, 4, -8], [1, 1, 1], [0, 0, 0]),
    ]
    r = certify(*model(), "y", observations(rows), tolerance=0)
    assert r["status"] == "unverified"
    assert r["pair_counts"] == {
        "matched_pairs": 0,
        "different_grids": 2,
        "unequal_initials": 1,
        "nonzero_time_origin": 0,
    }
    for row in rows[:2]:
        row[1][0] = 0.1
    rows[1][1][-1] = 2
    other = certify(*model(), "y", observations(rows[:2]), tolerance=0)
    assert other["pair_counts"]["nonzero_time_origin"] == 1


@pytest.mark.parametrize(
    "x,w",
    [
        ([2, -1, 2, 2], [0, 0, 0, 0]),  # forcing crossed before the output reversal
        ([1, 1, 1, 1], [0, 0, 1, 0]),  # other focal input differs mid-prefix
        (
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ),  # fully identical inputs have no ordered excitation
    ],
)
def test_whole_prefix_is_required(x, w):
    rows = [
        ("train_a", [0, 1, 2, 3], [0] * 4, [0] * 4, [0] * 4),
        ("train_b", [0, 1, 2, 3], [0, 3, 0, -9], x, w),
    ]
    assert (
        certify(*model(), "y", observations(rows), tolerance=0)["status"]
        == "unverified"
    )


def test_sign_witnesses_for_distinct_drivers_do_not_combine():
    t = [0, 1, 2]
    rows = [
        ("train_a", t, [0, 0, 0], [0] * 3, [0] * 3),
        ("train_b", t, [0, 2, 4], [1] * 3, [0] * 3),
        ("train_c", t, [0, -2, -4], [0] * 3, [1] * 3),
    ]
    r = certify(*model("a*y+b*x+w"), "y", observations(rows), tolerance=0)
    assert r["status"] == "unverified" and r["drivers_checked"] == ["w", "x"]


def test_previous_hold_ignores_an_input_step_exactly_at_the_observed_discrepancy():
    rows = [
        ("train_a", [0, 1, 2], [0] * 3, [0] * 3, [0] * 3),
        ("train_b", [0, 1, 2], [0, 5, -8], [0, 1, 1], [0] * 3),
    ]
    result = certify(
        *model(),
        "y",
        observations(
            rows,
            interpolation={
                "x": "previous_complete_knots",
                "w": "linear_complete_knots",
            },
        ),
        tolerance=0,
    )
    assert result["status"] == "unverified"
    assert (
        certify(*model(), "y", observations(rows), tolerance=0)["status"] == "conflict"
    )


def test_corrupt_evidence_nontraining_data_and_negative_tolerance():
    g, f = model()
    with pytest.raises(ValueError, match="digest"):
        certify(g, f, "y", observations().model_copy(update={"sha256": "0" * 64}))
    with pytest.raises(ValueError, match="nonnegative"):
        certify(g, f, "y", observations(), tolerance=-1)
    e = observations([("val_a", [0, 1], [0, 0], [0, 0], [0, 0])])
    with pytest.raises(ValueError, match="training"):
        certify(g, f, "y", e)
