"""Local derivative probes must not inherit a distant incumbent's huge rate."""

from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting.prefix_multioutput_profile import make_oracle
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from tests.test_prefix_multioutput_profile import profile_case


@pytest.fixture
def oracle():
    _, oracle, _ = make_oracle(*profile_case())
    return oracle


def test_local_signed_rate_check_recovers_without_restricting_its_domain(oracle):
    index = oracle.names.index("coefficient:y:y")
    point = oracle.vector(oracle.start)
    point[index] = -1.0
    oracle.units[index] = 2363901.1952044982
    original_units = oracle.units.copy()
    original_bounds = oracle.lower.copy(), oracle.upper.copy()
    calls = []

    def evaluate(value, deadline, *, jacobian=True):
        calls.append(float(value[0]))
        # A positive rate at the historical probe sizes overflows on this span.
        with np.errstate(over="ignore"):
            prediction = np.exp(np.array([1.0, 300.0]) * value[0])
        if not np.isfinite(prediction).all():
            raise NumericalInvalidTrial("explosive decay-control trial")
        derivative = (np.array([1.0, 300.0]) * prediction)[:, None]
        return prediction - np.array([0.5, 0.0]), derivative if jacobian else None

    def scales(value):
        full = point.copy()
        full[index] = value[0]
        return oracle.local_coordinate_scales(full)[[index]]

    view = SimpleNamespace(
        evaluate=evaluate,
        local_coordinate_scales=scales,
        units=original_units[[index]],
        lower=oracle.lower[[index]],
        upper=oracle.upper[[index]],
    )
    policy = adequacy.Policy(
        precision_aware=True,
        local_adaptive=True,
        local_bounds_aware=True,
        local_max_scales=6,
    )
    before = recovery._derivative(view, np.array([-1.0]), monotonic() + 5, policy)
    assert not before.search_qualified
    oracle.configure_local_probes({"coefficient:y:y": 1 / 300})
    calls.clear()
    after = recovery._derivative(view, np.array([-1.0]), monotonic() + 5, policy)
    assert after.search_qualified, after
    assert max(abs(value + 1) for value in calls) <= 1.0001e-4
    np.testing.assert_array_equal(oracle.units, original_units)
    np.testing.assert_array_equal(oracle.lower, original_bounds[0])
    np.testing.assert_array_equal(oracle.upper, original_bounds[1])
    assert oracle.lower[index] == -np.inf and oracle.upper[index] == np.inf


def test_configured_radius_handles_zero_and_both_rate_signs(oracle):
    index = oracle.names.index("coefficient:y:y")
    oracle.configure_local_probes({"coefficient:y:y": 1 / 300})
    for value in (-1.0, 0.0, 1.0):
        point = oracle.vector(oracle.start)
        point[index] = value
        assert oracle.local_coordinate_scales(point)[index] == max(
            abs(value), 1 / 300
        )
    receipt = oracle.recovery_audit["local_probe_policy"]
    assert not receipt["optimizer_units_changed"]
    assert not receipt["parameter_domains_changed"]


def test_reciprocal_coordinate_still_uses_its_small_current_magnitude(oracle):
    assert "tau_g" in oracle.sensitive_names
    point = oracle.vector(oracle.start)
    index = oracle.names.index("tau_g")
    point[index] = 1e-10
    oracle.configure_local_probes({"tau_g": 1.0})
    assert oracle.local_coordinate_scales(point)[index] == 1e-10


def test_configured_delay_does_not_inherit_a_smaller_historical_unit(oracle):
    point = oracle.vector(oracle.start)
    index = oracle.names.index("tau_g")
    point[index] = 300.0
    oracle.units[index] = 1.0
    assert oracle.local_coordinate_scales(point)[index] == 1.0
    oracle.configure_local_probes({"tau_g": 1.0})
    assert oracle.local_coordinate_scales(point)[index] == 300.0
    assert oracle.units[index] == 1.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_floor_cannot_change_the_policy(oracle, value):
    with pytest.raises(ValueError, match="finite and positive"):
        oracle.configure_local_probes({"coefficient:y:y": value})
    assert "local_probe_policy" not in oracle.recovery_audit


def test_unknown_empty_and_changed_policies_are_rejected(oracle):
    for floors in ({}, {"absent": 1.0}):
        with pytest.raises(ValueError, match="known coordinates"):
            oracle.configure_local_probes(floors)
    floors = {"coefficient:y:y": 1 / 300}
    oracle.configure_local_probes(floors)
    oracle.configure_local_probes(floors.copy())
    with pytest.raises(ValueError, match="already frozen"):
        oracle.configure_local_probes({"coefficient:y:y": 1.0})
    floors["coefficient:y:y"] = 999.0
    assert oracle.recovery_audit["local_probe_policy"]["floors"] == {
        "coefficient:y:y": 1 / 300
    }


def test_default_retains_historical_scale_and_receipt(oracle):
    point = oracle.vector(oracle.start)
    index = oracle.names.index("coefficient:y:y")
    oracle.units[index] = 2363901.0
    point[index] = -1.0
    assert oracle.local_coordinate_scales(point)[index] == 2363901.0
    assert "local_probe_policy" not in oracle.recovery_audit
