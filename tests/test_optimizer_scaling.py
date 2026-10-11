"""Scaling invariances, weak directions, profiled derivatives and legacy behavior."""

from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting.optimizer_scaling import RelativeScaling, relative_scales
from tests.test_recovery_numerics import QuadraticOracle


def test_relative_rule_is_bounded_and_invariant_to_loss_magnitude_and_row_repeats():
    j = np.diag([1e-10, 1, 1e10, 0])
    policy = RelativeScaling()
    baseline = relative_scales(j, policy)
    assert min(baseline["scales"]) == policy.minimum
    assert max(baseline["scales"]) == policy.maximum
    for transformed in (j * 1e-200, j * 1e200, np.tile(j, (4, 1))):
        actual = relative_scales(transformed, policy)
        assert actual["scales"] == pytest.approx(baseline["scales"])
        assert actual["weak_columns"] == baseline["weak_columns"]
    assert relative_scales(j, RelativeScaling(exponent=0))["scales"] == [1] * 4
    zero = relative_scales(np.zeros((3, 2)), policy)
    assert zero["status"] == "zero_information_identity"
    assert zero["scales"] == [1, 1] and zero["weak_columns"] == [0, 1]


@pytest.mark.parametrize("j", [np.array([]), np.ones((0, 2)), [[np.nan]], [[np.inf]]])
def test_invalid_jacobian_rejected(j):
    with pytest.raises(ValueError, match="finite nonempty"):
        relative_scales(j, RelativeScaling())


def run(oracle, method, **kwargs):
    return numerical.search(
        oracle,
        oracle.parameters(oracle.start),
        monotonic() + 10,
        lambda _: None,
        calls=100,
        target=1e-20,
        telemetry=True,
        record_steps=True,
        normalized_objective=True,
        optimizer_scaling=method,
        **kwargs,
    )


class UnequalOracle(QuadraticOracle):
    """Same physical problem in configurable units; initial Jacobian is imbalanced."""

    def __init__(self, conversion=1):
        self.conversion = conversion
        self.start = np.array([1.0, 0.0]) * conversion
        self.lower = super().lower * conversion
        self.upper = super().upper * conversion
        self.units = super().units * conversion
        self.visited = []

    def evaluate(self, vector, deadline):
        self.visited.append(vector / self.conversion)
        v = vector / self.conversion
        return np.array([20 * (v[0] - 2), v[1] - 3]), np.diag([20, 1]) / self.conversion


@pytest.mark.parametrize("method", ["fixed_coordinates", "guarded_relative"])
def test_physical_search_paths_invariant_to_units_and_objective_multiplier(method):
    baseline = UnequalOracle()
    answer = run(baseline, method)
    assert answer["stop_reason"] == "numerical_target_reached"
    for conversion, multiplier in [(1000, 1), (1, 1e-3), (1, 1e6)]:
        oracle = UnequalOracle(conversion)
        other = run(oracle, method, objective_multiplier=multiplier)
        assert other["stop_reason"] == "numerical_target_reached"
        assert len(oracle.visited) == len(baseline.visited)
        np.testing.assert_allclose(
            oracle.visited, baseline.visited, rtol=1e-7, atol=1e-8
        )
        assert other["solver_scaling"]["absolute_gradient_stop_enabled"] is False


def test_guard_uses_outer_profile_jacobian_and_charges_initial_evaluation():
    oracle = UnequalOracle()

    class Profile:
        outer = (0,)

        def evaluate(self, vector, end):
            fitted = vector.copy()
            fitted[1] = 3
            return fitted, fitted - [2, 3], np.array([[1], [0]]), {"active_mask": [0]}

    result = run(oracle, "guarded_relative", profile=Profile())
    assert result["best"]["parameters"] == pytest.approx({"a": 2, "initial": 3})
    assert result["solver_scaling"]["scales"] == [1]
    assert result["calls"] == result["completed_calls"] == len(result["evaluations"])
    assert result["evaluations"][0]["attempt"] == 1


def test_initial_target_and_failed_guard_preserve_checkpoint(monkeypatch):
    oracle = UnequalOracle()
    oracle.start = np.array([2, 3])
    result = run(oracle, "guarded_relative")
    assert result["calls"] == 1 and result["stop_reason"] == "numerical_target_reached"
    oracle.start = np.array([1, 0])
    monkeypatch.setattr(
        numerical,
        "relative_scales",
        lambda *a: (_ for _ in ()).throw(ValueError("invalid derivative geometry")),
    )
    failed = run(oracle, "guarded_relative")
    assert failed["stop_reason"] == "numerical_failure" and failed["best"] is not None
