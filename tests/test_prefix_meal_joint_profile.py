"""Shared forcing profiling keeps one coefficient across all trajectory blocks."""

from copy import deepcopy
from functools import partial
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_meal_joint_profile as joint
from autoformalism.fitting import prefix_portfolio_polish as polish
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit
from tests import test_prefix_meal_coordinates as meals

MEAL_OPTIONS = {
    "state": "A",
    "rate": "k",
    "input_gain": "q",
    "input_name": "r",
    "output_state": "x",
}
factory = partial(joint.JointMealProfile, drift="s", meal_options=MEAL_OPTIONS)


def fixture(*, output="r-x+k*A+s", observation="2*x", drift_bounds=None, known=False):
    source, _ = meals.fixture()
    payload = source.problem.model_dump(mode="json")
    payload["base_candidate"]["state_equations"][0]["rhs"] = output
    payload["base_candidate"]["observation_mappings"][0]["expression"] = observation
    payload["base_candidate"]["parameters"].append(
        {
            "name": "s",
            "scope": "global",
            "role": "nonnegative_coefficient",
            **({"bounds": drift_bounds} if drift_bounds else {}),
        }
    )
    payload["parameter_guesses"]["s"] = 0.2
    problem = PrefixFitProblem.model_validate(payload)
    data = source.training
    start = kernel.TrainingOracle(problem, data, kernel.NumericalPolicy()).start
    original, oracle = polish.make_oracle(
        problem, data, kernel.NumericalPolicy(), start
    )
    truth = oracle.start | {"q": 1.3, "s": 0.4}
    truth.update(
        {
            name: 1.1 + i
            for i, name in enumerate(oracle.names)
            if name.startswith("trajectory:")
        }
    )
    if known:
        residual, _ = oracle.evaluate(
            oracle.vector(truth), monotonic() + 10, jacobian=False
        )
        values, offset = data.model_dump(mode="json"), 0
        for row in values["rows"]:
            count = len(row["time"])
            row["targets"]["y"] = (
                np.asarray(row["targets"]["y"])
                + original.scales["y"] * residual[offset : offset + count]
            ).tolist()
            offset += count
        data = PublicSplit.model_validate(values)
        original, oracle = polish.make_oracle(
            problem, data, kernel.NumericalPolicy(), start
        )
    return original, oracle, truth


def test_global_design_and_solve_recover_known_shared_coefficients_and_roots():
    original, oracle, truth = fixture(known=True)
    profiler = factory(oracle)
    initial = deepcopy(oracle.start)
    block = profiler.design(truth, monotonic() + 10)
    point = profiler.meal.encode(truth)[list(profiler.inner)]
    actual, _ = oracle.evaluate(oracle.vector(truth), monotonic() + 10, jacobian=False)
    assert block["design"] @ point + block["offset"] == pytest.approx(actual, abs=1e-10)
    assert len(profiler.inner) == 4 and len(profiler.outer) == 1
    assert profiler.inner_names[:2] == ("meal_flux_gain:A", "s")
    endpoint = profiler.profile(initial, monotonic() + 10)
    assert endpoint["parameters"] == pytest.approx(truth, abs=1e-9)
    assert endpoint["training_nmse"] < 1e-20
    assert endpoint["joint_coefficient_optimality_checked"]
    assert not endpoint["outer_convergence_certified"]
    assert endpoint["joint_solve"]["design_columns"] == 4
    assert endpoint["joint_solve"]["design_rank"] == 4
    assert len(endpoint["joint_solve"]["coordinate_names"]) == 4
    assert endpoint["joint_solve"]["normalized_kkt_residual"] < 1e-10
    independent, _ = original.evaluate(
        original.vector(endpoint["original_parameters"]),
        monotonic() + 10,
        jacobian=False,
        method="Radau",
    )
    assert np.mean(independent**2) < 1e-16
    assert oracle.start == initial


def test_global_shared_columns_are_not_independent_per_trajectory():
    _, oracle, _ = fixture()
    profiler = factory(oracle)
    block = profiler.design(oracle.start, monotonic() + 10)
    n = len(oracle.original.rows[0].time)
    a = block["design"]
    assert np.linalg.norm(a[:n, :2]) > 0 and np.linalg.norm(a[n:, :2]) > 0
    assert np.all(a[:n, 3] == 0) and np.all(a[n:, 2] == 0)
    endpoint = profiler.profile(oracle.start, monotonic() + 10)
    assert endpoint["joint_solve"]["scope"].startswith("all supplied trajectories")
    old = roots.RootProfile(oracle).profile(oracle.start, monotonic() + 10)
    assert endpoint["training_nmse"] <= old["training_nmse"] + 1e-10


def test_nonnegative_drift_bound_is_enforced():
    _, oracle, _ = fixture(drift_bounds={"lower": 0, "upper": 0.25})
    profiler = factory(oracle)
    result = profiler.profile(oracle.start, monotonic() + 10)
    assert 0 <= result["parameters"]["s"] <= 0.25
    assert result["joint_solve"]["bounds_satisfied"]
    with pytest.raises(ValueError, match="nondegenerate"):
        fixture(drift_bounds={"lower": 0.2, "upper": 0.2})


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"output": "r-(1+s)*x+k*A"}, "dynamics matrix"),
        ({"output": "r-x+k*A+s*s"}, "jointly affine"),
        ({"observation": "2*x+s"}, "readouts"),
    ],
)
def test_nonlinear_shared_uses_and_direct_shared_readout_fail_closed(kwargs, match):
    _, oracle, _ = fixture(**kwargs)
    with pytest.raises(ValueError, match=match):
        factory(oracle)


def test_shared_initialization_dependence_fails_closed(monkeypatch):
    original, oracle, _ = fixture()
    system = original.system
    initial = type(system).initial_symbolic

    def coupled(self, row, theta):
        value = initial(self, row, theta)
        value[0] += theta[self.names.index("s")]
        return value

    # Exercise the executable certificate even though the current proposer
    # initializer namespace already prevents direct dynamic-parameter reuse.
    monkeypatch.setattr(type(system), "initial_symbolic", coupled)
    with pytest.raises(ValueError, match="initializers"):
        factory(oracle)


def test_invalid_drift_and_foreign_domains_are_rejected():
    _, oracle, _ = fixture()
    for drift in (
        "missing",
        "k",
        "q",
        next(n for n in oracle.names if n.startswith("trajectory:")),
    ):
        with pytest.raises(ValueError, match="distinct shared"):
            joint.JointMealProfile(oracle, drift=drift, meal_options=MEAL_OPTIONS)
    oracle.original.lower[oracle.original.names.index("s")] = -1
    with pytest.raises(ValueError, match="domains differ"):
        factory(oracle)


def test_source_zero_rate_and_unsupported_meal_equations_fail_closed():
    _, oracle, _ = fixture()
    profiler = factory(oracle)
    with pytest.raises(NumericalInvalidTrial, match="zero rate"):
        profiler.profile(oracle.start | {"k": 0}, monotonic() + 10)
    assert profiler.counts["profile_calls_failed"] == 1
    _, bad, _ = fixture(output="r-x+2*k*A+s")
    with pytest.raises(ValueError, match="exactly"):
        factory(bad)


def test_deadline_and_wrong_full_rollout_are_not_accepted(monkeypatch):
    _, oracle, _ = fixture()
    profiler = factory(oracle, failure_diagnostics=True)
    with pytest.raises(TimeoutError):
        profiler.profile(oracle.start, monotonic() - 1)
    actual = oracle.evaluate

    def wrong(*args, **kwargs):
        r, j = actual(*args, **kwargs)
        return r + 1, j

    monkeypatch.setattr(oracle, "evaluate", wrong)
    with pytest.raises(NumericalInvalidTrial, match="pairing differs") as caught:
        profiler.profile(oracle.start, monotonic() + 10)
    assert caught.value.root_profile_diagnostic["stage"] == "pairing"
    assert profiler.counts["profile_calls_completed"] == 0


def test_qualification_and_search_factory_profile_the_two_shared_coefficients():
    _, oracle, _ = fixture()
    profiler = factory(oracle)
    qualification = roots.root_profile_qualification(
        oracle, oracle.start, 10, lambda _: None, profiler_factory=factory
    )
    assert qualification["passed"]
    assert qualification["joint_coefficient_optimality_checked"]
    result = roots.root_profile_search(
        oracle,
        oracle.start,
        10,
        5,
        lambda _: None,
        profiler_factory=factory,
        coordinate_chart=profiler.meal,
    )
    assert result["search_calls_started"] == 5
    assert result["outer_coordinates"] == ["k"]
    assert "selected shared coefficients" in result["optimizer"]
    assert not result["outer_coordinate_chart"]["original_root_profiler_unchanged"]
    assert result["best"]["joint_coefficient_optimality_checked"]
    assert (
        result["best"]["training_nmse"]
        <= result["unoptimized_control"]["training_nmse"]
    )


def test_profiled_shared_coefficient_cannot_be_fixed_by_outer_search():
    _, oracle, _ = fixture()
    result = roots.root_profile_search(
        oracle,
        oracle.start,
        10,
        5,
        lambda _: None,
        profiler_factory=factory,
        fixed={"s": 0.2},
    )
    assert result["search_calls_started"] == 0
    assert result["best"] is None
