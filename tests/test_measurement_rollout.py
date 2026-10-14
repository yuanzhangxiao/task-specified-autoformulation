"""Known-noise whitening, explicit boundaries and certified weighted profiles."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting.measurement_rollout import MeasurementRollouts, build
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.measurement_fitting import MeasurementProblem
from tests.test_measurement_contract import measurement_payload


def linear_problem(*, shared=False, missing=False):
    """Return x'=rate*u with known u=2 and an independently specified boundary."""
    payload = measurement_payload(shared=shared, missing=missing)
    payload["request"]["base_candidate"]["state_equations"][0]["rhs"] = "rate*u"
    payload["manifest"]["noise"]["sigma"] = [0.2, 0.8] if missing else [0.2, 0.4, 0.8]
    for parameter in payload["request"]["base_candidate"]["parameters"]:
        parameter["bounds"] = {"lower": -20, "upper": 20}
    return MeasurementProblem.model_validate(payload)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_weighted_affine_optimum_including_zero_outer_block(shared, missing):
    problem = linear_problem(shared=shared, missing=missing)
    before = content_sha256(problem)
    oracle, profile, route = build(problem)
    assert route["selected"] == "affine_profile"
    assert len(profile.outer) == 0
    x = oracle.vector(problem.start)
    fitted, residual, jacobian, audit = profile.evaluate(x, monotonic() + 10)
    time = np.asarray(oracle.layout.analysis_times)[oracle.mask]
    y = oracle.observations[oracle.mask]
    sigma = oracle.sigma[oracle.mask]
    design = (
        np.column_stack([2 * time, np.ones(len(time))])
        if shared
        else (2 * time)[:, None]
    )
    expected = np.linalg.lstsq(
        design / sigma[:, None], (y - (0 if shared else 4)) / sigma, rcond=None
    )[0]
    assert fitted == pytest.approx(expected, abs=1e-12)
    direct, derivative = oracle.evaluate(fitted, monotonic() + 10)
    assert residual == pytest.approx(direct, abs=1e-12)
    assert derivative == pytest.approx(design / sigma[:, None], abs=1e-12)
    assert jacobian.shape == (len(y), 0)
    assert audit["optimality"] < 1e-8
    assert content_sha256(problem) == before
    # This observation is not a hard initial condition or a second likelihood factor.
    assert len(residual) == len(y)
    if not shared:
        assert residual[0] == pytest.approx(5.0)


def test_shared_initial_sensitivity_and_independent_integrators():
    problem = linear_problem(shared=True, missing=True)
    oracle, _, _ = build(problem)
    vector = oracle.vector({"rate": 0.7, "initial": 5.0})
    residual, jacobian = oracle.evaluate(vector, monotonic() + 10)
    assert oracle.system.initial_parameter_names == ("initial",)
    for method in ("DOP853", "Radau", "BDF"):
        other, derivative = oracle.evaluate(vector, monotonic() + 10, method=method)
        assert other == pytest.approx(residual, abs=1e-7)
        assert derivative == pytest.approx(jacobian, abs=1e-7)
    for index in range(len(vector)):
        step = np.eye(len(vector))[index] * 1e-5
        plus = oracle.evaluate(vector + step, monotonic() + 10)[0]
        minus = oracle.evaluate(vector - step, monotonic() + 10)[0]
        assert jacobian[:, index] == pytest.approx((plus - minus) / 2e-5, abs=1e-8)
    row = oracle.train.trajectories[0]
    assert oracle.system.initial_for(row, vector) == pytest.approx([5.0])
    with pytest.raises(TimeoutError):
        oracle.evaluate(vector, monotonic() - 1)


def test_explicit_exact_boundaries_can_differ_across_trajectories():
    payload = linear_problem().model_dump(mode="json")
    row = deepcopy(payload["training"]["rows"][0])
    row["trajectory_id"] = "train_b"
    for reading in row["readings"]:
        reading["id"] = "b_" + reading["id"]
        reading["source_row"] += 100
    payload["training"]["rows"].append(row)
    clock = deepcopy(payload["manifest"]["time_bases"][0])
    clock["trajectory"] = "train_b"
    payload["manifest"]["time_bases"].append(clock)
    forcing = deepcopy(payload["manifest"]["forcing_rules"][0])
    forcing["trajectory"] = "train_b"
    forcing["knot_reading_ids"] = ["b_" + name for name in forcing["knot_reading_ids"]]
    payload["manifest"]["forcing_rules"].append(forcing)
    noise = payload["manifest"]["noise"]
    noise["reading_ids"] += ["b_" + name for name in noise["reading_ids"]]
    noise["sigma"] *= 2
    payload["initial_rules"][0]["exact_boundaries"].append(
        {"trajectory": "train_b", "value": -2.0, "preparation_source": "preparation"}
    )
    oracle = MeasurementRollouts(MeasurementProblem.model_validate(payload))
    x = oracle.vector(oracle.problem.start)
    residual, _ = oracle.evaluate(x, monotonic() + 10)
    assert residual[[0, 3]] == pytest.approx([5.0, -25.0])
    assert [
        oracle.system.initial_for(row, x)[0] for row in oracle.train.trajectories
    ] == [4, -2]


def test_full_likelihood_and_raw_analysis_residual_decomposition():
    problem = linear_problem(missing=True)
    oracle, _, _ = build(problem)
    x = oracle.vector(problem.start)
    residual, _ = oracle.evaluate(x, monotonic() + 10)
    evidence = oracle.evidence(x, monotonic() + 10)
    cached = oracle.residual_evidence(residual)
    quadratic = 0.5 * residual @ residual
    constant = np.log([0.2, 0.8]).sum() + np.log(2 * np.pi)
    assert evidence["quadratic_objective"] == pytest.approx(quadratic)
    assert evidence["gaussian_nll"] == pytest.approx(quadratic + constant)
    assert oracle.likelihood_constant == pytest.approx(constant)
    rows = evidence["reading_audit"]
    assert rows[0]["raw_value"] == 1
    assert rows[0]["analysis_value"] == 3
    assert rows[0]["prediction_analysis"] == 4
    assert rows[0]["prediction_raw"] == 1.5
    assert rows[0]["residual_raw"] == 0.5
    assert rows[0]["sigma_raw"] == 0.1
    assert rows[1]["raw_value"] is rows[1]["residual_analysis"] is None
    assert rows[1]["prediction_analysis"] == pytest.approx(6)
    assert cached["reading_audit"][1]["prediction_analysis"] is None
    assert cached["gaussian_nll"] == evidence["gaussian_nll"]
    assert sum(r["quadratic_contribution"] or 0 for r in rows) == pytest.approx(
        quadratic
    )
    with pytest.raises(ValueError, match="layout"):
        oracle.residual_evidence(np.zeros(3))


def test_profiled_initial_and_outer_derivative_use_same_heteroskedastic_weights():
    payload = measurement_payload(shared=True)
    payload["manifest"]["noise"]["sigma"] = [0.1, 0.5, 2.0]
    problem = MeasurementProblem.model_validate(payload)
    oracle, profile, route = build(problem)
    assert route["selected"] == "affine_profile"
    x = oracle.vector({"rate": 0.4, "initial": 1.0})
    fitted, residual, jacobian, _ = profile.evaluate(x, monotonic() + 10)
    direct, _ = oracle.evaluate(fitted, monotonic() + 10)
    assert residual == pytest.approx(direct, abs=1e-12)
    time = np.asarray([0, 1, 2])
    design = np.exp(-0.4 * time)
    offset = 2 / 0.4 * (1 - design) - oracle.observations
    weights = 1 / oracle.sigma**2
    expected_initial = -sum(design * offset * weights) / sum(design**2 * weights)
    assert fitted[1] == pytest.approx(expected_initial, abs=1e-12)
    step = np.array([1e-5, 0])
    plus = profile.evaluate(x + step, monotonic() + 10)[1]
    minus = profile.evaluate(x - step, monotonic() + 10)[1]
    assert jacobian[:, 0] == pytest.approx((plus - minus) / 2e-5, abs=1e-8)


def test_multi_output_falls_back_to_joint_and_applies_masks_per_reading():
    payload = linear_problem().model_dump(mode="json")
    payload["request"]["context"]["targets"].append("z")
    payload["request"]["base_candidate"]["observation_mappings"].append(
        {"channel": "z", "expression": "2*x"}
    )
    channel = deepcopy(payload["manifest"]["channels"][0])
    channel["id"] = "z"
    payload["manifest"]["channels"].append(channel)
    rows = payload["training"]["rows"][0]["readings"]
    targets = []
    for reading in list(rows):
        if reading["channel"] != "y":
            continue
        targets.append(reading["id"])
        second = deepcopy(reading)
        second.update(id=reading["id"].replace("y", "z"), channel="z")
        second["raw_value"] = 2 * second["raw_value"] + 0.5
        if second["clock_value"] == 12:
            second.update(raw_value=None, condition="missing")
        else:
            targets.append(second["id"])
        rows.append(second)
    payload["manifest"]["noise"].update(
        reading_ids=targets, sigma=[0.1, 0.2, 0.4, 0.8, 1.6]
    )
    oracle, profile, route = build(MeasurementProblem.model_validate(payload))
    assert profile is None and route["selected"] == "joint_rollout"
    assert len(route["inapplicable"]) == 2
    residual, jacobian = oracle.evaluate(
        oracle.vector(oracle.problem.start), monotonic() + 10
    )
    expected = np.array([0, 0, 2 / 0.4, 4 / 0.8, 8 / 1.6])
    assert jacobian[:, 0] == pytest.approx(expected)
    assert len(residual) == 5


def test_rank_deficient_inner_problem_is_unavailable():
    payload = linear_problem().model_dump(mode="json")
    candidate = payload["request"]["base_candidate"]
    candidate["parameters"].append({"name": "other", "scope": "global"})
    candidate["state_equations"][0]["rhs"] = "rate*u+other*u"
    payload["coordinates"]["parameters"]["other"] = {"center": 1, "scale": 1}
    payload["start"]["other"] = payload["incumbent"]["other"] = 1
    oracle, profile, _ = build(MeasurementProblem.model_validate(payload))
    with pytest.raises(ValueError, match="rank-deficient"):
        profile.evaluate(oracle.vector(oracle.problem.start), monotonic() + 10)


def test_gain_initial_confounding_is_visible_in_joint_jacobian():
    payload = measurement_payload(shared=True)
    candidate = payload["request"]["base_candidate"]
    candidate["state_equations"][0]["rhs"] = "0*x"
    candidate["observation_mappings"][0]["expression"] = "rate*x"
    oracle, profile, _ = build(MeasurementProblem.model_validate(payload))
    assert profile is None
    _, jacobian = oracle.evaluate(oracle.vector(oracle.problem.start), monotonic() + 10)
    assert np.linalg.matrix_rank(jacobian) == 1
    assert np.linalg.norm(jacobian[:, 0]) > 0
    assert np.linalg.norm(jacobian[:, 1]) > 0


def test_weighted_gain_optimum_at_active_bound():
    payload = linear_problem().model_dump(mode="json")
    payload["request"]["base_candidate"]["parameters"][0]["bounds"] = {
        "lower": -2,
        "upper": 0.1,
    }
    payload["start"]["rate"] = payload["incumbent"]["rate"] = 0.0
    oracle, profile, _ = build(MeasurementProblem.model_validate(payload))
    fitted, residual, derivative, audit = profile.evaluate(
        oracle.vector(oracle.problem.start), monotonic() + 10
    )
    assert fitted == pytest.approx([0.1])
    assert audit["active_mask"] == [1]
    assert derivative.shape == (3, 0)
    assert residual == pytest.approx(oracle.evaluate(fitted, monotonic() + 10)[0])


@pytest.mark.parametrize("shared", [False, True])
def test_missing_first_reading_does_not_change_boundary(shared):
    payload = linear_problem(shared=shared).model_dump(mode="json")
    payload["training"]["rows"][0]["readings"][0].update(
        condition="missing", raw_value=None
    )
    noise = payload["manifest"]["noise"]
    noise["reading_ids"], noise["sigma"] = noise["reading_ids"][1:], noise["sigma"][1:]
    oracle, _, _ = build(MeasurementProblem.model_validate(payload))
    x = oracle.vector(oracle.problem.start)
    expected = 1 if shared else 4
    assert oracle.system.initial_for(oracle.train.trajectories[0], x)[0] == expected
    r, j = oracle.evaluate(x, monotonic() + 10)
    assert len(r) == 2 and j.shape == (2, len(x))
    assert oracle.problem.training.rows[0].readings[0].raw_value is None


def test_nonlinear_terminal_profile_whitens_basis_and_outer_derivatives():
    payload = linear_problem(missing=True).model_dump(mode="json")
    candidate = payload["request"]["base_candidate"]
    candidate["parameters"].append({"name": "gain", "scope": "global"})
    candidate["states"].append({"name": "z", "kind": "latent"})
    candidate["state_equations"] = [
        {"state": "x", "rhs": "-rate*x+gain*tanh(z)"},
        {"state": "z", "rhs": "-z*z+u"},
    ]
    payload["initial_rules"].append(
        {
            "state": "z",
            "mode": "exact_boundary",
            "assumption": "initial",
            "exact_boundaries": [
                {
                    "trajectory": "train_a",
                    "value": 0.7,
                    "preparation_source": "preparation",
                }
            ],
        }
    )
    payload["coordinates"]["parameters"]["gain"] = {"center": 1, "scale": 1}
    payload["coordinates"]["states"]["z"] = {"center": 0, "scale": 1}
    payload["start"]["gain"] = payload["incumbent"]["gain"] = 1
    oracle, profile, route = build(MeasurementProblem.model_validate(payload))
    assert route["selected"] == "terminal_output_profile"
    assert not oracle.is_affine
    vector = oracle.vector({"rate": 0.4, "gain": 1.0})
    fitted, residual, derivative, _ = profile.evaluate(vector, monotonic() + 10)
    direct, _ = oracle.evaluate(fitted, monotonic() + 10)
    assert residual == pytest.approx(direct, abs=1e-8)
    row = oracle.train.trajectories[0]
    a, b, _, _, _ = profile.kernel.trajectory(
        row, vector, oracle.settings, monotonic() + 10
    )
    sigma = oracle.sigma[oracle.mask]
    design = a[oracle.mask, 0] / sigma
    offset = (b[oracle.mask] - oracle.observations[oracle.mask]) / sigma
    assert fitted[1] == pytest.approx(-design @ offset / (design @ design), abs=1e-10)
    step = np.array([1e-4, 0])
    plus = profile.evaluate(vector + step, monotonic() + 10)[1]
    minus = profile.evaluate(vector - step, monotonic() + 10)[1]
    assert derivative[:, 0] == pytest.approx((plus - minus) / 2e-4, abs=1e-6)
