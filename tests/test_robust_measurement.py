"""Convex Huber optima, exact profiled derivatives and explicit guard failures."""

from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import least_squares

from autoformalism.fitting import robust_measurement as robust
from autoformalism.schemas.measurement_fitting import MeasurementProblem
from tests.test_measurement_contract import measurement_payload
from tests.test_measurement_rollout import linear_problem


def arrays(design, offset, *, outer=1, lower=-100.0, upper=100.0):
    """Prepare an independent affine numerical problem with fixed derivatives."""
    a, b = np.asarray(design, float), np.asarray(offset, float)
    return (
        a,
        b,
        np.zeros((*a.shape, outer)),
        np.zeros((len(a), outer)),
        np.full(a.shape[1], lower),
        np.full(a.shape[1], upper),
    )


@pytest.mark.parametrize("kind", ["gaussian", "huber"])
def test_loss_transform_score_and_exact_chain_rule(kind):
    loss = robust.Loss(kind=kind)
    r = np.array([-8.0, -1.345, -0.2, 0, 0.2, 1.345, 9.0])
    q, derivative = robust.transformed_residual(r, loss)
    assert 0.5 * q**2 == pytest.approx(robust.loss_values(r, loss))
    assert q * derivative == pytest.approx(robust.score(r, loss))
    h = 1e-6
    qp, _ = robust.transformed_residual(r + h, loss)
    qm, _ = robust.transformed_residual(r - h, loss)
    assert derivative == pytest.approx((qp - qm) / (2 * h), abs=3e-7)
    assert derivative[3] == 1
    if kind == "huber":
        assert derivative[-1] != pytest.approx(np.sqrt(loss.delta / r[-1]))


@pytest.mark.parametrize("delta", [0, -1, float("nan"), float("inf"), True, "1.3"])
def test_loss_policy_rejects_invalid_or_mutable_configuration(delta):
    with pytest.raises(ValidationError):
        robust.Loss(delta=delta)
    with pytest.raises(ValidationError):
        robust.Loss().delta = 2.0
    with pytest.raises(ValidationError):
        robust.Loss(kind="cauchy")


@pytest.mark.parametrize("residual", [[np.nan], [np.inf], [[0.0]], 1.0])
def test_loss_inputs_are_finite_vectors(residual):
    for function in (robust.loss_values, robust.score, robust.transformed_residual):
        with pytest.raises(ValueError, match="finite vector"):
            function(residual, robust.Loss())


def test_convex_location_optimum_and_independent_bounded_solver():
    y = np.array([0.0, 0.2, 0.1, 100.0])
    inputs = arrays(np.ones((4, 1)), -y)
    result = robust.project(*inputs, loss=robust.Loss())
    expected = (0.3 + 1.345) / 3
    assert result["gains"] == pytest.approx([expected], abs=2e-10)
    reference = least_squares(
        lambda g: g[0] - y,
        [0.0],
        jac=lambda _: np.ones((4, 1)),
        bounds=(-100, 100),
        loss="huber",
        f_scale=1.345,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    )
    assert result["gains"] == pytest.approx(reference.x, abs=2e-7)
    assert result["audit"]["objective_value"] == pytest.approx(
        reference.cost, abs=1e-12
    )
    audit = result["audit"]
    assert audit["convex_optimum_certified"]
    assert audit["optimality"] < 1e-9
    assert audit["iterations"] > 0 and audit["inner_evaluations"] > 1
    assert audit["inlier_count"] == 3
    assert not audit["approximate_profile"]


def test_full_design_derivative_keeps_outlier_score_contribution():
    slope = np.array([0.1, 0.3, -0.2, 0.6])
    offset_slope = np.array([0.2, -0.1, 0.4, -0.2])
    y = np.array([0, 0.2, 0.1, 8])

    def at(beta):
        a = (1 + beta * slope)[:, None]
        return robust.project(
            a,
            -y + beta * offset_slope,
            slope[:, None, None],
            offset_slope[:, None],
            [-100],
            [100],
            loss=robust.Loss(),
        )

    point, h = at(0.2), 1e-5
    numerical = (at(0.2 + h)["residual"] - at(0.2 - h)["residual"]) / (2 * h)
    assert point["jacobian"][:, 0] == pytest.approx(numerical, abs=2e-8)
    # Omitting the saturated outlier's design derivative gives a different result.
    r = point["residual"]
    inlier = np.abs(r) < 1.345
    a = 1 + 0.2 * slope
    partial = slope * point["gains"][0] + offset_slope
    wrong_dg = -(a[inlier] @ partial[inlier] + slope[inlier] @ r[inlier]) / (
        a[inlier] @ a[inlier]
    )
    assert np.linalg.norm(numerical - (partial + a * wrong_dg)) > 0.1


def test_strict_active_bounds_have_valid_piecewise_derivative():
    inputs = list(arrays(np.ones((3, 1)), [-3, -4, -8], upper=0.1))
    inputs[3][:, 0] = [0.1, -0.2, 0.3]
    result = robust.project(*inputs, loss=robust.Loss())
    assert result["gains"] == pytest.approx([0.1], abs=1e-12)
    assert result["audit"]["active_mask"] == [1]
    assert result["audit"]["optimality"] == 0
    assert result["jacobian"] == pytest.approx(inputs[3])


def test_decoding_column_units_preserves_physical_bounds_exactly():
    scale = 3.6401748474461374e-10
    a = np.array([[scale], [scale / 2]])
    result = robust.project(
        *arrays(a, -a[:, 0] * 10, outer=0, lower=0.05, upper=0.8),
        loss=robust.Loss(),
    )
    assert 0.05 <= result["gains"][0] <= 0.8
    assert result["residual"] == pytest.approx(a @ result["gains"] - a[:, 0] * 10)


@pytest.mark.parametrize(
    ("offset", "lower", "reason"),
    [
        ([-10, 10], -100, "inlier Hessian"),
        ([-1.345, 1.345], -100, "threshold proximity"),
        ([-0.2, 0.2], 0, "weak or changing active bound"),
    ],
)
def test_nonsmooth_or_nonunique_profile_requests_joint_fallback(offset, lower, reason):
    with pytest.raises(robust.ProfileUnavailable, match=reason) as caught:
        robust.project(
            *arrays(np.ones((2, 1)), offset, lower=lower), loss=robust.Loss()
        )
    assert caught.value.audit["convex_optimum_certified"]
    assert caught.value.audit["status"] == "unavailable"
    assert caught.value.audit["derivative_guards"]


def test_zero_outer_nonunique_optimum_is_valid_without_uniqueness_claim():
    result = robust.project(
        *arrays(np.ones((2, 1)), [-10, 10], outer=0), loss=robust.Loss()
    )
    assert result["audit"]["convex_optimum_certified"]
    assert not result["audit"]["local_uniqueness_certified"]
    assert result["jacobian"].shape == (2, 0)
    assert result["audit"]["derivative_guards"]


def test_rank_deficient_design_is_unavailable_for_outer_derivative():
    with pytest.raises(robust.ProfileUnavailable, match="rank-deficient"):
        robust.project(*arrays(np.ones((4, 2)), [0, 1, 2, 3]), loss=robust.Loss())


@pytest.mark.parametrize("changed", ["design", "offset", "derivative", "bounds", "nan"])
def test_malformed_projection_inputs_are_rejected(changed):
    inputs = list(arrays(np.ones((3, 1)), [0, 0.1, 6]))
    if changed == "design":
        inputs[0] = np.ones(3)
    elif changed == "offset":
        inputs[1] = np.ones(4)
    elif changed == "derivative":
        inputs[2] = np.ones((3, 1))
    elif changed == "bounds":
        inputs[4][:] = inputs[5]
    else:
        inputs[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid bounded"):
        robust.project(*inputs, loss=robust.Loss())


def test_timeout_and_inner_failure_preserve_audit(monkeypatch):
    inputs = arrays(np.ones((3, 1)), [-0.1, -0.2, -7])
    with pytest.raises(TimeoutError) as caught:
        robust.project(*inputs, loss=robust.Loss(), deadline=monotonic() - 1)
    assert caught.value.audit["inner_evaluations"] == 0
    with pytest.raises(robust.ProfileUnavailable, match="iteration allowance"):
        robust.project(*inputs, loss=robust.Loss(), max_iterations=0)

    def failed(*args, **kwargs):
        kwargs["callback"](np.array([0.0]))
        return SimpleNamespace(
            x=np.array([0.0]), success=False, status=1, message="limit"
        )

    monkeypatch.setattr(robust, "minimize", failed)
    with pytest.raises(robust.ProfileUnavailable, match="KKT") as caught:
        robust.project(*inputs, loss=robust.Loss(), max_iterations=1)
    assert not caught.value.audit["convex_optimum_certified"]
    assert caught.value.audit["iterations"] == 1
    assert caught.value.audit["optimality"] > caught.value.audit["kkt_tolerance"]


@pytest.mark.parametrize("kind", ["gaussian", "huber"])
def test_rollout_wrapper_keeps_raw_evidence_and_method_propagation(kind):
    oracle, profile, _ = robust.build(
        linear_problem(shared=True), robust.Loss(kind=kind)
    )
    vector, deadline = oracle.vector(oracle.problem.start), monotonic() + 10
    raw, j = oracle.base.evaluate(vector, deadline)
    q, jq = oracle.evaluate(vector, deadline, method="Radau")
    expected, chain = robust.transformed_residual(raw, oracle.loss)
    assert q == pytest.approx(expected, abs=1e-7)
    assert jq == pytest.approx(chain[:, None] * j, abs=1e-7)
    assert oracle.objective(raw) == oracle.base.objective(raw)
    assert oracle.residual_evidence(raw) == oracle.base.residual_evidence(raw)
    assert oracle.evaluate(vector, deadline, jacobian=False)[1] is None
    fitted, q, derivative, _ = profile.evaluate(vector, deadline)
    raw, _ = oracle.base.evaluate(fitted, deadline)
    assert q == pytest.approx(robust.transformed_residual(raw, oracle.loss)[0])
    assert derivative.shape == (3, 0)
    assert len(profile.inner_audits) == 1
    with pytest.raises(TimeoutError):
        profile.evaluate(vector, monotonic() - 1)
    assert profile.inner_audits[-1]["status"] == "unavailable"


def test_shared_initial_profile_exact_huber_chain_derivative():
    payload = measurement_payload(shared=True)
    payload["manifest"]["noise"]["sigma"] = [0.1, 0.5, 2.0]
    problem = MeasurementProblem.model_validate(payload)
    oracle, profile, route = robust.build(problem, robust.Loss())
    assert route["selected"] == "affine_profile"
    vector = oracle.vector({"rate": 0.4, "initial": 1.0})
    fitted, q, derivative, audit = profile.evaluate(vector, monotonic() + 10)
    direct, _ = oracle.evaluate(fitted, monotonic() + 10)
    assert q == pytest.approx(direct, abs=1e-12)
    h = np.array([1e-5, 0])
    plus = profile.evaluate(vector + h, monotonic() + 10)[1]
    minus = profile.evaluate(vector - h, monotonic() + 10)[1]
    assert derivative[:, 0] == pytest.approx((plus - minus) / 2e-5, abs=1e-7)
    assert audit["convex_optimum_certified"]


def test_nonlinear_joint_rollout_chain_rule_includes_shared_initial():
    payload = measurement_payload(shared=True)
    payload["request"]["base_candidate"]["state_equations"][0]["rhs"] = "-rate*x*x+u"
    oracle, profile, route = robust.build(
        MeasurementProblem.model_validate(payload), robust.Loss()
    )
    assert profile is None and route["selected"] == "joint_rollout"
    vector = oracle.vector({"rate": 0.4, "initial": 1.0})
    _, derivative = oracle.evaluate(vector, monotonic() + 10)
    for index in range(len(vector)):
        h = np.eye(len(vector))[index] * 1e-5
        plus = oracle.evaluate(vector + h, monotonic() + 10)[0]
        minus = oracle.evaluate(vector - h, monotonic() + 10)[0]
        assert derivative[:, index] == pytest.approx((plus - minus) / 2e-5, abs=2e-6)


def test_nonlinear_terminal_profile_exact_huber_derivative():
    payload = linear_problem().model_dump(mode="json")
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
    oracle, profile, route = robust.build(
        MeasurementProblem.model_validate(payload), robust.Loss()
    )
    assert route["selected"] == "terminal_output_profile"
    vector = oracle.vector({"rate": 0.4, "gain": 1.0})
    fitted, q, derivative, audit = profile.evaluate(vector, monotonic() + 10)
    assert q == pytest.approx(oracle.evaluate(fitted, monotonic() + 10)[0], abs=1e-7)
    h = np.array([1e-4, 0])
    plus = profile.evaluate(vector + h, monotonic() + 10)[1]
    minus = profile.evaluate(vector - h, monotonic() + 10)[1]
    assert derivative[:, 0] == pytest.approx((plus - minus) / 2e-4, abs=2e-6)
    assert audit["convex_optimum_certified"]
