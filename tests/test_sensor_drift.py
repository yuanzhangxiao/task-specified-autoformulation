"""Frozen drift contracts, observation-only sensitivities and exact convex profiles."""

from time import monotonic

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import least_squares, lsq_linear

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import measurement_fit as fit
from autoformalism.fitting import measurement_fixtures as fixtures
from autoformalism.fitting import robust_measurement as robust
from autoformalism.fitting import sensor_drift
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.sensor_drift import DriftProblem, SensorDrift


def drift_problem(case="exact_varied", **changes):
    """Add only a predeclared shared observation slope to a public fixture."""
    base = fixtures.make(case, 0)[0]
    definition = {
        "channel": "y",
        "lower": -0.2,
        "upper": 0.2,
        "start": 0.0,
        "incumbent": 0.0,
        "scale": 0.1,
        "unit": "m/min",
        "sources": ["calibration"],
        "rationale": "Optional shared sensor drift hypothesis.",
    } | changes
    return DriftProblem(base=base, drift=SensorDrift.model_validate(definition))


def test_contract_preserves_base_identity_and_separate_immutable_nuisance():
    base = fixtures.make()[0]
    before = content_sha256(base)
    problem = drift_problem()
    assert content_sha256(problem.base) == before
    assert problem.request == base.request
    assert problem.training == base.training
    assert problem.initial_rules == base.initial_rules
    assert set(problem.model_dump()) == {"protocol", "base", "drift"}
    assert set(problem.start) == set(problem.incumbent) == {"g", "sensor_drift"}
    assert "sensor_drift" not in {
        p.name for p in problem.request.base_candidate.parameters
    }
    assert problem.coordinates.arrays("parameters", ["g", "sensor_drift"])[
        1
    ] == pytest.approx([1, 0.1])
    assert DriftProblem.model_validate_json(problem.model_dump_json()) == problem
    with pytest.raises(TypeError, match="immutable"):
        problem.start["sensor_drift"] = 0.2
    with pytest.raises(TypeError, match="immutable"):
        problem.coordinates.parameters["sensor_drift"] = None
    with pytest.raises(ValidationError):
        problem.drift.upper = 1.0
    assert content_sha256(problem.base) == before


@pytest.mark.parametrize(
    "change",
    [
        {"lower": 0.2},
        {"lower": float("nan")},
        {"upper": float("inf")},
        {"start": 0.3},
        {"incumbent": -0.3},
        {"scale": 0.0},
        {"scale": True},
        {"time_origin": float("nan")},
        {"sources": []},
        {"sources": ["absent"]},
        {"sources": ["calibration", "calibration"]},
        {"unit": "cm/min"},
        {"unit": " "},
        {"rationale": " "},
        {"channel": "u"},
        {"channel": "missing"},
        {"name": "g"},
        {"name": "x"},
        {"name": "u"},
        {"name": "y"},
        {"name": "t"},
    ],
)
def test_invalid_drift_domains_units_namespace_and_provenance_fail_closed(change):
    with pytest.raises(ValidationError):
        drift_problem(**change)


def test_mixed_time_units_and_evaluator_fields_are_rejected():
    payload = drift_problem().model_dump(mode="json")
    payload["base"]["manifest"]["time_bases"][1]["analysis_unit"] = "seconds"
    with pytest.raises(ValidationError, match="common analysis time units"):
        DriftProblem.model_validate(payload)
    payload = drift_problem().model_dump(mode="json")
    payload["truth"] = {"sensor_drift": 0.1}
    with pytest.raises(ValidationError):
        DriftProblem.model_validate(payload)
    payload = drift_problem().model_dump(mode="json")
    payload["base"]["request"]["base_candidate"]["processes"].append(
        {"name": "sensor_drift", "expression": "g*u"}
    )
    with pytest.raises(ValidationError, match="collides"):
        DriftProblem.model_validate(payload)


@pytest.mark.parametrize(
    "case", ["exact_varied", "shared_initial", "missing_readings", "multi_output"]
)
def test_drift_sensitivity_preserves_physical_dynamics_masks_and_units(case):
    problem = drift_problem(case, time_origin=0.5)
    before = content_sha256(problem)
    optimizer, _, _ = sensor_drift.build(problem, robust.Loss(kind="gaussian"))
    raw = optimizer.base
    vector = raw.vector(problem.start | {"sensor_drift": 0.07})
    residual, jacobian = raw.evaluate(vector, monotonic() + 10)
    physical, physical_j = raw.physical.evaluate(vector[:-1], monotonic() + 10)
    assert residual - physical == pytest.approx(0.07 * raw.column)
    assert jacobian[:, :-1] == pytest.approx(physical_j)
    assert jacobian[:, -1] == pytest.approx(raw.column)
    assert raw.system.names == raw.physical.names
    assert "sensor_drift" not in raw.system.names
    assert raw.names[-1] == "sensor_drift"
    for method in ("DOP853", "Radau"):
        other, derivative = raw.evaluate(vector, monotonic() + 10, method=method)
        assert other == pytest.approx(residual, abs=1e-7)
        assert derivative == pytest.approx(jacobian, abs=1e-7)
    h = np.zeros(len(vector))
    h[-1] = 1e-6
    plus = raw.evaluate(vector + h, monotonic() + 10)[0]
    minus = raw.evaluate(vector - h, monotonic() + 10)[0]
    assert jacobian[:, -1] == pytest.approx((plus - minus) / 2e-6, abs=1e-8)
    if case == "multi_output":
        channels = np.asarray(raw.layout.channels)[raw.mask]
        assert not np.any(raw.column[channels == "z"])
    assert content_sha256(problem) == before


def test_raw_evidence_preserves_readings_and_exposes_total_and_physical_predictions():
    problem = drift_problem("missing_readings", time_origin=0.5)
    raw = sensor_drift.build(problem, robust.Loss(kind="gaussian"))[0].base
    vector = raw.vector(problem.start | {"sensor_drift": 0.1})
    residual, _ = raw.evaluate(vector, monotonic() + 10)
    # A diagnostic on the underlying physical oracle must not poison cached totals.
    raw.physical.evaluate(np.array([0.8]), monotonic() + 10)
    evidence = raw.evidence(vector, monotonic() + 10)
    rows = evidence["reading_audit"]
    assert rows[0]["raw_value"] is None
    assert rows[0]["prediction_analysis"] == pytest.approx(0.3 - 0.05)
    assert rows[0]["physical_prediction_analysis"] == pytest.approx(0.3)
    assert rows[0]["prediction_raw"] == pytest.approx((0.25 - 0.03) / 0.01)
    assert all(row["prediction_analysis"] is not None for row in rows)
    assert sum(row["quadratic_contribution"] or 0 for row in rows) == pytest.approx(
        0.5 * residual @ residual
    )
    assert raw.residual_evidence(residual)["raw_readings_modified"] is False
    assert raw.objective(residual)["quadratic_objective"] == pytest.approx(
        0.5 * residual @ residual
    )


def test_retained_prediction_evidence_uses_selected_parameters_without_integration(
    monkeypatch,
):
    problem = drift_problem("missing_readings")
    raw = sensor_drift.build(problem, robust.Loss())[0].base
    selected = problem.start | {"g": 1.1, "sensor_drift": 0.04}
    residual, _ = raw.evaluate(raw.vector(selected), monotonic() + 10)
    raw.evaluate(raw.vector(problem.start | {"sensor_drift": -0.1}), monotonic() + 10)
    monkeypatch.setattr(
        raw, "evaluate", lambda *a, **k: pytest.fail("reintegrated retained endpoint")
    )
    evidence = raw.prediction_evidence(residual, selected)
    for reading in evidence["reading_audit"]:
        if not reading["included"]:
            assert reading["physical_prediction_analysis"] is None
            assert reading["drift_contribution_analysis"] is None
            continue
        expected = 0.04 * reading["analysis_time"]
        assert reading["drift_contribution_analysis"] == pytest.approx(expected)
        assert reading["physical_prediction_analysis"] + expected == pytest.approx(
            reading["prediction_analysis"]
        )
        assert reading["physical_prediction_raw"] == pytest.approx(
            (reading["physical_prediction_analysis"] - 0.03) / 0.01
        )
    assert evidence["parameters"] == selected


@pytest.mark.parametrize("kind", ["gaussian", "huber"])
@pytest.mark.parametrize("case", ["exact_varied", "shared_initial"])
def test_zero_outer_profile_matches_independent_bounded_convex_fit(kind, case):
    problem = drift_problem(case)
    optimizer, profile, route = sensor_drift.build(problem, robust.Loss(kind=kind))
    raw = optimizer.base
    vector = raw.vector(problem.start)
    r, design = raw.evaluate(vector, monotonic() + 10)
    offset = r - design @ vector
    fitted, q, derivative, audit = profile.evaluate(vector, monotonic() + 10)
    if kind == "gaussian":
        reference = lsq_linear(
            design, -offset, bounds=(raw.lower, raw.upper), tol=1e-13
        )
    else:
        reference = least_squares(
            lambda v: design @ v + offset,
            vector,
            jac=lambda _: design.copy(),
            bounds=(raw.lower, raw.upper),
            loss="huber",
            f_scale=1.345,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
        )
        assert audit["convex_optimum_certified"]
    assert fitted == pytest.approx(reference.x, abs=2e-7)
    assert q == pytest.approx(
        optimizer.evaluate(fitted, monotonic() + 10)[0], abs=1e-10
    )
    assert derivative.shape == (len(r), 0)
    assert profile.inner[-1] == len(raw.names) - 1
    assert route["certificate"]["gain_names"][-1] == "sensor_drift"


@pytest.mark.parametrize("kind", ["gaussian", "huber"])
def test_profile_outer_derivative_includes_physical_drift_cross_terms(kind):
    problem = drift_problem("nonlinear_decay")
    optimizer, profile, _ = sensor_drift.build(problem, robust.Loss(kind=kind))
    vector = optimizer.vector(problem.start)
    fitted, q, derivative, _ = profile.evaluate(vector, monotonic() + 10)
    assert profile.outer == (1,)
    assert q == pytest.approx(
        optimizer.evaluate(fitted, monotonic() + 10)[0], abs=1e-10
    )
    h = np.array([0, 1e-5, 0])
    plus = profile.evaluate(vector + h, monotonic() + 10)[1]
    minus = profile.evaluate(vector - h, monotonic() + 10)[1]
    assert derivative[:, 0] == pytest.approx((plus - minus) / 2e-5, abs=2e-6)


def test_shared_initial_and_drift_profile_preserves_outer_derivative():
    payload = drift_problem("shared_initial").model_dump(mode="json")
    base = payload["base"]
    base["request"]["base_candidate"]["parameters"].append(
        {"name": "k", "scope": "global", "bounds": {"lower": 0.1, "upper": 1.5}}
    )
    base["request"]["base_candidate"]["state_equations"][0]["rhs"] = "g*u-k*x"
    base["coordinates"]["parameters"]["k"] = {"center": 0, "scale": 1}
    base["start"]["k"] = base["incumbent"]["k"] = 0.7
    optimizer, profile, _ = sensor_drift.build(
        DriftProblem.model_validate(payload), robust.Loss()
    )
    vector = optimizer.vector(optimizer.problem.start)
    _, _, derivative, _ = profile.evaluate(vector, monotonic() + 10)
    h = np.zeros(len(vector))
    h[2] = 1e-5
    plus = profile.evaluate(vector + h, monotonic() + 10)[1]
    minus = profile.evaluate(vector - h, monotonic() + 10)[1]
    assert derivative[:, 0] == pytest.approx((plus - minus) / 2e-5, abs=2e-6)


@pytest.mark.parametrize("kind", ["gaussian", "huber"])
def test_terminal_nonlinear_profile_and_drift_have_complete_derivatives(kind):
    payload = drift_problem("nonlinear_decay").model_dump(mode="json")
    base = payload["base"]
    candidate = base["request"]["base_candidate"]
    candidate["states"].append({"name": "z", "kind": "latent"})
    candidate["state_equations"] = [
        {"state": "x", "rhs": "g*tanh(z)-k*x"},
        {"state": "z", "rhs": "u-z*z"},
    ]
    base["initial_rules"].append(
        {
            "state": "z",
            "mode": "exact_boundary",
            "assumption": "initial_known",
            "exact_boundaries": [
                {"trajectory": name, "value": 0.7, "preparation_source": "preparation"}
                for name in ("train_0", "train_1")
            ],
        }
    )
    base["coordinates"]["states"]["z"] = {"center": 0, "scale": 1}
    problem = DriftProblem.model_validate(payload)
    optimizer, profile, route = sensor_drift.build(problem, robust.Loss(kind=kind))
    assert route["selected"] == "terminal_output_profile"
    vector = optimizer.vector(problem.start)
    fitted, q, derivative, _ = profile.evaluate(vector, monotonic() + 10)
    assert q == pytest.approx(optimizer.evaluate(fitted, monotonic() + 10)[0], abs=1e-7)
    h = np.array([0, 1e-5, 0])
    plus = profile.evaluate(vector + h, monotonic() + 10)[1]
    minus = profile.evaluate(vector - h, monotonic() + 10)[1]
    assert derivative[:, 0] == pytest.approx((plus - minus) / 2e-5, abs=2e-6)


def test_active_drift_bound_and_confounding_are_explicit():
    problem = drift_problem(lower=-0.00001, upper=0.00001)
    optimizer, profile, _ = sensor_drift.build(problem, robust.Loss())
    fitted, _, _, audit = profile.evaluate(
        optimizer.vector(problem.start), monotonic() + 10
    )
    assert abs(fitted[-1]) <= 0.00001
    assert audit["active_mask"][-1] != 0
    payload = problem.model_dump(mode="json")
    payload["base"]["request"]["base_candidate"]["state_equations"][0]["rhs"] = "g"
    for kind in ("gaussian", "huber"):
        optimizer, profile, _ = sensor_drift.build(
            DriftProblem.model_validate(payload), robust.Loss(kind=kind)
        )
        vector = optimizer.vector(optimizer.problem.start)
        _, jacobian = optimizer.base.evaluate(vector, monotonic() + 10)
        assert np.linalg.matrix_rank(jacobian) == 1
        if kind == "gaussian":
            with pytest.raises(ValueError, match="rank-deficient"):
                profile.evaluate(vector, monotonic() + 10)
        else:
            _, _, _, audit = profile.evaluate(vector, monotonic() + 10)
            assert audit["convex_optimum_certified"]
            assert not audit["design_full_rank"]
            assert not audit["local_uniqueness_certified"]


def test_ineligible_physical_profile_uses_joint_drift_and_checks_deadlines():
    payload = drift_problem().model_dump(mode="json")
    payload["base"]["request"]["base_candidate"]["state_equations"][0]["rhs"] = (
        "g*u-0.65*x*x"
    )
    problem = DriftProblem.model_validate(payload)
    optimizer, profile, route = sensor_drift.build(problem, robust.Loss())
    assert profile is None and route["selected"] == "joint_rollout"
    vector = optimizer.vector(problem.start)
    _, derivative = optimizer.evaluate(vector, monotonic() + 10)
    for i in range(len(vector)):
        h = np.eye(len(vector))[i] * 1e-5
        plus = optimizer.evaluate(vector + h, monotonic() + 10)[0]
        minus = optimizer.evaluate(vector - h, monotonic() + 10)[0]
        assert derivative[:, i] == pytest.approx((plus - minus) / 2e-5, abs=2e-6)
    with pytest.raises(TimeoutError):
        optimizer.evaluate(vector, monotonic() - 1)
    with pytest.raises(ValueError, match="incomplete"):
        optimizer.vector({"g": 1})
    with pytest.raises(ValueError, match="outside"):
        optimizer.vector(problem.start | {"sensor_drift": 1})


@pytest.mark.parametrize("kind", ["gaussian", "huber"])
def test_fit_dispatch_identity_resume_and_parameter_grouping(
    tmp_path, monkeypatch, kind
):
    monkeypatch.setattr(fit.public, "_source_identity", lambda: "drift-test-source")
    problem = drift_problem()
    policy = fit.Policy(
        restart_count=0, search_seconds=5, verification_seconds=5, maximum_calls=30
    )
    loss = robust.Loss(kind=kind)
    direct = fit.run(problem, policy, tmp_path / "direct", strategy="direct", loss=loss)
    profiled = fit.run(problem, policy, tmp_path / "profiled", loss=loss)
    assert direct["status"] == profiled["status"] == "complete"
    assert profiled["objective_value"] == pytest.approx(
        direct["objective_value"], abs=1e-7
    )
    assert profiled["parameters"] == pytest.approx(direct["parameters"], abs=1e-7)
    assert profiled["identity"]["protocol"] == "sensor-drift-fitting-execution-1"
    assert profiled["protocol"] == "sensor-drift-fit-result-1"
    assert profiled["identity"]["base_request_sha256"] == content_sha256(
        problem.base.request
    )
    assert profiled["physical_parameters"] == {"g": profiled["parameters"]["g"]}
    assert profiled["observation_nuisance_parameters"] == {
        "sensor_drift": profiled["parameters"]["sensor_drift"]
    }
    assert profiled["verification"]["parameter_order"] == ["g", "sensor_drift"]
    assert profiled["confidence_probability"] is None
    for reading in profiled["residual_evidence"]["reading_audit"]:
        assert reading["physical_prediction_analysis"] + reading[
            "drift_contribution_analysis"
        ] == pytest.approx(reading["prediction_analysis"])
    monkeypatch.setattr(
        sensor_drift, "build", lambda *args: pytest.fail("repeated setup")
    )
    assert fit.run(problem, policy, tmp_path / "profiled", loss=loss) == profiled
    assert read_seal(tmp_path / "profiled/result.json") == profiled
    changed = problem.model_dump(mode="json")
    changed["drift"]["time_origin"] = 0.1
    with pytest.raises(ValueError, match="differs"):
        fit.run(
            DriftProblem.model_validate(changed),
            policy,
            tmp_path / "profiled",
            loss=loss,
        )
    with pytest.raises(ValueError, match="differs"):
        fit.run(problem.base, policy, tmp_path / "profiled", loss=loss)


def test_default_drift_loss_is_explicit_gaussian(tmp_path, monkeypatch):
    monkeypatch.setattr(fit.public, "_source_identity", lambda: "drift-test-source")
    problem = drift_problem()
    result = fit.run(problem, fit.Policy(restart_count=0), tmp_path)
    assert result["objective_definition"] == {"kind": "gaussian", "delta": 1.345}
    assert result["negative_log_likelihood"] is not None
    assert result["raw_readings_modified"] is False


def test_revalidation_catches_untrusted_model_copy_before_numerical_work(tmp_path):
    problem = drift_problem()
    invalid = problem.model_copy(
        update={"drift": problem.drift.model_copy(update={"unit": "bad"})}
    )
    with pytest.raises(ValidationError, match="unit must be"):
        fit.run(invalid, fit.Policy(), tmp_path)
