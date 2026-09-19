"""Fixed equations are evaluated, not optimized; learned initials still require fit."""

import json

import pytest

from autoformalism.fitting import fixed_model
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitRequest
from tests.test_public_fitting import _control


def fixed_control(profile="collocation-single-target-v2"):
    request, train, val = _control(profile)
    data = request.model_dump(mode="json")
    data["base_candidate"]["parameters"] = []
    data["parameter_guesses"] = {}
    data["base_candidate"]["state_equations"] = [
        {"state": "x", "rhs": "-0.7*x + z"},
        {"state": "z", "rhs": "-1.2*z"},
    ]
    data["initialization_plan"]["rules"]["z"]["initial"] = {
        "mode": "map",
        "expression": "0.5*v01",
        "parameters": [],
    }
    return PublicFitRequest.model_validate(data), train, val


@pytest.mark.parametrize(
    "profile",
    ["general-rollout-v1", "collocation-feasible-v1", "collocation-single-target-v2"],
)
def test_fixed_model_rollout_and_resume_without_optimization(
    tmp_path, monkeypatch, profile
):
    from autoformalism.fitting import collocation_sensitivity as collocation

    def forbidden(*args, **kwargs):
        pytest.fail("a fixed model must not enter any optimizer")

    monkeypatch.setattr(public, "fit_candidate", forbidden)
    monkeypatch.setattr(collocation, "fit_collocation_forward_sensitivity", forbidden)
    request, train, val = fixed_control(profile)
    public.prepare_fit(request, train, val, tmp_path)
    result = public.execute_fit(tmp_path)
    assert result.status == "complete", result.message
    assert result.parameters == {}
    assert result.training.normalized_mse < 1e-12
    assert result.validation.normalized_mse < 1e-12
    assert result.actual_residual_calls == 0
    assert result.native_optimizer_converged is None
    assert not result.budget_exhausted
    assert result.independent_replay == "not_performed"
    raw = json.loads((tmp_path / "backend_result.json").read_text())
    assert raw["execution_mode"] == fixed_model.POLICY
    assert raw["optimizer_calls"] == 0
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    monkeypatch.setattr(fixed_model, "evaluate", forbidden)
    public.prepare_fit(request, train, val, tmp_path)
    assert public.execute_fit(tmp_path) == result
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}


def test_no_equation_parameters_with_fitted_initial_still_optimizes(
    tmp_path, monkeypatch
):
    from autoformalism.fitting import collocation_sensitivity as collocation

    request, train, val = fixed_control()
    data = request.model_dump(mode="json")
    data["initialization_plan"]["rules"]["z"]["initial"] = {
        "mode": "map",
        "expression": "scale*v01",
        "parameters": [{"name": "scale", "role": "coefficient", "guess": 0.5}],
    }
    request = PublicFitRequest.model_validate(data)
    seen = []

    def fit(model, *args, **kwargs):
        seen.append(model.parameter_names)
        raise RuntimeError("reached nonempty optimization route")

    monkeypatch.setattr(collocation, "fit_collocation_forward_sensitivity", fit)
    monkeypatch.setattr(fixed_model, "evaluate", lambda *a: pytest.fail("wrong bypass"))
    public.prepare_fit(request, train, val, tmp_path)
    result = public.execute_fit(tmp_path)
    assert seen == [("init_z_scale",)]
    assert "reached nonempty" in result.message


def test_invalid_fixed_rollout_is_unavailable_not_a_penalty_score(tmp_path):
    request, train, val = fixed_control()
    data = request.model_dump(mode="json")
    data["base_candidate"]["state_equations"][0]["rhs"] = "x*x"
    public.prepare_fit(PublicFitRequest.model_validate(data), train, val, tmp_path)
    result = public.execute_fit(tmp_path)
    assert result.status == "fit_failed"
    assert result.parameters == {}
    assert not result.training.available
    assert result.training.normalized_mse is None
    assert result.training.failed_trajectories
    assert result.actual_residual_calls == 0


def test_fixed_rollout_deadline_is_not_a_score(tmp_path, monkeypatch):
    original = public.profile_settings
    monkeypatch.setattr(
        public,
        "profile_settings",
        lambda r: {
            **original(r),
            "maximum_wall_time_seconds": 1e-12,
        },
    )
    request, train, val = fixed_control("general-rollout-v1")
    public.prepare_fit(request, train, val, tmp_path)
    result = public.execute_fit(tmp_path)
    assert result.status == "fit_failed"
    assert result.budget_exhausted
    assert not result.training.available and not result.validation.available


@pytest.mark.parametrize(
    "target,expression,forcing,equilibrium",
    [
        ("v01", "(u01) + (-v01)", {"u01": 2.0}, 2.0),
        (
            "T",
            "(Cf*Tf) - (Cf*(T-T)) + (C) + (Tj) - (T)",
            {"Cf": 2.0, "Tf": 3.0, "Tjf": 1.0, "C": 4.0, "Tj": 5.0},
            15.0,
        ),
    ],
)
def test_reported_equations_have_analytic_fixed_rollouts(
    tmp_path, target, expression, forcing, equilibrium
):
    import numpy as np

    from autoformalism.schemas.public_fitting import PublicSplit

    inputs = {k: v for k, v in forcing.items() if k not in ("C", "Tj")}
    auxiliaries = {k: v for k, v in forcing.items() if k in ("C", "Tj")}
    candidate = {
        "candidate_id": "fixed_reported_equation",
        "parent_candidate_id": None,
        "states": [{"name": target, "kind": "observed"}],
        "state_equations": [{"state": target, "rhs": expression}],
        "observation_mappings": [{"channel": target, "expression": target}],
        "initial_conditions": [{"state": target, "scope": "global", "fixed_value": 0}],
    }
    request = PublicFitRequest.model_validate(
        {
            "base_candidate": candidate,
            "context": {
                "targets": [target],
                "external_inputs": list(inputs),
                "auxiliaries": list(auxiliaries),
            },
            "initialization_plan": {"rules": {}},
            "profile": "collocation-single-target-v2",
            "source": {
                "stage": "synthetic_control",
                "task_id": "reported-equation",
                "artifact_sha256": public.content_sha256(candidate),
            },
        }
    )
    t = np.linspace(0, 2, 21)

    def split(name, initial):
        return PublicSplit.model_validate(
            {
                "name": name,
                "fingerprint": name,
                "rows": [
                    {
                        "trajectory_id": name,
                        "time": t.tolist(),
                        "targets": {
                            target: (
                                equilibrium + (initial - equilibrium) * np.exp(-t)
                            ).tolist()
                        },
                        "external_inputs": {k: [v] * len(t) for k, v in inputs.items()},
                        "auxiliaries": {
                            k: [v] * len(t) for k, v in auxiliaries.items()
                        },
                    }
                ],
            }
        )

    public.prepare_fit(request, split("train", 1), split("val", 4), tmp_path)
    result = public.execute_fit(tmp_path)
    assert result.status == "complete", result.message
    assert result.parameters == {} and result.actual_residual_calls == 0
    assert result.training.normalized_mse < 1e-12
    assert result.validation.normalized_mse < 1e-12
