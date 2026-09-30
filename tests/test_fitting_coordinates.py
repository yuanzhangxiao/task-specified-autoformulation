"""Affine equivalence, bounds, causal initialization and native solver regression."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.fitting.coordinates import (
    AffineAxis,
    NumericalCoordinates,
    coordinate_least_squares,
    training_coordinates,
)
from autoformalism.fitting.public_fitting import _lower
from autoformalism.schemas.public_fitting import PublicFitRequest


def small_problem():
    request = PublicFitRequest.model_validate(
        {
            "base_candidate": {
                "candidate_id": "signed_coordinate_smoke",
                "parent_candidate_id": None,
                "states": [{"name": n, "kind": "latent"} for n in ("z", "y")],
                "state_equations": [
                    {"state": "z", "rhs": "-rate*z+u"},
                    {"state": "y", "rhs": "gain*z-rate*(y-350)"},
                ],
                "observation_mappings": [{"channel": "output", "expression": "y"}],
                "parameters": [
                    {"name": n, "scope": "global", "role": "nonnegative_coefficient"}
                    for n in ("rate", "gain")
                ],
                "initial_conditions": [
                    {"state": n, "scope": "global", "fixed_value": 0}
                    for n in ("z", "y")
                ],
            },
            "context": {"targets": ["output"], "external_inputs": ["u"]},
            "initialization_plan": {
                "rules": {"z": {"initial": {"mode": "value", "guess": -1.9}}}
            },
            "parameter_guesses": {"rate": 0.6, "gain": 1.2},
            "profile": "collocation-feasible-v1",
            "source": {
                "stage": "synthetic_control",
                "task_id": "smoke",
                "artifact_sha256": "a" * 64,
            },
        }
    )
    splits = []
    for split in (SplitName.TRAIN, SplitName.VALIDATION):
        rows = []
        for i, amplitude in enumerate(
            (0.5, 1.0) if split is SplitName.TRAIN else (0.8,)
        ):
            t = np.linspace(0, 3, 16)
            rate, gain, z0 = 0.7, 1.3, -2.0
            y0 = 350 + i
            e = np.exp(-rate * t)
            y = (
                350
                + (y0 - 350) * e
                + gain * z0 * t * e
                + gain * amplitude / rate * ((1 - e) / rate - t * e)
            )
            rows.append(
                Trajectory(
                    trajectory_id=f"{split.value}_{i}",
                    time=t,
                    targets={"output": y},
                    auxiliaries={},
                    external_inputs={"u": np.full_like(t, amplitude)},
                    fixed_covariates={},
                    derivatives={},
                )
            )
        splits.append(DatasetSplit(split, tuple(rows), split.value))
    model, start, _ = _lower(request)
    return model, start, *splits


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_transform_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        AffineAxis(center=0, scale=scale)


def test_design_uses_training_and_keeps_signed_latents():
    model, start, train, val = small_problem()
    coord = training_coordinates(model, train, start)
    assert coord.states["z"].center == -1.9
    assert coord.states["y"].center > 300
    with pytest.raises(ValueError, match="training only"):
        training_coordinates(model, val, start)
    with pytest.raises(ValueError, match="symbols"):
        coord.arrays("parameters", ["misspelled"])


def test_least_squares_chain_rule_bounds_and_physical_callbacks(monkeypatch):
    from autoformalism.fitting import coordinates as module

    coord = NumericalCoordinates(
        parameters={"p": AffineAxis(center=300, scale=20)},
        states={"x": AffineAxis(center=0, scale=1)},
        provenance={},
    )
    seen = []

    def solver(fun, start, *, jac, bounds, callback, **kwargs):
        assert start == pytest.approx([0])
        assert bounds[0] == pytest.approx([-15])
        assert bounds[1] == pytest.approx([5])
        assert fun(np.array([1.0])) == pytest.approx([10])
        np.testing.assert_allclose(jac(np.array([1.0])), [[20]])
        result = OptimizeResult(
            x=np.array([0.5]),
            jac=np.array([[20.0]]),
            grad=np.array([40.0]),
            cost=0,
            nfev=1,
            optimality=40,
        )
        callback(result)
        return result

    monkeypatch.setattr(module, "least_squares", solver)
    result = coordinate_least_squares(
        lambda x: x - 310,
        np.array([300.0]),
        jac=lambda x: np.ones((1, 1)),
        bounds=(np.array([0.0]), np.array([400.0])),
        callback=lambda r: seen.append(r.x.copy()),
        coordinates=coord,
        names=["p"],
    )
    assert result.x == pytest.approx([310])
    np.testing.assert_allclose(result.jac, [[1]])
    assert result.grad == pytest.approx([2])
    assert seen[0] == pytest.approx([310])
    assert result.optimality == 40  # native scaled-coordinate certificate


def test_scaled_joint_fit_free_rollout_and_physical_checkpoints(tmp_path):
    pytest.importorskip("casadi")
    from autoformalism.fitting.collocation_sensitivity import (
        CollocationSensitivityConfig,
        fit_collocation_forward_sensitivity,
    )
    from autoformalism.rebuttal.fitter_diagnostic import read_json

    model, start, train, val = small_problem()
    coords = training_coordinates(model, train, start)
    result = fit_collocation_forward_sensitivity(
        model,
        train,
        val,
        CollocationSensitivityConfig(
            initializer_seconds=20,
            refinement_seconds=20,
            maximum_function_evaluations=50,
            recovery_policy="feasible",
            collocation_diagnostics=True,
            collocation_node_start="rollout_or_observed",
        ),
        tmp_path,
        initial_parameters=start,
        numerical_coordinates=coords,
    )
    assert result["status"] == "complete", result
    assert result["training"]["normalized_mse"] < 1e-6
    assert result["validation"]["normalized_mse"] < 1e-6
    assert result["parameters"]["init_z_value"] < -1.5
    assert not result["validation_used_for_fitting"]
    assert result["refinement"]["native_optimality_coordinates"] == "scaled"
    progress = read_json(tmp_path / "collocation/progress.json")
    assert progress["checkpoints"]
    assert all(
        set(p["parameters"]) == set(start) for p in progress["checkpoints"].values()
    )
