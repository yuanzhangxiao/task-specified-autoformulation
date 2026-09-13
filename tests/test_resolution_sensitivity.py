"""Resolution starvation, sparse derivative parity and distinct recovery trials."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit
from autoformalism.fitting.collocation_mesh import plan_meshes
from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.fitting.feasibility import (
    SensitivityUnavailable,
    recover_refinement,
    restart_points,
)
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.rebuttal.attainability_controls import ordinary_start, system_for
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import unpack_split


def control():
    raw, _ = synthetic_problem("shared", 0.0, 0)
    system, _ = system_for(raw)
    training = unpack_split(raw["splits"]["train"])
    return raw, system, training


def test_input_corner_floor_cannot_starve_constant_input_state_resolution():
    _, system, training = control()
    row = training.trajectories[0]
    t = np.linspace(0, 60, 601)

    def data(name, u):
        return replace(
            row,
            trajectory_id=name,
            time=t,
            targets={"v01": np.sin(t)},
            external_inputs={"u01": u},
        )

    training = DatasetSplit(
        training.name,
        (data("oscillatory_input", np.sin(t)), data("autonomous", np.zeros_like(t))),
        "mesh-floor",
    )
    legacy, _ = plan_meshes(system, training, 200)
    assert len(legacy[1].time) == 2
    meshes, report = plan_meshes(system, training, 200, minimum_intervals=120)
    assert max(np.diff(meshes[1].time)) <= 0.5 + 1e-12
    assert report["target_exceeded_for_resolution"]
    assert report["target_exceeded_for_input_fidelity"]
    assert report["input_interpolation_preserved"]
    again, audit = plan_meshes(system, training, 200, minimum_intervals=120)
    assert report == audit
    assert np.array_equal(again[1].time, meshes[1].time)
    assert sum(len(m.observation_intervals) for m in meshes) == 1202


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_mesh_rejects_invalid_resolution(value):
    _, system, training = control()
    with pytest.raises(ValueError, match="minimum intervals"):
        plan_meshes(system, training, 100, minimum_intervals=value)


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_sparse_augmented_rollout_preserves_values_and_parameter_derivatives(method):
    raw, dense, training = control()
    sparse = SymbolicODE(dense.model, solver_jacobian_format="sparse")
    start = ordinary_start(raw)
    theta = np.array([start[n] for n in dense.names])
    settings = CollocationSensitivityConfig(integration_method=method).fit_config()
    row = training.trajectories[0]
    left = symbolic_rollout(dense, row, theta, settings, None, sensitivities=True)
    right = symbolic_rollout(sparse, row, theta, settings, None, sensitivities=True)
    for a, b in zip(left[:3], right[:3], strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)
    direction = np.linspace(0.3, 0.8, len(theta))
    eps = 1e-4
    high = symbolic_rollout(sparse, row, theta + eps * direction, settings, None)[0]
    low = symbolic_rollout(sparse, row, theta - eps * direction, settings, None)[0]
    np.testing.assert_allclose(
        right[1] @ direction, (high - low) / (2 * eps), rtol=5e-4, atol=1e-5
    )


def test_initial_alias_survives_checkpoint_deduplication():
    start = {"a": 1.0}
    rows = restart_points(
        {"progress": {"checkpoints": {"latest": {"parameters": start}}}},
        start,
        [],
        prioritize_initial=True,
    )
    assert len(rows) == 1
    assert rows[0]["source"] == "ordinary"
    assert rows[0]["sources"] == ["ordinary", "collocation_checkpoint:latest"]


@pytest.mark.parametrize("converged", [False, True])
def test_augmented_retry_uses_distinct_parameter_vector(
    tmp_path, monkeypatch, converged
):
    from autoformalism.fitting import feasibility

    raw, system, training = control()
    start = ordinary_start(raw)
    other = {n: v * 0.8 for n, v in start.items()}
    attempted = []

    def failed(oracle, point, **kwargs):
        attempted.append(point)
        raise SensitivityUnavailable("test first evaluation timeout")

    monkeypatch.setattr(feasibility, "instrumented_fit", failed)
    monkeypatch.setattr(
        feasibility,
        "poll_fit",
        lambda *args, **kwargs: {"message": "test polling skipped"},
    )
    config = CollocationSensitivityConfig(
        recovery_retry_policy="distinct",
        recovery_prioritize_initial=True,
        refinement_seconds=20,
        maximum_function_evaluations=10,
    )
    result = recover_refinement(
        system,
        training,
        1.0,
        config.fit_config(),
        config,
        tmp_path,
        {"success": converged, "parameters": other, "initializer_objective": 1.0},
        start,
        [{"source": "alternative", "parameters": other}],
        "test",
        False,
    )
    assert len(attempted) == 2 and attempted[0] != attempted[1]
    assert result["parameters"] is not None
    assert all(s["evaluation_status"] == "evaluated" for s in result["screens"])


def test_screen_timeout_remains_explicit(tmp_path, monkeypatch):
    from autoformalism.fitting import feasibility, sensitivity_probe

    raw, system, training = control()

    def timeout(*args, **kwargs):
        raise TimeoutError("deliberate point deadline")

    monkeypatch.setattr(sensitivity_probe, "symbolic_rollout", timeout)
    monkeypatch.setattr(feasibility, "poll_fit", lambda *args, **kwargs: {})
    config = CollocationSensitivityConfig(
        refinement_seconds=10, maximum_function_evaluations=2
    )
    result = recover_refinement(
        system,
        training,
        1.0,
        config.fit_config(),
        config,
        tmp_path,
        {"success": False},
        ordinary_start(raw),
        [],
        "timeout",
        False,
    )
    screen = result["screens"][0]
    assert not screen["valid"] and screen["evaluation_status"] == "timeout"
    assert screen["error"] == "deliberate point deadline"
    assert result["parameters"] is None


def test_profile_comparison_uses_output_scale_at_zero_loss():
    from autoformalism.rebuttal.resolution_campaign import profile_differences

    a = {"residual": np.array([0.0, 1e-10]), "jacobian": np.eye(2)}
    b = {"residual": np.array([1e-10, -1e-10]), "jacobian": np.eye(2)}
    r = profile_differences(a, b)
    assert r["maximum_scaled_prediction_difference"] == pytest.approx(2e-10)
    assert r["relative_jacobian_difference"] == 0
    b["residual"][0] = float("nan")
    with pytest.raises(ValueError, match="aligned and finite"):
        profile_differences(a, b)
