"""Mapped equations, full observation objectives and faithful forcing meshes."""

from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

import casadi as ca

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import matching_probe
from autoformalism.fitting.collocation_assembly import mapped_collocation
from autoformalism.fitting.collocation_mesh import plan_meshes
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.rebuttal.fitter_diagnostic import read_json
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.schemas import CandidateModel


def polynomial_problem(inputs=False):
    model = compile_candidate(
        CandidateModel.model_validate(
            {
                "candidate_id": "polynomial",
                "parent_candidate_id": None,
                "states": [{"name": "x", "kind": "latent"}],
                "state_equations": [
                    {"state": "x", "rhs": "a*u01" if inputs else "a*t"}
                ],
                "observation_mappings": [{"channel": "v01", "expression": "x**2"}],
                "parameters": [
                    {"name": "a", "role": "nonnegative_coefficient", "scope": "global"}
                ],
                "initial_conditions": [
                    {"state": "x", "scope": "global", "fixed_value": 1.0}
                ],
            }
        ),
        ValidationContext(targets=("v01",), external_inputs=("u01",) if inputs else ()),
    )
    t = np.linspace(0, 1, 101)
    row = Trajectory(
        "train",
        t,
        {"v01": (1 + t * t) ** 2},
        {},
        {"u01": np.zeros_like(t)} if inputs else {},
        {},
        {},
    )
    return SymbolicODE(model), DatasetSplit(SplitName.TRAIN, (row,), "poly")


def test_coarse_polynomial_scores_every_observation_and_nonlinear_mapping(tmp_path):
    system, training = polynomial_problem()
    row = training.trajectories[0]
    y = row.targets["v01"].copy()
    y[17] += 1.0  # Not a collocation point: it must still contribute to the loss.
    row = Trajectory(row.trajectory_id, row.time, {"v01": y}, {}, {}, {}, {})
    training = DatasetSplit(SplitName.TRAIN, (row,), "modified")
    opti = ca.Opti()
    theta = opti.variable(1)

    def seed(data):
        return np.ones((len(data.time), 1)), {"policy": "rollout_or_observed"}

    objective, nodes, _ = mapped_collocation(
        system,
        opti,
        theta,
        training,
        np.array([2.0]),
        1.0,
        tmp_path,
        monotonic() + 10,
        seed,
        target_variables=3,
    )
    function = ca.Function("check", [opti.x], [objective, opti.g])
    value, constraints = function([2.0, 2.0, 1 + 1 / 9])
    assert nodes == 2
    assert float(value) == pytest.approx(1.0)
    assert np.max(np.abs(constraints)) < 1e-12
    mesh = read_json(tmp_path / "mesh.json")
    assert mesh["trajectories"][0]["observations"] == 101


@pytest.mark.parametrize("magnitude", [1.0, 1e-15])
def test_budget_never_removes_input_pulse(magnitude):
    system, training = polynomial_problem(True)
    row = training.trajectories[0]
    u = np.zeros_like(row.time)
    u[49:52] = magnitude
    row = Trajectory(row.trajectory_id, row.time, row.targets, {}, {"u01": u}, {}, {})
    training = DatasetSplit(SplitName.TRAIN, (row,), "pulse")
    meshes, report = plan_meshes(system, training, target_variables=3)
    assert report["target_exceeded_for_input_fidelity"]
    np.testing.assert_allclose(
        np.interp(row.time, meshes[0].time, np.interp(meshes[0].time, row.time, u)),
        u,
        atol=magnitude * 1e-12,
    )
    meshes2, report2 = plan_meshes(system, training, target_variables=3)
    assert report == report2
    assert np.array_equal(meshes[0].time, meshes2[0].time)


@pytest.mark.parametrize("kind", ["shared", "causal_map", "piecewise"])
def test_mapped_full_mesh_matches_unrolled_objective_constraints_and_derivatives(
    tmp_path, monkeypatch, kind
):
    raw, _ = synthetic_problem(kind, 0.0, 0)
    model = compile_candidate(
        CandidateModel.model_validate(raw["candidate"]),
        ValidationContext.model_validate(raw["context"]),
    )
    model, guesses, _ = apply_initialization_plan(
        model, LatentInitializationPlan.model_validate(raw["initialization_plan"])
    )
    system = SymbolicODE(model, allow_piecewise=True)
    training = unpack_split(raw["splits"]["train"])
    # Use a small prefix to verify the actual production construction twice.
    rows = tuple(
        Trajectory(
            r.trajectory_id,
            r.time[:6],
            {k: v[:6] for k, v in r.targets.items()},
            {},
            {k: v[:6] for k, v in r.external_inputs.items()},
            {},
            {},
        )
        for r in training.trajectories[:2]
    )
    training = DatasetSplit(SplitName.TRAIN, rows, "construction")
    captured = []
    original = ca.Opti

    def factory():
        opti = original()
        captured.append(opti)
        return opti

    def stop(self):
        raise RuntimeError("stop after graph construction")

    monkeypatch.setattr(original, "solve", stop)
    monkeypatch.setattr(matching_probe.ca, "Opti", factory)
    starts = {**raw["start"], **guesses}
    theta = np.array([starts[k] for k in system.names])
    for assembly in ("unrolled", "mapped"):
        matching_probe.latent_start(
            system,
            training=training,
            lower=np.full(len(theta), -np.inf),
            upper=np.full(len(theta), np.inf),
            start=theta,
            scale=1.0,
            settings=CollocationSensitivityConfig().fit_config(),
            method="collocation_init",
            seconds=10,
            directory=tmp_path / assembly,
            node_start="rollout_or_observed",
            warmup_seconds=0.001,
            assembly=assembly,
        )
    assert len(captured) == 2
    x = np.linspace(0.2, 1.2, int(captured[0].nx))
    outputs = []
    for i, opti in enumerate(captured):
        assert int(opti.nx) == len(x)
        fun = ca.Function(
            f"audit{i}",
            [opti.x],
            [opti.f, opti.g, ca.gradient(opti.f, opti.x), ca.jacobian(opti.g, opti.x)],
        )
        outputs.append(fun(x))
    for left, right in zip(*outputs, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)


def test_coarse_initialization_then_full_ode_refinement_recovers(tmp_path):
    raw, _ = synthetic_problem("shared", 0.0, 0)
    model = compile_candidate(
        CandidateModel.model_validate(raw["candidate"]),
        ValidationContext.model_validate(raw["context"]),
    )
    result = fit_collocation_forward_sensitivity(
        model,
        unpack_split(raw["splits"]["train"]),
        unpack_split(raw["splits"]["val"]),
        CollocationSensitivityConfig(
            initializer_seconds=20,
            refinement_seconds=20,
            maximum_function_evaluations=50,
            collocation_assembly="mapped",
            collocation_target_variables=30,
            collocation_node_start="rollout_or_observed",
            collocation_diagnostics=True,
            recovery_policy="feasible",
            recovery_handoff="best_valid",
            sensitivity_invalid_trials="reject",
            least_squares_ftol=None,
        ),
        tmp_path,
        initial_parameters=raw["start"],
        initialization_plan=LatentInitializationPlan.model_validate(
            raw["initialization_plan"]
        ),
    )
    assert result["status"] == "complete", result
    assert result["validation"]["normalized_mse"] < 1e-8
    mesh = read_json(tmp_path / "collocation/mesh.json")
    assert mesh["actual_variables"] <= 30
    assert mesh["all_observations_retained"]
