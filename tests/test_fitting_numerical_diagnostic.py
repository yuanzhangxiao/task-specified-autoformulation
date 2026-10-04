"""Reduced meshes preserve data; restart decisions never use validation or truth."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.fitting import adaptive_mesh as mesh
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import numerical_diagnostic as d
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import transcription_campaign as campaign
from autoformalism.fitting import transcription_fit as fit
from autoformalism.fitting import transcription_solver as solver
from autoformalism.fitting.collocation_mesh import plan_meshes
from autoformalism.fitting.identifiable_campaign import scale_for
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from tests.test_fitting_strategies import small_payload


@pytest.fixture(scope="module")
def inputs():
    return controls.make_inputs()


def reduced(payload, target=20):
    model, _, _ = public._lower(PublicFitRequest.model_validate(payload["request"]))
    training = public.unpack_split(PublicSplit.model_validate(payload["training"]))
    planned, audit = plan_meshes(
        SymbolicODE(model), training, target, minimum_intervals=2
    )
    for row, grid in zip(training.trajectories, planned, strict=True):
        key = row.trajectory_id
        payload["nodes"][key] = mesh.interpolate_nodes(
            payload["meshes"][key], payload["nodes"][key], grid.time
        ).tolist()
        payload["meshes"][key] = grid.time.tolist()
    payload["collocation_dense_output"] = True
    return audit


def test_reduced_mesh_all_samples_contribute_without_new_variables(inputs, tmp_path):
    payload = small_payload(inputs, "collocation")
    dense = solver.build_problem(payload, tmp_path / "dense")
    audit = reduced(payload)
    problem = solver.build_problem(payload, tmp_path / "coarse")
    assert problem.opti.nx == audit["actual_variables"] < dense.opti.nx
    assert problem.observation_count == dense.observation_count == 21

    def value(expression):
        return problem.opti.debug.value(expression, problem.opti.initial())

    # Linear physical node guesses produce a linear Radau polynomial; every
    # interior observation must contribute to the objective with the same weight.
    row = problem.layout.training.trajectories[0]
    states = np.asarray(payload["nodes"][row.trajectory_id])
    time = payload["meshes"][row.trajectory_id]
    expected = np.interp(row.time, time, states[:, 0])
    scale = scale_for(problem.layout.training)["y"]
    assert float(value(problem.objective)) == pytest.approx(
        np.mean(((expected - row.targets["y"]) / scale) ** 2)
    )
    assert audit["input_interpolation_preserved"]
    assert audit["all_observations_retained"]


def test_reduced_mesh_cannot_hide_a_short_input_pulse(inputs, tmp_path):
    payload = small_payload(inputs, "collocation")
    row = payload["training"]["rows"][0]
    row["external_inputs"]["u"] = [0.0] * len(row["time"])
    row["external_inputs"]["u"][9] = 1.0
    payload["training"]["fingerprint"] = public.content_sha256(
        payload["training"]["rows"]
    )
    key = next(iter(payload["meshes"]))
    payload["meshes"][key] = [row["time"][0], row["time"][-1]]
    payload["nodes"][key] = [payload["nodes"][key][0], payload["nodes"][key][-1]]
    payload["collocation_dense_output"] = True
    with pytest.raises(ValueError, match="input interpolation"):
        solver.build_problem(payload, tmp_path)


@pytest.mark.parametrize("hessian", ["exact", "limited-memory"])
def test_native_shooting_profile_and_hessian_options(inputs, tmp_path, hessian):
    payload = small_payload(inputs, "shooting")
    payload.update(hessian_approximation=hessian, profile_derivatives=True, seconds=30)
    result = solver.solve(payload, tmp_path)
    assert result["native_success"]
    profile = public._read(tmp_path / "derivative_profile.json")["phases"]
    assert profile["objective"]["status"] == profile["jacobian"]["status"] == "complete"
    assert profile["hessian"]["status"] == (
        "complete" if hessian == "exact" else "not_requested"
    )
    assert "t_wall_nlp_f" in result["solve_chunks"][0]["native_function_statistics"]
    assert public._read(tmp_path / "layout.json")["hessian_approximation"] == hessian


def test_reduced_native_solve_and_invalid_options(inputs, tmp_path):
    payload = small_payload(inputs, "collocation")
    reduced(payload, 40)
    assert solver.solve(payload, tmp_path)["native_success"]
    payload["hessian_approximation"] = "guess"
    with pytest.raises(ValueError, match="Hessian"):
        solver.build_problem(payload, tmp_path)


def test_frozen_six_way_pairs_and_training_only_worker(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.controls, "make_inputs", lambda: deepcopy(inputs))
    config = campaign.CampaignConfig(
        starts=1,
        include_cstr=False,
        cases=("linear",),
        numerical_diagnostic=d.NumericalDiagnosticPolicy(),
    )
    campaign.prepare(tmp_path, config)
    plan, frozen = campaign.verify(tmp_path)
    assert plan["protocol"] == d.PROTOCOL
    assert len(plan["tasks"]) == 6
    common = plan["commons"]["linear_s0"]
    for task in plan["tasks"]:
        payload = campaign.worker_payload(plan, frozen, task)
        assert set(payload) == {
            "request",
            "coordinates",
            "nodes",
            "training",
            "arm",
            "policy",
            "numerical_diagnostic",
            "seed",
        }
        assert payload["request"] == common["request"]
    campaign.prepare(tmp_path, config)  # deterministic plan resume
    with pytest.raises(ValueError, match="separate"):
        campaign.CampaignConfig(include_reuse=True, numerical_diagnostic={})
    with pytest.raises(ValueError, match="reserve"):
        campaign.CampaignConfig(
            numerical_diagnostic={}, strategy={"maximum_rollout_calls": 41}
        )


def history(values):
    return [{"call": i + 1, "best_nmse": v} for i, v in enumerate(values)]


def test_stagnation_requires_sufficient_complete_training_history():
    policy = d.NumericalDiagnosticPolicy()
    assert not d.stagnation(history([1] * 19), policy)["triggered"]
    assert not d.stagnation(history(np.linspace(1, 0.1, 25)), policy)["triggered"]
    assert d.stagnation(history([1] * 25), policy)["triggered"]
    with pytest.raises(ValueError, match="history"):
        d.stagnation(history([float("nan")] * 25), policy)


def rollout_payload(inputs, arm):
    payload = small_payload(inputs, "collocation")
    return {k: payload[k] for k in ("request", "training", "coordinates", "nodes")} | {
        "arm": arm,
        "policy": {"seconds": 60, "maximum_rollout_calls": 90},
        "numerical_diagnostic": {},
        "seed": 2,
    }


@pytest.mark.parametrize("arm", ["rollout_continue", "rollout_restart"])
def test_two_phase_call_budget_and_incumbent_survives_worse_restart(
    inputs, tmp_path, monkeypatch, arm
):
    starts, budgets = [], []
    original = None

    def fake_refine(
        system,
        training,
        scales,
        settings,
        coordinates,
        points,
        policy,
        mode,
        folder,
        **kwargs,
    ):
        nonlocal original
        starts.append(points[0]["parameters"])
        budgets.append(policy.maximum_calls)
        original = original or starts[0]
        public._write(folder / "training_history.json", history([0.2] * 25))
        return {
            "parameters": points[0]["parameters"],
            "training_nmse": 0.2 if len(starts) == 1 else 0.5,
            "actual_residual_calls": 41 if len(starts) == 1 else 10,
            "stop_reason": "budget_exhausted",
        }

    monkeypatch.setattr(d, "refine", fake_refine)
    result = d.rollout(rollout_payload(inputs, arm), tmp_path)
    assert budgets == [41, 49]
    assert result["actual_residual_calls"] == 51
    assert result["parameters"] == original
    assert result["training_nmse"] == 0.2
    assert (starts[1] != starts[0]) == (arm == "rollout_restart")
    assert result["decision"]["used"] == (arm == "rollout_restart")


def test_accurate_point_does_not_restart(inputs, tmp_path, monkeypatch):
    calls = []

    def fake_refine(*args, **kwargs):
        calls.append(1)
        return {
            "parameters": args[5][0]["parameters"],
            "training_nmse": 1e-15,
            "actual_residual_calls": 2,
            "stop_reason": "training_accuracy_reached_pending_replay",
        }

    monkeypatch.setattr(d, "refine", fake_refine)
    result = d.rollout(rollout_payload(inputs, "rollout_restart"), tmp_path)
    assert len(calls) == 1 and result["actual_residual_calls"] == 2
    assert not result["decision"]["triggered"]


def test_partial_recovery_keeps_better_second_phase_without_budget_reset(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        fit, "invoke", lambda *args: {"wall_timeout": True, "elapsed_seconds": 60}
    )
    for name, loss in (("refinement-0", 0.2), ("refinement-1", 0.1)):
        (tmp_path / name).mkdir()
        public._write(
            tmp_path / name / "best.json",
            {"training_nmse": loss, "parameters": {"a": loss}},
        )
    result = fit.run(
        {
            "policy": {"seconds": 60},
            "arm": "rollout_restart",
            "numerical_diagnostic": {},
        },
        tmp_path,
    )
    assert result["training_nmse"] == 0.1
    assert result["actual_residual_calls"] is None
    assert result["budget_exhausted"]


def test_fixed_mesh_does_not_refine_after_timeout(inputs, tmp_path, monkeypatch):
    payload = rollout_payload(inputs, "fixed_reduced_collocation")
    calls = []

    def timeout(payload, folder, seconds, **kwargs):
        calls.append(payload)
        assert payload["collocation_dense_output"]
        return {"wall_timeout": True, "elapsed_seconds": seconds}

    monkeypatch.setattr(fit, "invoke", timeout)
    result = fit.fit(payload, tmp_path)
    assert len(calls) == 1
    assert result["stop_reason"] == "native_stage_unavailable"
    assert result["parameters"] is not None  # screened generic incumbent retained
    assert public._read(tmp_path / "mesh_audit.json")["all_observations_retained"]
