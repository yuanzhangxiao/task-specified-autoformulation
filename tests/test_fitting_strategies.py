"""Mesh/input invariants, native transcriptions and campaign provenance."""

from copy import deepcopy
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import adaptive_mesh as m
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import nonlinear_shape_case as shape
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import transcription_campaign as c
from autoformalism.fitting import transcription_fit as f
from autoformalism.fitting import transcription_solver as solver
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.fitting.matching_probe import observed_node_guess
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas.public_fitting import PublicSplit


@pytest.fixture(scope="module")
def inputs():
    return controls.make_inputs()


@pytest.fixture
def plan(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(c.controls, "make_inputs", lambda: deepcopy(inputs))
    c.prepare(tmp_path, c.CampaignConfig(starts=1, include_cstr=False))
    return tmp_path


def test_nonlinear_shape_is_identifiable_but_not_affine(inputs):
    case = shape.make_case(inputs["cases"]["nonlinear"])
    assert case["identifiability"]["passed"]
    assert case["identifiability"]["singular_ratio"] > 0.02
    system = SymbolicODE(public._lower(shape.request(0))[0])
    assert not system.rhs_affine
    for split in ("training", "validation"):
        for row in case[split]["rows"]:
            assert set(row["targets"]) == {"y"}
            assert row["auxiliaries"] == {}


def test_physical_start_does_not_depend_on_private_coefficients(monkeypatch):
    expected = shape.request(0)
    monkeypatch.setattr(shape, "TRUTH", {"q": 987})
    assert shape.request(0) == expected


def test_radau_polynomial_and_derivative():
    def p(t):
        return np.array([2 + 3 * t - 4 * t * t, -1 + t * t])

    for t in (0, 0.15, 1 / 3, 0.55, 0.85, 1):
        assert m.polynomial(p(0), p(1 / 3), p(1), t) == pytest.approx(p(t))
        assert m.polynomial_derivative(p(0), p(1 / 3), p(1), t) == pytest.approx(
            [3 - 8 * t, 2 * t]
        )


def test_shooting_retains_short_pulse_and_all_observations():
    times = np.linspace(0, 10, 101)
    inputs = np.zeros(101)
    inputs[37:40] = [0.2, 1, 0.2]
    row = SimpleNamespace(
        time=times, external_inputs={"u": inputs}, targets={"y": np.zeros(101)}
    )
    boundaries = m.initial_mesh(row, "shooting")
    assert len(boundaries) < len(times)
    propagation = np.unique(
        np.concatenate(
            [m.integration_times(times, a, b) for a, b in pairwise(boundaries)]
        )
    )
    assert set(times).issubset(propagation)
    assert np.trapezoid(
        np.interp(propagation, times, inputs), propagation
    ) == pytest.approx(np.trapezoid(inputs, times))
    assert m.initial_mesh(row, "collocation") == times.tolist()


def test_mesh_refines_worst_intervals_and_preserves_every_boundary():
    assert m.refine_mesh(
        [0, 1, 2, 3, 4], [0, 2, 8, 0], threshold=1, maximum_intervals=8
    ) == [0, 1, 2, 2.5, 3, 4]
    assert m.refine_mesh([0, 1], [99], threshold=1, maximum_intervals=1) == [0, 1]
    with pytest.raises(ValueError, match="indicators"):
        m.refine_mesh([0, 1], [float("nan")], threshold=1, maximum_intervals=8)
    with pytest.raises(ValueError, match="mesh"):
        m.validate_mesh([0, 1], [0, 0.5, 0.4, 1])


def test_forcing_is_not_approximated_over_shooting_window(inputs):
    model, _, _ = public._lower(controls.request("linear", 0))
    system = SymbolicODE(model)
    integrate = solver.interval_integrator(system, 1e-9)
    x = np.array([0.2, 0.4])
    theta = np.array(
        [dict(controls.TRUTHS["linear"], init_z_value=0.4)[n] for n in system.names]
    )
    times = [0.0, 0.49, 0.5, 0.51, 1.0]
    values = [0, 0, 5, 0, 0]
    for i, (a, b) in enumerate(pairwise(times)):
        x = np.asarray(
            integrate(x0=x, p=np.r_[theta, a, b - a, values[i], values[i + 1]])["xf"]
        ).ravel()
    truth = controls.reference(
        {"time": times, "targets": {"y": [0.2] * 5}, "external_inputs": {"u": values}},
        dict(controls.TRUTHS["linear"], init_z_value=0.4),
    )
    assert x == pytest.approx(truth[-1], abs=1e-7)


def test_frozen_pairs_and_worker_no_private_data(plan):
    p, inputs = c.verify(plan)
    assert len(p["tasks"]) == 16
    for common in p["commons"]:
        tasks = [t for t in p["tasks"] if t["common"] == common]
        assert len(tasks) == 4
        payloads = [c.worker_payload(p, inputs, t) for t in tasks]
        for payload in payloads:
            assert set(payload) == {
                "request",
                "coordinates",
                "nodes",
                "training",
                "arm",
                "policy",
            }
            assert payload["nodes"] == payloads[0]["nodes"]
            assert payload["request"] == payloads[0]["request"]
    with pytest.raises(ValueError, match="configuration"):
        c.prepare(plan, c.CampaignConfig(starts=2, include_cstr=False))


def test_resume_never_restarts_native_budget(plan, monkeypatch):
    p, _ = c.verify(plan)
    seal(
        plan / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(p)},
    )
    directory = plan / "results" / p["tasks"][0]["task_id"]
    seal(directory / "started.json", {"started": True})
    monkeypatch.setattr(c, "run", lambda *a: pytest.fail("budget reset"))
    result = c.run_task(plan, 0)
    assert result["status"] == "interrupted"
    assert c.run_task(plan, 0) == result
    assert c.report(plan)["recorded"] == 1


def test_source_drift_and_failed_qualification_block_execution(plan, monkeypatch):
    p, _ = c.verify(plan)
    seal(
        plan / "qualification/result.json",
        {"passed": False, "plan_sha256": public.content_sha256(p)},
    )
    with pytest.raises(ValueError, match="qualification"):
        c.run_task(plan, 0)
    monkeypatch.setattr(public, "_source_identity", lambda: "different")
    with pytest.raises(ValueError, match="source/runtime"):
        c.verify(plan)
    assert c.report(plan)["status"] == "incomplete"


def small_payload(inputs, method):
    request = controls.request("linear", 0)
    data = deepcopy(inputs["cases"]["linear"]["training"])
    data["rows"] = data["rows"][:1]
    row = data["rows"][0]
    row["time"] = row["time"][:21]
    row["targets"]["y"] = row["targets"]["y"][:21]
    row["external_inputs"]["u"] = row["external_inputs"]["u"][:21]
    data["fingerprint"] = public.content_sha256(data["rows"])
    train = public.unpack_split(PublicSplit.model_validate(data))
    model, start, _ = public._lower(request)
    system = SymbolicODE(model)
    vector = np.array([start[n] for n in system.names])
    grids = {r.trajectory_id: m.initial_mesh(r, method) for r in train.trajectories}
    nodes = {
        r.trajectory_id: m.interpolate_nodes(
            r.time, observed_node_guess(system, r, vector), grids[r.trajectory_id]
        ).tolist()
        for r in train.trajectories
    }
    return {
        "request": request.model_dump(mode="json"),
        "training": data,
        "start": start,
        "coordinates": training_coordinates(model, train, start).model_dump(
            mode="json"
        ),
        "meshes": grids,
        "nodes": nodes,
        "method": method,
        "tolerance": 1e-9,
        "seconds": 15,
    }


@pytest.mark.parametrize("method", ["collocation", "shooting"])
def test_native_transcription_fits_latent_initial_and_keeps_samples(
    inputs, tmp_path, method
):
    payload = small_payload(inputs, method)
    result = solver.solve(payload, tmp_path)
    assert result["native_success"], result
    layout = public._read(tmp_path / "layout.json")
    assert layout["observation_residuals"] == 21
    saved = public._read(tmp_path / "checkpoints.json")["latest"]
    assert saved["maximum_scaled_defect"] < 1e-6
    assert saved["collocation_nmse"] < 1e-6
    assert "init_z_value" in saved["parameters"]


def test_validation_and_bad_boundary_rejected_before_native_solver(inputs, tmp_path):
    payload = small_payload(inputs, "collocation")
    payload["training"]["name"] = "val"
    with pytest.raises(ValueError, match="training"):
        solver.solve(payload, tmp_path)
    payload["training"]["name"] = "train"
    key = next(iter(payload["nodes"]))
    payload["nodes"][key][0][1] += 1
    with pytest.raises(ValueError, match="boundary"):
        solver.solve(payload, tmp_path)


def test_deadline_process_recovery_and_no_false_finite_endpoint(tmp_path):
    # A nonexistent worker arm fails. An exit is not silently called convergence.
    result = f.run({"policy": {"seconds": 5}, "arm": "invalid"}, tmp_path)
    assert result["parameters"] is None
    assert result["stop_reason"] == "worker_failed"
    assert result["process"]["returncode"] != 0
    assert result["total_seconds"] < 7


def test_offnode_defect_detects_dynamics_even_when_observed_flat():
    system = SimpleNamespace(
        model=None, inputs=(), rhs=lambda t, x, theta, u: np.array([0, 1])
    )
    from unittest.mock import patch

    with patch(
        "autoformalism.fitting.simulation.trajectory_forcing", return_value=None
    ):
        scores = m.collocation_indicators(
            system,
            None,
            np.array([]),
            np.zeros((2, 2)),
            np.zeros((1, 2)),
            [0, 1],
            np.ones(2),
        )
    assert scores == [1]


def test_submission_uncertain_receipt_never_retries(tmp_path, monkeypatch):
    import sys

    from scripts import submit_phase_c_fitting_strategies as submit

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(submit.q, "prepare", lambda *a: None)
    monkeypatch.setattr(submit.q, "verify", lambda *a: ({"tasks": [{}]}, {}))
    calls = []

    def uncertain(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="\n", stderr="Socket timed out")

    monkeypatch.setattr(submit.subprocess, "run", uncertain)
    monkeypatch.setattr(submit.subprocess, "check_output", lambda *a, **kw: "frozen\n")
    config = tmp_path / "config.json"
    config.write_text('{"include_cstr": false}')
    root = tmp_path / "campaign"
    root.mkdir()
    with pytest.raises(ValueError, match="unconfirmed"):
        submit.submit(root, config, account="cpu-test", concurrency=2)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Unconfirmed"):
        submit.submit(root, config, account="cpu-test", concurrency=2)
    assert len(calls) == 1


def test_checkpoint_recovery_uses_only_training_verified_incumbents(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        f, "invoke", lambda *a: {"wall_timeout": True, "elapsed_seconds": 5}
    )
    (tmp_path / "refinement").mkdir()
    (tmp_path / "mesh-0").mkdir()
    public._write(
        tmp_path / "refinement/best.json",
        {"parameters": {"a": 1}, "training_nmse": 0.02},
    )
    public._write(
        tmp_path / "best.json", {"parameters": {"a": 2}, "training_nmse": 0.03}
    )
    # A zero collocation loss at infeasible nodes is not a verified incumbent.
    public._write(
        tmp_path / "mesh-0/checkpoints.json",
        {"latest": {"parameters": {"a": 999}, "collocation_nmse": 0}},
    )
    result = f.run({"policy": {"seconds": 5}, "arm": "collocation_rollout"}, tmp_path)
    assert result["parameters"] == {"a": 1}
    assert result["stop_reason"] == "wall_budget_exhausted"
    assert result["budget_exhausted"]


def test_process_deadline_covers_startup(tmp_path):
    record = f.invoke({}, tmp_path, 0.001)
    assert record["wall_timeout"]
    assert record["elapsed_seconds"] < 3
    assert record["returncode"] is not None


def test_keyboard_interrupt_terminates_native_descendants(tmp_path, monkeypatch):
    killed = []

    class InterruptedProcess:
        pid = 123456789
        returncode = None

        def wait(self, *, timeout):
            if not killed:
                raise KeyboardInterrupt
            self.returncode = -9

    monkeypatch.setattr(f.subprocess, "Popen", lambda *a, **kw: InterruptedProcess())
    monkeypatch.setattr(f.os, "killpg", lambda pid, sig: killed.append(pid))
    with pytest.raises(KeyboardInterrupt):
        f.invoke({}, tmp_path, 5)
    assert killed == [123456789]
    assert public._read(tmp_path / "process.json")["interrupted"]
