"""Public handoff integrity, initialization boundaries and honest fit evidence."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_fit_smoke", ROOT / "scripts/smoke_public_fitting.py"
)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def _write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def _control(profile="general-rollout-v1", **kwargs):
    return SMOKE.control(profile, **kwargs)


def _raw(score=0.2, *, failed=False):
    metric = {
        "normalized_mse": score,
        "per_target_normalized_mse": {"v01": score},
        "failed_trajectories": ["train_0"] if failed else [],
    }
    return {
        "global_parameters": {"a": 0.7, "b": 1.2, "init_z_scale": 0.5},
        "training_metrics": metric,
        "validation_metrics": metric,
        "best_start_index": 0,
        "diagnostics": [
            {
                "start_index": 0,
                "success": False,
                "status": 0,
                "actual_residual_evaluations": 3,
            }
        ],
    }


def test_public_lowering_provenance_and_unchanged_resume(tmp_path, monkeypatch):
    request, train, val = _control()
    prepared = public.prepare_fit(request, train, val, tmp_path)
    frozen = json.loads((tmp_path / "freeze.json").read_text())
    lowered = frozen["lowered_candidate"]
    assert {i["state"]: i["expression"] for i in lowered["initial_conditions"]} == {
        "x": "v01",
        "z": "init_z_scale * v01",
    }
    assert frozen["initial_parameters"] == {"a": 0.7, "b": 1.2, "init_z_scale": 0.5}

    calls = []

    def backend(req, model, training, validation, guesses, settings, directory):
        calls.append((training, validation))
        assert training.name.value == "train" and validation.name.value == "val"
        assert all(not row.derivatives for row in training.trajectories)
        assert training.fingerprint == public.content_sha256(train)
        assert validation.fingerprint == public.content_sha256(val)
        assert len(model.parameter_names) == 3  # initializer lowered exactly once
        assert guesses["init_z_scale"] == 0.5
        assert settings["maximum_wall_time_seconds"] == 300
        return _raw()

    monkeypatch.setattr(public, "_run_backend", backend)
    result = public.execute_fit(tmp_path)
    assert result.status == "complete"
    assert result.training.normalized_mse == 0.2
    assert result.native_optimizer_converged is False
    assert result.budget_exhausted is True
    assert result.independent_replay == "not_performed"
    assert result.identity == prepared["identity"]
    assert result.request_sha256 == public.content_sha256(request)
    assert result.lowered_candidate_sha256 == public.content_sha256(lowered)
    assert result.initialization_plan_sha256 == public.content_sha256(
        request.initialization_plan
    )
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.glob("*.json")
    }
    public.prepare_fit(request, train, val, tmp_path)
    assert public.execute_fit(tmp_path) == result
    assert before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.glob("*.json")
    }
    assert len(calls) == 1


@pytest.mark.parametrize("score,failed", [(1e12, True), (None, False), (-1, False)])
def test_failure_penalty_is_not_an_available_nmse(tmp_path, monkeypatch, score, failed):
    request, train, val = _control()
    public.prepare_fit(request, train, val, tmp_path)
    monkeypatch.setattr(public, "_run_backend", lambda *a: _raw(score, failed=failed))
    result = public.execute_fit(tmp_path)
    assert result.status == "fit_failed"
    assert result.training.normalized_mse is None
    assert not result.training.available
    assert result.training.per_target_normalized_mse == {}


def test_collocation_native_placeholder_is_not_convergence_evidence():
    request, _, _ = _control("collocation-feasible-v1")
    raw = {
        "parameters": {"a": 1},
        "training": _raw()["training_metrics"],
        "validation": _raw()["validation_metrics"],
        "refinement": {
            "optimizer_native_success": False,
            "budget_exhausted": True,
            "actual_residual_calls": 9,
        },
    }
    result = public._evidence(request, raw)
    assert result["native_optimizer_converged"] is None
    assert result["actual_residual_calls"] == 9
    assert result["budget_exhausted"]


def test_multi_target_collocation_is_capability_failure_without_fallback(
    tmp_path, monkeypatch
):
    request, train, val = _control("collocation-feasible-v1", multiple_targets=True)
    public.prepare_fit(request, train, val, tmp_path)

    def forbidden(*args):
        pytest.fail("unsupported request cannot run any optimizer")

    monkeypatch.setattr(public, "_run_backend", forbidden)
    assert not public.inspect_fit(tmp_path)["capability_supported"]
    result = public.execute_fit(tmp_path)
    assert result.status == "capability_unsupported"
    assert "v01" in result.message
    assert not (tmp_path / "started.json").exists()


def test_interrupted_attempt_never_gets_a_fresh_budget(tmp_path, monkeypatch):
    request, train, val = _control()
    prepared = public.prepare_fit(request, train, val, tmp_path)
    _write(tmp_path / "started.json", {"identity": prepared["identity"]})
    (tmp_path / "saved_checkpoint.txt").write_text("retain me")
    monkeypatch.setattr(public, "_run_backend", lambda *a: pytest.fail("fresh attempt"))
    result = public.execute_fit(tmp_path)
    assert result.status == "interrupted"
    assert (tmp_path / "saved_checkpoint.txt").read_text() == "retain me"
    assert public.execute_fit(tmp_path) == result


def test_backend_exception_remains_terminal(tmp_path, monkeypatch):
    request, train, val = _control()
    public.prepare_fit(request, train, val, tmp_path)

    def fail(*args):
        raise RuntimeError("simulated integration failure")

    monkeypatch.setattr(public, "_run_backend", fail)
    result = public.execute_fit(tmp_path)
    assert result.status == "fit_failed"
    assert "simulated integration failure" in result.message
    monkeypatch.setattr(public, "_run_backend", lambda *a: pytest.fail("fresh attempt"))
    assert public.execute_fit(tmp_path) == result


@pytest.mark.parametrize("field", ["data", "source", "settings", "lowering"])
def test_frozen_contract_tampering_rejected_before_execution(tmp_path, field):
    request, train, val = _control()
    public.prepare_fit(request, train, val, tmp_path)
    frozen = json.loads((tmp_path / "freeze.json").read_text())
    if field == "data":
        frozen["training"]["rows"][0]["targets"]["v01"][1] += 0.1
    elif field == "source":
        frozen["source_sha256"] = "a" * 64
    elif field == "settings":
        frozen["settings"]["maximum_wall_time_seconds"] = 3360
    else:
        frozen["lowered_candidate"]["state_equations"][0]["rhs"] = "0"
    _write(tmp_path / "freeze.json", frozen)
    with pytest.raises(ValueError, match="changed"):
        public.execute_fit(tmp_path)
    assert not (tmp_path / "started.json").exists()


def test_changed_arrays_under_same_loader_fingerprint_cannot_resume(tmp_path):
    request, train, val = _control()
    public.prepare_fit(request, train, val, tmp_path)
    payload = train.model_dump(mode="json")
    payload["rows"][0]["targets"]["v01"][1] += 0.1
    with pytest.raises(ValueError, match="differ"):
        public.prepare_fit(request, PublicSplit.model_validate(payload), val, tmp_path)


@pytest.mark.parametrize("file", ["result.json", "backend_result.json"])
def test_terminal_evidence_tampering_is_rejected(tmp_path, monkeypatch, file):
    request, train, val = _control()
    public.prepare_fit(request, train, val, tmp_path)
    monkeypatch.setattr(public, "_run_backend", lambda *a: _raw())
    public.execute_fit(tmp_path)
    payload = json.loads((tmp_path / file).read_text())
    if file == "result.json":
        payload["result"]["training"]["normalized_mse"] = 0
    else:
        payload["global_parameters"]["a"] = 100
    _write(tmp_path / file, payload)
    with pytest.raises(ValueError, match="digest differs"):
        public.execute_fit(tmp_path)


@pytest.mark.parametrize("change", ["extra", "test", "length", "time", "nonfinite"])
def test_public_arrays_schema_rejects_bad_or_private_payload(change):
    _, train, _ = _control()
    data = train.model_dump(mode="json")
    if change == "extra":
        data["rows"][0]["derivatives"] = {"v01": [0] * 21}
    elif change == "test":
        data["name"] = "test"
    elif change == "length":
        data["rows"][0]["targets"]["v01"].pop()
    elif change == "time":
        data["rows"][0]["time"][1] = 0
    else:
        data["rows"][0]["time"][1] = float("nan")
    with pytest.raises(ValidationError):
        PublicSplit.model_validate(data)


def test_unregistered_channel_and_split_swap_rejected(tmp_path):
    request, train, val = _control()
    data = train.model_dump(mode="json")
    data["rows"][0]["targets"]["private_z"] = [0] * 21
    with pytest.raises(ValueError, match="targets differ"):
        public.prepare_fit(request, PublicSplit.model_validate(data), val, tmp_path)
    with pytest.raises(ValueError, match="train then val"):
        public.prepare_fit(request, val, train, tmp_path)


@pytest.mark.parametrize("change", ["plan", "guesses", "lowered", "range", "lagged"])
def test_ambiguous_initialization_and_parameter_scope_rejected(tmp_path, change):
    request, train, val = _control()
    data = request.model_dump(mode="json")
    if change == "plan":
        data["initialization_plan"]["rules"] = {}
    elif change == "guesses":
        data["parameter_guesses"]["init_z_scale"] = 0.9
    elif change == "lowered":
        data["context"]["fitted_initialization"] = True
    elif change == "lagged":
        data["context"]["lagged_targets"] = ["v01"]
    else:
        initial = data["base_candidate"]["initial_conditions"][1]
        initial["fixed_value"] = None
        initial["initialization_range"] = {"lower": -2, "upper": 2}
    with pytest.raises(ValueError):
        public.prepare_fit(PublicFitRequest.model_validate(data), train, val, tmp_path)


def test_profile_matches_existing_frozen_repair_config():
    pytest.importorskip("casadi")
    from autoformalism.fitting.collocation_sensitivity import (
        CollocationSensitivityConfig,
    )

    request, _, _ = _control("collocation-feasible-v1")
    expected = CollocationSensitivityConfig(
        initializer_seconds=120,
        refinement_seconds=180,
        maximum_function_evaluations=240,
        collocation_node_start="rollout_or_observed",
        node_warmup_seconds=5,
        least_squares_ftol=None,
        collocation_diagnostics=True,
        recovery_policy="feasible",
        recovery_max_starts=10,
        recovery_probe_seconds=10,
    )
    assert public.profile_settings(request) == expected.model_dump(mode="json")


def test_real_general_fit_causal_validation_and_multiple_targets(tmp_path):
    request, train, val = _control(multiple_targets=True)
    public.prepare_fit(request, train, val, tmp_path)
    result = public.execute_fit(tmp_path)
    assert result.status == "complete", result.message
    assert result.training.normalized_mse < 1e-8
    assert result.validation.normalized_mse < 1e-8
    assert set(result.validation.per_target_normalized_mse) == {"v01", "v02"}
    raw = json.loads((tmp_path / "backend_result.json").read_text())
    assert all(
        not values
        for values in raw["validation_trajectory_initial_conditions"].values()
    )
    assert abs(result.parameters["init_z_scale"] - 0.5) < 1e-5


def test_cli_round_trip(tmp_path):
    request, train, val = _control()
    for name, value in (("request", request), ("training", train), ("validation", val)):
        (tmp_path / f"{name}.json").write_text(value.model_dump_json())
    command = [sys.executable, str(ROOT / "scripts/run_public_fitting.py")]
    prepared = subprocess.run(
        [
            *command,
            "prepare",
            "--request",
            str(tmp_path / "request.json"),
            "--training",
            str(tmp_path / "training.json"),
            "--validation",
            str(tmp_path / "validation.json"),
            "--output",
            str(tmp_path / "fit"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    identity = json.loads(prepared.stdout)["identity"]
    inspected = subprocess.run(
        [*command, "inspect", "--output", str(tmp_path / "fit")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(inspected.stdout)["identity"] == identity
    fitted = subprocess.run(
        [*command, "run", "--output", str(tmp_path / "fit")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(fitted.stdout)["status"] == "complete"
