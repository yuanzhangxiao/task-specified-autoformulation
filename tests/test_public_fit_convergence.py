"""Diagnostic continuation lineage, training-only routing, budgets and resume."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from autoformalism.fitting import fit_continuation as pilot
from autoformalism.fitting import fit_convergence as diagnostic
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.fit_convergence import ConvergenceSelection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "convergence_parent_smoke", ROOT / "scripts/smoke_public_fit_continuation.py"
    )
    SMOKE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(SMOKE)
finally:
    sys.path.pop(0)


def fake_window(
    state, *, drop=0.05, success=False, optimality=None, failure=None, validation=0.2
):
    """Explicit fake numerical history; used only to exercise the controller."""
    parameters = {
        **state["parameters"],
        "init_z_scale": state["parameters"]["init_z_scale"] + 0.01,
    }
    cost = state["training_cost"] * (1 - drop)
    selected = drop > 0
    parameters = parameters if selected else state["parameters"]
    training = {
        **state["training"],
        "normalized_mse": state["training"]["normalized_mse"] * (1 - max(drop, 0)),
    }
    val = {
        "available": validation is not None,
        "normalized_mse": validation,
        "per_target_normalized_mse": {} if validation is None else {"v01": validation},
        "failed_trajectories": [],
    }
    if not selected:
        val = state["validation"]
    best = {"parameters": parameters, "cost": min(cost, state["training_cost"])}
    fields = {
        "status": "complete",
        "selected": "extension" if selected else "parent",
        "parameters": parameters,
        "training": training,
        "validation": val,
        "extension_budget_exhausted": not success,
        "extension_residual_calls": 3,
        "cumulative_residual_calls": state["actual_residual_calls"] + 3,
        "extension_numerical_seconds": 180.0,
        "native_optimizer_converged": success if success else None,
        "feedback_status": "local_optimizer_stopped"
        if success
        else "budget_limited_unresolved",
        "message": "Synthetic diagnostic fixture",
    }
    return {
        "initial_parameters": state["parameters"],
        "best_evaluated": best,
        "start_check": {"agrees": True, "cost": state["training_cost"]},
        "result_fields": fields,
        "actual_residual_calls": 3,
        "numerical_seconds": 180.0,
        "setup_seconds": 1.0,
        "scoring_seconds": 2.0,
        "budget_exhausted": not success,
        "integration_failures": 0,
        "failure": failure,
        "scoring_error": None,
        "optimizer": {
            "optimizer_native_success": success,
            "native_optimizer_parameters": parameters if success else None,
            "optimizer_status": 3 if success else -2,
            "optimality": optimality,
            "selected_training_rollout_verified": success,
            "message": "synthetic history",
            "iterations": [],
        },
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    parent, continuation, output = (
        tmp_path / name / child
        for name, child in (
            ("parent", "fit"),
            ("pilot", "continuation"),
            ("new", "diagnostic"),
        )
    )
    old_selection = SMOKE.fixture_parent(parent)
    pilot.prepare_continuation(parent, old_selection, continuation)

    def extension(frozen, directory):
        state = {
            **frozen["parent"]["result"]["result"],
            "training_cost": frozen["gate"]["parent_training_cost"],
        }
        return fake_window(state)

    monkeypatch.setattr(pilot, "_run_extension", extension)
    result = pilot.execute_continuation(continuation)
    selection = ConvergenceSelection(
        continuation_identity=result.identity,
        continuation_backend_sha256=result.backend_result_sha256,
        maximum_windows=3,
    )
    diagnostic.prepare_convergence(parent, continuation, selection, output)
    return parent, continuation, selection, output


def test_chain_carries_all_parameters_and_keeps_training_best(prepared, monkeypatch):
    parent, continuation, _, output = prepared
    originals = {
        str(p): p.read_bytes()
        for root in (parent, continuation)
        for p in root.rglob("*")
        if p.is_file()
    }
    starts = []

    def run(frozen, state, directory):
        starts.append(copy.deepcopy(state))
        assert (
            diagnostic._numerical_input(frozen, state)["parent"]["result"]["result"]
            == state
        )
        return fake_window(state, validation=1000 + len(starts))

    monkeypatch.setattr(diagnostic, "_run_window", run)
    result = diagnostic.execute_convergence(output)
    assert result["status"] == "diagnostic_cap_reached"
    assert len(starts) == result["completed_windows"] == 3
    assert result["additional_numerical_seconds"] == 540
    assert result["additional_residual_calls"] == 9
    for index, state in enumerate(starts[1:]):
        assert state == result["windows"][index]["state"]
    assert starts[0]["parameters"]["init_z_scale"] == 0.31
    assert result["retained_state"]["training_cost"] < starts[0]["training_cost"]
    assert result["retained_state"]["validation"]["normalized_mse"] == 1003
    assert not result["final_protocol_limit_decided"]
    assert originals == {
        str(p): p.read_bytes()
        for root in (parent, continuation)
        for p in root.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(diagnostic, "_run_window", lambda *a: pytest.fail("renewed"))
    assert diagnostic.execute_convergence(output) == result
    assert diagnostic.report_convergence(output) == result


@pytest.mark.parametrize("drop", [0.0, 0.0001])
def test_slow_progress_does_not_stop_this_diagnostic(prepared, monkeypatch, drop):
    output = prepared[-1]
    monkeypatch.setattr(
        diagnostic, "_run_window", lambda f, s, d: fake_window(s, drop=drop)
    )
    result = diagnostic.execute_convergence(output)
    assert result["completed_windows"] == 3
    assert all(row["slow_progress"] for row in result["windows"])


@pytest.mark.parametrize(
    "success,optimality,verified,expected",
    [
        (True, 1e-8, True, "local_stationarity_reached"),
        (True, 5000.0, True, "diagnostic_cap_reached"),
        (True, None, True, "diagnostic_cap_reached"),
        (False, 0.0, True, "diagnostic_cap_reached"),
        (True, 1e-8, False, "diagnostic_cap_reached"),
    ],
)
def test_native_success_needs_verified_small_optimality(
    prepared, monkeypatch, success, optimality, verified, expected
):
    def run(f, s, d):
        raw = fake_window(s, success=success, optimality=optimality)
        raw["optimizer"]["selected_training_rollout_verified"] = verified
        return raw

    monkeypatch.setattr(diagnostic, "_run_window", run)
    result = diagnostic.execute_convergence(prepared[-1])
    assert result["status"] == expected
    assert result["completed_windows"] == (
        1 if expected == "local_stationarity_reached" else 3
    )


def test_native_success_at_different_point_cannot_certify_incumbent(
    prepared, monkeypatch
):
    def run(f, s, d):
        raw = fake_window(s, success=True, optimality=0.0)
        raw["optimizer"]["native_optimizer_parameters"] = s["parameters"]
        return raw

    monkeypatch.setattr(diagnostic, "_run_window", run)
    assert (
        diagnostic.execute_convergence(prepared[-1])["status"]
        == "diagnostic_cap_reached"
    )


@pytest.mark.parametrize("validation", [None, 1000.0])
def test_validation_cannot_stop_or_select(prepared, monkeypatch, validation):
    def run(f, s, d):
        raw = fake_window(s, validation=validation)
        if validation is None:
            raw["scoring_error"] = "validation: failed integration"
            raw["result_fields"]["status"] = "extension_failed"
        return raw

    monkeypatch.setattr(diagnostic, "_run_window", run)
    result = diagnostic.execute_convergence(prepared[-1])
    assert result["status"] == "diagnostic_cap_reached"
    assert result["retained_state"]["validation"]["normalized_mse"] == validation


def test_numerical_error_stops_but_preserves_incumbent(prepared, monkeypatch):
    def run(f, s, d):
        raise RuntimeError("numerical failure fixture")

    monkeypatch.setattr(diagnostic, "_run_window", run)
    result = diagnostic.execute_convergence(prepared[-1])
    assert result["status"] == "numerical_failure"
    assert (
        result["retained_state"]["parameters"] == result["initial_state"]["parameters"]
    )
    assert result["retained_state"]["actual_residual_calls"] is None
    assert result["timing_incomplete"]
    assert diagnostic.execute_convergence(prepared[-1]) == result


def test_mid_window_interruption_consumed_and_not_rerun(prepared, monkeypatch):
    output = prepared[-1]
    calls = []

    def run(f, s, d):
        calls.append(d.name)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return fake_window(s)

    monkeypatch.setattr(diagnostic, "_run_window", run)
    with pytest.raises(KeyboardInterrupt):
        diagnostic.execute_convergence(output)
    assert diagnostic.report_convergence(output)["completed_windows"] == 1
    monkeypatch.setattr(diagnostic, "_run_window", lambda *a: pytest.fail("renewed"))
    result = diagnostic.execute_convergence(output)
    assert result["status"] == "interrupted"
    assert result["timing_incomplete"] and result["residual_accounting_incomplete"]
    assert (
        result["retained_state"]["parameters"]
        == result["windows"][0]["state"]["parameters"]
    )
    assert result["completed_windows"] == 1
    assert result["recorded_windows"] == 2
    assert diagnostic.execute_convergence(output) == result


def test_resume_after_complete_window_only_runs_remaining(prepared, monkeypatch):
    output = prepared[-1]
    monkeypatch.setattr(diagnostic, "_run_window", lambda f, s, d: fake_window(s))
    publish = diagnostic._publish_report

    def stop(f, d, rows, state):
        if len(rows) == 1:
            raise KeyboardInterrupt
        return publish(f, d, rows, state)

    monkeypatch.setattr(diagnostic, "_publish_report", stop)
    with pytest.raises(KeyboardInterrupt):
        diagnostic.execute_convergence(output)
    monkeypatch.setattr(diagnostic, "_publish_report", publish)
    calls = []

    def resume(f, s, d):
        calls.append(d.name)
        return fake_window(s)

    monkeypatch.setattr(diagnostic, "_run_window", resume)
    assert diagnostic.execute_convergence(output)["completed_windows"] == 3
    assert calls == ["002", "003"]


def test_backend_publication_recovery_does_not_rerun_window(prepared, monkeypatch):
    output = prepared[-1]
    monkeypatch.setattr(diagnostic, "_run_window", lambda f, s, d: fake_window(s))
    record = diagnostic._record
    monkeypatch.setattr(
        diagnostic, "_record", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        diagnostic.execute_convergence(output)
    monkeypatch.setattr(diagnostic, "_record", record)
    calls = []

    def resume(f, s, d):
        calls.append(d.name)
        return fake_window(s)

    monkeypatch.setattr(diagnostic, "_run_window", resume)
    assert diagnostic.execute_convergence(output)["completed_windows"] == 3
    assert calls == ["002", "003"]


@pytest.mark.parametrize("file", ["backend_result.json", "result.json", "started.json"])
def test_window_evidence_corruption_rejected(prepared, monkeypatch, file):
    output = prepared[-1]
    monkeypatch.setattr(diagnostic, "_run_window", lambda f, s, d: fake_window(s))
    diagnostic.execute_convergence(output)
    path = output / "windows/001" / file
    value = public._read(path)
    if file == "started.json":
        value["request"]["index"] = 99
    else:
        value.setdefault("result_fields", {})["selected"] = "fake"
        if file == "result.json":
            value["result"]["retained_cost"] = 0
    public._write(path, value)
    with pytest.raises(ValueError):
        diagnostic.report_convergence(output)


def test_second_output_and_budget_expansion_rejected(prepared):
    parent, continuation, selection, output = prepared
    with pytest.raises(ValueError, match="already reserved"):
        diagnostic.prepare_convergence(
            parent, continuation, selection, output.with_name("other")
        )
    with pytest.raises(ValueError, match="already reserved"):
        diagnostic.prepare_convergence(
            parent,
            continuation,
            selection.model_copy(update={"maximum_windows": 4}),
            output,
        )
    for source in (parent.parent, continuation.parent):
        with pytest.raises(ValueError, match="separate"):
            diagnostic.prepare_convergence(
                parent, continuation, selection, source / "new"
            )


@pytest.mark.parametrize("source", ["parent", "continuation"])
@pytest.mark.parametrize("file", ["freeze.json", "result.json", "backend_result.json"])
def test_historical_artifacts_remain_hash_bound(prepared, source, file):
    parent, continuation, _, output = prepared
    path = (parent if source == "parent" else continuation) / file
    value = public._read(path)
    if file == "result.json":
        value["result"]["parameters"]["a"] = 999
    else:
        value["tampered"] = True
    public._write(path, value)
    with pytest.raises(ValueError):
        diagnostic.report_convergence(output)


@pytest.mark.parametrize("kind", ["source", "runtime"])
def test_current_execution_contract_cannot_drift(prepared, monkeypatch, kind):
    monkeypatch.setattr(
        public,
        "_source_identity" if kind == "source" else "_runtime",
        lambda: "changed",
    )
    with pytest.raises(ValueError, match="changed"):
        diagnostic.execute_convergence(prepared[-1])


def test_bad_training_selection_rejected(prepared, monkeypatch):
    def run(f, s, d):
        raw = fake_window(s)
        raw["result_fields"]["training"]["normalized_mse"] = 1000
        return raw

    monkeypatch.setattr(diagnostic, "_run_window", run)
    with pytest.raises(ValueError, match="training-only"):
        diagnostic.execute_convergence(prepared[-1])


def test_actual_numerical_diagnostic_smoke(tmp_path):
    pytest.importorskip("casadi")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from smoke_public_fit_convergence import smoke

        result = smoke(tmp_path)
    finally:
        sys.path.pop(0)
    assert result["status"] == "passed"
    assert result["training_nmse"] < 1e-8


def test_report_locks_before_reading_snapshot(prepared, monkeypatch):
    output = prepared[-1]
    with public._lock(output):
        monkeypatch.setattr(
            diagnostic, "_load", lambda *a: pytest.fail("unlocked read")
        )
        with pytest.raises(RuntimeError, match="in use"):
            diagnostic.report_convergence(output)
