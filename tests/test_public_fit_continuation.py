"""Lineage, budget, handoff and training-only continuation selection."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.fitting import fit_continuation as continuation
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.fit_continuation import ContinuationSelection

ROOT = Path(__file__).resolve().parents[1]
# The standalone smoke has a normal sibling-script import. Keep test collection
# from permanently changing the import search path for unrelated tests.
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "continuation_smoke", ROOT / "scripts/smoke_public_fit_continuation.py"
    )
    SMOKE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(SMOKE)
finally:
    sys.path.pop(0)


@pytest.fixture
def parent(tmp_path):
    source = tmp_path / "parent/fit"
    selection = SMOKE.fixture_parent(source)
    return source, selection, tmp_path / "extension"


def _reseal_parent(source, mutate):
    raw = public._read(source / "backend_result.json")
    mutate(raw)
    public._write(source / "backend_result.json", raw)
    envelope = public._read(source / "result.json")
    request = public.PublicFitRequest.model_validate(
        public._read(source / "freeze.json")["request"]
    )
    envelope["result"].update(public._jsonable(public._evidence(request, raw)))
    envelope["result"]["backend_result_sha256"] = public.content_sha256(raw)
    envelope["sha256"] = public.content_sha256(envelope["result"])
    public._write(source / "result.json", envelope)
    return ContinuationSelection(
        parent_lowered_candidate_sha256=envelope["result"]["lowered_candidate_sha256"],
        parent_backend_result_sha256=envelope["result"]["backend_result_sha256"],
    )


def test_historical_source_retained_without_rewriting_parent(parent):
    source, selection, output = parent
    frozen = public._read(source / "freeze.json")
    frozen["source_sha256"] = "f" * 64
    frozen["identity"] = public.content_sha256(
        {k: v for k, v in frozen.items() if k != "identity"}
    )
    public._write(source / "freeze.json", frozen)
    envelope = public._read(source / "result.json")
    envelope["result"]["identity"] = frozen["identity"]
    envelope["sha256"] = public.content_sha256(envelope["result"])
    public._write(source / "result.json", envelope)
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.glob("*.json")
    }
    inspection = continuation.prepare_continuation(source, selection, output)
    assert inspection["gate"]["eligible"]
    assert inspection["parameter_count"] == 3
    assert inspection["budgets"]["cumulative_fit_seconds"] == 480
    assert before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.glob("*.json")
    }
    assert continuation.prepare_continuation(source, selection, output) == inspection


@pytest.mark.parametrize("file", ["freeze.json", "result.json", "backend_result.json"])
def test_parent_corruption_rejected(parent, file):
    source, selection, output = parent
    value = public._read(source / file)
    value["tampering"] = True
    if file == "result.json":
        value["result"]["parameters"]["a"] = 9
    public._write(source / file, value)
    with pytest.raises(ValueError, match="digest"):
        continuation.prepare_continuation(source, selection, output)
    assert not (output / "started.json").exists()


@pytest.mark.parametrize("change", ["settings", "lowering", "private_data"])
def test_resealed_invalid_parent_contract_rejected(parent, change):
    source, selection, output = parent
    value = public._read(source / "freeze.json")
    if change == "settings":
        value["settings"]["refinement_seconds"] = 3360
    elif change == "lowering":
        value["lowered_candidate"]["state_equations"][0]["rhs"] = "0"
    else:
        value["training"]["rows"][0]["hidden"] = {"z": [0] * 21}
    value["identity"] = public.content_sha256(
        {k: v for k, v in value.items() if k != "identity"}
    )
    public._write(source / "freeze.json", value)
    with pytest.raises(ValueError):
        continuation.prepare_continuation(source, selection, output)


@pytest.mark.parametrize(
    "change", ["no_budget_stop", "converged", "poll", "flat", "few", "different_point"]
)
def test_training_progress_gate_is_fail_closed(parent, monkeypatch, change):
    source, _, output = parent

    def mutate(raw):
        refinement = raw["refinement"]
        stage = refinement["stages"][0]["result"]
        if change == "no_budget_stop":
            refinement["budget_exhausted"] = False
        elif change == "converged":
            stage.update(optimizer_status=1, optimizer_native_success=True)
        elif change == "poll":
            refinement["stages"][0]["mode"] = "directional_poll"
        elif change == "flat":
            for row in stage["iterations"]:
                row["cost"] = stage["cost"]
        elif change == "few":
            stage["iterations"] = stage["iterations"][:1]
        else:
            stage["parameters"] = {**stage["parameters"], "a": 2}

    selection = _reseal_parent(source, mutate)
    assert not continuation.prepare_continuation(source, selection, output)["gate"][
        "eligible"
    ]
    monkeypatch.setattr(
        continuation, "_run_extension", lambda *a: pytest.fail("must not optimize")
    )
    result = continuation.execute_continuation(output)
    assert result.status == "ineligible" and result.selected == "parent"
    assert not result.structural_failure_established
    assert not (output / "started.json").exists()


def test_gate_never_reads_validation_values(parent):
    source, selection, _ = parent
    snapshot = continuation._verified_parent(source)
    before = continuation.progress_gate(snapshot, selection)
    snapshot["result"]["result"]["validation"] = {"not_available_to_gate": True}
    snapshot["backend"]["validation"] = {"not_available_to_gate": True}
    assert continuation.progress_gate(snapshot, selection) == before


@pytest.mark.parametrize("change", ["missing_initializer", "negative_rate"])
def test_invalid_retained_vector_rejected_before_budget_reservation(parent, change):
    source, _, output = parent

    def mutate(raw):
        if change == "missing_initializer":
            del raw["parameters"]["init_z_scale"]
        else:
            raw["parameters"]["a"] = -1

    selection = _reseal_parent(source, mutate)
    with pytest.raises(ValueError, match="parameter"):
        continuation.prepare_continuation(source, selection, output)
    assert not (output / "started.json").exists()


@pytest.mark.parametrize(
    "score_case",
    ["worse_training", "failed_training", "worse_validation", "failed_validation"],
)
def test_handoff_selection_uses_training_and_preserves_parent(
    tmp_path, monkeypatch, score_case
):
    pytest.importorskip("casadi")
    from autoformalism.fitting import continuation_numerics as numerics

    source, output = tmp_path / "parent/fit", tmp_path / "extension"
    selection = SMOKE.fixture_parent(source, real_scores=True)
    continuation.prepare_continuation(source, selection, output)
    original = public._read(source / "result.json")["result"]
    optimum = {"a": 0.7, "b": 1.2, "init_z_scale": 0.5}

    def instrumented(oracle, start, **kwargs):
        assert start == original["parameters"]
        oracle.best = {"cost": 0.0, "parameters": optimum, "call": 2}
        return {
            "native_optimizer_parameters": optimum,
            "optimizer_native_success": True,
        }

    calls = []

    def score(model, split, parameters, scale, settings):
        calls.append(split.name.value)
        assert parameters == optimum
        value = 0.0
        failed = []
        if split.name.value == "train" and score_case == "worse_training":
            value = original["training"]["normalized_mse"] * 2
        if split.name.value == "val":
            value = original["validation"]["normalized_mse"] * 2
        if (score_case == "failed_training" and split.name.value == "train") or (
            score_case == "failed_validation" and split.name.value == "val"
        ):
            value, failed = 1e12, [split.trajectories[0].trajectory_id]
        return {
            "metrics": {
                "normalized_mse": value,
                "per_target_normalized_mse": {"v01": value},
                "failed_trajectories": failed,
            },
            "local_initials": {},
        }

    monkeypatch.setattr(numerics, "instrumented_fit", instrumented)
    monkeypatch.setattr(numerics, "_score", score)
    result = continuation.execute_continuation(output)
    if score_case.endswith("training"):
        assert (
            result.selected == "parent" and result.parameters == original["parameters"]
        )
        assert calls == ["train"]
        assert (
            result.validation.normalized_mse == original["validation"]["normalized_mse"]
        )
    else:
        assert result.selected == "extension" and result.parameters == optimum
        assert calls == ["train", "val"]
        if score_case == "failed_validation":
            assert not result.validation.available
        else:
            assert (
                result.validation.normalized_mse
                > original["validation"]["normalized_mse"]
            )
    if score_case.startswith("failed"):
        assert result.status == "extension_failed"
        assert result.feedback_status == "numerical_failure_unresolved"


def test_changed_objective_stops_before_optimizer_and_preserves_incumbent(
    parent, monkeypatch
):
    pytest.importorskip("casadi")
    from autoformalism.fitting import continuation_numerics as numerics

    source, selection, output = parent  # Fixture cost intentionally not real.
    continuation.prepare_continuation(source, selection, output)
    monkeypatch.setattr(
        numerics, "instrumented_fit", lambda *a, **k: pytest.fail("different objective")
    )
    result = continuation.execute_continuation(output)
    assert result.status == "extension_failed" and result.selected == "parent"
    raw = public._read(output / "backend_result.json")
    assert not raw["start_check"]["agrees"]
    assert result.extension_residual_calls == 1


def test_native_convergence_at_retained_parent_is_attributed(tmp_path, monkeypatch):
    pytest.importorskip("casadi")
    from autoformalism.fitting import continuation_numerics as numerics

    source, output = tmp_path / "parent/fit", tmp_path / "extension"
    selection = SMOKE.fixture_parent(source, real_scores=True)
    continuation.prepare_continuation(source, selection, output)

    def instrumented(oracle, start, **kwargs):
        return {
            "native_optimizer_parameters": start,
            "optimizer_native_success": True,
            "optimizer_status": 1,
        }

    monkeypatch.setattr(numerics, "instrumented_fit", instrumented)
    result = continuation.execute_continuation(output)
    assert result.selected == "parent"
    assert result.native_optimizer_converged is True
    assert result.feedback_status == "local_optimizer_stopped"
    assert not result.structural_failure_established


def test_empty_remaining_budget_never_becomes_a_new_optimizer_window(
    parent, monkeypatch
):
    pytest.importorskip("casadi")
    from autoformalism.fitting import continuation_numerics as numerics

    source, selection, output = parent
    continuation.prepare_continuation(source, selection, output)

    def timeout(oracle, values):
        oracle.budget.calls = oracle.budget.maximum
        raise TimeoutError("shared refinement budget exhausted")

    monkeypatch.setattr(numerics.GuardedOracle, "__call__", timeout)
    monkeypatch.setattr(
        numerics, "instrumented_fit", lambda *a, **k: pytest.fail("spent budget")
    )
    result = continuation.execute_continuation(output)
    assert result.selected == "parent"
    assert result.feedback_status == "budget_limited_unresolved"
    assert result.extension_budget_exhausted
    assert result.extension_residual_calls == 240
    assert continuation.execute_continuation(output) == result


def test_no_second_window_even_in_different_output(parent):
    source, selection, output = parent
    continuation.prepare_continuation(source, selection, output)
    with pytest.raises(ValueError, match="already reserved"):
        continuation.prepare_continuation(source, selection, output.with_name("second"))
    with pytest.raises(ValueError, match="separate"):
        continuation.prepare_continuation(source, selection, source / "extension")


def test_interruption_preserves_parent_and_cannot_renew(parent, monkeypatch):
    source, selection, output = parent
    prepared = continuation.prepare_continuation(source, selection, output)
    public._write(output / "started.json", {"identity": prepared["identity"]})
    (output / "partial.json").write_text("{}")
    monkeypatch.setattr(
        continuation, "_run_extension", lambda *a: pytest.fail("fresh budget")
    )
    result = continuation.execute_continuation(output)
    assert result.status == "interrupted" and result.selected == "parent"
    assert result.parameters["init_z_scale"] == 0.3
    assert continuation.execute_continuation(output) == result
    assert (output / "partial.json").exists()


def test_extension_error_keeps_complete_vector_and_terminal_result(parent, monkeypatch):
    source, selection, output = parent
    continuation.prepare_continuation(source, selection, output)

    def fail(frozen, directory):
        assert frozen["parent"]["result"]["result"]["parameters"] == {
            "a": 0.9,
            "b": 0.8,
            "init_z_scale": 0.3,
        }
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(continuation, "_run_extension", fail)
    result = continuation.execute_continuation(output)
    assert result.status == "extension_failed"
    assert result.selected == "parent" and result.training.normalized_mse == 0.2
    monkeypatch.setattr(
        continuation, "_run_extension", lambda *a: pytest.fail("fresh budget")
    )
    assert continuation.execute_continuation(output) == result
    assert not result.automatic_followup


@pytest.mark.parametrize(
    "field", ["additional_seconds", "additional_residual_calls", "maximum_extensions"]
)
def test_limits_not_silently_expanded(parent, field):
    _, selection, _ = parent
    value = selection.model_dump(mode="json")
    value[field] *= 2
    with pytest.raises(ValueError):
        ContinuationSelection.model_validate(value)


def test_real_sensitivity_extension_and_initial_map_recovery(tmp_path):
    pytest.importorskip("casadi")
    parent, output = tmp_path / "parent/fit", tmp_path / "extension"
    selection = SMOKE.fixture_parent(parent, real_scores=True)
    continuation.prepare_continuation(parent, selection, output)
    result = continuation.execute_continuation(output)
    assert result.status == "complete", public._read(output / "backend_result.json")
    assert result.selected == "extension"
    assert result.training.normalized_mse < 1e-8
    assert result.validation.normalized_mse < 1e-8
    assert abs(result.parameters["init_z_scale"] - 0.3) > 1e-3
    raw = public._read(output / "backend_result.json")
    assert raw["initial_parameters"] == {"a": 0.9, "b": 0.8, "init_z_scale": 0.3}
    assert raw["start_check"]["agrees"]
    assert raw["collocation_reruns"] == raw["additional_starts"] == 0
    assert all(not x for x in raw["scoring"]["validation"]["local_initials"].values())
    assert result.cumulative_residual_calls == 6 + result.extension_residual_calls
    assert continuation.execute_continuation(output) == result
    # Terminal evidence is immutable and checked on read.
    value = json.loads((output / "backend_result.json").read_text())
    value["initial_parameters"]["init_z_scale"] = 0
    public._write(output / "backend_result.json", value)
    with pytest.raises(ValueError, match="backend digest"):
        continuation.execute_continuation(output)


def test_cli_prepare_inspect_and_report_are_read_only_before_run(parent):
    source, selection, output = parent
    selection_path = output.parent / "selection.json"
    selection_path.write_text(selection.model_dump_json())
    command = [sys.executable, str(ROOT / "scripts/run_public_fit_continuation.py")]

    def invoke(*arguments):
        process = subprocess.run(
            [*command, *arguments, "--output", str(output)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(process.stdout)

    prepared = invoke(
        "prepare", "--parent", str(source), "--selection", str(selection_path)
    )
    assert invoke("inspect")["identity"] == prepared["identity"]
    pending = invoke("report")
    assert pending["status"] == "pending_or_interrupted"
    assert not (output / "started.json").exists()
