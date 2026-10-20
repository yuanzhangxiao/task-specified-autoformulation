"""M46 duration tuning, strict prefix access and durable audit/replay."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("casadi")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_multiroot_campaign as q
from autoformalism.schemas.public_fitting import PublicSplit


def test_inputs_independent_roots_split_and_training_start_identity():
    data = q.make_inputs()
    q._validate(data)
    assert len(q.roster()) == 12
    a = data["records"]["separated_noisy_s0"]
    b = data["records"]["separated_noisy_s1"]
    assert a["training"] == b["training"]
    assert a["validation"] == b["validation"]
    assert a["followup"] == b["followup"]
    assert a["problem"] != b["problem"]
    graph = q.kernel.PrefixFitProblem.model_validate(a["problem"]).graph
    assert all(rule.initial.scope == "trajectory" for rule in graph.rules.values())
    assert q.make_inputs() == data
    changed = deepcopy(data)
    changed["records"]["separated_noisy_s1"]["training"]["rows"][0]["targets"]["y"][
        1
    ] += 1
    changed["records"]["separated_noisy_s1"]["training"] = q.controls.fingerprint(
        changed["records"]["separated_noisy_s1"]["training"]
    )
    with pytest.raises(ValueError, match="identical observations"):
        q._validate(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scenario", "bad"),
        ("noise_sigma", 0.3),
        ("seed", 9),
        ("repeat", 1),
        ("test_data_opened", True),
    ],
)
def test_rejects_acquisition_metadata_changes(field, value):
    data = q.make_inputs()
    data["records"]["separated_clean_s0"][field] = value
    with pytest.raises(ValueError, match="acquisition identity"):
        q._validate(data)


def test_rejects_stale_observation_hash():
    data = q.make_inputs()
    data["records"]["separated_clean_s0"]["training"]["fingerprint"] = "stale"
    with pytest.raises(ValueError, match="fingerprint"):
        q._validate(data)


def test_diagnostic_failure_not_counted_as_completed(monkeypatch):
    def diagnostic(available):
        return {
            "available": available,
            "diagnostic_calls_started": 1,
            "diagnostic_calls_completed": int(available),
        }

    monkeypatch.setattr(
        q.controls, "prefix_information", lambda *args: diagnostic(False)
    )
    monkeypatch.setattr(
        q.controls, "conditional_linear_optimum", lambda *args: diagnostic(True)
    )
    progress = []
    result = q._diagnostic(
        None,
        None,
        None,
        {},
        q.Policy(),
        0,
        lambda value: progress.append(deepcopy(value)),
    )
    assert result["diagnostic_calls_attempted"] == 2
    assert result["diagnostic_calls_started"] == 2
    assert result["diagnostic_calls_completed"] == 1
    assert progress[0]["diagnostic_calls_completed"] == 0
    assert result["timing"]["callback_calls"] == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"durations": (0.1, 0.2)},
        {"durations": (0, 0.15)},
        {"durations": (0, 0.3)},
        {"durations": (0, 3.5)},
        {"durations": (0, 0.2, 0.1)},
        {"root_starts": ((0.1, 0.2), (0.1, 0.2))},
        {"root_starts": ((-1, 0.2), (0.1, 0.2))},
        {"root_starts": ((0.1,), (0.1, 0.2))},
        {"immediate_duration": 0.15},
    ],
)
def test_policy_rejects_invalid_grids_and_roots(kwargs):
    with pytest.raises(ValidationError):
        q.Policy(**kwargs)


def test_prepare_resume_and_source_identity(tmp_path, monkeypatch):
    policy = q.Policy()
    initial = q.prepare(tmp_path, policy)
    assert initial["tasks"] == 12
    before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
    monkeypatch.setattr(q, "make_inputs", lambda: pytest.fail("redrew noise"))
    assert q.prepare(tmp_path, policy) == initial
    assert before == {p: p.read_bytes() for p in tmp_path.glob("*.json")}
    plan, _ = q.verify(tmp_path)
    monkeypatch.setattr(q.public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="source/runtime"):
        q.verify(tmp_path)
    assert q.verify(tmp_path, runtime=False)[0] == plan
    with pytest.raises(ValueError, match="sealed artifact differs"):
        q.prepare(tmp_path, policy)


def test_both_horizons_are_required_and_missing_grid_not_dropped():
    policy = q.Policy(durations=(0, 0.2))
    metrics = [
        {"duration": 0, "complete": True, "adequate": False},
        {"duration": 0.2, "complete": True, "adequate": True},
    ]
    assert q.select_duration(metrics, policy)["duration"] == 0.2
    assert q.select_duration(metrics[1:], policy)["status"] == "unavailable"
    metrics[1]["adequate"] = False
    assert q.select_duration(metrics, policy)["status"] == "no_adequate_window"
    metrics[1]["complete"] = False
    assert q.select_duration(metrics, policy)["status"] == "unavailable"


def test_immediate_window_exposes_error_hidden_by_late_horizon():
    policy = q.Policy(durations=(0, 2))
    row = PublicSplit.model_validate(
        q.controls.fixture("separated")["validation"]
    ).rows[0]
    prediction = {
        "trajectory_id": row.trajectory_id,
        "forecast_success": True,
        "time": list(row.time),
        "cutoff": 0.2,
        "predictions": {"y": list(row.targets["y"])},
    }
    prediction["predictions"]["y"][3] += 1
    scored = q._scores(prediction, row, 0.2, {"training_scales": {"y": 1}}, policy)
    assert scored["scores"]["common"]["normalized_mse"] == 0
    assert scored["scores"]["immediate"]["normalized_mse"] > 0.1
    assert scored["scores"]["immediate"]["sample_count"] == 5
    prediction["predictions"]["y"][3] -= 1
    prediction["predictions"]["y"][2] += 100
    assert (
        q._scores(prediction, row, 0.2, {"training_scales": {"y": 1}}, policy)[
            "scores"
        ]["immediate"]["normalized_mse"]
        < 1e-20
    )


def test_start_choice_uses_prefix_loss_not_forecast_accuracy():
    attempts = [
        {
            "inference": {"fit_success": True, "normalized_mse": loss},
            "scores": {"common": score},
        }
        for loss, score in ((0.1, 100), (0.2, 0))
    ]
    assert q.previous._choose_prefix_start(attempts) == 0
    attempts[1]["inference"]["normalized_mse"] = 0.1
    assert q.previous._choose_prefix_start(attempts) == 0


def test_followup_targets_never_enter_prefix_or_forecast_schedule():
    record = q.controls.fixture("separated")
    row = PublicSplit.model_validate(record["followup"]).rows[0]
    policy = q.kernel.PrefixPolicy(duration=0.2)
    view = q.kernel.make_prefix(row, policy)
    schedule = q.kernel.make_forecast_schedule(row, policy)
    changed = row.model_copy(
        update={"targets": {"y": row.targets["y"][:3] + (99.0,) * 38}}
    )
    assert q.kernel.make_prefix(changed, policy) == view
    assert q.kernel.make_forecast_schedule(changed, policy) == schedule


def test_partial_operation_remains_charged_without_retry(tmp_path):
    identity = {"task": {"task_id": "test"}}

    def interrupted(save):
        save({"search_calls_attempted": 2, "search_calls_started": 1})
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        q._operation(tmp_path, identity, "training", 20, interrupted)
    assert (
        q._operation(
            tmp_path,
            identity,
            "training",
            20,
            lambda save: pytest.fail("repeated interrupted work"),
        )["status"]
        == "interrupted"
    )
    operations = q._read_operations(tmp_path, identity)
    cost = q._cost(operations)
    assert cost["search_calls_started_observed"] == 1
    assert cost["search_calls_started_total"] is None
    assert cost["wall_seconds_total"] is not None
    with pytest.raises(ValueError, match="identity"):
        q._operation(tmp_path, identity, "training", 21, lambda save: None)


def test_abrupt_exit_reservation_and_unknown_cost(tmp_path):
    identity = {"task": {"task_id": "test"}}
    bound = {"campaign": identity, "stage": "training", "allowance_seconds": 20}
    folder = tmp_path / "operations" / "training"
    seal(folder / "started.json", bound)
    q.public._write(folder / "progress.json", {"search_calls_started": 2})
    result = q._operation(
        tmp_path, identity, "training", 20, lambda save: pytest.fail("retry")
    )
    assert not result["accounting_complete"]
    cost = q._cost(q._read_operations(tmp_path, identity))
    assert cost["wall_seconds_total"] is None
    assert cost["budget_charge_seconds"] == 20
    assert cost["search_calls_started_observed"] == 2
    assert cost["search_calls_started_total"] is None


def test_absent_tasks_not_scored_as_zero(tmp_path):
    q.prepare(tmp_path, q.Policy())
    report = q.report(tmp_path)
    assert report["recorded"] == 0 and report["expected"] == 12
    assert report["status_counts"] == {"missing": 12}
    assert report["cost"]["wall_seconds_total"] is None
    assert report["actual_construction_export_qualified"] is False


def test_all_recorded_but_unavailable_does_not_claim_complete(tmp_path, monkeypatch):
    q.prepare(tmp_path, q.Policy())
    monkeypatch.setattr(q.execute, "fit_training", lambda *a, **k: None)
    for index in range(12):
        q.run_task(tmp_path, index)
    report = q.report(tmp_path)
    assert report["recorded"] == 12 and report["all_results_recorded"]
    assert report["status"] == "incomplete"
    assert report["status_counts"] == {"incomplete": 12}
    assert report["selection_counts"] == {"unavailable": 12}


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("m46_complete")
    q.prepare(root, q.Policy(durations=(0, 0.2), root_starts=((0.3, 1.3), (1.3, 0.3))))
    result = q.run_task(root, 0)
    assert result["status"] == "complete", result["selection"]
    assert result["selection"]["duration"] == 0.2
    return root, result


def test_known_structure_smoke_forecasts_and_diagnostics(completed):
    _, result = completed
    assert result["training_fit"]["fit_success"]
    assert result["windows"][0]["metrics"]["adequate"] is False
    assert result["windows"][1]["metrics"]["adequate"] is True
    assert result["followup"][0]["metrics"]["adequate"] is True
    assert result["diagnostics_available"] == result["diagnostics_expected"] == 12
    fit = result["training_fit"]
    for group in (result["windows"], result["followup"]):
        for window in group:
            for entry in window["trajectories"]:
                for attempt in entry["attempts"]:
                    assert (
                        attempt["inference"]["shared_parameters"]
                        == fit["shared_parameters"]
                    )
                    assert set(attempt["inference"]["roots"]) == {
                        "init_z_value",
                        "init_r_value",
                    }
    assert result["cost"]["search_calls_started_total"] > 0
    assert result["evaluation"]["truth_used_for_selection"] is False


def _snapshot(root: Path):
    return {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*.json")
    }


def test_exact_resume_does_not_repeat_work(completed, monkeypatch):
    root, result = completed
    before = _snapshot(root)
    monkeypatch.setattr(
        q.execute, "fit_training", lambda *a, **kw: pytest.fail("fit repeated")
    )
    monkeypatch.setattr(
        q.execute, "infer_prefix", lambda *a, **kw: pytest.fail("prefix repeated")
    )
    monkeypatch.setattr(
        q.verification, "forecast", lambda *a, **kw: pytest.fail("forecast repeated")
    )
    assert q.run_task(root, 0) == result
    assert _snapshot(root) == before
    assert q.report(root)["recorded"] == 1


def test_resealed_score_and_selection_tampering_rejected(completed):
    root, _ = completed
    path = root / "results/separated_clean_s0/result.json"
    original = path.read_bytes()
    changed = deepcopy(read_seal(path))
    changed["windows"][1]["metrics"]["scores"]["common"]["mean_nmse"] += 1
    path.unlink()
    seal(path, changed)
    try:
        with pytest.raises(ValueError, match="terminal artifact differs"):
            q.report(root)
    finally:
        path.write_bytes(original)


def test_scalar_score_rejects_nonfinite_predictions():
    row = PublicSplit.model_validate(
        q.controls.fixture("separated")["validation"]
    ).rows[0]
    prediction = {
        "trajectory_id": row.trajectory_id,
        "forecast_success": True,
        "time": list(row.time),
        "cutoff": 0.2,
        "predictions": {"y": [np.nan] * 41},
    }
    with pytest.raises(ValueError, match="invalid forecast"):
        q._scores(prediction, row, 0.2, {"training_scales": {"y": 1}}, q.Policy())
