"""M45 preserves fits/history and recovers only explicitly failed verification."""

import shutil
from copy import deepcopy

import pytest

pytest.importorskip("casadi")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_window_selection as old
from autoformalism.fitting import prefix_window_supplement as q


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(q.public, "_source_identity", lambda: "test-source")


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    source = tmp_path_factory.mktemp("m44")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(q.public, "_source_identity", lambda: "test-source")
        old.prepare(source, None, old.Policy(durations=(0, 0.2)))
        original_infer = old.kernel.infer_prefix
        original_forecast = old.kernel.forecast
        failures = {"inference": False, "forecast": False}

        def infer(*args, **kwargs):
            result = original_infer(*args, **kwargs)
            if not failures["inference"] and result["cutoff"] == 0.2:
                failures["inference"] = True
                result.update(
                    fit_success=False,
                    parameters=None,
                    roots=None,
                    verification_error="injected historical timeout",
                )
            return result

        def forecast(*args, **kwargs):
            result = original_forecast(*args, **kwargs)
            if failures["inference"] and not failures["forecast"]:
                failures["forecast"] = True
                result.update(
                    forecast_success=False,
                    predictions={},
                    states={},
                    verification_error="injected historical timeout",
                )
            return result

        patch.setattr(old.kernel, "infer_prefix", infer)
        patch.setattr(old.kernel, "forecast", forecast)
        for index in range(6):
            old.run_task(source, index)
    return source


def bytes_and_times(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*.json")
    }


def test_real_recovery_keeps_fits_candidates_and_history(
    historical, tmp_path, monkeypatch
):
    before = bytes_and_times(historical)
    root = tmp_path / "m45"
    assert q.prepare(root, historical, q.Policy())["tasks"] == 1
    baseline = q.report(root)
    assert baseline["source_status_counts"] == {"incomplete": 1, "complete": 5}
    assert baseline["supplemental_cost"]["wall_seconds_total"] is None
    # No search, training fit or original inference/forecast can run in recovery.
    for name in ("fit_training", "infer_prefix", "forecast"):
        monkeypatch.setattr(
            old.kernel,
            name,
            lambda *a, **k: pytest.fail("historical computation repeated"),
        )
    result = q.run_task(root, 0)
    assert result["status"] == "complete"
    assert result["recovered_forecast_gaps"] == 2
    assert result["selection"]["duration"] == 0.2
    report = q.report(root)
    assert report["source_status_counts"] == baseline["source_status_counts"]
    assert report["source_cost"] == baseline["source_cost"]
    assert report["supplemental_cost"]["search_calls_started_total"] == 0
    assert report["supplemental_cost"]["verification_calls_completed_total"] == 6
    assert (
        report["combined_cost"]["search_calls_started_total"]
        == baseline["source_cost"]["search_calls_started_total"]
    )
    assert (
        report["combined_cost"]["mixed_start_counter_total"]
        == baseline["source_cost"]["verification_calls_started_total"] + 6
    )
    assert report["fitting_performed"] is False
    assert report["test_data_opened"] is False
    assert bytes_and_times(historical) == before
    terminal = bytes_and_times(root)
    monkeypatch.setattr(
        q.verifier, "verify_inference", lambda *a, **k: pytest.fail("retry")
    )
    monkeypatch.setattr(q.verifier, "forecast", lambda *a, **k: pytest.fail("retry"))
    assert q.run_task(root, 0) == result
    assert bytes_and_times(root) == terminal
    # A portable copy does not require the old absolute source path.
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert q.run_task(relocated, 0) == result
    assert q.report(relocated) == report


def test_changed_policy_source_and_saved_results_fail(historical, tmp_path):
    root = tmp_path / "m45"
    q.prepare(root, historical, q.Policy())
    q.prepare(root, None, q.Policy())
    with pytest.raises(ValueError, match="policy differs"):
        q.prepare(
            root,
            None,
            q.Policy(execution=q.verifier.ExecutionPolicy(overall_seconds=61)),
        )
    q.run_task(root, 0)
    path = root / "results/clean_r0_s0/result.json"
    value = read_seal(path)
    value["selection"]["duration"] = 0
    path.unlink()
    seal(path, value)
    with pytest.raises(ValueError, match="terminal artifact"):
        q.report(root)
    (root / "source_snapshot/inputs.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot/identity"):
        q.verify(root)


def test_interruption_is_charged_and_not_retried(historical, tmp_path, monkeypatch):
    root = tmp_path / "m45"
    q.prepare(root, historical, q.Policy())

    def interrupted(*args, checkpoint):
        checkpoint({"verification_calls_started": 1, "verification_calls_completed": 0})
        raise KeyboardInterrupt

    monkeypatch.setattr(q.verifier, "verify_inference", interrupted)
    with pytest.raises(KeyboardInterrupt):
        q.run_task(root, 0)
    monkeypatch.setattr(
        q.verifier, "verify_inference", lambda *a, **kw: pytest.fail("retry")
    )
    result = q.run_task(root, 0)
    assert result["status"] == "incomplete"
    assert result["selection"]["status"] == "unavailable"
    report = q.report(root)
    assert report["supplemental_cost"]["verification_calls_started_observed"] >= 1
    assert report["supplemental_cost"]["verification_calls_started_total"] is None
    assert report["source_cost"]["operations"] > 0


def test_failed_supplement_not_replaced_by_old_failure_or_good_subset(
    historical, tmp_path, monkeypatch
):
    root = tmp_path / "m45"
    q.prepare(root, historical, q.Policy())

    def failed(*args, **kwargs):
        raise TimeoutError("overall verification deadline")

    monkeypatch.setattr(q.verifier, "verify_inference", failed)
    monkeypatch.setattr(q.verifier, "forecast", failed)
    result = q.run_task(root, 0)
    assert result["status"] == "incomplete"
    assert result["recovered_forecast_gaps"] == 0
    assert result["selection"]["duration"] is None
    assert q.report(root)["selections"][0]["status"] == "unavailable"


def test_no_candidate_cannot_trigger_search_or_silent_recovery():
    row = {
        "windows": [
            {
                "duration": 1,
                "trajectories": [
                    {
                        "trajectory_id": "x",
                        "attempts": [{"inference": None, "forecast": None}],
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="no search retry"):
        q._gaps(row)


def test_extra_operation_cannot_enter_report(historical, tmp_path):
    root = tmp_path / "m45"
    q.prepare(root, historical, q.Policy())
    plan, _ = q.verify(root)
    _, data, rows = q._source_rows(root / "source_snapshot")
    row = rows[0]
    policy = old.Policy.model_validate(
        read_seal(root / "source_snapshot/plan.json")["policy"]
    )
    with pytest.raises(ValueError, match="unexpected supplemental operation"):
        q._compose(row, data["records"][row["task_id"]], policy, {"unapproved": {}})
    assert plan["tasks"][0]["task_id"] == row["task_id"]


@pytest.mark.parametrize("status", ["unfinished", "interrupted"])
def test_partial_checkpoint_timing_is_not_an_exact_total(status):
    operation = {
        "identity": {"stage": "x_verify"},
        "status": status,
        "value": None,
        "accounting_complete": False,
        "budget_charge_seconds": 60,
        "wall_seconds": None,
        "cpu_seconds": None,
        "calls": None,
        "observed_progress": {
            "verification_calls_attempted": 1,
            "timing": {
                "callback_wall_seconds": 0.2,
                "callback_cpu_seconds": 0.01,
                "numerical_active_wall_seconds": 0.1,
                "numerical_wall_seconds": 0.05,
                "numerical_cpu_seconds": 0.03,
            },
        },
    }
    cost = q._supplement_cost({"x_verify": operation})
    assert cost["callback_wall_seconds_observed"] == 0.2
    assert cost["callback_wall_seconds_total"] is None
    assert cost["numerical_active_wall_seconds_total"] is None
    assert cost["verification_calls_attempted_observed"] == 1
    assert cost["verification_calls_attempted_total"] is None


def test_operation_interrupt_receipt_and_unknown_abrupt_termination(tmp_path):
    identity = {"id": 1}

    def interrupt(save):
        err = KeyboardInterrupt()
        err.verification_receipt = {
            "verification_calls_started": 1,
            "timing": {"elapsed_wall_seconds": 0.1},
        }
        raise err

    with pytest.raises(KeyboardInterrupt):
        q._operation(tmp_path, identity, "verify", 60, interrupt)
    receipt = q._operation(
        tmp_path, identity, "verify", 60, lambda _: pytest.fail("retry")
    )
    assert receipt["status"] == "interrupted"
    assert receipt["value"]["verification_calls_started"] == 1
    folder = tmp_path / "operations/abrupt"
    seal(
        folder / "started.json",
        {"campaign": identity, "stage": "abrupt", "allowance_seconds": 60},
    )
    result = q._operation(
        tmp_path, identity, "abrupt", 60, lambda _: pytest.fail("retry")
    )
    assert result["wall_seconds"] is None
    assert not result["accounting_complete"]
    assert result["budget_charge_seconds"] == 60


def test_reject_overlapping_roots_and_missing_initial_source(historical, tmp_path):
    with pytest.raises(ValueError, match="separate directories"):
        q.prepare(historical / "bad", historical, q.Policy())
    with pytest.raises(ValueError, match="source root"):
        q.prepare(tmp_path / "new", None, q.Policy())


def test_compose_rejects_root_substitution(historical, tmp_path):
    root = tmp_path / "m45"
    q.prepare(root, historical, q.Policy())
    q.run_task(root, 0)
    plan, _ = q.verify(root)
    source_plan, data, rows = q._source_rows(root / "source_snapshot")
    source = rows[0]
    identity = q._identity(plan, plan["tasks"][0], source)
    operations = old._read_operations(root / "results/clean_r0_s0", identity)
    for key, op in operations.items():
        if key.endswith("_verify"):
            op["value"] = deepcopy(op["value"])
            op["value"]["roots"]["init_r_value"] += 0.01
    with pytest.raises(ValueError, match="changed saved root"):
        q._compose(
            source,
            data["records"][source["task_id"]],
            old.Policy.model_validate(source_plan["policy"]),
            operations,
        )
