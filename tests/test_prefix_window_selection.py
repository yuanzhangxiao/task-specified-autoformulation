"""M44 frozen acquisitions, common horizons, causal selection and durable resume."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_window_selection as q


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(q.public, "_source_identity", lambda: "a" * 64)


def test_three_acquisitions_six_starts_known_noise_exact_boundary():
    data = q.make_inputs()
    assert data == q.make_inputs()
    q._validate(data)
    assert len(data["records"]) == 6
    for task in q.roster():
        record = data["records"][task["task_id"]]
        for split, count in (("training", 3), ("validation", 4)):
            assert len(record[split]["rows"]) == count
            for noisy, clean in zip(
                record[split]["rows"],
                record["evaluation"]["clean"][split]["rows"],
                strict=True,
            ):
                assert noisy["targets"]["y"][0] == clean["targets"]["y"][0] == 0.4
                residual = (
                    np.asarray(noisy["targets"]["y"])[1:]
                    - np.asarray(clean["targets"]["y"])[1:]
                )
                assert bool(np.linalg.norm(residual) > 0) == (task["case"] == "noisy")
        assert "evaluation" not in record["problem"]
    a, b = (data["records"][f"noisy_r{rep}_s0"] for rep in (0, 1))
    assert a["training"] != b["training"]
    assert a["validation"] != b["validation"]


@pytest.mark.parametrize(
    "payload",
    [
        {"durations": []},
        {"durations": [0, 0.1, 0.1]},
        {"durations": [0.1, 1]},
        {"durations": [0, 4]},
        {"durations": [0, 0.15]},
        {"root_starts": [0.5, 0.5]},
        {"root_starts": [-0.1, 1]},
        {"validation_nmse_limit": 0},
        {"durations": [0, float("nan")]},
    ],
)
def test_invalid_policy_rejected(payload):
    with pytest.raises(ValidationError):
        q.Policy.model_validate(payload)


def test_prepare_no_redraw_identity_and_missing_report(tmp_path, monkeypatch):
    policy = q.Policy(durations=(0, 0.2))
    assert q.prepare(tmp_path, None, policy)["tasks"] == 6
    monkeypatch.setattr(q, "make_inputs", lambda: pytest.fail("redrew acquisitions"))
    q.prepare(tmp_path, None, policy)
    summary = q.report(tmp_path)
    assert summary["recorded"] == 0 and summary["status_counts"] == {"missing": 6}
    assert summary["selection_counts"] == {"unavailable": 2}
    assert summary["cost"]["wall_seconds_total"] is None
    with pytest.raises(ValueError, match="sealed artifact differs"):
        q.prepare(tmp_path, None, q.Policy())
    monkeypatch.setattr(q.public, "_source_identity", lambda: "b" * 64)
    with pytest.raises(ValueError, match="source/runtime"):
        q.verify(tmp_path)
    assert q.report(tmp_path) == summary


@pytest.mark.parametrize("change", ["fingerprint", "same_acquisition", "y0", "overlap"])
def test_input_errors_are_not_silently_accepted(change):
    data = q.make_inputs()
    record = data["records"]["clean_r0_s1"]
    row = record["validation"]["rows"][0]
    if change == "fingerprint":
        row["targets"]["y"][1] += 0.2
    elif change == "same_acquisition":
        row["targets"]["y"][1] += 0.2
        record["validation"] = q._fingerprint(record["validation"])
    elif change == "y0":
        row["targets"]["y"][0] += 0.2
        record["validation"] = q._fingerprint(record["validation"])
    else:
        row["trajectory_id"] = "train_0"
        record["validation"] = q._fingerprint(record["validation"])
    with pytest.raises(ValueError, match=r"prefix-window|same acquisition"):
        q._validate(data)


def test_real_sweep_common_horizon_no_truth_selection_and_resume(tmp_path, monkeypatch):
    policy = q.Policy(durations=(0, 0.2))
    q.prepare(tmp_path, None, policy)
    original = q._evaluate

    def evaluator(record, fit, windows, selection):
        saved = read_seal(tmp_path / "results/clean_r0_s0/selection.json")
        assert saved == selection
        return original(record, fit, windows, selection)

    monkeypatch.setattr(q, "_evaluate", evaluator)
    result = q.run_task(tmp_path, 0)
    assert result["status"] == "complete"
    assert result["selection"]["duration"] == 0.2
    for window in result["windows"]:
        for entry in window["trajectories"]:
            for attempt in entry["attempts"]:
                assert attempt["common_score"]["sample_count"] == 38
                assert attempt["common_score"]["score_start"] == 0.2
                assert (
                    attempt["inference"]["shared_parameters"]
                    == result["training_fit"]["shared_parameters"]
                )
                assert attempt["inference"]["root_start"] == {
                    "init_r_value": attempt["start"]
                }
                assert attempt["forecast"]["cutoff"] == window["duration"]
    assert result["windows"][0]["metrics"]["adequate"] is False
    assert result["windows"][1]["metrics"]["adequate"] is True
    assert result["cost"]["diagnostic_calls_completed_total"] == 8
    paths = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*.json")
    }
    for function in ("fit_training", "infer_prefix", "forecast"):
        monkeypatch.setattr(
            q.kernel, function, lambda *a, **kw: pytest.fail("repeated")
        )
    assert q.run_task(tmp_path, 0) == result
    assert paths == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*.json")
    }
    assert q.report(tmp_path)["recorded"] == 1
    tampered = deepcopy(result)
    tampered["windows"][1]["trajectories"][0]["attempts"][0]["common_score"][
        "normalized_mse"
    ] = 99
    path = tmp_path / "results/clean_r0_s0/result.json"
    path.unlink()
    seal(path, tampered)
    with pytest.raises(ValueError, match="retained score"):
        q.report(tmp_path)


def test_interrupted_training_remains_charged_without_budget_reset(
    tmp_path, monkeypatch
):
    q.prepare(tmp_path, None, q.Policy(durations=(0, 0.2)))

    def interrupted(*args, checkpoint):
        checkpoint({"calls": 2, "completed_calls": 1})
        raise KeyboardInterrupt

    monkeypatch.setattr(q.kernel, "fit_training", interrupted)
    with pytest.raises(KeyboardInterrupt):
        q.run_task(tmp_path, 0)
    summary = q.report(tmp_path)
    assert summary["cost"]["budget_charge_seconds"] == 40
    assert summary["cost"]["search_calls_started_observed"] == 2
    assert summary["cost"]["search_calls_completed_observed"] == 1
    monkeypatch.setattr(q.kernel, "fit_training", lambda *a, **kw: pytest.fail("retry"))
    result = q.run_task(tmp_path, 0)
    assert result["status"] == "incomplete"
    assert result["selection"]["status"] == "unavailable"
    assert result["cost"]["wall_seconds_total"] is None
    assert result["operations"]["training"]["status"] == "interrupted"
    assert q.run_task(tmp_path, 0) == result
