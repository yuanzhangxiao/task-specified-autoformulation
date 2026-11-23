"""Verify post-freeze reporting using synthetic trajectories only."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scripts import positive_insulin_refit_report as trial
from scripts import t2_challenge_folds as metric


def inputs():
    rows = [{"trajectory_id": f"val_{n:03d}"} for n in range(4)]
    models = {
        tag: {
            "formula": {"parameters": {"p": 2}},
            "scales": dict.fromkeys(trial.inputs.TARGETS, 1.0),
        }
        for tag in trial.TAGS
    }
    frozen = {
        tag: {
            stage: {
                "parameters": {
                    "p": 1 if stage == "joint" and tag.endswith("positive_EGP") else 2
                },
                "diagnostics": {"submitted": stage},
            }
            for stage in trial.STAGES
        }
        for tag in models
    }
    return {"identity": "frozen", "models": frozen}, models, rows


def forecast(model, rows):
    events = []
    for row in rows:
        y = [1.0, 5.0, 5.0]
        yp = [1.0, 4.0, 5.0 + model["formula"]["parameters"]["p"]]
        curve = {
            "trajectory_id": row["trajectory_id"],
            "time": [0.0, 10.0, 17.5],
            "observed": dict.fromkeys(trial.inputs.TARGETS, y),
            "predicted": {"Gp": y, "I": yp, "U": y},
        }
        measured = metric.event_metrics(curve, model["scales"])
        events.append(curve | measured)
    return {"events": events, "summary": metric.aggregate(events, len(events))}


def test_matched_metrics_and_stage_identity_without_mutating_cached_replay():
    seal, models, rows = inputs()
    frozen_copy = deepcopy(seal)
    cache = [
        forecast(
            {
                "formula": {"parameters": {"p": k}},
                "scales": models[trial.TAGS[0]]["scales"],
            },
            rows,
        )
        for k in (1, 2)
    ]

    def cached(model, _rows):
        return cache[int(model["formula"]["parameters"]["p"]) - 1]

    result = trial.report(seal, models, rows, cached, "val")
    stage = result["models"]["015-positive_EGP"]["joint"]
    assert seal == frozen_copy
    assert "insulin_behavior" not in cache[0]["events"][0]
    assert stage["diagnostics"] == {"submitted": "joint"}
    assert stage["summary"]["max_abs_I_error_pmol_L"] == 1
    assert stage["summary"]["minimum_predicted_I_pmol_L"] == 1
    assert stage["summary"]["all_events_within_1_pmol_L"]
    assert stage["summary"]["median_abs_peak_time_error_min"] == 7.5
    assert stage["events"][0]["insulin_behavior"]["predicted_peak_at_boundary"]
    assert stage["events"][0]["insulin_behavior"]["observed_peak_time_min"] == 10.0
    assert stage["events"][0]["insulin_behavior"]["predicted_peak_pmol_L"] == 6
    comparisons = result["paired"]["015"]
    assert comparisons["start"]["child_minus_parent"]["pooled_nmse"] == 0
    assert comparisons["joint"]["lower_by_target"] == {"Gp": 0, "U": 0, "I": 4}
    assert comparisons["joint"]["child_minus_parent"]["pooled_nmse"] == pytest.approx(
        -1 / 3
    )
    assert (
        result["models"]["015-restoring"]["joint"]["summary"]["events_within_1_pmol_L"]
        == 0
    )


def test_unresolved_event_preserves_partial_status():
    seal, models, rows = inputs()

    def partial(model, rows):
        scored = forecast(model, rows)
        failed = {"event": rows[-1]["trajectory_id"], "status": "integration_failed"}
        events = [*scored["events"][:-1], failed]
        return {"events": events, "summary": metric.aggregate(events, len(events))}

    result = trial.report(seal, models, rows, partial, "val")
    base = result["models"]["002-restoring"]["joint"]["summary"]
    new = result["models"]["002-positive_EGP"]["joint"]["summary"]
    pairs = result["paired"]["002"]["joint"]
    assert pairs["compared"] == 3 and pairs["child_minus_parent"] is None
    assert new["per_target_nmse"] is None and new["max_abs_I_error_pmol_L"] is None
    assert base["all_events_within_1_pmol_L"] is False
    assert new["all_events_within_1_pmol_L"] is None


def test_incomplete_freeze_or_wrong_split_rejected_before_replay():
    seal, models, rows = inputs()

    def forbidden(*_):
        pytest.fail("Unexpected rollout")

    with pytest.raises(ValueError, match="Only training"):
        trial.report(seal, models, rows, forbidden, "test")
    with pytest.raises(ValueError, match="registered split"):
        trial.report(seal, models, rows[:-1], forbidden, "val")
    seal["models"]["015-positive_EGP"]["joint"]["parameters"]["p"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        trial.report(seal, models, rows, forbidden, "val")


def test_plot_uses_frozen_development_curves_and_refuses_missing_event(tmp_path):
    pytest.importorskip("matplotlib")
    from autoformalism.rebuttal.prefit_replay import sealed_write
    from scripts.plot_positive_insulin_refits import plot

    seal, models, rows = inputs()
    report = trial.report(seal, models, rows, forecast, "val")
    data = {"freeze": "test_only", "report": report}
    sealed_write(tmp_path / "val.json", data)
    plot(tmp_path / "val.json", tmp_path / "insulin.png")
    assert (tmp_path / "insulin.png").read_bytes().startswith(b"\x89PNG")
    report["models"]["015-positive_EGP"]["joint"]["events"].pop()
    sealed_write(tmp_path / "missing.json", data)
    with pytest.raises(ValueError, match="four unique development events"):
        plot(tmp_path / "missing.json", tmp_path / "invalid.png")
