"""Train-only basin law reconstruction and post-initial open-loop scoring."""

import numpy as np
import pytest

from autoformalism.research import basin_observed_process as basin


def event(index, initial_up, initial_down, prefix="train"):
    """Generate exact linear, freely draining synthetic trajectories at zero inflow."""
    t = np.linspace(0, 15, 31)
    up = max(initial_up - 0.3, 0)
    # Areas 10 and 20, rate gains 1 and 4. The upstream discharge decays at 0.1.
    down = (
        0.2
        + (initial_down - 0.2) * np.exp(-0.2 * t)
        + 0.5 * up * (np.exp(-0.1 * t) - np.exp(-0.2 * t))
    )
    return {
        "trajectory_id": f"{prefix}_{index:03}",
        "time": t.tolist(),
        "targets": {"h_down": down.tolist()},
        "inputs": {"inflow_up": [0] * len(t), "inflow_down": [0] * len(t)},
        "auxiliaries": {},
        "covariates": {
            "initial_up": initial_up,
            "area_up": 10,
            "area_down": 20,
            "crest_up": 0.3,
            "crest_down": 0.2,
        },
    }


def test_conditional_reconstruction_and_unseen_event_uses_only_initial_depth():
    rows = [
        event(0, 0.1, 0.5),
        event(1, 0.2, 0.9),
        event(2, 0.7, 1.2),
        event(3, 0.8, 0.6),
        event(4, 0.9, 0.8),
    ]
    held = event(5, 0.75, 0.7)
    result = basin.run_fold(rows, held)
    assert result["training_events"] == [r["trajectory_id"] for r in rows]
    assert result["outlet_evidence"]["isolated_intervals"] == {
        "train_000": 30,
        "train_001": 30,
        "train_002": 0,
        "train_003": 0,
        "train_004": 0,
    }
    control = result["models"][1]
    assert result["held_time_min"] == held["time"]
    assert result["held_observed_depth_m"] == held["targets"]["h_down"]
    for model in result["models"]:
        assert model["upstream_evidence"] == "conditional_on_fitted_downstream_law"
        errors = np.subtract(
            model["held_predicted_depth_m"], result["held_observed_depth_m"]
        )
        assert np.sqrt(np.mean(errors[1:] ** 2)) == pytest.approx(model["held_rmse_m"])
    assert control["status"] == "scored"
    assert control["held_rmse_m"] < 0.001 and control["held_samples"] == 30
    assert control["upstream"]["family"] == control["downstream"]["family"] == "linear"
    assert control["reconstruction"]["train_003"]["conditional_intervals"] == 30
    assert control["reconstruction"]["train_000"]["conditional_intervals"] == 0
    # An arbitrary shift in every subsequent held measurement leaves the rollout
    # and both selected laws unchanged, but increases the measured forecast error.
    shifted = event(5, 0.75, 0.7)
    shifted["targets"]["h_down"][1:] = [2.0] * 30
    assert np.array_equal(
        basin._depth_forecast(control, basin._public_event(held)),
        basin._depth_forecast(control, basin._public_event(shifted)),
    )
    changed = basin.run_fold(rows, shifted)["models"][1]
    assert changed["held_rmse_m"] > 1 and changed["upstream"] == control["upstream"]


def test_nonisolated_or_nonpublic_events_cannot_give_direct_outlet_measurements():
    active = [event(1, 0.6, 1.0), event(2, 0.8, 1.2)]
    result = basin.run_fold(active, event(0, 0.7, 0.9))
    assert result["outlet_evidence"]["head_range_m"] is None
    assert all(m["status"] == "outlet_unavailable" for m in result["models"])
    assert all(m["held_nmse"] is None for m in result["models"])
    with pytest.raises(ValueError, match="appears in training"):
        basin.run_fold(active, active[0])
    with pytest.raises(ValueError, match="training event"):
        basin.run_fold(active, event(7, 0.7, 0.9, "test"))
    hidden = event(1, 0.6, 1.0)
    hidden["auxiliaries"] = {"h_up": [0.6] * 31}
    with pytest.raises(ValueError, match="public runoff"):
        basin.fit_basin_laws([hidden])


def test_isolation_uses_both_endpoints_and_cumulative_inflow():
    row = event(1, 0.2, 1.0)
    # Available upstream storage is 1 m3. A 3 m3/min pulse during a pair
    # of half-minute intervals fills it before the second interval ends.
    row["inputs"]["inflow_up"][1] = 3.0
    interval = basin._Intervals.collect(basin._public_event(row))
    assert interval.isolated.tolist() == [True] + [False] * 29
