import numpy as np
import pytest

from scripts.trajectory_horizon_audit import event_horizons, summarize


def test_time_weighted_error_handles_irregular_observation_density():
    for t in (np.arange(17) / 16, np.array([0, 0.01, 0.02, 0.03, 0.125, 0.25, 0.5, 1])):

        def predict(n, time=t):
            return 2 * time[:n], 2 * time[:n]

        out = event_horizons("train_00", t, np.zeros_like(t), 2, predict)
        assert [r["nrmse"] for r in out] == pytest.approx(
            [r["fraction"] / np.sqrt(3) for r in out]
        )


def test_prefix_never_scores_future_observation_or_truncated_failure():
    t = [0, 0.2, 0.5, 0.7, 1]

    def predict(n):
        if n == 5:
            raise ArithmeticError("solver domain")
        return np.zeros(n), np.zeros(n) + (0.05 if n == 3 else 0)

    rows = event_horizons("train_00", t, [0, 0, 0, 0, 999], 1, predict)
    assert [r["status"] for r in rows] == [
        "insufficient_observations",
        "scored",
        "resolution_unverified",
        "rollout_failed",
    ]
    assert rows[1]["last_time"] == 0.2
    assert "nrmse" not in rows[-1]
    result = summarize({"train_00": rows})
    assert result["first_warning"]["train_00"]["fraction"] == 0.5
    assert result["horizons"][-1]["median_scored_nrmse"] is None


def test_split_and_resolution_provenance():
    with pytest.raises(ValueError, match="training"):
        event_horizons("val_00", [0, 1], [0, 0], 1, None)
    with pytest.raises(ValueError, match="coarse"):
        event_horizons("train_00", [0, 1], [0, 0], 1, lambda n: ([0, 0], None))
    with pytest.raises(ValueError, match="training SD"):
        event_horizons("train_00", [0, 1], [0, 0], 0, None)
