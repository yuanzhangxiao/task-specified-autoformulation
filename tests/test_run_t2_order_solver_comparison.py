from collections import Counter

import pytest

from scripts import run_t2_order_solver_comparison as campaign


def test_schedule_matches_starts_budgets_and_counterbalances_timing():
    jobs = list(campaign._schedule())
    assert len(jobs) == len(set(jobs)) == 16
    counts = Counter((arm, start) for _, arm, start in jobs)
    for start in campaign.STARTS:
        assert counts["reference", start] == counts["feasible", start] == 3
        assert counts["one_percent", start] == counts["unrestricted", start] == 1
    assert [j[1:] for j in jobs if j[0] == 1] == [j[1:] for j in jobs[:4]][::-1]


def test_timing_replicates_cannot_change_the_fitted_vector():
    with pytest.raises(ValueError, match="different parameters"):
        campaign._timing([{"parameters": {"p": 1}}, {"parameters": {"p": 2}}])


def test_replay_change_records_native_error_and_observation(monkeypatch):
    saved = {}
    for name, y in [("control", 3), ("half_gap", 6)]:
        observed = {k: [1, 2] for k in ("Gp", "U", "I")}
        saved[name] = {
            "key": name,
            "result": {
                "events": [
                    {
                        "event": "train_000",
                        "time": [0, 1],
                        "observed": observed,
                        "predicted": {k: [1, y] for k in observed},
                        "per_target_nmse": dict.fromkeys(observed, (y - 2) ** 2),
                    }
                ]
            },
        }
    monkeypatch.setattr(campaign, "sealed_read", lambda path: saved[path.stem])
    refs = {"arms": {k: {"replay": k} for k in saved}}
    metrics = campaign._replay_changes(refs)["events"][0]["targets"]["Gp"]
    assert metrics == {
        "control": 3.0,
        "half_gap": 6.0,
        "observed": 2.0,
        "delta_nmse": 15,
        "worst_increase": 3.0,
        "time_of_largest_absolute_error_increase": 1,
    }
    saved["half_gap"]["result"]["events"][0]["time"][0] = -1
    with pytest.raises(ValueError, match="aligned"):
        campaign._replay_changes(refs)
