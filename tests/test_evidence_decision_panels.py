"""Train-only family evidence, input isolation, and receipt boundaries."""

from math import exp, sqrt

import pytest

from scripts import evidence_decision_panels as adapter


def test_dry_recession_accepts_known_law_rejects_incompatible_power():
    # Use two widely separated initial heads and an exact forward solution.
    readings = []
    for i, initial in enumerate((1.0, 4.0)):
        final = (1 / sqrt(initial) + 0.05) ** -2
        readings.append(
            {
                "event": f"train_{i}",
                "times": [0.0, 1.0],
                "heads": [initial, final],
                "area_over_dt": 10,
            }
        )
    good = adapter.dry_check(readings, 1.5, 1e-6)
    assert good["status"] == "inconclusive"
    assert good["k_min"] < 1 < good["k_max"]
    bad = adapter.dry_check(readings, 1.0, 1e-6)
    assert bad["status"] == "certified_reject"
    assert bad["lower_witness"]["event"] != bad["upper_witness"]["event"]
    wide = adapter.dry_check(readings, 1, 0.15)
    assert wide["status"] == "inconclusive"
    assert adapter.dry_check([], 1, 0.1)["intervals"] == 0
    with pytest.raises(ValueError, match="Invalid depth tolerance or power"):
        adapter.dry_check(readings, 1.0, -0.1)


def test_linear_recession_recovers_its_shared_coefficient():
    samples = [
        {
            "event": "train_0",
            "times": [0.0, 1.0],
            "heads": [h, h * exp(-0.3)],
            "area_over_dt": 2,
        }
        for h in [1.0, 2.0, 3.0]
    ]
    ck = adapter.dry_check(samples, 1.0, 1e-8)
    assert ck["k_min"] < 0.6 < ck["k_max"]
    rising = [{**samples[0], "heads": [1.0, 2.0]}]
    assert adapter.dry_check(rising, 1.0, 0.001)["status"] == "certified_reject"


def test_cumulative_positive_upstream_volume_and_downstream_forcing():
    row = {
        "trajectory_id": "train_000",
        "covariates": {
            "crest_up": 2,
            "initial_up": 1,
            "area_up": 1,
            "area_down": 2,
            "crest_down": 1,
        },
        "time": [0, 1, 2, 3, 4],
        "inputs": {"inflow_up": [0, 0, 1, 1, 0], "inflow_down": [0, 0, 1, 0, 0]},
        "targets": {"h_down": [2] * 5},
    }
    assert [r["times"] for r in adapter.dry_intervals([row])] == [[0, 1]]
    starts_above = row | {"covariates": row["covariates"] | {"initial_up": 3}}
    assert adapter.dry_intervals([starts_above]) == []
    with pytest.raises(ValueError, match="fitting trajectories"):
        adapter.dry_intervals([row | {"trajectory_id": "val_000"}])


def test_heldout_ids_cannot_be_used_to_select():
    adapter._valid_ids(["train_0"], ["train_1"], 1, 1)
    with pytest.raises(ValueError, match="split"):
        adapter._valid_ids(["train_0"], ["train_0"], 1, 1)
    with pytest.raises(ValueError, match="training observations"):
        adapter._valid_ids(["train_0"], ["val_0"], 1, 1)
    with pytest.raises(ValueError, match="Unknown model panel"):
        adapter.validation("basin/fold42", "simple")
