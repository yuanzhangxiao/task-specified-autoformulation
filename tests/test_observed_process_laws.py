import json

import numpy as np
import pytest
from scipy.integrate import quad

from autoformalism.research.observed_process_laws import (
    fit_law_menu,
    interval_mean_law,
    predict_law,
)


def observed_events(truth):
    events = []
    for index in range(4):
        endpoints = np.array([-0.12, 0.01, 0.08, 0.4, 0.9, 2, 5]) * (1 + index / 6)
        left, right = endpoints[:-1], endpoints[1:]
        # Independently generate the interval means rather than using the fitter.
        rates = []
        for a, b in zip(left, right, strict=True):
            value, _ = quad(lambda x: truth(max(x, 0)), a, b, points=[0])
            rates.append(value / (b - a))
        events.append(
            {
                "trajectory_id": f"train_{index}",
                "left": left,
                "right": right,
                "mean_rate": rates,
            }
        )
    return events


@pytest.mark.parametrize(
    ("name", "truth"),
    [
        ("square_root", lambda x: 0.7 * np.sqrt(x)),
        ("linear", lambda x: 0.7 * x),
        ("power", lambda x: 0.7 * x**1.5),
        ("power", lambda x: 0.7 * x**0.72),
        ("saturating", lambda x: 0.7 * x / (0.11 + x)),
    ],
)
def test_recovers_full_law_on_other_training_events(name, truth):
    report = fit_law_menu(observed_events(truth))
    json.dumps(report, allow_nan=False)
    assert report["status"] == "ranked"
    candidate = report["candidates"][0]
    assert candidate["family"] == name
    assert candidate["cv_normalized_rmse"] < 1e-7
    assert all(f["held_out"] not in f["train_events"] for f in candidate["folds"])
    assert predict_law(candidate, [-1, 0, 0.2, 1.2]) == pytest.approx(
        [0, 0, truth(0.2), truth(1.2)], abs=1e-7
    )
    assert isinstance(predict_law(candidate, 1.0), float)


@pytest.mark.parametrize(
    ("family", "params", "truth"),
    [
        (
            "power",
            {"exponent": 0.74, "amplitude": 2.0, "source_scale": 1.0},
            lambda x: 2 * x**0.74,
        ),
        (
            "saturating",
            {"maximum": 3.0, "half_saturation": 0.23},
            lambda x: 3 * x / (0.23 + x),
        ),
    ],
)
def test_interval_average_across_threshold_and_small_increments(family, params, truth):
    law = {"family": family, "parameters": params}
    for a, b in [(-3, 2), (0, 4), (-0.1, -1), (1.2, 1.2), (1, 1 + 1e-9)]:
        expected, _ = quad(
            lambda t, a=a, b=b: truth(max(a + t * (b - a), 0)),
            0,
            1,
            points=[-a / (b - a)] if a * b < 0 else None,
        )
        assert interval_mean_law(law, a, b) == pytest.approx(expected, abs=1e-10)
        assert interval_mean_law(law, b, a) == pytest.approx(expected, abs=1e-10)


def test_fold_fit_independent_of_held_measurements_and_equal_event_weight():
    events = observed_events(lambda x: 0.7 * x**1.5)
    original = fit_law_menu(events)
    events[1]["mean_rate"] = [20 * rate for rate in events[1]["mean_rate"]]
    changed = fit_law_menu(events)
    for candidate in changed["candidates"]:
        other = next(
            c for c in original["candidates"] if c["family"] == candidate["family"]
        )
        fold = candidate["folds"][1]
        assert fold["parameters"] == other["folds"][1]["parameters"]
        assert fold["rmse"] > other["folds"][1]["rmse"]
    # Repeating all intervals of one event changes its sampling density only.
    for key in ("left", "right", "mean_rate"):
        events[1][key] = np.tile(events[1][key], 5)
    dense = fit_law_menu(events)
    for candidate, other in zip(
        dense["candidates"], changed["candidates"], strict=True
    ):
        assert candidate["family"] == other["family"]
        assert candidate["cv_rmse"] == pytest.approx(other["cv_rmse"], rel=1e-7)


def test_no_excitation_and_one_event_have_distinct_outcomes():
    assert fit_law_menu([])["status"] == "insufficient_excitation"
    event = {
        "trajectory_id": "train_one",
        "left": [0, -3],
        "right": [1, 1],
        "mean_rate": [1, 2],
    }
    assert fit_law_menu([event])["candidates"] == []
    events = observed_events(lambda x: x)
    result = fit_law_menu(events[:1])
    assert result["status"] == "fit_only"
    assert all(c["cv_rmse"] is None for c in result["candidates"])
    assert result["candidates"][0]["family"] == "linear"


def test_reject_nontraining_duplicate_and_nonfinite_inputs():
    event = observed_events(lambda x: x)[0]
    bad = dict(event, trajectory_id="val_001")
    with pytest.raises(ValueError, match="training"):
        fit_law_menu([bad])
    with pytest.raises(ValueError, match="once"):
        fit_law_menu([event, event])
    with pytest.raises(ValueError, match="finite"):
        fit_law_menu([dict(event, left=[np.nan])])
    with pytest.raises(ValueError, match="sized"):
        fit_law_menu([dict(event, right=[3])])


def _two_rates(first=8.0, second=2.0):
    return [
        {
            "trajectory_id": "train_lo",
            "left": [1.0],
            "right": [2.0],
            "mean_rate": [first],
            "absolute_error_bound": [0.5],
        },
        {
            "trajectory_id": "train_hi",
            "left": [3.0],
            "right": [2.0],
            "mean_rate": [second],
            "absolute_error_bound": 1.5,
        },
    ]


def test_ordering_rejects_whole_menu_with_bounded_training_witness():
    result = fit_law_menu(_two_rates())
    assert result["candidates"] == []
    assert result["status"] == "menu_contradicted"
    check = result["shape_check"]
    assert check["status"] == "contradicted"
    assert check["observed_rate_gap"] == 6
    assert check["gap_after_supplied_bounds"] == 4
    assert check["lower_source_interval"]["trajectory_id"] == "train_lo"
    assert check["higher_source_interval"]["source_range"] == [2.0, 3.0]
    assert json.loads(json.dumps(check, allow_nan=False)) == check


@pytest.mark.parametrize("first", [1.0, 4.0, np.nextafter(4.0, np.inf)])
def test_matching_or_within_bounds_and_roundoff_preserves_menu(first):
    result = fit_law_menu(_two_rates(first=first))
    assert result["shape_check"]["status"] == "no_witness"
    assert result["candidates"]


def test_overlapping_source_ranges_cannot_certify_rate_reversal():
    events = _two_rates()
    events[0]["right"] = [2.5]
    assert fit_law_menu(events)["shape_check"]["status"] == "no_witness"


def test_missing_error_bounds_warn_and_do_not_mask_certified_evidence():
    events = _two_rates()
    del events[0]["absolute_error_bound"]
    check = fit_law_menu(events)
    assert check["shape_check"]["status"] == "advisory"
    assert check["candidates"]
    # Same source point in two independent intervals also implies an ordering.
    events.append(
        {
            "trajectory_id": "train_extra",
            "left": [3, 3, 3],
            "right": [3, 3, 3],
            "mean_rate": [4, 14, 120],
            "absolute_error_bound": [0, 0, None],
        }
    )
    result = fit_law_menu(events)
    assert result["shape_check"]["status"] == "contradicted"
    assert result["shape_check"]["observed_rate_gap"] == 10
    assert result["candidates"] == []


@pytest.mark.parametrize("bad", [-1, np.inf, [0, 0]])
def test_invalid_rate_error_bounds(bad):
    events = _two_rates()
    events[0]["absolute_error_bound"] = bad
    with pytest.raises(ValueError, match="bound"):
        fit_law_menu(events)
