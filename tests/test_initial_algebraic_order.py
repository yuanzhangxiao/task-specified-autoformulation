"""An initial ordering witness concerns the supplied parameter vector only."""

import pytest

from autoformalism.research.initial_algebraic_order import check_initial_order


def event(name, observed, predicted):
    return {
        "event": name,
        "time": [0, 9],
        "observed": {"y": [observed, 999]},
        "predicted": {"y": [predicted, -999]},
    }


def test_reversal_witness_and_tied_prediction_use_only_initial_values():
    rows = [event("train_a", 1.5, 3), event("train_b", 2, 1), event("train_c", 4, 1)]
    out = check_initial_order(rows, "y", tolerance=0.1)
    assert out == check_initial_order(rows[::-1], "y", tolerance=0.1)
    assert out["scope"] == "fixed_parameter_vector_at_time_zero"
    assert out["status"] == "ordering_conflict"
    assert out["comparable_pairs"] == out["pair_count"] == 3
    assert out["reversed_pairs"] == 2
    assert out["tied_prediction_pairs"] == 1
    assert out["worst_order_witness"] == {
        "low_event": "train_a",
        "high_event": "train_c",
        "observed_low": 1.5,
        "observed_high": 4.0,
        "predicted_low": 3.0,
        "predicted_high": 1.0,
        "max_abs_error_lower_bound": 2.25,
    }
    assert out["events_exceeding_tolerance"] == 3
    assert out["max_abs_error"] == 3 and out["worst_error_event"] == "train_c"


def test_exact_allowance_boundary_and_observed_ties_do_not_impose_order():
    rows = [event("train_a", 1, 3), event("train_b", 1, 3), event("train_c", 3, 1)]
    out = check_initial_order(rows, "y", tolerance=1)
    assert out["status"] == "no_ordering_conflict"
    assert out["worst_order_witness"] is None and out["comparable_pairs"] == 0
    assert out["unresolved_observed_pairs"] == 3 and out["observed_tie_pairs"] == 1
    assert out["events_exceeding_tolerance"] == 3
    rows[2]["predicted"]["y"][0] = 4
    result = check_initial_order(rows, "y", tolerance=0.5)
    assert result["comparable_pairs"] == 2 and result["reversed_pairs"] == 0
    assert result["status"] == "no_ordering_conflict"


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        check_initial_order([event("train_a", 0, 2)], "y", tolerance=tolerance)


@pytest.mark.parametrize(
    "changes",
    [
        {"event": "train_"},
        {"event": "val_00"},
        {"event": "test_00"},
        {"split": "val"},
        {"event": None},
        {"trajectory_id": "val_a"},
        {"time": [1, 2]},
        {"time": []},
        {"time": [False, 2]},
        {"observed": {"y": [float("nan"), 1]}},
        {"predicted": {"y": ["2", 1]}},
        {"predicted": {}},
        {"predicted": {"y": 8}},
        {"predicted": {"y": [0]}},
    ],
)
def test_rejects_incomplete_or_nontraining_rows(changes):
    with pytest.raises(ValueError):
        check_initial_order([event("train_a", 0, 2) | changes], "y", tolerance=0)


def test_missing_duplicate_and_single_event():
    example = event("train_a", 1, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        check_initial_order([example, example], "y", tolerance=0)
    with pytest.raises(ValueError):
        check_initial_order([], "y", tolerance=0)
    example["trajectory_id"] = example.pop("event")
    single = check_initial_order([example], "y", tolerance=0)
    assert single["pair_count"] == single["events_exceeding_tolerance"] == 0
