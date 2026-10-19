"""Adversarial M44 selection guards without numerical fitting or evaluator truth."""

from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_window_diagnostics as diagnostic
from autoformalism.fitting import prefix_window_selection as selection
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicTrajectory


def entries(duration: float = 0.2) -> list[dict]:
    """Make four complete rows with two starts and authentic score receipts."""
    policy = selection.Policy()
    result = []
    time = np.linspace(0, 4, 41).tolist()
    for index in range(4):
        name = f"val_{index}"
        row = PublicTrajectory(
            trajectory_id=name, time=tuple(time), targets={"y": (0.0,) * len(time)}
        )
        attempts = []
        for start_index, start in enumerate(policy.root_starts):
            inference = {
                "protocol": "prefix-inference-1",
                "fit_success": True,
                "normalized_mse": 0.001 + start_index * 0.001,
                "roots": {"init_r_value": 0.8},
                "root_start": {"init_r_value": start},
                "training_scales": {"y": 1.0},
                "shared_parameters": {"theta": 0.35, "init_z_phi": 0.7},
                "cutoff": duration,
                "prefix_end_time": duration,
                "fit_sha256": "same-frozen-fit",
                "prefix_sha256": f"prefix-{name}-{duration}",
                "rank_diagnostic": {"coordinates": 1, "rank": 1},
            }
            forecast = {
                "protocol": "prefix-forecast-1",
                "forecast_success": True,
                "trajectory_id": name,
                "cutoff": duration,
                "time": time,
                "predictions": {"y": [0.02] * len(time)},
                "inference_sha256": content_sha256(inference),
                "schedule_sha256": f"schedule-{name}-{duration}",
            }
            attempts.append(
                {
                    "label": (
                        f"w{policy.durations.index(duration)}_{name}_s{start_index}"
                    ),
                    "start": start,
                    "inference": inference,
                    "forecast": forecast,
                    "common_score": diagnostic.common_horizon_score(
                        forecast, row, policy.durations[-1], {"y": 1.0}
                    ),
                    "own_suffix_score_descriptive_only": {"normalized_mse": 100.0},
                }
            )
        result.append(
            {
                "trajectory_id": name,
                "attempts": attempts,
                "chosen_start": 0,
                "information": {"available": True, "rank": 1},
            }
        )
    return result


def refresh_receipts(data: list[dict]) -> None:
    """Keep endpoint hashes valid when a test changes actual endpoint contents."""
    for row in data:
        for attempt in row["attempts"]:
            attempt["forecast"]["inference_sha256"] = content_sha256(
                attempt["inference"]
            )
            attempt["common_score"]["forecast_sha256"] = content_sha256(
                attempt["forecast"]
            )
            attempt["common_score"]["inference_sha256"] = attempt["forecast"][
                "inference_sha256"
            ]
            attempt["common_score"]["schedule_sha256"] = attempt["forecast"][
                "schedule_sha256"
            ]


def assert_rejected(data: list[dict], duration: float = 0.2) -> None:
    """Malformed/incomplete windows may fail explicitly, but cannot be selected."""
    try:
        metric = selection.window_metrics(duration, data, selection.Policy())
    except ValueError:
        return
    assert metric["complete"] is False
    assert metric["adequate"] is False
    assert metric["mean_validation_nmse"] is None


def metric_grid() -> list[dict]:
    """Use the compact aggregate contract used by the campaign reporter."""
    return [
        {"duration": duration, "complete": True, "adequate": duration >= 0.2}
        for duration in selection.Policy().durations
    ]


def test_complete_window_and_shortest_adequate_selection():
    metric = selection.window_metrics(0.2, entries(), selection.Policy())
    assert metric["complete"] and metric["adequate"]
    assert metric["validation_rows_expected"] == 4
    assert len(metric["per_row_validation_nmse"]) == 4
    assert metric["mean_validation_nmse"] == pytest.approx(0.0004)
    decision = selection.select_duration(metric_grid(), selection.Policy())
    assert decision["status"] == "selected" and decision["duration"] == 0.2
    assert decision["scoring_start"] == 2
    assert decision["selection_data"] == "validation_common_suffix"
    assert decision["unbiased_generalization_estimate"] is False


@pytest.mark.parametrize("edit", ["missing", "duplicate", "unexpected", "extra"])
def test_exact_validation_roster_is_required(edit):
    data = entries()
    if edit == "missing":
        data.pop()
    elif edit == "duplicate":
        data[3] = deepcopy(data[2])
    elif edit == "unexpected":
        data[3]["trajectory_id"] = "val_99"
    else:
        data.append(deepcopy(data[0]))
    assert_rejected(data)


@pytest.mark.parametrize("edit", ["missing", "duplicate", "reordered", "wrong"])
def test_exact_predeclared_start_roster_is_required(edit):
    data = entries()
    row = data[0]
    if edit == "missing":
        row["attempts"].pop()
    elif edit == "duplicate":
        row["attempts"][1] = deepcopy(row["attempts"][0])
    elif edit == "reordered":
        row["attempts"].reverse()
    else:
        row["attempts"][1]["start"] = 7.0
    assert_rejected(data)


@pytest.mark.parametrize("component", ["inference", "forecast", "common_score"])
@pytest.mark.parametrize("missing", [False, True])
def test_failed_or_missing_start_cannot_reduce_the_denominator(component, missing):
    data = entries()
    attempt = data[2]["attempts"][1]
    if missing:
        attempt[component] = None
    else:
        key = {
            "inference": "fit_success",
            "forecast": "forecast_success",
            "common_score": "available",
        }[component]
        attempt[component][key] = False
    assert_rejected(data)


def test_cached_chosen_start_must_match_prefix_loss_winner():
    data = entries()
    # The second start has a better suffix score but a worse prefix objective.
    data[0]["attempts"][1]["common_score"]["normalized_mse"] = 0.0
    data[0]["chosen_start"] = 1
    assert_rejected(data)


def test_endpoint_choice_uses_only_prefix_loss_and_breaks_ties_by_order():
    attempts = entries()[0]["attempts"]
    attempts[0]["common_score"]["normalized_mse"] = 1e9
    attempts[1]["common_score"]["normalized_mse"] = 0.0
    assert selection._choose_prefix_start(attempts) == 0
    attempts[1]["inference"]["normalized_mse"] = 0.001
    assert selection._choose_prefix_start(attempts) == 0
    attempts[0]["inference"]["normalized_mse"] = 0.003
    assert selection._choose_prefix_start(attempts) == 1


@pytest.mark.parametrize("loss", [np.nan, np.inf, -np.inf, -0.1, None])
def test_invalid_prefix_objective_is_never_a_winning_start(loss):
    attempts = entries()[0]["attempts"]
    attempts[0]["inference"]["normalized_mse"] = loss
    assert selection._choose_prefix_start(attempts) == 1


def test_no_verified_prefix_start_has_no_winner():
    attempts = entries()[0]["attempts"]
    attempts[0]["inference"] = None
    attempts[1]["inference"]["fit_success"] = False
    assert selection._choose_prefix_start(attempts) is None


@pytest.mark.parametrize("loss", [np.nan, np.inf, -np.inf, -0.1, None])
def test_invalid_common_score_never_makes_a_window_complete(loss):
    data = entries()
    data[1]["attempts"][0]["common_score"]["normalized_mse"] = loss
    assert_rejected(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("forecast_cutoff", 1.0),
        ("score_start", 1.0),
        ("sample_count", 0),
        ("sample_count", 19),
        ("horizon_sample_count", 19),
        ("forecast_sha256", "stale-forecast"),
    ],
)
def test_common_score_must_match_the_actual_common_horizon(field, value):
    data = entries()
    data[1]["attempts"][0]["common_score"][field] = value
    assert_rejected(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("trajectory_id", "val_99"),
        ("cutoff", 0.1),
        ("inference_sha256", "stale-inference"),
    ],
)
def test_forecast_must_belong_to_its_row_window_and_endpoint(field, value):
    data = entries()
    data[0]["attempts"][0]["forecast"][field] = value
    assert_rejected(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cutoff", 0.1),
        ("root_start", {"init_r_value": 7.0}),
        ("training_scales", {"y": 0.0}),
        ("training_scales", {"y": float("nan")}),
        ("training_scales", {"y": 2.0}),
    ],
)
def test_inference_boundary_and_normalization_must_match(field, value):
    data = entries()
    data[0]["attempts"][1]["inference"][field] = value
    assert_rejected(data)


def test_different_time_grids_cannot_be_compared_pointwise():
    data = entries()
    data[0]["attempts"][1]["forecast"]["time"] = [
        value * 1.1 for value in data[0]["attempts"][1]["forecast"]["time"]
    ]
    assert_rejected(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_prediction_cannot_be_hidden_by_a_small_cached_score(value):
    data = entries()
    data[0]["attempts"][1]["forecast"]["predictions"]["y"][30] = value
    assert_rejected(data)


def test_one_unstable_row_is_not_diluted_by_other_rows():
    data = entries()
    data[0]["attempts"][0]["forecast"]["predictions"]["y"] = [0.0] * 41
    data[0]["attempts"][1]["forecast"]["predictions"]["y"] = [0.019] * 41
    refresh_receipts(data)
    metric = selection.window_metrics(0.2, data, selection.Policy())
    assert metric["complete"]
    assert metric["maximum_start_prediction_spread"] == pytest.approx(0.019)
    assert metric["adequate"] is False  # Pooling four rows would give 0.0095.


def test_accuracy_and_stability_thresholds_are_inclusive():
    data = entries()
    for row in data:
        row["attempts"][0]["forecast"]["predictions"]["y"] = [0.0] * 41
        row["attempts"][1]["forecast"]["predictions"]["y"] = [0.01] * 41
        row["attempts"][0]["common_score"]["normalized_mse"] = 0.01
        row["attempts"][0]["common_score"]["per_target_normalized_mse"] = {"y": 0.01}
    refresh_receipts(data)
    metric = selection.window_metrics(0.2, data, selection.Policy())
    assert metric["mean_validation_nmse"] == 0.01
    assert metric["maximum_start_prediction_spread"] == 0.01
    assert metric["adequate"]


def test_earlier_forecast_errors_cannot_change_common_horizon_stability():
    data = entries()
    for row in data:
        row["attempts"][0]["forecast"]["predictions"]["y"][:21] = [1000.0] * 21
        row["attempts"][1]["forecast"]["predictions"]["y"][:21] = [-1000.0] * 21
    refresh_receipts(data)
    metric = selection.window_metrics(0.2, data, selection.Policy())
    assert metric["complete"] and metric["adequate"]
    assert metric["maximum_start_prediction_spread"] == 0


def test_equivalent_predictions_do_not_require_unique_roots_or_full_rank():
    data = entries()
    for row in data:
        row["information"] = {"available": True, "rank": 0, "status": "uninformative"}
        for index, attempt in enumerate(row["attempts"]):
            attempt["inference"]["roots"]["init_r_value"] = float(index + 1)
            attempt["inference"]["rank_diagnostic"] = {"coordinates": 1, "rank": 0}
    refresh_receipts(data)
    metric = selection.window_metrics(0.2, data, selection.Policy())
    assert metric["complete"] and metric["adequate"]
    assert metric["maximum_start_root_spread"] == 1


@pytest.mark.parametrize("edit", ["missing", "duplicate", "reordered", "failed"])
def test_any_incomplete_duration_grid_prevents_selection(edit):
    metrics = metric_grid()
    if edit == "missing":
        metrics.pop()
    elif edit == "duplicate":
        metrics[-1] = deepcopy(metrics[-2])
    elif edit == "reordered":
        metrics.reverse()
    else:
        metrics[-1]["complete"] = False
    decision = selection.select_duration(metrics, selection.Policy())
    assert decision["status"] == "unavailable"
    assert decision["duration"] is None


def test_complete_but_inadequate_grid_is_distinct_from_missing_work():
    metrics = metric_grid()
    for metric in metrics:
        metric["adequate"] = False
    decision = selection.select_duration(metrics, selection.Policy())
    assert decision["status"] == "no_adequate_window"
    assert decision["duration"] is None


@pytest.mark.parametrize(
    "remove",
    [
        lambda item: item.pop("trajectory_id"),
        lambda item: item["attempts"][0].pop("start"),
        lambda item: item["attempts"][0]["inference"].pop("normalized_mse"),
        lambda item: item["attempts"][0]["forecast"].pop("time"),
        lambda item: item["attempts"][0]["common_score"].pop("normalized_mse"),
    ],
)
def test_missing_required_fields_fail_explicitly_or_mark_window_incomplete(
    remove: Callable[[dict], None],
):
    data = entries()
    remove(data[0])
    assert_rejected(data)
