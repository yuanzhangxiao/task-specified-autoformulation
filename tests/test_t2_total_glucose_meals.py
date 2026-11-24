from copy import deepcopy

import pytest

from scripts import t2_total_glucose_meals as meal


def _row(event, rates, insulin=False):
    n = len(rates)
    return {
        "trajectory_id": event,
        "time": list(range(n)),
        "inputs": {meal.MEAL: rates, meal.INFUSION: [float(insulin)] * n},
        "targets": {k: [2.0] * n for k in meal.TARGETS},
    }


def _curve(row, error):
    obs = row["targets"]
    return {
        "event": row["trajectory_id"],
        "time": row["time"],
        "observed": obs,
        "predicted": {k: [v + error for v in y] for k, y in obs.items()},
        "qualification": {"eligible": True, "numerically_resolved": True},
        "meal_grams": meal.meal_schedule(row)["total_g"],
        "per_target_nmse": dict.fromkeys(obs, error**2),
    }


def _fixture():
    rows, curves = {}, {}
    for held in meal.HOLDOUTS:
        rows[held] = [
            _row(
                event,
                [0, 1, 0]
                if event == "train_001"
                else ([0, 0, 1, 1, 0, 0] if event == "train_002" else [0, 0, 0]),
            )
            for event in meal.fit_ids(held)
        ]
        curves[held] = {
            arm: {
                r["trajectory_id"]: _curve(
                    r, int(r["trajectory_id"] == "train_001" and arm == "total")
                )
                for r in rows[held]
            }
            for arm in meal.ARMS
        }
    return rows, curves


def test_rate_integral_defines_meal_dose_spacing_and_insulin_regime():
    row = _row("train_015", [0, 2, 0, 0, 4, 0], insulin=True)
    schedule = meal.meal_schedule(row)
    assert [p["dose_g"] for p in schedule["pulses"]] == [2.0, 4.0]
    assert schedule["pulses"][0]["start_min"] == 0.0
    assert schedule["pulses"][0]["end_min"] == 2.0
    assert schedule["spacing_min"] == [3.0]
    assert schedule["group"] == "meal with external insulin"
    row["inputs"][meal.MEAL][2] = -1
    with pytest.raises(ValueError, match="negative"):
        meal.meal_schedule(row)


def test_meal_report_distinguishes_sample_and_event_weighting(tmp_path):
    rows, curves = _fixture()
    results = meal.review_meals(curves, rows, tmp_path)
    for held in meal.HOLDOUTS:
        result = results["folds"][held]
        assert result["all_meals"]["events"] == ["train_001", "train_002"]
        u = result["all_meals"]["U"]
        assert u["total"]["pooled_nmse"] == pytest.approx(1 / 3)
        assert u["total"]["macro_nmse"] == 0.5
        assert u["total"]["worst_event"] == "train_001"
        assert u["total"]["time_min"] == 0
        assert u["total_higher"] == u["tied_within_1e-10"] == 1
        assert (
            result["by_schedule"]["delayed single meal"]["U"]["total"]["pooled_nmse"]
            == 0
        )
        assert (tmp_path / f"meals-{held}.png").stat().st_size > 1000


def test_rejects_another_split_or_unmatched_curve():
    rows, curves = _fixture()
    h = "train_012"
    bad = deepcopy(rows[h])
    bad[-1]["trajectory_id"] = "validation_000"
    with pytest.raises(ValueError, match="registered"):
        meal._fold(h, curves[h], bad)
    broken = deepcopy(curves[h])
    broken["total"]["train_001"]["observed"]["Gp"][0] += 1
    with pytest.raises(ValueError, match="Unmatched observations"):
        meal._fold(h, broken, rows[h])
