"""Order, observed-path, and true prefix checks on an affine action graph."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import t2_order_checkpoint_diagnostics as trial
from tests.test_validate_positive_insulin_transfer import toy


def _fixture():
    row, model = toy("train_000")
    row.time = [0.0, 6.0, 15.0, 25.0]
    row.auxiliaries["Uii"] = [1.0] * 4
    form, graph = model["formula"], model["graph"]
    graph["variable_types"]["I_act"] = "differential"
    graph["sources"]["I_act"] = ["I"]
    graph["sources"]["U"] = ["Gp", "I_act", "Uii", "Gt"]
    graph["initializers"]["I_act"] = {
        "mode": "map",
        "state_initials": ["I"],
        "public_initials": [],
    }
    form["initializers"]["I_act"] = "2+state0_I"
    form["equations"]["I_act"] = "0.01*I"
    form["equations"]["U"] = "Gp+I_act+Uii+0.01*Gt"
    return vars(row), model


def test_pair_slack_counts_and_absolute_shift():
    ids = ["middle", "tied", "high", "low"]
    observed = [1.0, 1.001, 1.2, 0.9]
    # Leave a common bias and then shift the high prediction far enough to reverse it.
    predicted = [3.0, 3.001, 2.91, 2.9]
    report = trial._comparisons(ids, observed, predicted, 0.5, 0.01)
    assert report["pairs"] == 5
    assert report["worst_signed_slack"] == pytest.approx(-0.1905)
    assert report["reversed_pairs"] == 2 and report["violations"] == 3
    assert report["worst_pair"]["low"] == "tied"
    assert report["worst_absolute_error"] == pytest.approx(2.0)
    assert report["worst_order_violation"] == pytest.approx(0.1905)
    assert trial._comparisons(["x"], [0], [10], 0.5, 0.01)["worst_pair"] is None
    with pytest.raises(ValueError, match="Invalid"):
        trial._comparisons(ids, observed, [1.3], 0.5, 0.01)


def test_prefix_uses_no_future_or_within_event_target_and_verifies_full():
    row, model = _fixture()
    engine = trial._Rollout(model["graph"], model["formula"], trial.TARGETS, 8)
    # Early measured I and Gp can affect the conditional path only.
    altered = deepcopy(row)
    for name in ("I", "Gp", "U"):
        altered["targets"][name][1:] = [3e2, 4e2, 6e2]
    # Future forcing must not affect the simulated prefix.
    for group in ("inputs", "auxiliaries"):
        for v in altered[group].values():
            v[-1] += 80
    for candidate in [row, altered]:
        pub = trial.forecast.public_event(candidate)
        prefix = trial._event(pub, trial.MINUTE)
        a, steps = trial._flow(engine, model, prefix)
        b, full_steps = trial._flow(engine, model, trial._event(pub, None))
        assert steps < full_steps
        np.testing.assert_array_equal(a, b[:, :3])
        drive = trial._affine_insulin_drive(model, engine)
        conditional = trial._conditional(candidate, prefix, model, engine, drive)
        measured = np.array(candidate["targets"]["I"][:3])
        expected = (
            2
            + measured[0]
            + 0.01 * trial.trapezoid(measured, prefix.time)
            + candidate["targets"]["Gp"][2]
            + row["auxiliaries"]["Uii"][2]
            + 0.01 * row["auxiliaries"]["Gt"][2]
        )
        assert conditional == pytest.approx(expected)
        if candidate is row:
            clean = a.copy()
            observed_path = conditional
        else:
            np.testing.assert_array_equal(a, clean)
            assert conditional != observed_path
        with pytest.raises(ValueError, match="checkpoint"):
            trial._event(pub, 14)


def test_reject_wrong_conditional_law_and_foreign_cohort():
    row, model = _fixture()
    engine = trial._Rollout(model["graph"], model["formula"], trial.TARGETS, 8)
    model["formula"]["equations"]["I_act"] = "I*I"
    with pytest.raises(ValueError, match="affine"):
        trial._affine_insulin_drive(model, engine)
    row["trajectory_id"] = "val_001"
    with pytest.raises(ValueError, match="training cohort"):
        trial.diagnose(model, [row])


def test_diagnostic_keeps_minute15_separate_from_training_domain(monkeypatch):
    row, model = _fixture()
    # The scalar/graph registry check is for real intake models, not this fixture.
    monkeypatch.setattr(trial, "parameter_violations", lambda _: [])
    rows = [row | {"trajectory_id": k} for k in trial.TRAIN_IDS]
    with pytest.raises(ArithmeticError, match="differs from full"):
        trial.diagnose(model, rows)
    t = np.arange(0, 25.25, 0.25)
    for group in ("targets", "auxiliaries", "inputs"):
        row[group] = {
            name: np.interp(t, row["time"], values).tolist()
            for name, values in row[group].items()
        }
    row["time"] = t.tolist()
    rows = [row | {"trajectory_id": k} for k in trial.TRAIN_IDS]
    d = trial.diagnose(model, rows)
    assert d["domain"]["eligible"]
    assert d["minute15"]["pairs"] == 0 and d["minute15"]["worst_order_violation"] == 0
    assert (
        d["cost"]["partial_event_simulations"]
        == d["cost"]["matched_full_event_simulations"]
        == 16
    )
    assert d["cost"]["independent_dense_full_event_simulations"] == 32
    assert d["events"][0]["conditional15"] != d["events"][0]["actual15"]
    assert d["verification"]["max_partial_vs_matched_full_by_target"] == dict.fromkeys(
        trial.TARGETS, 0
    )
    assert set(d["per_target_nmse"]) == set(trial.TARGETS)


def test_fixed_bank_screen_executes_full_rollout_only_for_surviving_vectors():
    row, model = _fixture()
    model["formula"]["equations"]["U"] = "Gp+I_act+weight*Uii+0.01*Gt"
    model["formula"]["parameters"]["weight"] = 1.0
    bad = deepcopy(model)
    bad["formula"]["parameters"]["weight"] = -1.0
    rows = []
    for i, name in enumerate(trial.TRAIN_IDS):
        record = deepcopy(row)
        record["trajectory_id"] = name
        record["auxiliaries"]["Uii"] = [i + 1.0] * 4
        record["targets"]["U"] = [i + 1.0] * 4
        rows.append(record)
    d = trial.screen_bank({"good": model, "bad": bad}, rows)
    assert d["passed_candidates"] == 1 and d["rejected_candidates"] == 1
    assert d["decisions_agree"] and d["candidates"]["good"]["full_predictions_sha256"]
    assert d["cost"]["prefix_first"]["partial_event_simulations"] == 32
    assert d["cost"]["prefix_first"]["full_event_simulations"] == 16
    assert d["cost"]["full_only"]["full_event_simulations"] == 32
    assert d["cost"]["full_only"]["partial_event_simulations"] == 0
    with pytest.raises(ValueError, match="training events"):
        trial.screen_bank({"good": model}, [row])
