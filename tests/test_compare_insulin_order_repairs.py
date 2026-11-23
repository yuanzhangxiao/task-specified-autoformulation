from copy import deepcopy

import numpy as np
import pytest

from scripts import compare_insulin_order_repairs as trial


def event(seed, n=21):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.3, 1.2, size=n))
    return {
        "trajectory_id": f"train_{seed:03}",
        "time": t,
        "targets": {"I": [2.0 + seed] * n},
        "inputs": {trial.DRUG: rng.uniform(size=n), trial.MEAL: rng.uniform(size=n)},
        "auxiliaries": {
            "Gt": rng.uniform(10, 20, size=n),
            "EGP": rng.uniform(0, 4, size=n),
        },
    }


@pytest.mark.parametrize("arm", trial.ARMS)
def test_exact_and_independent_on_irregular_inputs_and_known_gains(arm):
    rows = [event(k) for k in range(4)]
    tz = 5 if arm == "adaptation" else None
    gain = [1, -0.5, 0.2] + ([] if arm == "reference" else [2.3])
    selected = {"tau": 5, "tau_z": tz, "gains": gain}
    for row in rows:
        off, x = trial.design(row, arm, 5, tz)
        row["targets"]["I"] = off + x @ gain
        np.testing.assert_allclose(
            trial.independent(row, arm, selected),
            row["targets"]["I"],
            atol=trial.ATOL,
            rtol=trial.RTOL,
        )
    fit, profiles = trial.fit(rows, arm, [5])
    np.testing.assert_allclose(fit["gains"], gain, rtol=1e-8)
    assert len(profiles) == 1 and fit["sse"] < 1e-17


def test_rollout_excludes_future_targets_and_knots():
    row = event(0)
    edited = deepcopy(row)
    edited["targets"]["I"][1:] = [999] * 20
    for x in (edited["auxiliaries"]["Gt"], edited["inputs"][trial.DRUG]):
        x[11:] = -3500
    args = ("adaptation", 5, 15)
    a = trial.design(row, *args)
    b = trial.design(edited, *args)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1][:11], b[1][:11])


def test_absent_input_gets_zero_gain_and_unverified_event_withholds_aggregate():
    row = event(1)
    row["inputs"][trial.DRUG] = [0] * len(row["time"])
    best, _ = trial.fit([row], "reference", [5])
    assert best["rank"] == 2 and best["gains"][2] == 0
    assert trial.combine([{"verified": False}]) == {"status": "unverified"}


def test_checkpointed_splits_and_no_held_target_in_fitting(monkeypatch, tmp_path):
    rows = {f"train_{k:03}": event(k, 5) for k in range(7)}
    folds = trial.folds(rows.values())
    plan = {"artifact_sha256": "test-plan", "folds": folds}
    captured = []
    original = trial.fit

    def capture(training, arm):
        captured.append((arm, [r["trajectory_id"] for r in training]))
        return original(training, arm, [5])

    monkeypatch.setattr(trial, "fit", capture)
    first = trial.experiment(tmp_path, rows, plan)
    assert all(ids == folds[i // 4]["fit"] for i, (_, ids) in enumerate(captured[:12]))
    assert "train_000" in folds[2]["fit"]
    monkeypatch.setattr(trial, "fit", lambda *_: pytest.fail("Repeated a fit"))
    assert trial.experiment(tmp_path, rows, plan) == first


def test_exact_solution_to_constant_forcing_and_invalid_timescale():
    row = event(0)
    off, x = trial.design(row, "reference", 5)
    t = row["time"] - row["time"][0]
    analytic = 10 + (row["targets"]["I"][0] - 10) * np.exp(-t / 5)
    np.testing.assert_allclose(off + x[:, 0] * 2, analytic, atol=1e-12)
    with pytest.raises(ValueError, match="timescales"):
        trial.design(row, "adaptation", 5)
