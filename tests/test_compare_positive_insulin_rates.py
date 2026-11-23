"""Train-split positive production laws, including exact basis-knot crossings."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import compare_positive_insulin_rates as trial
from tests.test_compare_insulin_order_repairs import event


def test_saturated_hats_partition_and_constant_sparse_inputs():
    x = np.array([-12.0, 0, 0.25, 1.5, 2, 80])
    h = trial.hats(x, [0, 1, 2])
    np.testing.assert_allclose(
        h, [[1, 0, 0], [1, 0, 0], [0.75, 0.25, 0], [0, 0.5, 0.5], [0, 0, 1], [0, 0, 1]]
    )
    assert np.all(h >= 0)
    np.testing.assert_allclose(trial.hats(x, [4]), np.ones((6, 1)))
    r = event(2, 5)
    r["inputs"][trial.prior.DRUG] = [0, 0, 0, 0, 1]
    r["auxiliaries"]["Gt"] = [130] * 5
    assert trial.calibrate([r], trial.SOURCES["base"]) == {
        "Gt": [130.0],
        trial.prior.DRUG: [0.0, 1.0],
    }
    for bad in ([], [1, 1], [2, 1]):
        with pytest.raises(ValueError, match="knots"):
            trial.hats(x, bad)


def test_crossing_inside_one_observation_interval_known_exact_nonlinearity():
    # The unit middle hat gives q(t)=t up to t=1 and q(t)=2-t after t=1.
    r = event(0, 2)
    r["time"] = [0, 2]
    r["targets"]["I"] = [0, 999]
    r["auxiliaries"]["Gt"] = [0, 2]
    knots = {"Gt": [0, 1, 2]}
    model = {"tau": 1, "weights": [0, 1, 0]}
    assert trial.predict(r, knots, model)[-1] == pytest.approx(
        (1 - np.exp(-1)) ** 2, abs=1e-12
    )
    r["auxiliaries"]["Gt"] = [2, 0]
    assert trial.predict(r, knots, model)[-1] == pytest.approx(
        (1 - np.exp(-1)) ** 2, abs=1e-12
    )
    with pytest.raises(ValueError, match="positive"):
        trial.design(r, knots, 0)


def test_local_future_observations_never_initialize_or_restart_a_forecast():
    r = event(1)
    k = trial.calibrate([r], trial.SOURCES["EGP"])
    model = {"tau": 15, "weights": np.arange(sum(map(len, k.values()))).tolist()}
    expected = trial.predict(r, k, model)

    class InitialOnly:
        def __getitem__(self, i):
            if i != 0:
                raise ValueError("intermediate target read")
            return expected[0]

    r["targets"]["I"] = InitialOnly()
    np.testing.assert_allclose(trial.predict(r, k, model), expected)
    np.testing.assert_allclose(
        trial.independent(r, k, model), expected, atol=trial.ATOL, rtol=trial.RTOL
    )
    edited = deepcopy(r)
    edited["auxiliaries"]["Gt"][5:] += 500
    np.testing.assert_array_equal(trial.predict(edited, k, model)[:5], expected[:5])
    edited["trajectory_id"] = "val_100"
    with pytest.raises(ValueError, match="training event"):
        trial.predict(edited, k, model)


def test_rank_deficient_gain_solvers_agree_and_enforce_sign():
    mat = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    for constraint in trial.WEIGHTS:
        solution = trial.solve(mat, np.array([2.0, -4.0, -2.0]), constraint)
        assert solution["rank"] == 2 and solution["columns"] == 3
        assert solution["relative_objective_disagreement"] < 1e-12
        assert solution["normalized_kkt_residual"] < 1e-12
        if constraint == "signed":
            assert solution["sse"] < 1e-20
        else:
            assert min(solution["weights"]) >= 0
            assert solution["sse"] == pytest.approx(24.0)


def test_positive_known_solution_fit_and_independent_rollout():
    rows = [event(k) for k in range(5)]
    knots = trial.calibrate(rows, trial.SOURCES["EGP"])
    golden = {"tau": 15, "weights": [1.0, 4.0, 2.0, 2.0, 5.0, 0.3, 10.0, 0.2, 0.8]}
    for r in rows:
        r["targets"]["I"] = trial.predict(r, knots, golden)
    _, profiles = trial.fit(rows, trial.SOURCES["EGP"])
    for constraint in trial.WEIGHTS:
        best = min(profiles[constraint], key=lambda p: p["sse"])
        assert best["sse"] < 1e-15 and best["tau"] == 15
        assert best["rank"] < best["columns"]
    best = min(profiles["nonnegative"], key=lambda p: p["sse"])
    altered = deepcopy(rows[0])
    altered["targets"]["I"][0] = 0
    for k in ("Gt", "EGP"):
        altered["auxiliaries"][k] *= 1e6
    altered["inputs"][trial.prior.DRUG] -= 1e6
    assert min(trial.predict(altered, knots, best)) >= 0
    assert min(best["weights"]) >= 0


def test_resume_skips_all_fits_and_held_events_do_not_define_knots(
    tmp_path, monkeypatch
):
    rows = {r["trajectory_id"]: r for r in [event(i, 5) for i in range(7)]}
    splits = trial.prior.folds(rows.values())
    for n in splits[0]["held"]:
        rows[n]["auxiliaries"]["Gt"] += 1e5
    real = trial.fit
    monkeypatch.setattr(trial, "fit", lambda rows, channels: real(rows, channels, [15]))
    plan = {"artifact_sha256": "test", "folds": splits}
    r = trial.experiment(tmp_path, plan, rows)
    sub = trial.sealed_read(tmp_path / "1-base.json")
    assert max(sub["knots"]["Gt"]) < 100
    assert {s["event"] for s in sub["arms"]["signed"]["scores"]} == set(
        splits[0]["held"]
    )
    assert len(r["arms"]["EGP-signed"]["events"]) == 6
    monkeypatch.setattr(trial, "fit", lambda *_: pytest.fail("Repeated frozen fit"))
    assert trial.experiment(tmp_path, plan, rows) == r


def test_sorted_json_restores_each_egp_coefficient_and_full_forecast(tmp_path):
    import json

    rows = [event(i) for i in range(4)]
    knots, profiles = trial.fit(rows, ("EGP", trial.prior.DRUG, "Gt"), [15])
    for label in trial.WEIGHTS:
        original = profiles[label][0]
        payload = json.loads(
            json.dumps({"selected": original, "knots": knots}, sort_keys=True)
        )
        restored, new_knots = payload["selected"], payload["knots"]
        assert list(new_knots) != list(knots)
        assert restored["coefficient_hats"] == [
            {"channel": n, "hat_index": i, "knot": k}
            for n in trial.CHANNEL_ORDER
            for i, k in enumerate(new_knots[n])
        ]
        for r in rows:
            np.testing.assert_array_equal(
                trial.predict(r, new_knots, restored), trial.predict(r, knots, original)
            )
            np.testing.assert_array_equal(
                trial.independent(r, new_knots, restored),
                trial.independent(r, knots, original),
            )
            assert trial.score(r, new_knots, restored) == trial.score(
                r, knots, original
            )
        wrong = restored | {"coefficient_hats": restored["coefficient_hats"][::-1]}
        with pytest.raises(ValueError, match="coefficient hats"):
            trial.predict(rows[0], new_knots, wrong)
