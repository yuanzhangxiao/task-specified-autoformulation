"""Free total-state replay, integrated glucose balance, and fitting-fold membership."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import t2_total_glucose_replay as verify
from tests.test_t2_initial_memory_forecast import example


def _total():
    row, model = example()
    row["auxiliaries"].update(E=[0.1] * 4)
    row["inputs"][verify.MEAL] = [0, 1, 0, 0]
    g, f = model["graph"], model["formula"]
    del g["initializers"]["Gp"]
    g["initializers"].update(
        S={"mode": "map", "public_initials": ["Gp", "Gt"], "state_initials": []},
        D_meal={"mode": "known"},
    )
    g["variable_types"].update(Gp="algebraic", S="differential", D_meal="differential")
    g["sources"].update(
        S=["EGP", "E", "U", "D_meal"], Gp=["S", "Gt"], D_meal=["D_meal", verify.MEAL]
    )
    f["initializers"].update(S="Gp+Gt", D_meal="0")
    f["equations"].update(
        S="EGP-E-U+meal_balance_gain*D_meal",
        Gp="S-Gt",
        D_meal="(meal_rate_g_per_min-D_meal)/meal_balance_tau",
    )
    f["parameters"].update(meal_balance_tau=4.0, meal_balance_gain=2.0)
    return row, model


def test_integral_tracks_forcing_and_checks_glucose_initial_readout():
    row, model = _total()
    row["targets"]["Gp"][0] = 50.0
    pub = verify.prior.public_event(row)
    f = verify.predict(model, pub)
    assert f["predicted"]["Gp"][0] == row["targets"]["Gp"][0]
    a = f["qualification"]["total_mass"]
    assert a["resolved"] and a["max_abs_integrated_mass_error"] < 1e-5
    assert a["initial_error"] < 1e-13 and a["max_abs_readout_error"] < 1e-13
    other = deepcopy(row)
    for key in ("Gp", "I", "U"):
        other["targets"][key][1:] = [float("nan")] * 3
    other["targets"]["U"][0] = float("nan")
    assert verify.predict(model, verify.prior.public_event(other)) == f


def test_wrong_excretion_sign_fails_independent_mass_audit():
    row, model = _total()
    model["formula"]["equations"]["S"] = "EGP+E-U+meal_balance_gain*D_meal"
    f = verify.predict(model, verify.prior.public_event(row))
    assert f["qualification"]["total_mass"][
        "max_abs_integrated_mass_error"
    ] == pytest.approx(0.8)
    assert not f["qualification"]["numerically_resolved"]
    assert not f["qualification"]["eligible"]


def test_replay_rejects_foreign_fit_and_scores_meals_on_fitting_fold():
    row, model = _total()
    ids = [i for i in verify.prior.TRAIN_IDS if i != "train_011"]
    model["fit_training_ids"] = ids
    rows = [row | {"trajectory_id": i} for i in ids]
    for target in verify.TARGETS:
        model["scales"][target] = float(np.std(np.tile(row["targets"][target], 15)))
    curves = verify.training_curves(model, rows)
    assert set(curves) == set(ids)
    assert all(c["meal_grams"] == pytest.approx(1.0) for c in curves.values())
    q = verify.training_replay(model, rows)
    assert q["ordering"]["eligible"] and q["summary"]["scored_events"] == 15
    with pytest.raises(ValueError, match="fitting fold"):
        verify.training_curves(
            model, [*rows[:-1], row | {"trajectory_id": "train_011"}]
        )
