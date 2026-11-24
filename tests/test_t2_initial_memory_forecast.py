from copy import deepcopy

import numpy as np
import pytest

from scripts import t2_initial_memory_forecast as run
from tests.test_validate_positive_insulin_transfer import toy


def example():
    row, model = toy("train_011")
    row.auxiliaries["Uii"] = [0.25] * len(row.time)
    return vars(row), model


def test_public_projection_only_reads_gp0_and_i0():
    row, model = example()
    pub = run.public_event(row)

    class Initial:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, ix):
            assert ix == 0
            return self.value

    class Guard:
        def __getitem__(self, key):
            assert key in {"I", "Gp"}
            return Initial(row["targets"][key][0])

    assert pub == run.public_event(row | {"targets": Guard()})
    pred = run.predict(model, pub)
    assert pred["qualification"]["eligible"]
    np.testing.assert_allclose(
        np.array(pred["predicted"]["I"]) / 2, pred["predicted"]["U"]
    )


def test_rejects_prohibited_forcing_and_foreign_fold():
    row, model = example()
    row["inputs"]["U"] = row["targets"]["U"]
    with pytest.raises(ValueError, match="prohibited"):
        run.public_event(row)
    with pytest.raises(ValueError, match="registered"):
        run.public_event(row | {"trajectory_id": "val_000"})
    with pytest.raises(ValueError, match="fifteen-event"):
        run.training_replay(model, [row] * 15)


def test_training_constraint_checks_every_strict_pair():
    q = run._ordering([2.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.5, 1.9])
    assert q["pairs"] == 5 and q["violations"] == 3 and not q["eligible"]
    assert q["min_slack"] == pytest.approx(-0.1)
    assert run._ordering([2, 3], [1, 1.5 - 1e-10])["eligible"]


def test_replay_verifies_fit_fold_scales_and_reports_feasibility():
    row, model = example()
    rows = [row | {"trajectory_id": i} for i in run.TRAIN_IDS if i != "train_011"]
    with pytest.raises(ValueError, match="normalization scale"):
        run.training_replay(model, rows)
    model["scales"] = {
        k: float(np.std(np.tile(row["targets"][k], 15))) for k in run.TARGETS
    }
    result = run.training_replay(model, rows)
    assert result["eligible"] and result["ordering"]["pairs"] == 0
    assert result["summary"]["scored_events"] == 15


def test_score_uses_training_scales_and_reports_measured_minute_300(tmp_path):
    row, model = example()
    axis = np.arange(301, dtype=float)
    old = row["time"]
    for group in ("inputs", "auxiliaries", "targets"):
        row[group] = {
            k: np.interp(axis, old, v).tolist() for k, v in row[group].items()
        }
    row["time"] = axis.tolist()
    model["scales"]["U"] = 2.0
    scores = {}
    for event in run.OMISSIONS:
        row["trajectory_id"] = event
        f = run.predict(model, run.public_event(row))
        s = run.score(model, f, row)
        scores[event] = {"persistent": s, "fading": s}
        assert s["per_target_nmse"]["U"] == pytest.approx(
            np.mean(np.square(np.subtract(f["predicted"]["U"], row["targets"]["U"])))
            / 4
        )
        assert s["predicted_U0"] == 2.0 and s["observed_U0"] == 4.0
        assert s["predicted_Gp300"] == f["predicted"]["Gp"][-1]
        assert s["observed_Gp300"] == 3.0
    other = deepcopy(row)
    other["targets"]["Gp"][0] += 1
    with pytest.raises(ValueError, match="public inputs"):
        run.score(model, f, other)
    # A physically inadmissible, numerically resolved held curve is still scored.
    f["qualification"]["physical"] = f["qualification"]["eligible"] = False
    assert not run.score(model, f, row)["qualification"]["physical"]
    run.plot(scores, tmp_path / "held.png")
    assert (tmp_path / "held.png").stat().st_size > 1000
