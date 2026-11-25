"""Checks for schedule isolation, step-held propagation and synthetic study resume."""

from copy import deepcopy

import numpy as np

from scripts import check_constructor_recovery as study


def test_shared_schedule_distinct_splits_and_exact_affine_forecast():
    base = study.event("affine", 0, 0, held=False)
    other = study.event("sine", 0, 0, held=False)
    assert base["inputs"] == other["inputs"] and base["targets"] != other["targets"]
    held = study.event("affine", 0, 0, held=True)
    assert held["inputs"] != base["inputs"] and held["time"] != base["time"]
    assert np.ptp(np.diff(held["time"])) > 0.1
    law = {
        "equations": {"x": "-decay*x+bias+s*a+q*b"},
        "initializers": {},
        "parameters": {"decay": 0.6, "bias": 0.3, "s": 0.5, "q": -0.25},
    }
    assert study.forecast(law, [held], 1)["nmse"] < 1e-14
    changed = deepcopy(held)
    changed["targets"]["x"][1:] = [y + 3 for y in held["targets"]["x"][1:]]
    assert abs(study.forecast(law, [changed], 1)["nmse"] - 9) < 1e-6


def test_trials_pass_only_anonymous_training_events_and_resume(tmp_path, monkeypatch):
    real = study.constructor.build_shape_shortlist
    calls = []

    def audited(graph, training, **kwargs):
        calls.append([r["trajectory_id"] for r in training])
        assert all(
            set(r)
            == {
                "trajectory_id",
                "time",
                "inputs",
                "targets",
                "auxiliaries",
                "covariates",
            }
            for r in training
        )
        assert graph == study.GRAPH and kwargs == {"interpolation": study.STEP}
        return real(graph, training, **kwargs)

    monkeypatch.setattr(study.constructor, "build_shape_shortlist", audited)
    monkeypatch.setattr(study, "LAWS", ("product a b", "sine"))
    monkeypatch.setattr(study, "SEEDS", (7,))
    result = study.run(tmp_path)
    assert len(calls) == 2 and all(k.startswith("train_") for k in calls[0])
    assert all(row["forecasts_scored"] == 1 for row in result["rows"])
    assert result["rows"][0]["top1"] == 1 and result["rows"][1]["top2"] == 0
    assert result == study.run(tmp_path) and len(calls) == 2
