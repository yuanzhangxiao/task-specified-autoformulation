from pathlib import Path

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import thermal_forecast_continuation as study
from tests.test_thermal_balance_trial import rows


@pytest.mark.parametrize("p", [[0.2, 0.3, 3], [0.2, 0.3, 3, 1.5]])
def test_exact_discrete_gradient_matches_finite_difference(p):
    records = rows()
    d = study.windows(records, 3)
    parameters = np.array(p)
    res, jac, clipped = study.residual_jacobian(d, parameters, 290, 360, 5)
    assert not clipped
    for i in range(len(parameters)):
        e = np.zeros_like(parameters)
        e[i] = 1e-5
        upper = study.residual_jacobian(d, parameters + e, 290, 360, 5)[0]
        lower = study.residual_jacobian(d, parameters - e, 290, 360, 5)[0]
        assert np.max(abs((upper - lower) / (2 * e[i]) - jac[:, i])) < 1e-6
    assert res.size == len(records) * 30


def test_each_window_uses_only_its_first_observation():
    source = rows()[0]
    changed = list(source.targets["T"])
    changed[4] += 7
    changed[10] += 3
    vectors = []
    for r in (source, source.model_copy(update={"targets": {"T": tuple(changed)}})):
        data = study.windows([r], 3)
        res = study.residual_jacobian(data, np.array([0.35, 0.6, 3]), 290, 355, 1)[0]
        vectors.append(res.reshape((10, 3)) + data.temperature[1:])
    assert vectors[0][:, 0] == pytest.approx(vectors[1][:, 0])
    assert np.max(abs(vectors[0][:, 1] - vectors[1][:, 1])) > 1
    assert vectors[0][:, 2] == pytest.approx(vectors[1][:, 2])
    with pytest.raises(ValueError, match="divide"):
        study.windows([source], 4)


def test_optimization_clips_disclosed_extreme_training_attempt():
    extreme = rows()[0].model_copy(update={"targets": {"T": tuple([400.0] * 31)}})
    batch = study.windows([extreme], 3)
    p = np.array([0.1, 0.05, 3.0, 2.0])
    *_, clipped = study.residual_jacobian(batch, p, 320, 331, 5)
    assert clipped
    solved = study.optimize(batch, p, 320, 331, 5, limit=2)
    assert solved["nfev"] <= 2
    assert solved["clipped_objective_evaluations"] > 0


def test_held_split_and_frozen_class_selections(monkeypatch, tmp_path):
    from scripts import thermal_balance_trial as base

    events = rows()
    spec = {"targets": ["T"], "external_inputs": ["Tf"], "auxiliaries": ["C", "Tj"]}
    reads = []

    def inputs(_, dataset, split):
        reads.append((dataset, split))
        if split == "specification":
            return spec if dataset == base.EASY else {**spec, "auxiliaries": []}
        return (
            events
            if split == "train"
            else [events[0].model_copy(update={"trajectory_id": "val_000"})]
        )

    monkeypatch.setattr(base, "_read", inputs)
    prior = tmp_path / "prior"
    root = tmp_path / "continue"
    base.train(Path("public"), prior)
    monkeypatch.setattr(study, "PARTITIONS", (3, 1))
    with pytest.raises(FileNotFoundError):
        study.validate(Path("public"), prior, root)
    baseline_calls = 0

    def limited(batch, initial, lo, hi, scale):
        nonlocal baseline_calls
        baseline_calls += 1
        return {"vector": initial, "nfev": 1}

    monkeypatch.setattr(study, "optimize", limited)
    frozen = study.train(Path("public"), prior, root)
    assert baseline_calls == 8
    assert frozen["preferred"] in study.FAMILIES
    assert all(frozen["selections"][n]["checkpoint"] for n in study.FAMILIES)
    assert all(split != "val" for _, split in reads)
    assert study.train(Path("public"), prior, root) == frozen and baseline_calls == 8
    first = study.validate(Path("public"), prior, root)
    assert set(first["scores"]) == set(study.FAMILIES)
    assert study.validate(Path("public"), prior, root) == first
    assert reads.count((base.EASY, "val")) == 1
    assert frozen == sealed_read(root / "training.json")
