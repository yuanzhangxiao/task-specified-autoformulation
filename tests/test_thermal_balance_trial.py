from pathlib import Path

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search.topology_compatibility import CompleteTrajectory
from scripts import thermal_balance_trial as trial


def event(name, initial, feed, jacket, concentration):
    t = np.linspace(0, 2, 31)
    a, b, c = 0.35, 0.6, 3.0
    settled = (a * feed + b * jacket + c * concentration) / (a + b)
    y = settled + (initial - settled) * np.exp(-(a + b) * t)
    return CompleteTrajectory(
        trajectory_id=name,
        time=t,
        inputs={"Tf": np.full_like(t, feed)},
        auxiliaries={
            "C": np.full_like(t, concentration),
            "Tj": np.full_like(t, jacket),
        },
        targets={"T": y},
        covariates={},
    )


def rows():
    return [
        event("train_001", 330, 350, 300, 1),
        event("train_002", 310, 325, 345, 2),
        event("train_003", 345, 342, 315, 3),
        event("train_004", 350, 352, 318, 4),
    ]


def test_integral_concentration_fit_recovers_positive_coefficients():
    fit = trial.fit(rows(), ("concentration",))["concentration"]
    assert fit["design_rank"] == fit["nonzero_terms"] == 3
    assert all(
        fit["best"][k] == pytest.approx(v, rel=0.001)
        for k, v in zip(("a", "b", "c"), (0.35, 0.6, 3.0), strict=True)
    )
    assert fit["best"]["rate_rmse"] < 1e-8
    assert trial.score(rows(), fit, 5)["pooled_nmse"] < 1e-6


def test_four_families_and_subset_conditioning():
    temperatures = np.array([300, 320, 340])
    for name in trial.FAMILIES:
        assert np.all(trial.multiplier(name, 0, temperatures, 300, 340) == 1)
    assert trial.multiplier("linear_temperature", 3, 270, 300, 340) == 0
    expected = np.exp(2 * (1 / 300 - 1 / temperatures) / (1 / 300 - 1 / 340))
    assert trial.multiplier(
        "exponential_inverse_temperature", 2, temperatures, 300, 340
    ) == pytest.approx(expected)
    with pytest.raises(ValueError, match="Unknown"):
        trial.multiplier("unregistered", 0, temperatures, 300, 340)
    same_t = rows()[0].model_copy(update={"targets": {"T": [330] * 31}})
    assert trial.fit([same_t])["linear_temperature"]["status"].endswith("uninformative")
    with pytest.raises(ValueError, match="training"):
        trial.fit([rows()[0].model_copy(update={"trajectory_id": "val_000"})])


def test_forecast_uses_only_initial_target():
    r = rows()[0]
    fit = trial.fit(rows(), ("concentration",))["concentration"]
    shifted = [r.targets["T"][0], *[v + 20 for v in r.targets["T"][1:]]]
    changed = r.model_copy(update={"targets": {"T": shifted}})
    before, after = (trial.score([v], fit, 5) for v in (r, changed))
    assert before["events"][0]["predicted_T"] == after["events"][0]["predicted_T"]
    assert after["pooled_nmse"] > before["pooled_nmse"]


def test_public_roles_freeze_and_deterministic_resume(monkeypatch, tmp_path):
    calls = []
    training = rows()
    validation = [event("val_000", 331, 335, 310, 2)]
    easy = {"targets": ["T"], "external_inputs": ["Tf"], "auxiliaries": ["C", "Tj"]}
    hard = {**easy, "auxiliaries": []}
    assert trial.eligibility(hard)["missing_supplied_auxiliaries"] == ["C", "Tj"]

    def load(_, dataset, split):
        calls.append((dataset, split))
        return {
            (trial.EASY, "specification"): easy,
            (trial.HARD, "specification"): hard,
            (trial.EASY, "train"): training,
            (trial.EASY, "val"): validation,
        }[dataset, split]

    monkeypatch.setattr(trial, "_read", load)
    root = tmp_path / "thermal"
    with pytest.raises(FileNotFoundError):
        trial.validate(Path("public"), root)
    first = trial.train(Path("public"), root)
    assert (trial.EASY, "val") not in calls
    assert (trial.HARD, "train") not in calls
    frozen = sealed_read(root / "training.json")
    assert frozen["selection"] in trial.FAMILIES
    for r in training:
        fold = sealed_read(root / "folds" / r.trajectory_id / "fit.json")
        assert r.trajectory_id not in fold["training_events"]
        members = [v for v in training if v != r]
        assert fold["models"]["concentration"]["T_max"] == max(
            max(v.targets["T"]) for v in members
        )
    monkeypatch.setattr(trial, "fit", lambda *_: pytest.fail("Repeated fitting"))
    assert trial.train(Path("public"), root) == first
    held = trial.validate(Path("public"), root)
    assert set(held["scores"]) == set(trial.FAMILIES)
    monkeypatch.setattr(trial, "predict", lambda *_: pytest.fail("Repeated rollout"))
    assert trial.validate(Path("public"), root) == held
    assert sealed_read(root / "training.json") == frozen
