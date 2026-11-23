"""Public volume inference and an independent open-loop forecast."""

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search.topology_compatibility import CompleteTrajectory
from scripts import basin_balance_inference as study


def event(i, up, down, prefix="train"):
    """A measured equilibrium event with synthetic, independently set flux powers."""
    q = 4 * max(up - 0.3, 0) ** 1.3
    d = 10 * max(down - 0.1, 0) ** 1.6
    assert d >= q
    return CompleteTrajectory(
        trajectory_id=f"{prefix}_{i:03}",
        time=tuple(range(10)),
        targets={"h_down": [down] * 10},
        inputs={"inflow_up": [q] * 10, "inflow_down": [d - q] * 10},
        auxiliaries={},
        covariates={
            "area_up": 80,
            "area_down": 120,
            "crest_up": 0.3,
            "initial_up": up,
            "crest_down": 0.1,
        },
    )


def events():
    return [
        event(i, u, d)
        for i, (u, d) in enumerate(
            [
                (0.2, 0.3),
                (0.3, 0.9),
                (0.2, 1.5),
                (0.6, 0.6),
                (1.0, 1.2),
                (1.5, 1.5),
            ]
        )
    ]


def test_integral_handles_crest_crossings_both_directions_and_flat_segments():
    a = np.array([-1, 1, 2, 2, -1, 1, 1.0])
    b = np.array([1, -1, 2, 3, -0.2, 1 + 1e-13, 3])
    p = 1.25
    expected = [
        0.5 / (p + 1),
        0.5 / (p + 1),
        2**p,
        (3 ** (p + 1) - 2 ** (p + 1)) / (p + 1),
        0,
        1 + p * 5e-14,
        (3 ** (p + 1) - 1) / (2 * (p + 1)),
    ]
    assert study.head_integral(a, b, np.full(7, 4), p) == pytest.approx(
        4 * np.array(expected)
    )
    with pytest.raises(ValueError, match="positive"):
        study.head_integral(1, 1, -1, p)


def test_hidden_depth_recovery_from_only_observed_targets_and_known_inflows():
    rows = events()
    for stride in study.STRIDES:
        estimated = study.estimate(rows, stride)
        for side, gain, power in (("upstream", 4, 1.3), ("downstream", 10, 1.6)):
            best = estimated[side]["best"]
            assert best["power"] == pytest.approx(power, abs=1e-7)
            assert best["gain"] == pytest.approx(gain, rel=1e-7)
            assert best["flow_rmse_m3_per_min"] < 1e-8
            assert {p["power"] for p in estimated[side]["profile"]} >= {1, 1.5, 2}
        for row in rows:
            h = estimated["reconstruction"][row.trajectory_id]["upstream_depth_m"]
            assert h == pytest.approx([row.covariates["initial_up"]] * len(h))


def test_unidentified_models_and_reconstructed_negative_flux_are_reported():
    assert (
        study.estimate([event(0, 0.5, 1)], 1)["downstream"]["status"] == "no_intervals"
    )
    s = study.estimate([event(0, 0.2, 0.3)], 1)
    assert s["downstream"]["status"] == "unidentified_power"
    assert s["upstream"]["status"] == "outlet_unidentified"
    zero = study.BalanceSample(np.array([-2]), np.array([-1]), np.ones(1), np.ones(1))
    assert study.fit_power([zero])["status"] == "no_positive_head"
    rows = events()
    bad = rows[-1].model_copy(
        update={"inputs": {**rows[-1].inputs, "inflow_down": (50,) * 10}}
    )
    result = study.estimate([*rows[:-1], bad], 1)
    assert (
        result["reconstruction"][bad.trajectory_id]["negative_transfer_intervals"] == 9
    )
    assert min(result["reconstruction"][bad.trajectory_id]["transfer_volume_m3"]) < 0


def test_thinning_preserves_all_input_knots_and_last_observation():
    row = event(7, 0.2, 0.8)
    row = row.model_copy(
        update={"inputs": {**row.inputs, "inflow_up": (0, 5, 0, 0, 0, 0, 0, 0, 0, 0)}}
    )
    e = study.event_volumes(row, 4)
    assert list(e.t) == [0, 4, 8, 9] and list(e.up_in) == [5, 0, 0]
    assert e.isolated.all()
    with pytest.raises(ValueError, match="training"):
        study.estimate([event(7, 0.2, 0.8, "val")], 1)


def test_training_freeze_and_validation_open_loop_are_resumable(tmp_path, monkeypatch):
    root = tmp_path / "out"
    campaign = tmp_path / "campaign"
    frozen = sealed_write(
        campaign / "intake.json",
        {
            "source": study.trial.TRAIN,
            "train": {study.BASIN: [r.model_dump(mode="json") for r in events()]},
        },
    )
    original = sealed_write(
        campaign / "plan.json", {"intake_sha256": frozen["artifact_sha256"]}
    )
    monkeypatch.setattr(study.family, "INTAKE", frozen["artifact_sha256"])
    monkeypatch.setattr(study.family, "SOURCE_PLAN", original["artifact_sha256"])
    hold = event(2, 1, 1.5, "val")
    # All later target observations are deliberately biased. A forecast that
    # resets from them would mask the resulting large performance loss.
    hold = hold.model_copy(update={"targets": {"h_down": (1.5,) + (3.0,) * 9}})
    path = tmp_path / "validation.json"
    valid = sealed_write(
        path,
        {
            "split": "val",
            "training_intake_sha256": study.trial.TRAIN,
            "rows": {study.BASIN: [hold.model_dump(mode="json")]},
        },
    )
    monkeypatch.setattr(study.trial, "VALIDATION", valid["artifact_sha256"])
    with pytest.raises(FileNotFoundError):
        study.validate(root, path)
    training = study.train(campaign, root)
    assert training["pooled_nmse"] < 1e-12
    test = study.validate(root, path)
    assert test["pooled_nmse"] > 1
    assert test["events"][0]["predicted"] == pytest.approx([1.5] * 10)

    def repeated(*args):
        pytest.fail("Repeated a completed estimator or integrator")

    monkeypatch.setattr(study, "estimate", repeated)
    monkeypatch.setattr(study, "score", repeated)
    assert study.train(campaign, root) == training
    assert study.validate(root, path) == test
    check = root / "estimates/stride-1.json"
    check.write_text(check.read_text().replace('"estimated"', '"changed"', 1))
    with pytest.raises(ValueError, match="digest"):
        study.validate(root, path)
    assert sealed_read(root / "training.json")["validation_opened"] is False
