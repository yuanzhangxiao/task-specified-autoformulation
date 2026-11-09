"""Frozen T2 readout calibration, physical pulse contrasts and resume contracts."""

import csv

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import evaluate_t2_latent_dynamics as probe


def sample(name):
    return {
        "id": name,
        "time": [0.0, 1.0, 2.0, 3.0],
        "Gp": [1.0, 2.0, 3.0, 4.0],
        "I": [2.0, 3.0, 4.0, 5.0],
        "Uii": [1.0, 1.0, 1.0, 1.0],
        "X": [0.0, 2.0, 3.0, 4.0],
        "meal_event_g": [3.0, 0.0, 0.0, 0.0],
        "insulin_pmol_per_kg_min": [0.0, 1.0, 1.0, 0.0],
    }


def test_probe_has_matched_fasting_control_and_unused_reference_measurements():
    cases = probe.schedules()
    assert {c["id"] for c in cases if c["calibration"]} == set(probe.CALIBRATION)
    assert all(not c["meals"] for c in cases if not c["calibration"])
    ref = sample("a")
    public = probe.public_trajectory(ref)
    assert all(
        set(x) == expected
        for x, expected in (
            (public.targets, {"Gp", "I"}),
            (public.auxiliaries, {"Uii"}),
            (public.external_inputs, set(probe.INPUTS)),
        )
    )
    for key in ref:
        if key not in {"Gp", "I", "X"}:
            continue
        ref[key][1:] = [-123.0, -456.0, -789.0]
    changed = probe.public_trajectory(ref)
    for key in public.targets:
        np.testing.assert_array_equal(changed.targets[key], public.targets[key])
    assert not public.derivatives


def test_single_latent_affine_readout_generalizes_and_names_resolve_ties():
    x = np.array([1.0, 4.0, 6.0, 3.0])
    best = probe.calibrate(
        {"latent.z": x, "latent.a": x, "flat": np.ones(4)}, 4 - 2 * x
    )
    assert best["name"] == "latent.a"
    assert best["calibration_mse"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        probe.decode(best, {"latent.a": [10.0, -1.0]}), [-16.0, 6.0]
    )
    assert probe.calibrate({}, x)["status"] == "no_latent_coordinate"
    assert probe.calibrate({"flat": np.ones(4)}, x)["status"] == "no_varying_coordinate"


def test_paired_effect_scoring_distinguishes_flat_wrong_and_delayed_responses():
    t = np.arange(4.0)
    r = np.array([0.0, 3.0, 2.0, 1.0])
    c = np.zeros(4)
    actual = probe.metrics(t, r, r + 9.0, c, c + 9.0)
    assert actual["rmse"] == pytest.approx(9)
    assert actual["effect_rse"] == 0
    assert actual["direction_correct"]
    flat = probe.metrics(t, r, c + 7.0, c, c + 7.0)
    assert flat["effect_rse"] == 1
    assert flat["peak_time_error_min"] is None
    reverse = probe.metrics(t, r, -r, c, c)
    assert reverse["effect_rse"] == 4
    assert not reverse["direction_correct"]
    assert probe.metrics(t, r, [0.0, 0.0, 3.0, 1.0], c, c)["peak_time_error_min"] == 1
    assert probe.metrics(t, c, c, c, c)["status"] == "negligible_reference_effect"
    with pytest.raises(ValueError, match="increasing grid"):
        probe.metrics(t, r[:-1], c, c, c)


def test_baseline_cancellation_checked_at_pulse_scale():
    refs = {k: sample(k) for k in ("a", "fasting_control")}
    refs["a"]["Gp"] = [1.0, 2.01, 3.02, 4.0]
    prim = {k: {"Gp": np.ones(4) * 2000} for k in refs}
    alt = {"fasting_control": prim["fasting_control"], "a": {"Gp": np.ones(4) * 2000.1}}
    assert (
        probe.score_curves(refs, prim, alt, "a", "Gp", {})["status"]
        == "paired_solver_disagreement"
    )


def test_frozen_roster_returns_eight_cells_even_when_endpoint_missing(
    tmp_path, monkeypatch
):
    model = {
        "benchmark_id": probe.CELL,
        "method": "control",
        "repetition": 0,
        "status": "unavailable",
        "error": "missing source",
    }
    source = tmp_path / "input.json"
    sealed_write(
        source, {"protocol": "public-mechanism-model-bundle-1", "rows": [model]}
    )
    root = tmp_path / "work"
    plan = probe.prepare(source, root)
    monkeypatch.setattr(probe, "reference", lambda case, solver: sample(case["id"]))
    first = probe.run(root, 1)
    assert first["plan_sha256"] == plan["artifact_sha256"]
    assert not first["test_split_opened"]
    assert (
        len(list(csv.DictReader((root / "metrics.csv").read_text().splitlines()))) == 8
    )
    assert {s["status"] for s in first["models"][0]["scores"]} == {"unavailable"}
    monkeypatch.setattr(probe, "reference", lambda *args: pytest.fail("replayed"))
    assert probe.run(root, 1) == first
    bad = tmp_path / "duplicate.json"
    sealed_write(
        bad, {"protocol": "public-mechanism-model-bundle-1", "rows": [model] * 2}
    )
    with pytest.raises(ValueError, match="distinct"):
        probe.prepare(bad, root)
