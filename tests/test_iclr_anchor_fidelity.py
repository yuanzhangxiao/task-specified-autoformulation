"""Physical closures, public information boundaries and prospective assay controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks.phase_c_release import SPEC_PATHS
from autoformalism.rebuttal import dalla_man as dm
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import iclr_anchor_fidelity as assay
from scripts import iclr_reference_fits as fits


def test_fresh_and_training_interventions_are_separate():
    for cell in fits.CELLS:
        dev = assay.cases(cell, "development")
        locked = assay.cases(cell, "confirmation")
        for k, item in dev.items():
            if item["role"] == "contrast":
                assert (
                    item["protocol"]["specification"]
                    != locked[k]["protocol"]["specification"]
                )
                assert item["window"][0] < item["window"][1]
                assert item["control"] in dev
            elif item["role"] == "train_calibration":
                assert item["protocol"]["split"] == "train"
                assert item == locked[k]


def test_readout_denominators_and_zero_effects():
    t = np.arange(4.0)
    r = np.array([0.0, 1.0, 0.0, -2.0])
    z = np.zeros_like(t)
    a = assay.scores(t, r, z, r, z, [0.0, 2.0], 1.0)
    assert a["effect_rse"] == 0 and a["window_direction_correct"]
    assert a["reference_window_auc"] == 1
    b = assay.scores(t, r, z, z, z, [0.0, 2.0], 1.0)
    assert b["effect_rse"] == 1 and not b["window_direction_correct"]
    assert (
        assay.scores(t, z, z, r, z, [0.0, 2.0], 0.0)["status"]
        == "negligible_reference_effect"
    )


def test_only_public_initial_targets_reach_reference_or_candidate():
    t = np.arange(21.0)
    rate = np.maximum(0, 1 - abs(t - 5) / 5) * 12
    raw = {
        "time": t,
        "Gp": 170.0 + t,
        "I": 25.0 + t,
        "Uid": 2 * t,
        "Ra": 3 * t,
        "meal_rate_g_per_min": rate,
        "insulin_pmol_per_kg_min": t * 0,
        "Uii": t * 0 + 1,
    }
    tr = assay.trajectory(fits.T2, raw)
    assert np.all(tr.targets["Gp"] == 170) and np.all(tr.targets["I"] == 25)
    assert set(tr.auxiliaries) == {"Uii"} and not tr.derivatives
    w = fits.public_inputs(fits.T2, tr)
    assert sum(w[-1]) == pytest.approx(60_000) and w[-1, 0] > 0
    raw["meal_rate_g_per_min"] = np.ones(len(t))
    with pytest.raises(ValueError, match="truncated"):
        fits.public_inputs(fits.T2, assay.trajectory(fits.T2, raw))


@pytest.mark.parametrize(
    "cell,case", [(fits.CSTR, "feed_high"), (fits.T2, "fed_long"), (fits.T1, "split")]
)
def test_privileged_equations_reproduce_independent_solver(cell, case):
    pytest.importorskip("casadi")
    data_root = Path("data_raw")
    if cell == fits.CSTR and not (data_root / SPEC_PATHS["cstr"]).exists():
        pytest.skip("reactor simulator specification not installed")
    item = assay.cases(cell, "development")[case]
    r = assay.reference(cell, item, data_root, "DOP853")
    tr = assay.trajectory(cell, r)
    center = fits.coefficients(cell)
    true = dict(center)
    if cell == fits.CSTR:
        system = json.loads((data_root / SPEC_PATHS["cstr"]).read_text())
        p = system["parameters"]
        base = system["equilibrium"]
        rate = p["k0"] * np.exp(-p["E_over_R"] / 350)
        true.update(
            flow=p["flow_rate"],
            rate=rate,
            activation=p["E_over_R"] / 350,
            heat_rate=p["source_gain"] * rate,
            exchange=p["exchange_rate"],
            jacket_flow=p["secondary_flow_rate"],
            jacket_exchange=p["secondary_exchange_rate"],
            C0=base["C"],
            Tj0=base["Tj"],
        )
    elif cell == fits.T2:
        b = dm.compute_dalla_man_basal(dm.DallaManParameters())
        vals = dict(zip(dm.STATE_NAMES, b.initial_state, strict=True))
        true.update(
            Gt0=vals["Gt"],
            Gp_b=vals["Gp"],
            I_b=b.Ib,
            Il0=vals["Il"],
            Ipo0=vals["Ipo"],
            secretion_b=b.Sb,
        )
    z = np.log([true[k] / v for k, v in center.items()])
    fn = fits.family(cell, len(tr.time) - 1, 8, jacobian=False)
    curves = np.asarray(
        fn(
            [tr.targets[k][0] for k in fits.TARGETS[cell]],
            fits.public_inputs(cell, tr),
            z,
        )
    )
    for name, curve in zip(fits.TARGETS[cell] + fits.HIDDEN[cell], curves, strict=True):
        assert np.max(abs(curve - r[name])) / max(1, np.max(np.abs(r[name]))) < 2e-4


def test_stress_alias_requires_exact_source(tmp_path):
    folder = tmp_path / "acquisition_00"
    folder.mkdir()
    (folder / "specification.json").write_text("wrong")
    row = {"benchmark_id": "acquisition_00", "public_spec_sha256": "stale"}
    with pytest.raises(ValueError, match="specification hash"):
        assay.canonicalize(row, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="stress-public-root"):
        assay.canonicalize(row, tmp_path, None)


def test_confirmation_requires_a_frozen_new_search(tmp_path):
    f = tmp_path / "selected.json"
    sealed_write(
        f,
        {
            "protocol": fits.VERSION,
            "rows": [
                {
                    "benchmark_id": fits.CSTR,
                    "method": "fitted_reference_structure",
                    "repetition": 0,
                    "status": "ready",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="frozen internal search"):
        assay.prepare(
            [f], tmp_path / "run", "confirmation", tmp_path, tmp_path, tmp_path
        )


def test_reference_freeze_includes_every_planned_start(tmp_path):
    plan = sealed_write(tmp_path / "plan.json", {"files": {fits.T2: {}}, "starts": 2})
    path = tmp_path / "fits" / fits.T2
    sealed_write(
        path / "0.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "cell": fits.T2,
            "start": 0,
            "status": "fit_failed",
        },
    )
    with pytest.raises(FileNotFoundError):
        fits.freeze(tmp_path)
    assert not (tmp_path / "frozen_models.json").exists()
    sealed_write(
        path / "1.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "cell": fits.T2,
            "start": 1,
            "status": "fit_failed",
        },
    )
    result = fits.freeze(tmp_path)
    assert result["rows"][0]["status"] == "fit_failed"
    assert fits.freeze(tmp_path) == result


def test_controlled_export_rejects_substitution_and_partial_roster(tmp_path):
    plan = sealed_write(
        tmp_path / "plan.json",
        {"blocks": [{"tasks": [{"task_id": "a"}, {"task_id": "b"}]}]},
    )
    rows = [{"source_id": n, "status": "no_retained_model"} for n in ("a", "b")]
    receipt = sealed_write(
        tmp_path / "evaluation_freeze.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "roster_sha256": assay.public.content_sha256(rows),
        },
    )
    bundle = {
        "rows": rows,
        "plan_sha256": plan["artifact_sha256"],
        "selection_freeze_sha256": receipt["artifact_sha256"],
    }
    assay.verify_controlled_bundle(tmp_path / "frozen_models.json", bundle)
    with pytest.raises(ValueError, match="incomplete"):
        assay.verify_controlled_bundle(
            tmp_path / "frozen_models.json", {**bundle, "rows": rows[:1]}
        )
    rows[0]["status"] = "pending"
    with pytest.raises(ValueError, match="incomplete"):
        assay.verify_controlled_bundle(tmp_path / "frozen_models.json", bundle)
