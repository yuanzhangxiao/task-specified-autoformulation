"""Confirm continuous-input replay, dense feasibility and handoff boundaries."""

import hashlib
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from scripts import t2_disposal_coupled_replay as trial


def test_verify_only_required_members_and_detect_tampering(tmp_path, monkeypatch):
    path = tmp_path / "bundle.zip"
    payload = b'{"fitted": true}'
    manifest = json.dumps(
        {
            "sha256": {
                "train.json": hashlib.sha256(payload).hexdigest(),
                "development.json": "unopened",
            }
        }
    ).encode()
    monkeypatch.setattr(trial, "MANIFEST", hashlib.sha256(manifest).hexdigest())

    def write(contents, extra=False):
        with ZipFile(path, "w") as z:
            z.writestr("manifest.json", manifest)
            z.writestr("train.json", contents)
            z.writestr("development.json", "unreadable data")
            if extra:
                z.writestr("unknown.json", "{}")

    write(payload)
    assert trial._members(path, ("train.json",)) == {"train.json": {"fitted": True}}
    write(b"{}")
    with pytest.raises(ValueError, match="Corrupt"):
        trial._members(path, ("train.json",))
    write(payload, extra=True)
    with pytest.raises(ValueError, match="unexpected"):
        trial._members(path, ("train.json",))


def test_continuous_disposal_can_dip_between_positive_samples():
    # The exact gap is (t-0.33)^2 - 0.00001. All 16-subdivision nodes
    # happen to be positive, so a fine grid alone would miss the minimum.
    row = SimpleNamespace(
        trajectory_id="train_000",
        time=[0, 1],
        targets={"Gp": [1.10889, 1.44889], "I": [3, 3], "U": [1.10889, 1.44889]},
        inputs={},
        auxiliaries={"Gt": [0, 1], "Uii": [1, 1]},
        covariates={},
    )
    model = {
        "graph": {
            "variable_types": {
                "Gp": "differential",
                "I": "differential",
                "U": "algebraic",
            },
            "sources": {"Gp": ["Gt"], "I": [], "U": ["Gp"]},
            "initializers": {
                k: {"mode": "observed_initial", "channel": k} for k in ("I", "Gp")
            },
        },
        "formula": {
            "equations": {"Gp": "2*Gt-0.66", "I": "0", "U": "Gp"},
            "parameters": {},
            "initializers": {},
        },
        "knots": {},
        "scales": dict.fromkeys(trial.TARGETS, 1.0),
    }
    engine = trial._Rollout(model["graph"], model["formula"], trial.TARGETS, 2)
    source = {"predicted": row.targets}
    check = trial._check_event(engine, model, row, source, "train")
    assert min(np.array(row.targets["U"]) - 1) > 0
    assert not check["eligible"]
    assert check["active_counts"]["U_minus_Uii"] == 0
    assert check["minimum_U_minus_Uii"] == pytest.approx(-1e-5, abs=1e-10)
    assert check["minimum_times"]["U_minus_Uii"] == pytest.approx(0.33, abs=1e-7)
    assert check["reasons"] == ["negative U_minus_Uii"]


def test_balance_coefficients_and_frozen_insulin_checked(monkeypatch):
    model = {
        "formula": {
            "equations": {
                "I": "-I",
                "Gp": trial.DELAYED_GP,
                "D_meal": trial.DELAYED_RATE,
            },
            "initializers": {"D_meal": "0"},
            "parameters": {
                "afcoef10": 1.0,
                "afcoef13": 1.0,
                "pos_tau": 2.0,
                "meal_balance_egp": 1.0,
                "meal_balance_gain": 2.0,
                "meal_balance_tau": 3.0,
            },
        },
        "graph": {"initializers": {"I": {"mode": "observed_initial", "channel": "I"}}},
        "knots": {},
        "scalar_model": {},
    }
    monkeypatch.setattr(
        trial,
        "INSULIN",
        trial.content_hash(
            ["-I", model["graph"]["initializers"]["I"], {}, {}, {"pos_tau": 2.0}]
        ),
    )
    assert trial.parameter_violations(model) == []
    model["formula"]["parameters"]["pos_tau"] += 1
    model["formula"]["parameters"]["meal_balance_egp"] = -1
    model["formula"]["parameters"]["meal_balance_tau"] = trial.SAFE_DIVISION_EPSILON
    model["formula"]["equations"]["Gp"] = "U"
    assert len(trial.parameter_violations(model)) == 4
