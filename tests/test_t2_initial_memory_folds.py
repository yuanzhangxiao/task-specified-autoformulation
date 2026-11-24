"""Fold membership precedes insulin fitting, scaling, and anchoring."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from autoformalism.staged_topology import content_hash
from scripts import t2_initial_memory_folds as folds


def events():
    time = np.linspace(0, 20, 21)
    rows = []
    for j, event in enumerate(folds.intake.TRAIN_IDS):
        rows.append(
            {
                "trajectory_id": event,
                "time": time.tolist(),
                "covariates": {},
                "inputs": {
                    "insulin_pmol_per_kg_min": (
                        1 + 0.3 * np.sin(time / 5 + j)
                    ).tolist(),
                    "meal_rate_g_per_min": np.zeros_like(time).tolist(),
                },
                "auxiliaries": {
                    "Gt": (100 + j + 0.2 * time).tolist(),
                    "EGP": (0.5 + np.exp(-time / (1 + j))).tolist(),
                },
                "targets": {
                    "Gp": (90 + time + 2 * j).tolist(),
                    "I": (30 + (j + 5) * np.exp(-time / 7) + 0.1 * time).tolist(),
                    "U": (0.2 + 0.01 * j + 0.02 * time).tolist(),
                },
            }
        )
    return rows


@pytest.mark.parametrize("held", folds.HOLDOUTS)
def test_two_folds_use_only_remaining_readings_and_verify_insulin(held):
    rows = [r for r in events() if r["trajectory_id"] != held]
    result = folds.fit_inputs(rows[::-1], held)
    assert result["fit_ids"] == [r["trajectory_id"] for r in rows]
    assert result["training_rows_sha256"] == content_hash(rows)
    assert result["centers"]["Gt_ref"] == rows[0]["auxiliaries"]["Gt"][0]
    for target in folds.TARGETS:
        values = np.concatenate([r["targets"][target] for r in rows])
        assert result["scales"][target] == pytest.approx(float(np.std(values)))
        assert result["centers"][target] == pytest.approx(float(np.mean(values)))
    insulin = result["insulin"]
    assert insulin["knots"] == folds.production.calibrate(
        rows, folds.production.SOURCES["EGP"]
    )
    model = insulin["scalar_model"]
    assert min(model["weights"]) >= 0
    assert model["sse"] <= min(p["sse"] for p in insulin["coarse_profile"]) + 1e-9
    assert {r["event"] for r in insulin["replay_checks"]} == set(result["fit_ids"])
    assert all(r["verified"] for r in insulin["replay_checks"])
    formula, params = folds.insulin_formula(result)
    assert "I/pos_tau" in formula
    assert params["pos_tau"] == model["tau"]


def test_loader_excludes_held_values_before_scientific_validation(
    tmp_path, monkeypatch
):
    rows = events()
    omitted = "train_011"
    rows[11] = {"trajectory_id": omitted, "targets": {"bogus": "unread"}}
    body = {"train": {folds.intake.original.T2: rows}}
    digest = content_hash(body)
    path = tmp_path / "train.json"
    path.write_text(json.dumps(body | {"artifact_sha256": digest}))
    monkeypatch.setattr(folds.intake.original.campaign, "TRAIN", digest)
    subset = folds.load_fit_rows(path, omitted)
    assert [r["trajectory_id"] for r in subset] == list(folds.fit_ids(omitted))
    with pytest.raises(ValueError, match="registered train"):
        folds.load_fit_rows(tmp_path / "validation.json", omitted)
    with pytest.raises(ValueError, match="train_011 or train_012"):
        folds.load_fit_rows(path, "train_000")
    with pytest.raises(ValueError, match="fifteen designated"):
        folds.fit_inputs(rows, omitted)
    with pytest.raises(ValueError, match="fifteen designated"):
        folds.fit_inputs([*subset, subset[0]], omitted)


def test_handoff_authenticates_family_and_exposes_only_structure(tmp_path, monkeypatch):
    graph = {"variable_types": {"I": "differential"}}
    model = {"formula": {"parameters": {"all_event_gain": 123456}}, "graph": graph}
    digest = content_hash(model)
    record = {"model": model, "model_sha256": digest}
    data = json.dumps(record | {"artifact_sha256": content_hash(record)}).encode()
    manifest = json.dumps(
        {"sha256": {folds.MEMBER: hashlib.sha256(data).hexdigest()}}
    ).encode()
    monkeypatch.setattr(folds, "MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest())
    monkeypatch.setattr(folds, "SELECTED_MODEL", digest)

    def bundle(path: Path, payload: bytes):
        with ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", manifest)
            archive.writestr(folds.MEMBER, payload)

    p = tmp_path / "handoff.zip"
    bundle(p, data)
    public = folds.authenticate_handoff(p)
    assert public["graph"] == graph
    assert "all_event_gain" not in json.dumps(public)
    bundle(p, data + b" ")
    with pytest.raises(ValueError, match="archive member differs"):
        folds.authenticate_handoff(p)
