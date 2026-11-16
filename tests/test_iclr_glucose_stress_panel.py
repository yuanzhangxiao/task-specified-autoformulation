"""Check panel separation, reference identity, pooling and immutable forecast reuse."""

import copy

import numpy as np
import pytest

from scripts import iclr_glucose_stress_panel as study


def test_existing_six_case_tasks_preserve_controls_and_input_contracts():
    t1, t2 = study.schedules("T1"), study.schedules("T2")
    assert set(t1) == set(t2) and len(t1) == 6
    for name in t1:
        p, q = t1[name], t2[name]
        assert p["perturbed"]["input_names"] == ["meal_rate_g_per_min"]
        assert q["perturbed"]["input_names"] == [
            "meal_rate_g_per_min",
            "insulin_pmol_per_kg_min",
        ]
        assert study.same_preparation(p["control"], q["control"])
        assert study.same_preparation(p["perturbed"], q["perturbed"]) == (
            name != "test_strong_orthogonal"
        )
        for panel in (p, q):
            assert panel["perturbed"]["split"] == "test"
            assert panel["perturbed"]["input_contract"] == "continuous-rates-1"
            assert panel["control"]["specification"]["meals"] == []
            if "initial_shift" in panel["perturbed"]["specification"]:
                assert panel["perturbed"] == panel["control"]
    with pytest.raises(ValueError, match="unsupported glucose task"):
        study.schedules("T3")


def test_event_adapter_preserves_amounts_and_rejects_ambiguous_grids():
    assert study.event_input([0, 1, 2], [[0, 20], [1, 15], [1, 10]]) == [20, 25, 0]
    assert study.event_input([0, 1], []) == [0, 0]
    for grid, meals in [
        ([0, 1], [[0.5, 10]]),
        ([0, 1], [[0, -1]]),
        ([0, 0], [[0, 1]]),
        ([0, 1], [[0, float("nan")]]),
    ]:
        with pytest.raises(ValueError):
            study.event_input(grid, meals)


def test_input_projection_and_runtime_do_not_supply_future_targets():
    row = {
        "endpoint": {
            "context": {
                "targets": ["Gp"],
                "auxiliaries": ["Gt"],
                "external_inputs": ["meal_event_g"],
            }
        }
    }
    ref = {
        "time": [0, 1, 2],
        "fields": {
            "Gp": [5, 100, 200],
            "Gt": [6, 7, 8],
            "meal_event_g": [20, 0, 0],
            "I": [999, 999, 999],
        },
    }
    pair = study.public_pair(row, "case", ref, ref)
    assert "I" not in pair["perturbed"]["auxiliaries"]
    public = study.stress.public_input(pair["perturbed"], {})
    assert public.targets["Gp"].tolist() == [5, 5, 5]
    assert public.auxiliaries["Gt"].tolist() == [6, 7, 8]
    assert public.external_inputs["meal_event_g"].tolist() == [20, 0, 0]


def test_rse_uses_native_integral_and_continuous_accuracy():
    a = {"time": [0, 1, 3], "fields": {"Gp": [0, 3, -1]}}
    b = {"time": [0, 1, 3], "fields": {"Gp": [0, 0, 0]}}
    value = {
        "status": "complete",
        "predicted": {"Gp": [0, 3, 1]},
        "control": {"Gp": [0, 0, 0]},
    }
    r = study.score(value, a, b, "Gp", 2)
    assert r["test_nmse"] == pytest.approx(1 / 3)
    assert r["response_rse"] == pytest.approx(4 / 14.5)
    assert r["directional_accuracy"] == 0.75
    assert r["shape_error"] == pytest.approx(0.36)
    zero = value | {"predicted": {"Gp": [0, 0, 0]}}
    r = study.score(zero, b, b, "Gp", 2)
    assert r["test_nmse"] == 0 and r["response_rse"] is None
    assert r["directional_accuracy"] is None and r["shape_error"] is None
    assert study.score({"status": "rollout_unavailable"}, a, b, "Gp", 2) == {
        "status": "rollout_unavailable"
    }
    with pytest.raises(ValueError, match="invalid normalization"):
        study.score(value, a, b, "Gp", 0)


def test_reference_rejects_legacy_physics_before_simulation(monkeypatch):
    monkeypatch.setattr(
        study, "simulate_phase_b", lambda *a, **k: pytest.fail("called")
    )
    p = copy.deepcopy(study.schedules("T1")["test_meal_20g_t0"]["perturbed"])
    p["input_contract"] = "legacy-events-1"
    p["input_names"] = ["meal_event_g"]
    with pytest.raises(ValueError, match="continuous-rate held-out"):
        study.reference(p)


def test_forecast_checkpoint_binds_full_model_and_references(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "import_forecast", lambda *a: None)
    calls = []

    def predict(*args):
        calls.append(1)
        return {"Gp": np.array([5, 6])}, {"Gp": np.array([5, 5])}

    monkeypatch.setattr(study.stress, "predict", predict)
    endpoint = {
        "context": {"targets": ["Gp"], "auxiliaries": [], "external_inputs": []},
        "parameters": {"k": 1},
    }
    row = {"key": "model", "endpoint": endpoint, "acquisition": {}}
    ref = {
        "artifact_sha256": "reference",
        "reference": {"time": [0, 1], "fields": {"Gp": [5, 6]}},
    }
    job = (tmp_path, tmp_path, "plan", row, "case", ref, ref)
    for _ in range(2):
        assert study.forecast(job)["status"] == "complete"
    assert len(calls) == 1
    row["endpoint"]["parameters"]["k"] = 2
    with pytest.raises(ValueError, match="stale six-case forecast"):
        study.forecast(job)


def test_failed_forecast_remains_checkpointed(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "import_forecast", lambda *a: None)

    def fail(*args):
        raise ValueError("solver failure")

    monkeypatch.setattr(study.stress, "predict", fail)
    row = {
        "key": "model",
        "endpoint": {
            "context": {
                "targets": ["Gp"],
                "auxiliaries": [],
                "external_inputs": [],
            }
        },
        "acquisition": {},
    }
    ref = {
        "artifact_sha256": "r",
        "reference": {"time": [0, 1], "fields": {"Gp": [5, 6]}},
    }
    job = (tmp_path, tmp_path, "plan", row, "case", ref, ref)
    assert study.forecast(job)["status"] == "rollout_unavailable"
    monkeypatch.setattr(
        study.stress, "predict", lambda *a: pytest.fail("reran failure")
    )
    assert study.forecast(job)["reason"] == "solver failure"


def test_import_rejects_mismatched_frozen_cohort(tmp_path):
    folder = tmp_path / study.QUALITY / "t2" / "acquisition_00"
    study.metrics.write(folder / "plan.json", {"cell": {}, "models": "wrong"})
    row = {
        "origin": "current",
        "condition": "acquisition_00",
        "acquisition": {},
        "source_bundle": "expected",
    }
    with pytest.raises(ValueError, match="another frozen cohort"):
        study.import_forecast(tmp_path, row, "case", {})


def test_source_identity_rejects_changed_bytes(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("original")
    sources = {"source.txt": study.metrics.digest(path)}
    study.sources_match(tmp_path, sources)
    path.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        study.sources_match(tmp_path, sources)
