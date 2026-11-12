import importlib
import json
from pathlib import Path

import pytest

from scripts import iclr_stress_evaluation as full
from scripts.noise_handoff import example_adapter, exchange, grade, package


def fixture_plan(tmp_path):
    root = tmp_path / "fit"
    cells = {}
    rows = [{"targets": {"Gp": [1, 3, 2], "I": [4, 2, 1]}}]
    for i, cell in enumerate(exchange.CELLS):
        path = root / "public" / cell
        path.mkdir(parents=True)
        for name in ("train.json", "val.json"):
            exchange.publish(path / name, {"rows": rows})
        exchange.publish(path / "specification.json", {"noise": i})
        (path / "proposer_prompt.txt").write_text("Development only")
        cells[cell] = {
            "public_files": {f: exchange.file_hash(path / f) for f in exchange.FILES}
        }
    exchange.publish(root / "release.json", {"cells": cells, "seeds": [0, 1, 2]})
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "method": "smoke_native",
                "semantics": "discrete_map",
                "implementation_revision": "fixture",
                "budget": 1,
                "selection_rule": "persistence",
            }
        )
    )
    work = tmp_path / "work"
    return root, work, exchange.make_plan(root, work, config, example_adapter.__name__)


def test_freeze_requires_12_terminal_rows_and_the_original_model_and_input(tmp_path):
    root, work, plan = fixture_plan(tmp_path)
    for t in plan["tasks"][:-1]:
        exchange.fit_task(root, work, plan, t, example_adapter)
    with pytest.raises(FileNotFoundError):
        exchange.freeze(work)
    exchange.fit_task(root, work, plan, plan["tasks"][-1], example_adapter)
    frozen = exchange.freeze(work)
    assert len(frozen["rows"]) == 12
    assert exchange.freeze(work) == frozen
    (work / "tasks/acquisition_00-s0/model.bin").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="selected model changed"):
        exchange.freeze(work)
    (root / "public/acquisition_00/train.json").write_text("{}")
    with pytest.raises(ValueError, match="public data differ"):
        exchange.verify(root)


def test_blind_input_discards_all_later_target_measurements():
    row = {
        "trajectory_id": "a",
        "time": [0, 1, 2],
        "targets": {"Gp": [1, 2, 3], "I": [4, 5, 6]},
        "external_inputs": {"meal": [0, 2, 0]},
        "auxiliaries": {"Uii": [2, 2, 2]},
    }
    cell = {
        "noise": 0.05,
        "scales": {"Gp": 2, "I": 3, "Uii": 0},
        "stride": 1,
        "mask": ["Uii"],
    }
    blind = package.blind_row(row, cell)
    row["targets"]["Gp"][1:] = [1e5, 0]
    row["targets"]["I"][2] = -500
    assert package.blind_row(row, cell) == blind
    assert set(blind) == {"time", "targets0", "auxiliaries", "external_inputs"}
    assert all(isinstance(v, float) for v in blind["targets0"].values())


@pytest.fixture
def official(tmp_path, monkeypatch):
    src = package.official_module(exchange.file_hash(Path(full.__file__)))
    (tmp_path / "test_official_noise_metrics.py").write_text(src)
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("test_official_noise_metrics")
    monkeypatch.setattr(
        importlib.import_module("scripts.noise_handoff"), "official", mod, raising=False
    )
    return mod


def test_ode_solver_check_and_native_effect_metric(official):
    p = {"Gp": [1.0, 2.0, 3.0], "I": [1.0, 1.0, 1.0]}
    q = {"Gp": [1.0, 1.0, 1.0], "I": [1.0, 1.0, 1.0]}

    def sample(curve):
        return {"time": [0.0, 1.0, 2.0], "targets": curve}

    pair = {
        "case": "fixture",
        "blind_id": "q00",
        "perturbed": sample(p),
        "control": sample(q),
    }
    record = {
        "q00": {
            "status": "complete",
            "curves": {
                "perturbed": {"Radau": p, "BDF": q},
                "control": {"Radau": q, "BDF": q},
            },
        }
    }
    scales = {"Gp": 2.0, "I": 1.0}
    cell = {"scales": {"Gp": 1.0, "I": 1.0}}

    def score(mode):
        return official.score(
            record,
            cell,
            [pair],
            scales,
            lambda ep, c, ref: grade.forecast_pair(official, mode, ep, ref),
        )

    r = score("continuous_ode")
    assert (
        r["trajectories"][0]["failure_kind"] == "solver_disagreement"
        and r["test_nmse"] is None
    )
    record["q00"]["curves"] = {"perturbed": {"native": q}, "control": {"native": q}}
    r = score("discrete_map")
    assert r["common_scale_test_nmse"] == pytest.approx(5 / 6)
    assert r["test_nmse"] == pytest.approx(5 / 24)
    assert r["response_rse"] == 1
    assert r["trajectories"][0]["channels"]["I"]["response_rse"] is None


def test_forecast_cannot_be_substituted_from_another_freeze(tmp_path, official):
    root, work, plan = fixture_plan(tmp_path)
    for task in plan["tasks"]:
        exchange.fit_task(root, work, plan, task, example_adapter)
    frozen = exchange.freeze(work)
    parts = {"time": [0.0, 1.0], "targets": {"Gp": [0.0, 1.0], "I": [1.0, 1.0]}}
    refs = [
        {"blind_id": "q00", "case": "fixture", "perturbed": parts, "control": parts}
    ]
    key = {
        "format": "t2-noise-answer-key-1",
        "release_sha256": plan["release_sha256"],
        "full_evaluator_sha256": official.SOURCE_SHA256,
        "queries_sha256": "test",
        "references": refs,
        "cells": exchange.verify(root)["cells"],
    }
    for cell in key["cells"].values():
        cell["noise_percent"] = 0
    answer = tmp_path / "key.json"
    exchange.publish(answer, key)
    path = tmp_path / "forecasts/acquisition_00-s0/q00.json"
    exchange.publish(
        path,
        {
            "id": "q00",
            "status": "unresolved",
            "binding": {"freeze_sha256": "other"},
            "reason": "numerical",
        },
    )
    with pytest.raises(ValueError, match="different freeze"):
        grade.evaluate(answer, work / "freeze.json", tmp_path / "forecasts")
    assert frozen["rows"][0]["validation_nmse"] > 0
