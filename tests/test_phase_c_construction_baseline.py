"""Stage reuse, public boundaries, independent metrics and truthful failure counts."""

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import construction_baseline as b
from autoformalism.research import construction_trace, phase_c_inputs
from autoformalism.research.construction_statistics import aggregate
from scripts import smoke_fresh_shared as smoke


def fixture(root):
    historical = smoke.fixture(root / "toy")
    cell = next(iter(historical["cells"].values()))
    cell["independent_rules"] = [
        {
            "id": "input_response",
            "public_requirement": "Positive input effect",
            "kind": "causal_response",
            "target": "v01",
            "driver": "u01",
            "expected_sign": 1,
            "readout": "output",
            "interpretation": "Finite effect",
        }
    ]
    cfg = b.Config.model_validate_json(
        (b.REPO / "configs/phase_c_construction_v1.json").read_text()
    )
    cfg = cfg.model_copy(
        update={
            "assessment": cfg.assessment.model_copy(
                update={
                    "trajectory_seconds": 5,
                    "maximum_training_trajectories": 1,
                }
            )
        }
    )
    task = {
        "task_id": "toy_seed0_full",
        "seed": 0,
        "arm": "full",
        "benchmark_id": "toy",
        "shared_processes": True,
        "scientific_verifier": True,
    }
    return sealed_write(
        root / "plan.json",
        {
            "protocol": b.PROTOCOL,
            "config": cfg.model_dump(mode="json"),
            "cells": {"toy": cell},
            "tasks": [task],
            "source_identity": b.source_identity(),
        },
    )


def test_end_to_end_actual_constructor_fit_and_independent_assessment(tmp_path):
    plan = fixture(tmp_path)
    task, calls = plan["tasks"][0], []
    p = b.propose(
        tmp_path, plan, task, "http://offline", transport=smoke.transport_for(calls)
    )
    assert p["status"] == "constructed", p
    assert calls
    assert all("independent_rules" not in json.dumps(c) for c in calls)
    assert all("validation" not in c and "test" not in c for c in calls)
    n = len(calls)
    assert (
        b.propose(
            tmp_path, plan, task, "http://offline", transport=smoke.transport_for(calls)
        )
        == p
    )
    assert len(calls) == n

    fit = b.fit(tmp_path, plan, task)
    assert fit["status"] == "complete", fit
    assert b.fit(tmp_path, plan, task) == fit
    a = b.assess(tmp_path, plan, task)
    assert a["status"] == "assessed"
    assert a["counts"].get("pass") == 1, a
    assert b.assess(tmp_path, plan, task) == a
    result = b.report(tmp_path, plan)
    assert result["aggregate"]["full"]["compliance_mean"] == 1
    trace = json.loads((b.location(tmp_path, task) / "trace.json").read_text())
    assert len(trace["calls"]) == p["cost"]["physical_requests"]
    assert all(r["raw_response"] for r in trace["calls"])
    assert any(r["runtime_events"] for r in trace["calls"])
    assert "fitted_parameters" in (tmp_path / "EQUATIONS.md").read_text()
    assert len(calls) == n

    # A saved model is insufficient for a valid resume without its call evidence.
    next((b.location(tmp_path, task) / "calls").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="call records"):
        b.propose(tmp_path, plan, task, "http://offline")


def test_pending_and_terminal_failure_differ(tmp_path):
    plan = fixture(tmp_path)
    task = plan["tasks"][0]
    assert b.fit(tmp_path, plan, task)["status"] == "pending_construction"
    assert b.report(tmp_path, plan)["aggregate"]["full"]["status"] == "pending"
    directory = b.location(tmp_path, task)
    sealed_write(
        directory / "proposal.json",
        {"identity": b.namespace(plan, task), "status": "construction_failed"},
    )
    assert b.fit(tmp_path, plan, task)["status"] == "construction_failed"
    assert b.assess(tmp_path, plan, task)["status"] == "model_unavailable"
    result = b.report(tmp_path, plan)["aggregate"]["full"]
    assert result["nmse_median"] == "+infinity"
    assert result["compliance_mean"] == 0 and result["possible_mean"] == 1


def test_source_change_and_outcome_misplacement_rejected(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    task = plan["tasks"][0]
    path = b.location(tmp_path, task) / "proposal.json"
    sealed_write(path, {"identity": "other", "status": "constructed"})
    with pytest.raises(ValueError, match="different task"):
        b.read_outcome(path, plan, task)
    monkeypatch.setattr(b, "source_identity", lambda: {"different": True})
    with pytest.raises(ValueError, match="source/protocol"):
        b.verify(tmp_path)


def test_trace_is_escaped_and_provenance_checked(tmp_path):
    from autoformalism.staged_topology import content_hash

    calls = tmp_path / "calls"
    calls.mkdir()
    request = {"namespace": "ns", "body": {"prompt": "<script>bad()</script>"}}
    key = content_hash(request)
    path = calls / f"{key}.json"
    value = {
        "request_hash": key,
        "request": request,
        "step": "variables",
        "attempt": 0,
        "status": "responded",
        "raw_response": "<script>bad()</script>",
    }
    path.write_text(json.dumps(value))
    construction_trace.render(tmp_path, "ns")
    text = (tmp_path / "TRACE.html").read_text()
    assert "<script>" not in text and "&lt;script&gt;" in text
    with pytest.raises(ValueError, match="provenance"):
        construction_trace.render(tmp_path, "different")


def test_case_balanced_statistics_and_unavailable_denominator():
    rows = []
    for name, errors, scores in (("a", [1, 3], [0, 1]), ("b", [4, None], [1, 0])):
        for error, score in zip(errors, scores, strict=True):
            rows.append(
                {
                    "arm": "full",
                    "benchmark_id": name,
                    "assessment_status": "assessed",
                    "status": "complete" if error else "fit_failed",
                    "validation_nmse": error,
                    "confirmed": score,
                    "possible": 1,
                }
            )
    value = aggregate(rows)["full"]
    assert value["nmse_median"] == "+infinity" and value["nmse_mad"] is None
    assert value["compliance_mean"] == 0.5 and value["compliance_sample_sd"] == 0


def test_qualified_release_is_public_only_and_renames_rates(monkeypatch, tmp_path):
    release = b.REPO / "artifacts/phase-c-development-final-v2"
    if not release.exists():
        pytest.skip("local qualified release is not a committed fixture")
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "diagnostic" not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    cfg = b.Config.model_validate_json(
        (b.REPO / "configs/phase_c_construction_v1.json").read_text()
    )
    _, cells = phase_c_inputs.qualified_public_cells(
        release, cfg.model_dump(mode="json")
    )
    assert len(cells) == 8
    for name, cell in cells.items():
        if "dalla_man" in name:
            assert "meal_rate_g_per_min" in cell["context"]["external_inputs"]
            assert "meal_event_g" not in json.dumps(cell["independent_rules"])
    assert len(cells[next(iter(cells))]["independent_rules"]) > 0
    plan = b.freeze(release, tmp_path, b.REPO / "configs/phase_c_construction_v1.json")
    assert len(plan["tasks"]) == 32
    assert len({t["task_id"] for t in plan["tasks"]}) == 32
    assert {t["seed"] for t in plan["tasks"]} == {0, 1}
    assert b.verify(tmp_path) == plan
    assert (
        b.freeze(release, tmp_path, b.REPO / "configs/phase_c_construction_v1.json")
        == plan
    )
    for name in phase_c_inputs.basin.BASINS:
        cell = cells[name]
        assert (
            cell["brief"]["scientific_context"]
            == cell["public_specification"]["public_prompt"].rstrip()
        )
        assert len(cell["independent_rules"]) == (5 if "_coupled_" in name else 3)
