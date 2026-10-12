"""Frozen denominators, evaluator isolation, source provenance and submission."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_integration_campaign as campaign
from autoformalism.fitting import public_rollout as bridge
from autoformalism.rebuttal.prefit_replay import content_hash
from tests.test_adapter_campaign import fake_backend
from tests.test_public_fitting import _control


@pytest.fixture
def prepared(tmp_path):
    request, train, val = _control(multiple_targets=True)
    record = {
        "case": "synthetic",
        "source_status": "constructed",
        "kind": "constructed",
        "request": request.model_dump(mode="json"),
        "training": train.model_dump(mode="json"),
        "validation": val.model_dump(mode="json"),
    }
    records = {name: deepcopy(record) for name in campaign.CONSTRUCTED}
    records[campaign.CONSTRUCTED[2]].update(
        request=None, source_status="construction_failed"
    )
    records[campaign.CONSTRUCTED[5]].update(request=None, source_status="missing")
    for name in campaign.CONTROLS:
        records["control_" + name] = deepcopy(record) | {
            "kind": "known_structure",
            "source_status": "available",
            "reference_parameters": {"a": 0.7, "b": 1.2, "init_z_scale": 0.5},
        }
    inputs = tmp_path / "input.json"
    seal(
        inputs,
        {
            "protocol": campaign.INPUT_PROTOCOL,
            "records": records,
            "test_data_opened": False,
            "historical_fits_used": False,
        },
    )
    root = tmp_path / "campaign"
    assert campaign.prepare(root, inputs, campaign.Policy())["tasks"] == 11
    return root


def test_fixed_denominator_and_missing_constructions_never_fit(prepared, monkeypatch):
    monkeypatch.setattr(bridge, "run", lambda *a: pytest.fail("no missing-source fit"))
    for index in (2, 5):
        r = campaign.run_task(prepared, index)
        assert r["status"] == "source_unavailable"
        assert r["fit"] is None and campaign.costs(r)["budget_charge_seconds"] == 0
        assert campaign.run_task(prepared, index) == r
    r = campaign.report(prepared)
    assert r["expected"] == 11 and r["recorded"] == 2
    assert r["status_counts"] == {"missing": 9, "source_unavailable": 2}
    assert not campaign.costs(r["rows"][0])["accounting_complete"]


def test_fit_sealed_before_validation_and_terminal_resume(prepared, monkeypatch):
    monkeypatch.setattr(bridge.adapter.recovery, "run", fake_backend)
    seen = []

    def evaluate(record, point, policy, folder):
        assert (folder.parent / "fit/result.json").exists()
        saved = bridge.completed(folder.parent / "fit")
        problem = saved.preparation["value"]["problem"]
        assert "validation" not in problem and "reference_parameters" not in problem
        assert point == saved.fit.parameters
        seen.append(record["kind"])
        return {"status": "verified", "solver_agreement_passed": True}

    monkeypatch.setattr(campaign, "evaluate", evaluate)
    row = campaign.run_task(prepared, 0)
    assert row["status"] == "complete" and seen == ["constructed"]
    monkeypatch.setattr(bridge, "run", lambda *a: pytest.fail("resumed fitting"))
    assert campaign.run_task(prepared, 0) == row
    with pytest.raises(ValueError, match="index"):
        campaign.run_task(prepared, -1)


def test_unavailable_setup_recorded_and_not_sent_to_evaluator(prepared, monkeypatch):
    monkeypatch.setattr(
        bridge,
        "prepare",
        lambda *a: (_ for _ in ()).throw(ValueError("unsupported grammar")),
    )
    monkeypatch.setattr(
        campaign.verified_replay, "run", lambda *a: pytest.fail("no parameters")
    )
    row = campaign.run_task(prepared, 0)
    assert row["status"] == "setup_unavailable"
    assert "unsupported grammar" in row["fit"]["preparation"]["value"]["error"]
    assert row["evaluation"]["status"] == "no_verified_parameters"
    assert campaign.report(prepared)["recorded"] == 1


def test_no_reference_comparison_for_discovered_models_and_per_target_scores(
    prepared, monkeypatch
):
    _, data = campaign.verify(prepared)
    record = data["records"][campaign.CONSTRUCTED[0]]
    monkeypatch.setattr(
        campaign, "coefficient_metrics", lambda *a: pytest.fail("unknown coefficients")
    )
    monkeypatch.setattr(
        campaign.verified_replay,
        "run",
        lambda *a: {
            "solver_agreement_passed": True,
            "attempts": [
                {
                    "value": {
                        "rows": [
                            {"split": s, "samples": n, "nmse": {"v01": x, "v02": x * 2}}
                            for s in ("train", "val")
                            for n, x in ((2, 1), (6, 3))
                        ]
                    }
                }
            ],
        },
    )
    r = campaign.evaluate(record, {"a": 1}, campaign.Policy().replay, prepared)
    assert r["per_target_nmse"] == {
        "train": {"v01": 2.5, "v02": 5},
        "val": {"v01": 2.5, "v02": 5},
    }
    assert "coefficient_recovery" not in r


def test_source_export_uses_identity_not_fitted_scores(prepared, tmp_path):
    _, data = campaign.verify(prepared)
    source = tmp_path / "source"
    source.mkdir()
    record = data["records"][campaign.CONSTRUCTED[0]]
    req = record["request"]
    tasks = [
        {"task_id": n, "arm": "full", "seed": 0, "benchmark_id": n}
        for n in campaign.CONSTRUCTED
    ]
    plan = {
        "protocol": "phase-c-construction-baseline-1",
        "tasks": tasks,
        "test_data_opened": False,
        "config": {"fit_profile": req["profile"]},
        "cells": {
            n: {
                "training": record["training"],
                "validation": record["validation"],
                "public_specification": "public task",
            }
            for n in campaign.CONSTRUCTED
        },
    }
    plan["artifact_sha256"] = content_hash(plan)
    (source / "plan.json").write_text(json.dumps(plan))
    for task in tasks:
        folder = source / "results" / task["task_id"]
        folder.mkdir(parents=True)
        proposal = {
            "identity": content_hash([plan["artifact_sha256"], task]),
            "status": "constructed",
            "bundle": {
                "context": req["context"],
                "initialization": {
                    "base_candidate": req["base_candidate"],
                    "plan": req["initialization_plan"],
                    "guesses": req["parameter_guesses"],
                },
            },
        }
        proposal["artifact_sha256"] = content_hash(proposal)
        (folder / "proposal.json").write_text(json.dumps(proposal))
        (folder / "fitted.json").write_text("DO NOT READ PREVIOUS FITS")
    m28 = tmp_path / "old.json"
    seal(
        m28,
        {
            "protocol": "phase-c-rollout-adapter-inputs-1",
            "test_data_opened": False,
            "development": {
                "cases": {
                    n: record | {"reference_parameters": {}} for n in campaign.CONTROLS
                },
                "commons": {n + "_s0": {"request": req} for n in campaign.CONTROLS},
            },
        },
    )
    exported = tmp_path / "exported.json"
    assert campaign.export_inputs(source, m28, exported)["tasks"] == 11
    out = read_seal(exported)
    assert not out["historical_fits_used"]
    assert (
        out["records"][campaign.CONSTRUCTED[0]]["request"]["base_candidate"]
        == req["base_candidate"]
    )
    path = source / "results" / campaign.CONSTRUCTED[0] / "proposal.json"
    bad = json.loads(path.read_text())
    bad["status"] = "tampered"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="digest"):
        campaign.export_inputs(source, m28, tmp_path / "bad.json")


def test_submission_idempotent_and_uncertain_reply_blocks(
    prepared, tmp_path, monkeypatch
):
    from scripts import submit_phase_c_public_integration as submitter

    policy = tmp_path / "policy.json"
    policy.write_text(campaign.Policy().model_dump_json())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    calls = []

    def sbatch(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "test-commit"
    )
    kw = {"account": "test-cpu", "concurrency": 3, "inputs": prepared / "inputs.json"}
    r = submitter.submit(prepared, policy, **kw)
    assert r["array_tasks"] == 11 and r["gpus"] == 0
    assert "--array=0-10%3" in calls[1]
    assert submitter.submit(prepared, policy, **kw) == r and len(calls) == 3
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr="timeout"),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit(tmp_path / "uncertain", policy, **kw)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "run", lambda *a, **kw: pytest.fail("duplicate")
    )
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        submitter.submit(tmp_path / "uncertain", policy, **kw)
