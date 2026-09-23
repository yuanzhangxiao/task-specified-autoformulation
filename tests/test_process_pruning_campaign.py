"""Scientific gates, selection, consumed budgets and durable HPC submission."""

import copy
import json

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import process_pruning as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import smoke_process_pruning as smoke
from scripts import submit_process_pruning as submitter


def backend(calls):
    def run(request, model, train, val, guesses, settings, directory):
        calls.append(
            {"parameters": guesses, "directory": directory, "settings": settings}
        )
        metrics = {
            "normalized_mse": 1e-8,
            "failed_trajectories": [],
            "per_target_normalized_mse": {"v01": 1e-8},
        }
        return {
            "parameters": guesses,
            "training": metrics,
            "validation": metrics,
            "refinement": {"budget_exhausted": True, "actual_residual_calls": 1},
        }

    return run


def test_gates_rank_once_and_pair_same_budget(tmp_path, monkeypatch):
    plan = smoke.fixture(tmp_path)
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    result = campaign.run_one(tmp_path, 0)
    assert result["choice"]["status"] == "ready"
    assert len(calls) == 2 and calls[0]["settings"] == calls[1]["settings"]
    assert result["selection"]["pruned_accepted"]
    assert result["fits"]["pruned"]["budget_exhausted"]  # Not a convergence claim.
    assert "eps" in calls[0]["parameters"] and "eps" not in calls[1]["parameters"]
    assert campaign.run_one(tmp_path, 0) == result and len(calls) == 2
    summary = campaign.report(tmp_path)
    assert summary["rows"][0]["parent"] == plan["rows"][0]["parent"]["fit"]
    assert not summary["test_data_opened"] and summary["llm_calls"] == 0
    with pytest.raises(ValueError, match="index"):
        campaign.run_one(tmp_path, -1)


def test_cstr_style_unanchored_requirement_is_ambiguous_not_failed(
    tmp_path, monkeypatch
):
    plan = smoke.fixture(tmp_path)
    cell = copy.deepcopy(plan["cells"]["toy"])
    cell["mechanism_spec"]["required_mechanisms"] = [{"id": "controlled_balance"}]
    monkeypatch.setattr(
        campaign.pruning,
        "contributions",
        lambda *a, **kw: pytest.fail("ranking unresolved model"),
    )
    choice = campaign.choose(plan["rows"][0], cell)
    assert choice["status"] == "public_requirements_unresolved"
    assert choice["certificate"]["eligible_for_development_selection"]
    assert not choice["certificate"]["all_public_graph_requirements_certified"]
    assert any(
        p["status"] == "ambiguous"
        for p in choice["certificate"]["mechanisms"]["predicates"]
    )


def test_ranking_ignores_validation_and_checks_pathway_before_fitting(tmp_path):
    plan = smoke.fixture(tmp_path)
    cell = copy.deepcopy(plan["cells"]["toy"])
    first = campaign.choose(plan["rows"][0], cell)
    cell["validation"] = {"name": "val", "rows": "DO NOT READ"}
    assert campaign.choose(plan["rows"][0], cell) == first
    assert any(
        b["reason"] == "public_requirements_not_certified" for b in first["blocked"]
    )
    assert first["audit"]["removed"]["parameters"] == ["eps"]


def test_pruned_must_compete_with_extra_refit_and_parent(tmp_path):
    plan = smoke.fixture(tmp_path)
    row = plan["rows"][0]
    choice = campaign.choose(row, plan["cells"]["toy"])
    parent = copy.deepcopy(row["parent"]["fit"])
    parent["validation"]["normalized_mse"] = 1.0
    control, child = copy.deepcopy(parent), copy.deepcopy(parent)
    control["validation"]["normalized_mse"] = 0.5
    child["validation"]["normalized_mse"] = 0.7
    assert campaign.decide(parent, control, child, choice)["selected"] == "control"
    child["validation"]["normalized_mse"] = 0.504
    assert campaign.decide(parent, control, child, choice)["selected"] == "pruned"
    child["validation"]["normalized_mse"] = 0.506
    assert campaign.decide(parent, control, child, choice)["selected"] == "control"
    child["status"] = "fit_failed"
    assert campaign.decide(parent, None, child, choice)["selected"] == "parent"


def test_consumed_ranking_does_not_replay_but_control_can_finish(tmp_path, monkeypatch):
    plan = smoke.fixture(tmp_path)
    directory = tmp_path / "results" / plan["rows"][0]["task"]["task_id"]
    sealed_write(
        directory / "ranking_started.json",
        {"identity": plan["artifact_sha256"], "index": 0},
    )
    monkeypatch.setattr(
        campaign, "choose", lambda *a: pytest.fail("new ranking budget")
    )
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    result = campaign.run_one(tmp_path, 0)
    assert result["choice"]["status"] == "ranking_interrupted"
    assert result["fits"]["pruned"] is None and len(calls) == 1


def test_missing_and_sealed_inputs(tmp_path):
    smoke.fixture(tmp_path)
    assert campaign.report(tmp_path)["status_counts"] == {"missing": 1}
    (tmp_path / "evaluation_freeze.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed for evaluation"):
        campaign.run_one(tmp_path, 0)
    (tmp_path / "evaluation_freeze.json").unlink()
    raw = json.loads((tmp_path / "plan.json").read_text())
    raw["policy"]["contribution_seconds"] = 1000
    (tmp_path / "plan.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="digest"):
        campaign.verify(tmp_path)


def test_submission_has_no_gpu_and_resumes_receipts(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "out"
    sealed_write(
        source / "plan.json",
        {"protocol": campaign.history.INTEGRATION_PROTOCOL, "tasks": [1, 2]},
    )
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setenv("AF_PYTHON", str(python))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda p: "a" * 40)
    calls = []

    def queue(*args):
        calls.append(args)
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", queue)
    result = submitter.submit(source, root)
    assert len(calls) == 3 and result["gpus"] == 0 and result["llm_calls"] == 0
    assert "--array=0-1%4" in calls[1][2]
    assert "--dependency=afterok:101" in calls[1][2]
    assert "--dependency=afterany:102" in calls[2][2]
    assert submitter.submit(source, root) == result and len(calls) == 3


def test_partial_scheduler_intent_never_resubmits(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "out"
    sealed_write(
        source / "plan.json",
        {"protocol": campaign.history.INTEGRATION_PROTOCOL, "tasks": [1]},
    )
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setenv("AF_PYTHON", str(python))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda p: "a" * 40)

    def uncertain(*args):
        raise ValueError("uncertain scheduler response")

    monkeypatch.setattr(submitter, "submit_job", uncertain)
    with pytest.raises(ValueError, match="uncertain scheduler"):
        submitter.submit(source, root)
    with pytest.raises(ValueError, match="partial/uncertain submission"):
        submitter.submit(source, root)
    assert (root / "submission-intent/identity.json").exists()


def test_equal_count_is_not_reduction():
    assert not campaign.pruning.smaller(
        {"states": 2, "parameters": 3}, {"states": 2, "parameters": 3}
    )
    assert not campaign.pruning.smaller(
        {"states": 2, "parameters": 3}, {"states": 1, "parameters": 4}
    )


def test_freeze_copies_public_inputs_and_does_not_modify_source(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "out"
    toy = smoke.fixture(source)
    old = {"artifact_sha256": toy["artifact_sha256"], "cells": toy["cells"]}
    monkeypatch.setattr(campaign, "snapshot", lambda p: (old, toy["rows"]))
    before = (source / "plan.json").read_bytes()
    frozen = campaign.freeze(source, root)
    assert campaign.freeze(source, root) == frozen
    assert campaign.verify(root) == frozen
    assert (source / "plan.json").read_bytes() == before
    assert sealed_read(root / "plan.json")["cells"] == toy["cells"]
    with pytest.raises(ValueError, match="separate"):
        campaign.freeze(source, source / "child")


def test_declared_balance_without_executable_verifier_is_not_pruned(tmp_path):
    plan = smoke.fixture(tmp_path)
    row = copy.deepcopy(plan["rows"][0])
    row["parent"]["request"]["base_candidate"]["constraints"] = [
        {"subject": "q", "kind": "conservation", "source": "proposer"}
    ]
    choice = campaign.choose(row, plan["cells"]["toy"])
    assert choice["status"] == "declared_constraints_require_review"
    assert choice["request"] is None


def test_snapshot_real_handoff_schema_and_untouched_source(tmp_path, monkeypatch):
    from autoformalism.rebuttal import review_deadline_pipeline as pipeline
    from scripts import smoke_shared_process_integration as integration

    source, output = tmp_path / "source", tmp_path / "output"
    plan = integration.fixture(source)
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    monkeypatch.setattr(pipeline, "replay_packet", lambda *a: None)
    for task in plan["tasks"]:
        for index in (0, 1):
            client = pipeline._client(
                source,
                plan,
                task,
                index,
                "http://offline",
                lambda: True,
                integration.transport_for([]),
            )
            pipeline.propose_one(source, plan, task, index, client)
            pipeline.fit_one(source, plan, task, index)
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    frozen = campaign.freeze(source, output)
    assert len(frozen["rows"]) == len(plan["tasks"])
    assert before == {str(p): p.read_bytes() for p in source.rglob("*.json")}
    assert campaign.freeze(source, output) == frozen
    row = frozen["rows"][0]
    selected = row["parent"]
    fit_dir = (
        pipeline.io.round_path(source, row["task"], selected["origin_round"]) / "fit"
    )
    (fit_dir / "result.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        campaign.snapshot(source)


def test_published_result_requires_its_fit_receipt(tmp_path, monkeypatch):
    smoke.fixture(tmp_path)
    monkeypatch.setattr(public, "_run_backend", backend([]))
    result = campaign.run_one(tmp_path, 0)
    path = tmp_path / "results" / result["task"]["task_id"] / "control/result.json"
    envelope = json.loads(path.read_text())
    envelope["result"]["parameters"]["a"] = 987
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="digest"):
        campaign.report(tmp_path)
    with pytest.raises(ValueError, match="digest"):
        campaign.run_one(tmp_path, 0)
