"""Read-only, resumable v5 audit uses saved decisions without inventing approval."""

import shutil
from copy import deepcopy

import pytest

from autoformalism.rebuttal import process_assembly_audit as audit
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search import function_dependencies as dep
from scripts import smoke_function_dependencies as old


@pytest.fixture(params=[False, True], ids=["isolated_review", "legacy_cold_review"])
def saved(tmp_path, request, monkeypatch):
    root = tmp_path / "v5"
    base = root / "construction/results/toy"
    with monkeypatch.context() as patch:
        if request.param:
            fixture, prepare = old.fixture, dep.prepare

            def linked_fixture():
                brief, context, source = fixture()
                bindings = source["shared_process_contract"]["bindings"]
                source["process_review"] = {
                    "bindings": list(bindings),
                    "suggestions": [b["proposal"] for b in bindings],
                }
                return brief, context, source

            def legacy_prepare(*args, **kwargs):
                return prepare(*args, **kwargs, _legacy_review_aliases=True)

            patch.setattr(old, "fixture", linked_fixture)
            patch.setattr(dep, "prepare", legacy_prepare)
        brief, context, source, functions = old.construct(base / "review", [])
    assert functions["complete_model"]
    sealed_write(base / "review/topology_stage.json", {"result": source})
    sealed_write(base / "review/function_stage.json", {"result": functions})
    shutil.copytree(base / "review/calls", base / "calls")
    sealed_write(
        root / "plan.json",
        {
            "protocol": "detention-process-pilot-3",
            "function_dependency_policy": dep.POLICY,
            "tasks": [{"case": "toy", "construction_task": {"task_id": "toy"}}],
            "cells": {
                "toy": {
                    "brief": brief.model_dump(mode="json"),
                    "context": context.model_dump(mode="json"),
                }
            },
        },
    )
    return root, brief, context, source, functions


def test_audit_records_historical_context_and_resumes_without_mutation(saved, tmp_path):
    root, *_ = saved
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "audit"
    result = audit.audit(root, output)
    assert not result["unexpected_acceptance_regressions"]
    assert bool(result["historical_alias_replays"]) == ("process_review" in saved[3])
    assert result["unavailable_saved_attempts"] == 0
    assert (
        result["llm_calls"]
        == result["optimizer_calls"]
        == result["whole_models_recovered"]
        == 0
    )
    assert audit.audit(root, output) == result
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert (output / "SUMMARY.md").exists()
    assert not sealed_read(output / "freeze.json")["trajectory_values_used"]


def test_saved_minus_normalizes_but_covariate_decision_is_not_invented(saved):
    _, brief, context, source, _ = saved
    raw = {
        "expression": "-max(0,x-crest)",
        "parameters": [],
        "revise_dependencies": True,
    }
    row = audit.assess(brief, context, source, "term_0_0", raw, True)
    assert row["classification"] == "outer_sign_normalized"
    raw["expression"] = "-max(0,x-crest)/area"
    row = audit.assess(brief, context, source, "term_0_0", raw, True)
    assert row["classification"] == "conversion_review_required"
    assert not raw.get("conversion_factor_is_intrinsic")


def test_audit_preserves_missing_saved_replies(saved, tmp_path):
    root, _, _, _, functions = saved
    event = next(
        e for e in functions["events"] if e["step"].startswith("atomic_repair_")
    )
    (
        root / "construction/results/toy/calls" / (event["request_hash"] + ".json")
    ).unlink()
    result = audit.audit(root, tmp_path / "audit")
    assert result["unavailable_saved_attempts"] == 1


def test_historical_drift_and_overlapping_output_fail(saved, tmp_path):
    root, *_ = saved
    with pytest.raises(ValueError, match="separate"):
        audit.audit(root, root / "new")
    output = tmp_path / "audit"
    audit.audit(root, output)
    path = root / "construction/results/toy/review/function_stage.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        audit.audit(root, output)


def test_dependency_tampering_not_misreported_as_model_failure(saved, tmp_path):
    root, _, _, _, functions = saved
    value = deepcopy(functions)
    value["dependency_revisions"][0]["before_sources"] = ["wrong"]
    path = root / "construction/results/toy/review/function_stage.json"
    path.unlink()
    sealed_write(path, {"result": value})
    with pytest.raises(ValueError, match="ledger differs") as failure:
        audit.audit(root, tmp_path / "audit")
    message = str(failure.value)
    assert "toy/review" in message
    assert '"interaction_id": "term_0_0"' in message
    assert (
        '"before_sources": {"replayed": ["x", "u", "crest"], "saved": ["wrong"]}'
        in message
    )
