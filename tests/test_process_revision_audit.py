"""Real saved-stage histories, no invented decisions, no mutation, honest counts."""

import shutil
from copy import deepcopy

import pytest

from autoformalism.rebuttal import process_revision_audit as audit
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import smoke_process_assembly_contract as old


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "v6"
    for name, intrinsic in (("corrected", False), ("intrinsic", True)):
        base = root / "construction/results" / name
        brief, context, topology, functions = old.construct(
            base / "review", [], intrinsic=intrinsic
        )
        sealed_write(base / "review/topology_stage.json", {"result": topology})
        sealed_write(base / "review/function_stage.json", {"result": functions})
        shutil.copytree(base / "review/calls", base / "calls")
    sealed_write(
        root / "plan.json",
        {
            "protocol": "detention-process-pilot-3",
            "process_assembly_policy": audit.assembly.POLICY,
            "function_dependency_policy": audit.dep.POLICY,
            "tasks": [
                {"case": "toy", "construction_task": {"task_id": name}}
                for name in ("corrected", "intrinsic")
            ],
            "cells": {
                "toy": {
                    "brief": brief.model_dump(mode="json"),
                    "context": context.model_dump(mode="json"),
                }
            },
        },
    )
    return root


def test_saved_audit_resume_readonly_and_unverified_overlap(source, tmp_path):
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = audit.audit(source, tmp_path / "audit")
    assert result["historical_counters"]["confirmed_actual_overlaps"] == 1
    assert result["historical_counters"]["intrinsic_flags_without_overlap"] == 0
    assert len(result["accepted_requiring_new_conversion_decision"]) == 1
    assert not result["unexpected_acceptance_regressions"]
    assert (
        result["llm_calls"]
        == result["optimizer_calls"]
        == result["whole_models_recovered"]
        == 0
    )
    assert audit.audit(source, tmp_path / "audit") == result
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_missing_records_are_not_successes(source, tmp_path):
    base = source / "construction/results/corrected"
    event = sealed_read(base / "review/function_stage.json")["result"]["events"][0]
    path = base / "calls" / (event["request_hash"] + ".json")
    path.unlink()
    result = audit.audit(source, tmp_path / "audit")
    assert result["unavailable_saved_attempts"] == 1


def test_drift_escape_and_overlap_fail_closed(source, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        audit.audit(source, source / "audit")
    audit.audit(source, tmp_path / "audit")
    path = source / "construction/results/corrected/review/function_stage.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        audit.audit(source, tmp_path / "audit")


def test_counters_separate_actual_overlap_flags_and_public_paths():
    functions = {
        "assembly_decisions": [
            {"intrinsic_factor_confirmed": True, "consumer_conversion_overlap": []},
            {
                "intrinsic_factor_confirmed": True,
                "consumer_conversion_overlap": [{"target": "x"}],
            },
        ],
        "batch_term_audits": [{"batch_error": "CONSUMER_CONVERSION_OVERLAP"}],
        "events": [
            {
                "step": "atomic_repair_term_0_0",
                "accepted": False,
                "error": "dependency revision breaks public pathways",
            },
            {
                "step": "equation_functions_1",
                "accepted": False,
                "error": "equation function count mismatch",
            },
        ],
    }
    counters = audit.historical_counters(functions)
    assert (
        counters["confirmed_actual_overlaps"]
        == counters["intrinsic_flags_without_overlap"]
        == 1
    )
    assert counters["public_pathway_rejected_attempts"] == 1
    assert counters["process_target_path_rejected_attempts"] == 0
    assert counters["batch_delivery_rejected_attempts"] == 1


def test_changed_ledger_is_not_classified_as_scientific_failure(source, tmp_path):
    path = source / "construction/results/corrected/review/function_stage.json"
    saved = sealed_read(path)
    functions = deepcopy(saved["result"])
    functions["dependency_revisions"][0]["before_sources"] = ["tampered"]
    path.unlink()
    sealed_write(path, {"result": functions})
    with pytest.raises(ValueError, match="ledger differs"):
        audit.audit(source, tmp_path / "audit")
