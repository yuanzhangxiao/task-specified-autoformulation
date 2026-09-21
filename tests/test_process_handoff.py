"""Empty additions, faithful advisory context and immutable signed handoffs."""

import json
import shutil
from copy import deepcopy

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.detention_process_pilot import _stage
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.process_handoff_audit import audit, replay_attempt
from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    ScientificVariable,
    equation_reply_model,
)
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.staged_topology import content_hash


def construct(root):
    """A full equation can be made exclusively of runtime-supplied processes."""
    brief = PublicScientificBrief(
        scientific_context="Two storage states exchange material; y is observed.",
        public_variables=[{"name": "y", "data_role": "target"}],
        requirements=[],
    )
    context = ValidationContext(targets=("y",))
    inventory = tuple(
        ScientificVariable(name=n, definition="differential", scientific_role=n)
        for n in ("x", "y")
    )
    calls = []

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        calls.append(payload)
        if payload.get("protocol") == "shared-process-contract-2":
            reply = {
                "processes": [
                    {
                        "name": "q",
                        "depends_on": ["x"],
                        "kind": "transfer",
                        "scientific_meaning": "Investigate threshold transfer.",
                        "uses": [
                            {"target": n, "sign": s, "conversion": "max(0,x)"}
                            for n, s in (("x", "negative"), ("y", "positive"))
                        ],
                    }
                ]
            }
        elif "selected_lhs" in payload:
            assert "one or more interaction terms" not in body["messages"][0]["content"]
            reply = {"terms": [], "inventory_revision": None}
        elif "selected_equation" in payload:
            selected = payload["selected_equation"]
            assert selected["lhs"] == "q", (
                "No LLM function call for automatic consumers"
            )
            advice = selected["terms"][0]["process_proposal_context"]
            assert advice["unresolved_conversions"][0]["expression"] == "max(0,x)"
            reply = {"functions": [{"expression": "max(0,x)", "parameters": []}]}
        else:
            reply = {"initial": {"fixed_value": 1.0}}
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 50},
        }

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        directory=root / "calls",
        namespace="fixture",
        seed=0,
        base_url="offline",
        transport=transport,
    )
    topology = _stage(
        root / "topology_stage.json",
        lambda: run_staged_topology(
            brief,
            context,
            client,
            root / "topology",
            initial_inventory=inventory,
            optional_process_review=True,
            bind_shared_processes=True,
            signed_shared_processes=True,
        ),
    )
    assert topology["complete_topology"], topology
    functions = _stage(
        root / "function_stage.json",
        lambda: run_staged_functions(
            brief,
            context,
            topology,
            client,
            root / "functions",
            generation_granularity="equation_batch_atomic_repair",
            function_repair_policy="certified_outer_gain",
        ),
    )
    assert functions["complete_model"], functions
    return topology, functions, calls


def test_complete_empty_additions_and_resume(tmp_path):
    topology, functions, calls = construct(tmp_path)
    assert all(e["accepted"] for e in topology["events"])
    assert len([c for c in calls if "selected_lhs" in c]) == 2
    assert len([c for c in calls if "selected_equation" in c]) == 1
    assert len(topology["equations"]) == 3
    assert all(len(e["terms"]) == 1 for e in topology["equations"])
    second = construct(tmp_path)
    # First-return tuples become JSON arrays in the saved stage artifact.
    assert content_hash(second[:2]) == content_hash((topology, functions))
    assert second[2] == []


def test_empty_only_allowed_with_runtime_terms():
    raw = {"terms": [], "inventory_revision": None}
    old = equation_reply_model(("x",), maximum_terms=32)
    with pytest.raises(ValueError, match="nonempty"):
        old.model_validate(raw)
    new = equation_reply_model(("x",), maximum_terms=32, runtime_supplies_terms=True)
    assert not new.model_validate(raw).terms
    assert "runtime_supplies_terms" not in new.model_json_schema()["properties"]
    both = {
        "terms": [
            {
                "sources": ["x"],
                "outer_weight_sign": "negative",
                "scientific_role": "loss",
            }
        ],
        "inventory_revision": {
            "variable": {
                "name": "z",
                "definition": "differential",
                "scientific_role": "memory",
            },
            "reason": "needed",
        },
    }
    with pytest.raises(ValueError, match="never both"):
        new.model_validate(both)


def test_unknown_source_still_rejected():
    model = equation_reply_model(("x",), maximum_terms=1, runtime_supplies_terms=True)
    with pytest.raises(ValueError):
        model.model_validate(
            {
                "terms": [
                    {
                        "sources": ["unknown"],
                        "outer_weight_sign": "positive",
                        "scientific_role": "source",
                    }
                ],
                "inventory_revision": None,
            }
        )


def source_fixture(source):
    """Emulate historical empty-addition rejection with complete saved requests."""
    common = source / "construction/results/example"
    topology, _, _ = construct(common / "review")
    shutil.copytree(common / "review/calls", common / "calls")
    for event in topology["events"]:
        if event["step"].startswith("equation_"):
            event.update(accepted=False, error="return nonempty terms")
    (common / "review/topology_stage.json").unlink()
    sealed_write(common / "review/topology_stage.json", {"result": topology})
    sealed_write(common / "proposal.json", {"status": "construction_failed"})
    sealed_write(
        source / "plan.json",
        {
            "protocol": "detention-process-pilot-3",
            "tasks": [
                {
                    "task_id": "example_free",
                    "construction_task": {"task_id": "example"},
                    "case": "fixture",
                }
            ],
            "cells": {"fixture": {"training": {"rows": []}}},
        },
    )
    return topology, common


def test_saved_audit_is_local_read_only_and_resumable(tmp_path):
    source, output = tmp_path / "source", tmp_path / "audit"
    source_fixture(source)
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = audit(source, output)
    assert result["newly_valid_attempts"] == 2
    assert not result["previously_accepted_now_blocked"]
    assert result["whole_models_recovered"] == 0
    assert result["llm_calls"] == result["optimizer_calls"] == 0
    assert result["historical_status_counts"] == {"missing": 1}
    assert audit(source, output) == result
    assert {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    with pytest.raises(ValueError, match="separate"):
        audit(source, source / "audit")
    stage = source / "construction/results/example/review/topology_stage.json"
    value = sealed_read(stage)
    value["result"]["events"][0]["error"] = "changed"
    stage.unlink()
    sealed_write(stage, {"result": value["result"]})
    with pytest.raises(ValueError, match="source or runtime changed"):
        audit(source, output)


def test_audit_preserves_checks_and_missing_records(tmp_path):
    source = tmp_path / "source"
    topology, common = source_fixture(source)
    event = next(e for e in topology["events"] if e["step"].startswith("equation_"))
    call = common / "calls" / (event["request_hash"] + ".json")
    record = json.loads(call.read_text())
    bindings = topology["shared_process_contract"]["bindings"]
    assert replay_attempt(record, event, bindings)["new_local_valid"]
    broken = deepcopy(record)
    broken["request"]["body"]["messages"][1]["content"] += "tampered"
    assert "hash differs" in replay_attempt(broken, event, bindings)["new_error"]
    assert not replay_attempt(record, event, [])["new_local_valid"]
    call.unlink()
    result = audit(source, tmp_path / "audit")
    assert result["newly_valid_attempts"] == 1
    assert any(
        a.get("status") == "missing_record"
        for c in result["constructions"] for a in c["attempts"]
    )


def test_source_symlink_escape_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    (source / "plan.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        audit(source, tmp_path / "audit")
