"""Offline evidence counts must preserve context, failures and provenance."""

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import construction_stage_audit as a
from autoformalism.staged_topology import content_hash


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def call(payload: dict, reply: object) -> dict:
    return {
        "status": "responded",
        "request": {
            "messages": [
                {"role": "user", "content": "JSON below\n" + json.dumps(payload)}
            ]
        },
        "response": {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ]
        },
        "decisions": [],
    }


def variable_payload() -> dict:
    return {
        "agenda": {
            "agenda_id": "upstream_memory",
            "targets": ["y"],
            "drivers": ["u"],
            "requires_dynamic_memory": True,
        },
        "current_inventory": [
            {
                "name": "h",
                "definition": "differential",
                "scientific_role": "upstream storage",
            }
        ],
        "public_brief": {
            "public_variables": [{"name": "y", "data_role": "target"}],
            "scientific_context": "The public prompt does not fix an output role.",
        },
    }


def test_empty_reply_does_not_choose_an_existing_state():
    c = call(variable_payload(), {"variables": []})
    c["decisions"] = [
        {"event": {"unresolved_obligations": ["dynamic_memory_mediator:upstream"]}}
    ]
    result = a.variable_call(c)
    assert result["schema_error"] is None
    assert result["empty_reply_with_existing_dynamic_candidates"]
    assert result["existing_dynamic_candidates"][0]["name"] == "h"
    assert not result["target_contract_marker_displayed"]
    assert "memory_candidates" not in result  # Audit cannot assign scientific roles.


def test_schema_and_transport_failure_remain_visible():
    c = call(
        variable_payload(), {"variables": [{"name": "h", "definition": "unknown"}]}
    )
    assert a.variable_call(c)["schema_error"]
    c["status"] = "interrupted"
    result = a.variable_call(c)
    assert result["schema_error"] and result["parsed_reply"] is None


def test_shared_conflicts_tolerate_exact_repetition():
    positive = {"sources": ["P"], "outer_weight_sign": "positive"}
    negative = {"sources": ["P"], "outer_weight_sign": "negative"}
    new_consumer = {"sources": ["Q"], "outer_weight_sign": "negative"}
    c = call(
        {"required_process_contributions": [positive]},
        {"terms": [positive, negative, new_consumer]},
    )
    bindings = [{"proposal": {"name": n}} for n in ("P", "Q")]
    result = a.process_conflicts(c, bindings)
    assert len(result) == 2
    assert result[0]["classification"] == "declared_use_sign_or_group_conflict"
    assert result[1]["classification"] == "undeclared_consumer"


def fixture(source: Path) -> tuple[Path, str]:
    task = {"task_id": "toy", "benchmark_id": "toy"}
    plan = sealed_write(
        source / "plan.json",
        {
            "protocol": "phase-c-construction-baseline-1",
            "tasks": [task],
            "cells": {
                "toy": {
                    "target_contract": {
                        "targets": [
                            {
                                "target_channel": "y",
                                "expected_representation": "instantaneous_process",
                            }
                        ]
                    }
                }
            },
        },
    )
    identity = content_hash([plan["artifact_sha256"], task])
    directory = source / "results/toy"
    c = call(variable_payload(), {"variables": []})
    request = {"namespace": identity, "body": c["request"]}
    key = content_hash(request)
    event = {"request_hash": key, "step": "variables_0", "attempt": 0, "accepted": True}
    save(
        directory / f"calls/{key}.json",
        {
            "request_hash": key,
            "request": request,
            "raw_response": c["response"],
            "status": "responded",
            "step": "variables_0",
            "attempt": 0,
        },
    )
    result = {
        "inventory": [
            {
                "name": "y",
                "definition": "differential",
                "scientific_role": "<script>bad</script>",
            }
        ],
        "memory_candidates": {},
        "status": "variables_complete",
    }
    save(directory / "construction/variables/result.json", result)
    save(
        directory / "construction/variables/progress.json",
        {**result, "events": [event]},
    )
    save(
        directory / "construction/variables.json", {"result": result, "events": [event]}
    )
    sealed_write(
        directory / "proposal.json",
        {
            "identity": identity,
            "status": "construction_failed",
            "cost": {"physical_requests": 1},
        },
    )
    return directory, key


def test_audit_counts_calls_once_and_never_changes_source(tmp_path):
    source = tmp_path / "source"
    fixture(source)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    report = a.audit(source)
    assert report["counts"]["accepted_variable_calls"] == 1
    assert report["counts"]["constructions_with_policy_mismatch"] == 1
    assert len(report["rows"][0]["variable_calls"][0]["decisions"]) == 1
    assert len(report["rows"][0]["variable_calls"][0]["decisions"][0]["files"]) == 2
    a.write_report(report, tmp_path / "report")
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    page = (tmp_path / "report/VARIABLES.html").read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page
    with pytest.raises(ValueError, match="outside the source"):
        a.write_report(report, source / "audit")
    assert a.audit(source) == report


@pytest.mark.parametrize(
    "fault", ["tampered_call", "missing_call", "conflicting_decision"]
)
def test_bad_evidence_is_not_silently_counted_as_success(tmp_path, fault):
    directory, key = fixture(tmp_path)
    path = directory / f"calls/{key}.json"
    if fault == "missing_call":
        path.unlink()
    elif fault == "tampered_call":
        raw = json.loads(path.read_text())
        raw["request"]["namespace"] = "different"
        save(path, raw)
    else:
        save(
            directory / "construction/conflict.json",
            {
                "request_hash": key,
                "accepted": False,
            },
        )
    with pytest.raises(ValueError):
        a.audit(tmp_path)
