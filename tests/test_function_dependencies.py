"""Source revisions preserve hard graphs, shared uses and immutable history."""

import json
import shutil
from copy import deepcopy

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal import function_dependency_audit as offline
from autoformalism.rebuttal.prefit_construction_audit import reconstruct
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.search import function_dependencies as dep
from scripts import smoke_function_dependencies as smoke


def reply(expression, explicit=False):
    return dep.DependencyFunctionReply(
        expression=expression, parameters=(), revise_dependencies=explicit
    )


def test_fixed_covariates_and_explicit_changes_are_distinct():
    brief, context, source = smoke.fixture()
    unchanged = deepcopy(source)
    changed, event = dep.prepare(brief, context, source, "term_1_0", reply("u/area"))
    assert event["kind"] == "public_fixed_covariates"
    assert event["after_sources"] == ["u", "area"]
    assert changed["public_structure_checks_passed"]
    with pytest.raises(ValueError, match="revision required"):
        dep.prepare(brief, context, source, "term_0_0", reply("max(0,x-crest)"))
    changed, event = dep.prepare(
        brief, context, source, "term_0_0", reply("max(0,x-crest)", True)
    )
    assert event["kind"] == "proposer_revision"
    binding = changed["shared_process_contract"]["bindings"][0]
    assert binding["proposal"]["depends_on"] == ["x", "crest"]
    assert binding["signed_declaration"]["depends_on"] == ["x", "crest"]
    assert binding["uses"] == source["shared_process_contract"]["bindings"][0]["uses"]
    assert source == unchanged


@pytest.mark.parametrize("expression", ["u", "crest", "0"])
def test_required_memory_cannot_be_deleted(expression):
    brief, context, source = smoke.fixture()
    with pytest.raises(ValueError, match=r"public pathways|at least 1"):
        dep.prepare(brief, context, source, "term_0_0", reply(expression, True))


@pytest.mark.parametrize(
    "expression", ["private_state", "future_y", "area_unknown", "t"]
)
def test_unavailable_names_rejected_even_with_explicit_decision(expression):
    brief, context, source = smoke.fixture()
    with pytest.raises(ValueError, match="unavailable"):
        dep.prepare(brief, context, source, "term_0_0", reply(expression, True))


def test_shared_consumer_identity_and_algebraic_cycles_stay_blocked():
    brief, context, source = smoke.fixture()
    with pytest.raises(ValueError, match="preserve the agreed"):
        dep.prepare(brief, context, source, "term_1_1", reply("x", True))
    with pytest.raises(ValueError, match="cycle"):
        dep.prepare(brief, context, source, "term_0_0", reply("q+x", True))


def test_offline_labels_never_invent_proposer_authorization():
    brief, context, source = smoke.fixture()
    row = offline.assess(
        brief,
        context,
        source,
        "term_0_0",
        {"expression": "max(0,x-crest)", "parameters": []},
        False,
    )
    assert row["classification"] == "requires_proposer_revision"
    assert row["explicit_valid"] and not row["newly_mechanically_valid"]
    assert not row["whole_model_recovered"]
    row = offline.assess(
        brief,
        context,
        source,
        "term_1_0",
        {"expression": "u/area", "parameters": []},
        False,
    )
    assert row["classification"] == "public_covariate_addition"
    assert row["newly_mechanically_valid"]
    row = offline.assess(
        brief,
        context,
        source,
        "term_0_0",
        {"expression": "u*(area/area)", "parameters": []},
        False,
    )
    assert row["classification"] == "blocked"
    assert row["self_division_factors"] == ["area / area"]


def test_complete_reconstruction_resume_and_visible_contract(tmp_path):
    assert smoke.run(tmp_path)["status"] == "passed"
    calls = []
    brief, context, source, result = smoke.construct(tmp_path / "fresh", calls)
    batch = next(c for c in calls if "selected_equation" in c)
    assert batch["selected_equation"]["terms"][0]["sources"] == ["x", "u", "crest"]
    assert batch["permitted_fixed_covariates"] == ["area", "crest"]
    assert batch["process_consumer_conversions"]["q"][0]["conversion"] == "1/area"
    assert not result["equation_connectivity"]["without_target_path"]
    cell = {
        "brief": brief.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
    }
    damaged = deepcopy(result)
    damaged["dependency_revisions"][0]["after_sources"] += ["u"]
    with pytest.raises(ValueError, match="ledger differs"):
        reconstruct(cell, {"topology": source, "functions": damaged})
    damaged = deepcopy(result)
    damaged["effective_source"]["equations"][1]["terms"][0]["sources"] = ["area"]
    with pytest.raises(ValueError, match="source differs"):
        reconstruct(cell, {"topology": source, "functions": damaged})
    damaged = deepcopy(result)
    damaged["dependency_policy"] = "strict"
    with pytest.raises(ValueError, match="explicit policy"):
        reconstruct(cell, {"topology": source, "functions": damaged})


def test_drain_and_resume_after_committed_edit(tmp_path):
    calls = []
    with pytest.raises(DeferredCall):
        smoke.construct(tmp_path, calls, can_start=lambda: len(calls) < 2)
    progress = json.loads((tmp_path / "functions/progress.json").read_text())
    assert len(progress["dependency_revisions"]) == 1
    assert smoke.construct(tmp_path, calls)[-1]["complete_model"]
    assert len(calls) == 5


def test_failed_binding_does_not_commit_a_dependency_edit(tmp_path):
    calls = []
    original = smoke.transport(calls)

    def bad_first(url, body, timeout):
        record = original(url, body, timeout)
        payload = calls[-1]
        if payload.get("selected_equation", {}).get("lhs") == "x":
            raw = {
                "functions": [
                    {
                        "expression": "u/area",
                        "parameters": [{"name": "unused", "role": "rate"}],
                    }
                ]
            }
            record["choices"][0]["message"]["content"] = json.dumps(raw)
        elif payload.get("selected_term", {}).get("lhs") == "x":
            raw = {"expression": "u", "parameters": [], "revise_dependencies": False}
            record["choices"][0]["message"]["content"] = json.dumps(raw)
        return record

    _, _, _, result = smoke.construct(tmp_path, calls, responder=bad_first)
    assert result["complete_model"], result["error"]
    assert len(result["dependency_revisions"]) == 1
    assert result["effective_source"]["equations"][1]["terms"][0]["sources"] == ["u"]


def saved_fixture(source):
    common = source / "construction/results/example"
    brief, context, topology, functions = smoke.construct(common / "review", [])
    # Emulate strict historical replies: use original source sets, not new policy.
    functions = {
        "status": "failed",
        "candidate": None,
        "events": [],
        "batch_term_audits": [
            {
                "interaction_id": "term_0_0",
                "batch_accepted": False,
                "batch_function": {"expression": "max(0,x-crest)", "parameters": []},
            },
            {
                "interaction_id": "term_1_0",
                "batch_accepted": False,
                "batch_function": {"expression": "u/area", "parameters": []},
            },
            {
                "interaction_id": "term_1_0",
                "batch_accepted": True,
                "batch_function": {"expression": "u", "parameters": []},
            },
        ],
    }
    sealed_write(common / "review/function_stage.json", {"result": functions})
    sealed_write(common / "review/topology_stage.json", {"result": topology})
    shutil.copytree(common / "review/calls", common / "calls")
    sealed_write(
        source / "plan.json",
        {
            "protocol": "detention-process-pilot-3",
            "tasks": [
                {
                    "task_id": "example_gain",
                    "construction_task": {"task_id": "example"},
                    "case": "fixture",
                }
            ],
            "cells": {
                "fixture": {
                    "brief": brief.model_dump(mode="json"),
                    "context": context.model_dump(mode="json"),
                }
            },
        },
    )


def test_saved_audit_read_only_resume_and_drift(tmp_path):
    source, output = tmp_path / "source", tmp_path / "audit"
    saved_fixture(source)
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = offline.audit(source, output)
    assert result["classification_counts"] == {
        "already_valid": 1,
        "public_covariate_addition": 1,
        "requires_proposer_revision": 1,
    }
    assert result["newly_mechanically_valid"] == 1
    assert not result["previously_accepted_now_blocked"]
    assert (
        result["llm_calls"]
        == result["optimizer_calls"]
        == result["whole_models_recovered"]
        == 0
    )
    assert offline.audit(source, output) == result
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    path = source / "construction/results/example/review/function_stage.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        offline.audit(source, output)


def test_audit_rejects_overlap_and_symlink_escape(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        offline.audit(tmp_path, tmp_path / "audit")
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (source / "plan.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        offline.audit(source, tmp_path / "audit")


@pytest.mark.parametrize("malformed", [False, True])
def test_saved_atomic_reply_is_checked_in_its_original_context(tmp_path, malformed):
    source = tmp_path / "source"
    saved_fixture(source)
    base = source / "construction/results/example"
    record_path, record = next(
        (p, json.loads(p.read_text()))
        for p in (base / "calls").glob("*.json")
        if "selected_term" in offline.request_payload(json.loads(p.read_text()))
    )
    # Model a historical repair without the newly introduced decision field.
    record["raw_response"]["choices"][0]["message"]["content"] = (
        "not json"
        if malformed
        else json.dumps({"expression": "max(0,x-crest)", "parameters": []})
    )
    record_path.write_text(json.dumps(record))
    path = base / "review/function_stage.json"
    saved = sealed_read(path)
    saved.pop("artifact_sha256")
    saved["result"]["events"] = [
        {
            "step": "atomic_repair_term_0_0",
            "attempt": 0,
            "request_hash": record["request_hash"],
            "accepted": False,
        }
    ]
    path.unlink()
    sealed_write(path, saved)
    result = offline.audit(source, tmp_path / "audit")
    attempts = result["constructions"][0]["routes"][0]["attempts"]
    assert attempts[-1]["classification"] == (
        "blocked" if malformed else "requires_proposer_revision"
    )
    assert not attempts[-1]["whole_model_recovered"]
    record["request_hash"] = "0" * 64
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="event request hash differs"):
        offline.audit(source, tmp_path / "tampered-audit")


def test_legacy_function_schema_does_not_silently_accept_revision_flag():
    with pytest.raises(ValueError, match="Extra inputs"):
        InteractionFunctionReply.model_validate(
            {"expression": "x", "parameters": [], "revise_dependencies": True}
        )
