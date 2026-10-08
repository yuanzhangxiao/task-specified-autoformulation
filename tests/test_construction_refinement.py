"""Minimal fixes, saved repair provenance, receipts and deterministic continuation."""

import json

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_comparison as campaign
from autoformalism.research import construction_stage_progress as progress
from autoformalism.research.construction_refinement import REPAIRS, validate_tasks
from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_prompts as prompts
from autoformalism.search import construction_repair_fidelity as fidelity
from autoformalism.search import construction_schedules as schedules
from tests.test_construction_bookkeeping import shared_draft
from tests.test_construction_comparison import source_fixture, transport
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    patch,
    variable,
)
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


def test_receipt_preserves_full_effects_of_types_deletions_and_propagated_uses():
    before = shared_draft()
    after = apply(
        before,
        variables=[variable("y", "algebraic")],
        equations=[equation("y", "x")],
        remove_processes=["p"],
    )
    receipt = fidelity.receipt(before, after)
    by_name = {r["target"]: r for r in receipt["balances"]}
    assert set(by_name) == {"x", "y"}
    assert by_name["y"]["type_changed"]
    assert by_name["x"]["removed_process_uses"][0]["process"] == "p"
    assert by_name["y"]["before"]["lhs"] == "d(y)/dt"
    assert by_name["y"]["after"]["lhs"] == "y"
    assert before == shared_draft()
    assert fidelity.removal_options(before)[0]["if_you_choose_to_delete"] == {
        "remove_processes": ["p"]
    }
    assert fidelity._removed([{"x": 1}, {"x": 1}], [{"x": 1}]) == [{"x": 1}]


def test_minimal_clarity_has_only_versioned_additions_and_keeps_valid_partial_inventory(
    tmp_path,
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        assert body["messages"][0]["content"] == prompts.SYSTEM
        assert "read_only_generated_process_definitions" in p["current_draft"]
        assert p["current_draft"]["editing_contract"] == prompts.CLARITY_EDITING
        reply = p["response_template"]
        reply["stage_complete"] = True
        if p["stage"] == "variables":
            assert "Set stage_complete=true" in p["stage_instructions"]
            assert "Fitted coefficients are parameters" in p["stage_instructions"]
            reply["variables"] = [variable("x" if len(calls) == 1 else "y")]
            if len(calls) == 2:
                assert p["last_edit_result"]["stage_completion"][
                    "missing_public_targets"
                ] == ["y"]
        if p["stage"] == "equations":
            reply["equations"] = [equation("x", "u"), equation("y", "x")]
        return response(reply)

    def run():
        client = ConstructionClient(
            settings=StagedModelSettings(),
            directory=tmp_path / "calls",
            namespace="minimal-v2",
            seed=0,
            base_url="http://offline",
            transport=send,
            token_transport=tokenize,
        )
        return schedules.run(
            brief(),
            context(),
            {},
            brief().model_dump(mode="json"),
            client,
            tmp_path / "construction",
            "joint_adaptive",
            process_question="dedicated",
            prompt_family="minimal",
            bookkeeping_policy=bookkeeping.MINIMAL_POLICY,
        )

    result = run()
    assert result["status"] == "topology_complete"
    assert [p["stage"] for p in calls] == [
        "variables",
        "variables",
        "shared_laws",
        "equations",
    ]
    assert run() == result and len(calls) == 4
    with pytest.raises(ValueError):
        bookkeeping.validate_policy(bookkeeping.MINIMAL_POLICY, "current")


def test_saved_repair_only_resumes_with_same_start_and_no_construction_calls(tmp_path):
    calls = []
    start = apply(variables=[variable("y")], equations=[equation("y", "missing")])

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        assert p["stage"] == "repair"
        assert fidelity.INSTRUCTION in p["stage_instructions"]
        assert p["current_draft"]["declarations"] == start.model_dump(mode="json")
        return response(
            patch(equations=[equation("y", "u")], stage_complete=True).model_dump(
                mode="json"
            )
        )

    def run(can_start=lambda: True):
        client = ConstructionClient(
            settings=StagedModelSettings(),
            directory=tmp_path / "calls",
            namespace="saved",
            seed=0,
            base_url="http://offline",
            transport=send,
            token_transport=tokenize,
            can_start=can_start,
        )
        return schedules.run(
            brief(),
            context(),
            {},
            brief().model_dump(mode="json"),
            client,
            tmp_path / "construction",
            "separate",
            starting_draft=start,
            bookkeeping_policy=bookkeeping.FIDELITY_POLICY,
        )

    with pytest.raises(DeferredCall):
        run(lambda: False)
    result = run()
    assert result["status"] == "topology_complete"
    assert result["before_repair"]["draft"] == start.model_dump(mode="json")
    assert result["before_repair"]["cost"]["physical_requests"] == 0
    assert result["before_repair"]["stage_outcomes"] == []
    assert run() == result and len(calls) == 1
    event = sealed_read(tmp_path / "construction/events/000.json")
    assert event["repair_receipt"]["balances"][0]["removed_ordinary_records"]


def test_refinement_freezes_sources_roster_and_provenance_and_audits_stages(tmp_path):
    source_fixture(tmp_path)
    old_root, root = tmp_path / "old", tmp_path / "refined"
    old = campaign.freeze(
        tmp_path / "new",
        old_root,
        study="shared_law_comparison",
        bookkeeping_policy=bookkeeping.FEEDBACK_POLICY,
    )
    # The synthetic fixture uses different base IDs; freeze this known diagnostic
    # roster before creating any calls or request namespaces.
    for task in old["tasks"]:
        index = "cell06" if "coupled" in task["benchmark_id"] else "cell07"
        task["task_id"] = (
            f"{index}_seed0_full_{task['policy']}_{task['process_question']}"
        )
    old.pop("artifact_sha256")
    (old_root / "plan.json").unlink()
    old = sealed_write(old_root / "plan.json", old)
    calls = []
    base = transport(calls)

    def send(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        if payload["stage"] == "shared_laws":
            calls.append(payload)
            return response({**payload["response_template"], "stage_complete": True})
        return base(url, body, timeout)

    for task in old["tasks"]:
        if task["task_id"] in REPAIRS:
            campaign.propose(
                old_root,
                old,
                task,
                "http://offline",
                transport=send,
                token_transport=tokenize,
            )
    plan = campaign.freeze(
        tmp_path / "new", root, study="refinement_confirmation", repair_source=old_root
    )
    assert campaign.verify(root) == plan and len(plan["tasks"]) == 11
    assert len([t for t in plan["tasks"] if "starting_checkpoint" in t]) == 3
    report = campaign.report(root, plan)
    assert report["status_counts"] == {"pending": 11}
    for task in [plan["tasks"][0], plan["tasks"][-1]]:
        campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
    campaign.report(root, plan)
    audit = progress.audit(root)
    assert (
        audit["rows"][0]["stages"]["variables"]["first_retained"][
            "missing_public_targets"
        ]
        == []
    )
    assert list(audit["rows"][-1]["stages"]) == ["repair"]
    progress.write_report(audit, tmp_path / "audit")
    with pytest.raises(ValueError, match="outside"):
        progress.write_report(audit, root / "audit")
    plan["tasks"][-1]["starting_checkpoint"]["draft"]["variables"] = []
    with pytest.raises(ValueError, match="roster"):
        validate_tasks(plan)
    # Actual deleted cache evidence is not silently treated as an absent stage.
    call = next(
        (root / "results" / plan["tasks"][0]["task_id"] / "calls").glob("*.json")
    )
    call.unlink()
    with pytest.raises(ValueError):
        progress.audit(root)


def test_transfer_facts_are_scoped_and_not_scientific_judgments():
    draft = shared_draft().model_dump(mode="json")
    draft["processes"][0]["uses"] = draft["processes"][0]["uses"][:1]
    values = progress.facts(brief(), draft)
    assert values["invalid_pairwise_transfers"] == ["p"]
    assert values["single_consumer_processes"] == ["p"]
