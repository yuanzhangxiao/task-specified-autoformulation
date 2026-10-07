"""Matched minimal/current wording with one typed, resumable construction schedule."""

import json
from pathlib import Path

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_prompts as prompts
from autoformalism.search import construction_schedules as schedules
from tests.test_construction_comparison import source_fixture
from tests.test_construction_ledger import brief, context, equation, process, variable
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


def execute(root, family, send, **kwargs):
    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=root / "calls",
        namespace=family,
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
        **kwargs,
    )
    return schedules.run(
        brief(),
        context(),
        {},
        brief().model_dump(mode="json"),
        client,
        root / "construction",
        "joint_adaptive",
        process_question="dedicated",
        prompt_family=family,
    )


@pytest.mark.parametrize("family", prompts.FAMILIES)
def test_stage_order_shared_assembly_multi_lhs_and_cached_resume(tmp_path, family):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append((body, p))
        reply = p["response_template"]
        if p["stage"] == "variables":
            assert "processes" not in reply and "equations" not in reply
            reply["variables"] = [variable("x"), variable("y")]
        elif p["stage"] == "shared_laws":
            assert "equations" not in reply
            reply["processes"] = [process()]
        else:
            reply["equations"] = [equation("x", "u"), {"name": "y", "terms": []}]
        reply["stage_complete"] = True
        return response(reply)

    result = execute(tmp_path, family, send)
    assert result["status"] == "topology_complete"
    assert [p["stage"] for _, p in calls] == list(prompts.STAGES)
    assert result["before_repair"]["assessment"]["eligible"]
    assert len(result["draft"]["processes"]) == 1
    assert execute(tmp_path, family, send) == result
    assert len(calls) == 3
    for body, payload in calls:
        assert payload["stage_schedule"] == prompts.SCHEDULE
        assert payload["public_brief"] == brief().model_dump(mode="json")
        assert body["messages"][0]["content"] == (
            prompts.SYSTEM if family == "minimal" else schedules.SYSTEM
        )
        if family == "minimal":
            assert "MEMORY AND RETURN TO BASELINE" not in json.dumps(body)
            assert "[A, C]" in payload["editing_rules"]
            assert payload["binding_context"] == {
                "mechanism_bindings": [],
                "feedback_bindings": {
                    "applicable_targets": [],
                    "needs_proposer_coordinates": [],
                    "assignments": [],
                },
            }


@pytest.mark.parametrize("family", prompts.FAMILIES)
def test_atomic_rejection_receipt_and_targeted_repair_keep_public_context(
    tmp_path, family
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        reply = p["response_template"]
        reply["stage_complete"] = True
        if p["stage"] == "variables":
            reply["variables"] = [variable("x"), variable("y")]
        elif p["stage"] == "equations":
            reply["equations"] = [equation("x", "u"), equation("y", "a")]
            # Unknown source is pending until overall checking.
            reply["equations"][1]["terms"][0]["sources"] = ["missing"]
        elif p["stage"] == "repair":
            n = sum(q["stage"] == "repair" for q in calls)
            reply["equations"] = [equation("x", "u"), equation("y", "x")]
            if n == 1:
                reply["remove_equations"] = ["x"]
                reply["variables"] = [variable("z")]
            else:
                assert not p["last_edit_result"]["transaction_applied"]
                assert not any(
                    v["name"] == "z"
                    for v in p["current_draft"]["declarations"]["variables"]
                )
                assert p["runtime_diagnostics"]["structural_failures"]
        return response(reply)

    result = execute(tmp_path, family, send)
    assert result["status"] == "topology_complete"
    assert not result["before_repair"]["assessment"]["eligible"]
    assert len(calls) == 5
    assert all(p["public_brief"] == calls[0]["public_brief"] for p in calls)
    assert all(
        p["public_source_catalog"] == calls[0]["public_source_catalog"] for p in calls
    )
    assert execute(tmp_path, family, send) == result
    assert len(calls) == 5


def test_interrupted_stage_resumes_same_requests_without_new_namespace(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        reply = p["response_template"]
        reply["stage_complete"] = True
        if p["stage"] == "variables":
            reply["variables"] = [variable("y")]
        if p["stage"] == "equations":
            reply["equations"] = [equation("y", "u")]
        return response(reply)

    with pytest.raises(DeferredCall):
        execute(tmp_path, "minimal", send, can_start=lambda: len(calls) < 1)
    result = execute(tmp_path, "minimal", send)
    assert result["status"] == "topology_complete"
    assert len(calls) == 3


def test_prompt_family_roster_report_and_budget_pairing(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "pilot"
    plan = campaign.freeze(tmp_path / "new", root, study="prompt_comparison")
    assert campaign.verify(root) == plan
    assert len(plan["tasks"]) == 16
    assert len({t["benchmark_id"] for t in plan["tasks"][:8]}) == 8
    assert sum("detention" in t["benchmark_id"] for t in plan["tasks"]) == 4
    for case in plan["cells"]:
        tasks = [t for t in plan["tasks"] if t["benchmark_id"] == case]
        assert {t["prompt_family"] for t in tasks} == set(prompts.FAMILIES)
        assert len({t["matched_task_id"] for t in tasks}) == 1
        assert {t["stage_schedule"] for t in tasks} == {prompts.SCHEDULE}
        assert {t["policy"] for t in tasks} == {"joint_adaptive"}
    result = campaign.report(root, plan)
    assert result["status_counts"] == {"pending": 16}
    assert set(result["groups"]) == {"current:full", "minimal:full"}
    assert all(
        g["planned"] == 8 and g["tokens"]["median"] is None
        for g in result["groups"].values()
    )
    assert "Both use variables -> shared processes" in (root / "SUMMARY.md").read_text()


@pytest.mark.parametrize(
    "key,value",
    [("prompt_family", "other"), ("stage_schedule", "other"), ("policy", "separate")],
)
def test_reject_changed_prompt_schedule_or_family(tmp_path, key, value):
    source_fixture(tmp_path)
    root = tmp_path / "pilot"
    plan = campaign.freeze(tmp_path / "new", root, study="prompt_comparison")
    plan["tasks"][0][key] = value
    plan.pop("artifact_sha256")
    (root / "plan.json").unlink()
    sealed_write(root / "plan.json", plan)
    with pytest.raises(ValueError, match="roster"):
        campaign.verify(root)


def test_reviewed_wording_matches_implemented_text_and_replacement_semantics():
    text = Path("docs/PHASE_C_MINIMAL_PROMPTS.txt").read_text()
    for value in (
        prompts.SYSTEM,
        prompts.VARIABLES,
        prompts.SHARED,
        prompts.TOPOLOGY,
        prompts.EDITING,
    ):
        assert " ".join(value.split()) in " ".join(text.split())
    original = ledger.apply_patch(
        brief(),
        ledger.Draft(),
        ledger.DraftPatch(
            variables=[variable("y")],
            equations=[
                {
                    "name": "y",
                    "terms": equation("y", "u")["terms"] + equation("y", "a")["terms"],
                }
            ],
            stage_complete=False,
        ),
    )
    changed = ledger.apply_patch(
        brief(),
        original,
        ledger.DraftPatch(
            equations=[equation("y", "area")],
            stage_complete=False,
        ),
    )
    assert [term.sources for term in changed.equations[0].terms] == [("area",)]


def test_campaign_dispatches_both_families_and_reports_saved_records(tmp_path):
    from tests.test_construction_comparison import transport

    source_fixture(tmp_path)
    root = tmp_path / "pilot"
    plan = campaign.freeze(tmp_path / "new", root, study="prompt_comparison")
    calls = []
    base = transport(calls)

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        if p["stage"] == "shared_laws":
            calls.append(p)
            return response({**p["response_template"], "stage_complete": True})
        result = base(url, body, timeout)
        message = result["choices"][0]["message"]
        edits = json.loads(message["content"])
        for e in edits.get("equations", []):
            if e["name"] in {"T", "h_down"}:
                e["terms"][0]["sources"].append(e["name"])
        message["content"] = json.dumps(edits)
        return result

    for task in plan["tasks"]:
        result = campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
        assert result["status"] == "topology_complete"
    assert len(calls) == 48
    report = campaign.report(root, plan)
    assert report["status_counts"] == {"topology_complete": 16}
    assert all(g["final_complete"] == 8 for g in report["groups"].values())
    assert all(
        sum(e["shared_law_question_displayed"] for e in r["process_history"]) == 1
        for r in report["rows"]
    )
    for task in plan["tasks"]:
        campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
    assert len(calls) == 48


def test_binding_questions_preserve_control_field_paths_and_only_ask_applicable_roles():
    from tests.test_public_graph_obligations import contract

    draft = ledger.Draft(variables=[variable("y", "algebraic"), variable("x")])
    result = prompts.binding_questions(brief(True), draft, contract("target_feedback"))
    assert result["mechanism_bindings"][0]["requirement_id"] == "memory"
    feedback = result["feedback_bindings"]
    assert feedback["applicable_targets"] == ["y"]
    assert feedback["needs_proposer_coordinates"] == ["y"]
    assert [q["target"] for q in feedback["assignments"]] == ["y"]
    assert "states" not in feedback["assignments"][0]  # Runtime assigns no science.
    differential = ledger.Draft(variables=[variable("y")])
    result = prompts.binding_questions(
        brief(), differential, contract("target_feedback")
    )
    assert result["feedback_bindings"]["applicable_targets"] == ["y"]
    assert result["feedback_bindings"]["assignments"] == []
