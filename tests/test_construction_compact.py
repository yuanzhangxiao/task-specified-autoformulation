"""Lossless payload references, diagnostic-only naming hints and cached repair."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.search import construction_compact as compact
from autoformalism.search import construction_deferred as deferred
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_schedules as schedules
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    patch,
    process,
    variable,
)
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


@pytest.mark.parametrize("local", [False, True])
def test_exact_deduplication_preserves_context_edits_errors_and_all_canonical_values(
    local,
):
    edits = {"equations": [equation("y", "x"), equation("x", "u")]}
    checklist = {"items": [{"name": "p", "status": "inconsistent"}]}
    receipt = {
        "transaction_applied": True,
        "edit_effects": edits,
        "repair_receipt": {"committed_edits": deepcopy(edits)},
        "shared_process_checklist": checklist,
    }
    errors = [{"code": "memory_driver_path", "driver": "u", "memory": "x"}]
    original = {
        "bookkeeping_policy": compact.POLICY,
        "public_brief": {"science": "Keep every scientific fact", "training": [1, 2]},
        "current_draft": {"declarations": {"variables": [variable("y")]}},
        "stage_instructions": "Short original wording",
        "shared_process_checklist": checklist,
        "last_edit_result": receipt,
        "runtime_diagnostics": deepcopy(receipt)
        if local
        else {
            "last_edit_result": deepcopy(receipt),
            "structural_failures": errors,
            "remaining_topology_failures": deepcopy(errors),
        },
    }
    before = deepcopy(original)
    result = compact.compact_payload(original)
    assert original == before
    assert result["payload_references"]["aliases"]
    assert compact.expand_payload(result) == original
    assert compact.compact_payload(result) == result
    for field in ("public_brief", "current_draft", "stage_instructions"):
        assert result[field] == original[field]


def test_nonidentical_feedback_and_empty_values_are_not_deduplicated():
    original = {
        "bookkeeping_policy": compact.POLICY,
        "runtime_diagnostics": {"last_edit_result": {"error": "older failure"}},
        "last_edit_result": {"error": "new failure"},
        "shared_process_checklist": {},
    }
    assert compact.compact_payload(original) == original


def test_name_feedback_distinguishes_existing_state_from_driver_and_changes_nothing():
    draft = apply(variables=[variable("y"), variable("x")])
    edit = patch(
        processes=[
            process(name="x", depends_on=["x"]),
            process(name="u", depends_on=["u"]),
            process(name="fresh", depends_on=["x"]),
        ]
    )
    before = draft.model_dump()
    rows = compact.naming_conflicts(brief(), draft, edit)
    assert [r["name"] for r in rows] == ["x", "u"]
    assert rows[0]["existing_role"] == "differential state"
    assert "external_input" in rows[1]["existing_role"]
    assert "P_X=phi(X)" in rows[0]["repair_options"][0]
    assert all(not r["runtime_renamed_anything"] for r in rows)
    assert draft.model_dump() == before
    # An explicitly coordinated change to algebraic is a scientific edit, not
    # an automatic collision fix. A new algebraic target is allowed too.
    edit = patch(variables=[variable("x", "algebraic")], processes=[process(name="x")])
    assert not compact.naming_conflicts(brief(), draft, edit)
    assert not compact.naming_conflicts(
        brief(), ledger.Draft(), patch(processes=[process(name="y")])
    )


@pytest.mark.parametrize("policy", [compact.POLICY, deferred.POLICY])
def test_rejected_state_name_gets_actionable_feedback_only_in_v2_and_resumes(
    tmp_path, policy
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            edit["variables"] = [variable("y"), variable("x")]
        elif p["stage"] == "shared_laws":
            retry = sum(c["stage"] == "shared_laws" for c in calls) > 1
            if retry:
                assert p["last_edit_result"]["transaction_applied"] is False
                if policy == compact.POLICY:
                    assert p["last_edit_result"]["name_conflicts"][0]["name"] == "x"
                    assert "runtime_diagnostics" not in p
                else:
                    assert "name_conflicts" not in p["last_edit_result"]
            edit["processes"] = [
                process(
                    name="release" if retry else "x",
                    depends_on=["x"],
                )
            ]
        elif p["stage"] == "equations":
            edit["equations"] = [equation("x", "u"), {"name": "y", "terms": []}]
        else:
            pytest.fail("No global repair needed after proposer fixes the name")
        return response(edit)

    def run(can_start=lambda: True):
        client = ConstructionClient(
            settings=StagedModelSettings(
                maximum_requests=128,
                maximum_total_tokens=524288,
            ),
            directory=tmp_path / "calls",
            namespace="compact",
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
            "joint_adaptive",
            prompt_family="minimal",
            process_question="dedicated",
            bookkeeping_policy=policy,
        )

    with pytest.raises(DeferredCall):
        run(lambda: len(calls) < 2)
    result = run()
    assert result["status"] == "topology_complete"
    assert len(calls) == 4
    assert run() == result
    assert len(calls) == 4
