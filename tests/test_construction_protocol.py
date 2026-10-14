"""Protocol names, minimal terminology and bounded substantive repair integration."""

import json
from copy import deepcopy

import pytest

from autoformalism.research.construction_comparison import Config
from autoformalism.search import construction_ownership_feedback as ownership
from autoformalism.search import construction_protocol as protocol
from scripts import jetstream_construction_campaign as pilot
from tests.test_construction_focus import execute
from tests.test_construction_ledger import apply, brief, context, equation, variable
from tests.test_explicit_variable_bindings import response
from tests.test_jetstream_construction_campaign import fixture


def packet(draft, stage="repair", last=None, public=None):
    b = public or brief(True)
    return protocol.payload(
        stage, b, context(), {}, b.model_dump(mode="json"), draft, None, last
    )


def test_terms_bindings_and_control_feedback_preserve_public_content():
    b = brief(True)
    d = apply(variables=[variable("y"), variable("m")])
    p = packet(d, "variables", public=b)
    assert p["public_brief"] == b.model_dump(mode="json")
    assert "algebraic readout" in protocol.SYSTEM
    assert "memory_state_assignments" in p
    assert p["memory_binding_options"] == [
        {"requirement_id": "memory", "choice": "required"}
    ]
    assert "memory_assignment_rules" not in p
    assert all(
        "memory_assignment_required" not in r for r in p["requirements_checklist"]
    )
    d = apply(
        d,
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["m"]}],
        equations=[equation("m", "u"), equation("y", "m")],
    )
    assert "memory_state_assignments" not in packet(d, "shared_laws")
    broken = apply(d, equations=[equation("stage_complete", "u")])
    last = {
        "error": "unknown variables",
        "rejected_reply": {"remove_variables": ["stage_complete"]},
    }
    p = packet(broken, last=last)
    assert any(
        r["code"] == "reply_control_is_not_quantity" for r in p["feedback"]["issues"]
    )
    assert "remove_equations" in json.dumps(p["feedback"])
    assert "reply_control_is_not_quantity" not in json.dumps(
        ownership.payload(
            "repair", b, context(), {}, b.model_dump(mode="json"), broken, None, last
        )
    )


def test_signature_ignores_prose_order_and_completion_flags_but_not_math():
    d = apply(
        variables=[variable("y"), variable("m")], equations=[equation("y", "u", "m")]
    )
    raw = {"remove_variables": ["unknown"], "stage_complete": False}
    old = protocol.repair_signature(d, {"rejected_reply": raw}, [])
    changed = d.model_copy(update={"variables": tuple(reversed(d.variables))})
    raw2 = {**raw, "stage_complete": True, "scientific_role": "new explanation"}
    assert protocol.repair_signature(changed, {"rejected_reply": raw2}, []) == old
    assert (
        protocol.repair_signature(
            d, {"rejected_reply": {"remove_equations": ["unknown"]}}, []
        )
        != old
    )
    assert protocol.substantive([1, 1]) != protocol.substantive([1])


@pytest.mark.parametrize(
    "mode,expected", [("prose", 3), ("structural", 6), ("rejected_alternatives", 6)]
)
def test_useful_repairs_and_cached_resume(tmp_path, mode, expected):
    calls, repairs = [], []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(body)
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            edit.update(
                variables=[variable("y"), variable("m")],
                mechanism_bindings=[
                    {"requirement_id": "memory", "memory_states": ["m"]}
                ],
            )
        elif p["stage"] == "equations":
            edit["equations"] = [equation("m", "u"), equation("y", "u")]
        elif p["stage"] == "repair":
            repairs.append(p)
            n = len(repairs)
            assert p["feedback"]["repair_budget"]["remaining_after_this_call"] == 10 - n
            assert "three consecutive" in p["feedback"]["repair_budget"]["early_stop"]
            if mode == "prose":
                edit["variables"] = [
                    {**variable("m"), "scientific_role": f"explanation {n}"}
                ]
            elif n == 6:
                edit["equations"] = [equation("y", "m")]
            elif mode == "structural":
                edit["variables"] = [variable(f"z{n}")]
                edit["equations"] = [equation(f"z{n}", "u")]
            else:
                edit["remove_variables"] = [f"unknown{n}"]
        return response(edit)

    result = execute(tmp_path, send, policy=protocol.POLICY)
    assert len(repairs) == expected
    assert result["status"] == (
        "topology_incomplete" if mode == "prose" else "topology_complete"
    )
    if mode == "prose":
        assert "semantically repeated" in result["stop_reason"]
    else:
        assert [c["repair_calls"] for c in result["repair_checkpoints"]] == [5]
        assert not result["repair_checkpoints"][0]["assessment"]["eligible"]
    n = len(calls)
    assert execute(tmp_path, send, policy=protocol.POLICY) == result
    assert len(calls) == n


def test_new_roster_budget_and_preview_agree_without_repurposing_history(tmp_path):
    args = fixture(tmp_path)
    plan = pilot.prepare(*args, study="protocol_clarity_confirmation")
    assert len(plan["tasks"]) == 40
    assert all(
        t["task_id"].endswith("_minimal_protocol_clarity") for t in plan["tasks"]
    )
    c = plan["config"]
    assert c["model_settings"]["maximum_requests"] - c["repair_requests"] == 123
    assert c["model_settings"]["maximum_total_tokens"] - c["repair_tokens"] == 393216
    assert c["repair_requests"] == 10
    assert pilot.verify(args[-1]) == plan
    with pytest.raises(ValueError, match="frozen artifact differs"):
        pilot.prepare(*args, study="ownership_feedback_confirmation")
    bad = deepcopy(c)
    bad["repair_requests"] = 9
    with pytest.raises(ValueError, match="repair allowance"):
        pilot.validate_settings(bad)
    bad["repair_requests"] = 5
    bad["bookkeeping_policy"] = ownership.POLICY
    with pytest.raises(ValueError, match="extended repair"):
        Config.model_validate(bad)
