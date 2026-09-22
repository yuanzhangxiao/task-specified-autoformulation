"""Delivery repair preserves scientific contracts, cached calls and reconstruction."""

import copy
import json

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.search import function_delivery as delivery
from autoformalism.search import identified_function_stage as stage
from autoformalism.search.staged_function_runner import run_staged_functions
from scripts import smoke_process_revisions as smoke


def response(value):
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
        ],
        "usage": {"total_tokens": 50},
    }


def transport(calls, *, repair_law=False, wrong_count=False):
    def respond(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        if "selected_slot" in payload:
            return response(
                {
                    "expression": "max(0,x-crest)",
                    "parameters": [],
                    "revise_dependencies": True,
                }
            )
        values = {"term_0_0": "q = x+u+crest", "term_1_0": "u", "term_2_0": "y"}
        functions = [
            {"interaction_id": n, "expression": values[n], "parameters": []}
            for n in payload["requested_slots"]
        ]
        if repair_law and "term_0_0" in payload["requested_slots"]:
            functions[0]["expression"] = "x/area"
        if wrong_count:
            functions.append(
                {"interaction_id": "term_9_0", "expression": "typo", "parameters": []}
            )
        return response({"functions": functions})

    return respond


def client(root, calls, **kwargs):
    c = smoke.client(root, calls, [{}])
    c.transport = transport(calls, **kwargs)
    return c


def test_keyed_delivery_never_guesses_order_or_duplicate_slot():
    def f(n, e):
        return {"interaction_id": n, "expression": e, "parameters": []}

    mapped, errors = delivery.unpack(
        {"functions": [f("term_0_1", "u"), f("term_0_0", "x"), f("term_9_0", "z")]},
        ["term_0_0", "term_0_1"],
    )
    assert mapped["term_0_0"]["expression"] == "x"
    assert mapped["term_0_1"]["expression"] == "u"
    assert len(errors) == 1
    mapped, errors = delivery.unpack(
        {"functions": [f("term_0_0", "x"), f("term_0_0", "u"), f("term_0_1", "y")]},
        ["term_0_0", "term_0_1"],
    )
    assert set(mapped) == {"term_0_1"}
    assert errors


@pytest.mark.parametrize(
    "expression",
    [
        "wrong = x",
        "q = r = x",
        "q = x; q = y",
        "q += x",
        "q = __import__('os')",
        "q = q*area",
        "q = (x if x else u)",
    ],
)
def test_label_normalization_does_not_approve_invalid_math(expression):
    brief, context, source = smoke.fixture()
    with pytest.raises((ValueError, ModelValidationError)):
        delivery.interpret(
            brief,
            context,
            source,
            {},
            "term_0_0",
            {"expression": expression, "parameters": []},
        )


def test_exact_label_and_extra_delivery_no_repair_or_overwrite(tmp_path):
    args = smoke.fixture()
    calls = []
    c = client(tmp_path, calls, wrong_count=True)
    result = stage.run(
        *args, c, tmp_path / "stage", public_brief=args[0].model_dump(mode="json")
    )
    assert result["complete"], result["error"]
    assert len(calls) == 3
    assert all("term_1_1" not in p["requested_slots"] for p in calls)
    assert result["functions"]["term_0_0"]["expression"] == "x + u + crest"
    assert stage.replay(*args, result)[1] == result["functions"]
    assert (
        stage.run(
            *args,
            client(tmp_path, calls, wrong_count=True),
            tmp_path / "stage",
            public_brief=args[0].model_dump(mode="json"),
        )
        == result
    )
    assert len(calls) == 3
    changed = copy.deepcopy(result)
    changed["functions"]["term_0_0"]["expression"] = "x"
    with pytest.raises(ValueError, match="differs"):
        stage.replay(*args, changed)


def test_local_conversion_repair_then_cached_resume(tmp_path):
    args, calls = smoke.fixture(), []
    c = client(tmp_path, calls, repair_law=True)
    result = stage.run(
        *args, c, tmp_path / "stage", public_brief=args[0].model_dump(mode="json")
    )
    assert result["complete"], result["error"]
    assert len(calls) == 4
    assert stage.replay(*args, result)[1] == result["functions"]
    assert any(e["kind"] == "repair" for e in result["ledger"])


def test_partial_delivery_retains_good_slot_without_resending(tmp_path):
    args, calls = smoke.fixture(), []
    base = transport(calls)

    def respond(url, body, timeout):
        result = base(url, body, timeout)
        if len(calls) == 1:
            return response({"functions": []})
        return result

    c = client(tmp_path, calls)
    c.transport = respond
    result = stage.run(
        *args, c, tmp_path / "stage", public_brief=args[0].model_dump(mode="json")
    )
    assert result["complete"]
    assert len(calls) == 4
    assert calls[0]["requested_slots"] == calls[1]["requested_slots"]


def test_deferred_resume_uses_same_calls(tmp_path):
    args, calls = smoke.fixture(), []
    c = client(tmp_path, calls)
    c.can_start = lambda: len(calls) < 1
    with pytest.raises(DeferredCall):
        stage.run(
            *args, c, tmp_path / "stage", public_brief=args[0].model_dump(mode="json")
        )
    result = stage.run(
        *args,
        client(tmp_path, calls),
        tmp_path / "stage",
        public_brief=args[0].model_dump(mode="json"),
    )
    assert result["complete"]
    assert len(calls) == 3


def test_complete_stage_uses_existing_finalization(tmp_path):
    args, calls = smoke.fixture(), []
    c = client(tmp_path, calls)
    base = c.transport

    def respond(url, body, timeout):
        if body["messages"][1]["content"].__contains__('"selected_state"'):
            return response({"initial": {"fixed_value": 0.0}})
        return base(url, body, timeout)

    c.transport = respond
    result = run_staged_functions(
        *args,
        c,
        tmp_path / "full",
        generation_granularity="equation_batch_atomic_repair",
        function_repair_policy="certified_outer_gain",
        dependency_policy="local-function-dependencies-1",
        assembly_policy="process-assembly-contract-1",
        function_delivery_policy=delivery.POLICY,
    )
    assert result["complete_model"], result["error"]


def test_topology_repair_can_update_previously_accepted_companion(tmp_path):
    brief, context, source = smoke.fixture()
    source = copy.deepcopy(source)
    source["equations"] = source["equations"][1:] + source["equations"][:1]
    topology, _ = stage.revision.bind_known(brief, context, source, {})
    source["topology"] = topology.model_dump(mode="json")
    calls = []

    def respond(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        if "requested_slots" in payload:
            functions = []
            for n, slot in payload["requested_slots"].items():
                expression = "y" if slot["lhs"] == "y" else "u"
                functions.append(
                    {"interaction_id": n, "expression": expression, "parameters": []}
                )
            return response({"functions": functions})
        value = {"expression": "u", "parameters": [], "revise_dependencies": True}
        if payload["scope"] == "topology":
            assert payload["retained_functions"]["term_1_0"]["expression"] == "y"
            value["companion_laws"] = [
                {
                    "interaction_id": "term_1_0",
                    "expression": "y+x",
                    "parameters": [],
                    "revise_dependencies": True,
                }
            ]
        return response(value)

    c = client(tmp_path, calls)
    c.transport = respond
    result = stage.run(
        brief,
        context,
        source,
        c,
        tmp_path / "stage",
        public_brief=brief.model_dump(mode="json"),
    )
    assert result["complete"], result["error"]
    assert result["functions"]["term_1_0"]["expression"] == "y+x"
    assert [p["scope"] for p in calls if "scope" in p] == ["local", "topology"]
    assert stage.replay(brief, context, source, result)[1] == result["functions"]


def test_no_new_budget_after_missing_batch_delivery(tmp_path):
    args, calls = smoke.fixture(), []
    c = client(tmp_path, calls)
    c.settings = c.settings.model_copy(update={"maximum_requests": 1})
    c.transport = lambda url, body, timeout: response({"functions": []})
    result = stage.run(
        *args, c, tmp_path / "stage", public_brief=args[0].model_dump(mode="json")
    )
    assert not result["complete"]
    assert "total provider request budget" in result["error"]
    assert len(c.records) == 1


def test_no_matching_label_escape_for_self_reference(tmp_path):
    brief, context, source = smoke.fixture()
    with pytest.raises((ValueError, ModelValidationError)):
        delivery.interpret(
            brief,
            context,
            source,
            {},
            "term_0_0",
            {"expression": "q = q*x", "parameters": [], "revise_dependencies": True},
        )
