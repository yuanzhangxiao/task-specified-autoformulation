"""Check the two choices in the revised function prompt and provider schema."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.llm.staged_topology import strict_provider_schema
from scripts import trajectory_function_revision_prompts as prompts

GRAPH = {
    "variable_types": {"height": "differential", "memory": "differential"},
    "sources": {"height": ["memory"], "memory": ["flow"]},
    "initializers": {
        "height": {"mode": "observed_initial", "channel": "initial_height"},
        "memory": {"mode": "map", "public_initials": ["height"]},
    },
}
MODEL = {
    "equations": {"height": "memory", "memory": "k*flow"},
    "initializers": {"memory": "k*height"},
    "parameters": {"k": 1.0},
}
MENU = {"evidence": {"split": "train"}, "candidates": [MODEL, deepcopy(MODEL)]}


def authored(route):
    return {
        "candidate": {
            **MODEL,
            "route": route,
            "parameters": [{"name": "k", "initial": 1.0}],
        }
    }


def test_distinct_explicit_routes():
    route, form, origin, reason = prompts.decode_reply(
        GRAPH,
        {"candidate": {"route": "fixed_graph", "shortlist_id": "B"}},
        MENU["candidates"],
    )
    assert (route, form, origin, reason) == ("fixed_graph", MODEL, "selected_B", None)
    a = authored("fixed_graph")
    assert prompts.decode_reply(GRAPH, a, MENU["candidates"], False)[1] == MODEL
    b = authored("request_topology_change")
    b["candidate"]["scientific_reason"] = " Runoff also fills this compartment. "
    b["candidate"]["equations"] = {"height": "memory+flow/area", "memory": "k*flow"}
    route, form, origin, reason = prompts.decode_reply(GRAPH, b, MENU["candidates"])
    assert route == "request_topology_change" and reason.startswith("Runoff")
    assert form["equations"]["height"] == "memory+flow/area" and origin == "authored"
    assert "route" not in form and "scientific_reason" not in form


@pytest.mark.parametrize(
    "fields",
    [
        {"route": "request_topology_change"},
        {"route": "request_topology_change", "scientific_reason": "   "},
        {"route": "fixed_graph", "scientific_reason": "change"},
        {"route": "fixed_graph", "new_sources": {}},
        {
            "route": "request_topology_change",
            "scientific_reason": "reason",
            "equations": {},
        },
        {"route": "change"},
    ],
)
def test_incomplete_or_ambiguous_route_rejected(fields):
    reply = authored("fixed_graph")
    reply["candidate"].update(fields)
    with pytest.raises(ValidationError):
        prompts.decode_reply(GRAPH, reply, MENU["candidates"])


def test_only_hybrid_can_select_and_changes_require_equations():
    select = {"candidate": {"route": "fixed_graph", "shortlist_id": "A"}}
    with pytest.raises(ValidationError):
        prompts.decode_reply(GRAPH, select, MENU["candidates"], False)
    select["candidate"].update(route="request_topology_change", scientific_reason="why")
    with pytest.raises(ValidationError):
        prompts.decode_reply(GRAPH, select, MENU["candidates"])
    duplicate = authored("fixed_graph")
    duplicate["candidate"]["parameters"] *= 2
    with pytest.raises(ValueError, match="each shared parameter"):
        prompts.decode_reply(GRAPH, duplicate, MENU["candidates"])


def test_prompt_names_public_sources_and_respects_the_arm():
    frozen = {
        "public_brief": {"scientific_context": "Model water storage."},
        "histories": ["flow"],
        "measured_initial_bindings": {"height": "initial_height"},
        "initials": ["initial_height", "boundary_only"],
        "evidence": {
            "context": {
                "time_symbol": "t",
                "targets": ["height"],
                "external_inputs": ["flow"],
                "auxiliaries": [],
                "fixed_covariates": ["area", "initial_height"],
            },
        },
    }
    for hybrid in (True, False):
        system, packet, schema = prompts.request(GRAPH, frozen, [], MENU, hybrid)
        assert set(packet["sources_available_to_add"]) == {
            "t",
            "area",
            "flow",
            "height",
            "memory",
        }
        assert "state0_NAME" in system and "time zero" in system
        assert packet["measured_initial_bindings"] == {"height": "initial_height"}
        assert ("shortlist" in packet) == ("shortlist_id" in system) == hybrid
        assert strict_provider_schema(schema.model_json_schema())["required"] == [
            "candidate"
        ]
