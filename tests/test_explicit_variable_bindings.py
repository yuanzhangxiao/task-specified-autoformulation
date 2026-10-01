"""Memory is an explicit scientific decision, including reuse during local repair."""

import json

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import variable_bindings as b
from autoformalism.search.staged_topology_runner import run_staged_topology


def brief():
    return PublicScientificBrief.model_validate(
        {
            "scientific_context": "x is the measured output; u drives delayed action.",
            "public_variables": [
                {"name": "x", "data_role": "target"},
                {"name": "u", "data_role": "external_input"},
            ],
            "requirements": [
                {
                    "id": "memory",
                    "public_requirement": "Delayed action",
                    "targets": ["x"],
                    "drivers": ["u"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )


def variable(name, definition="differential"):
    return {
        "name": name,
        "definition": definition,
        "scientific_role": f"role of {name}",
    }


def reply(states, variables=()):
    return b.BoundVariableReply.model_validate(
        {
            "variables": variables,
            "mechanism_bindings": [
                {"requirement_id": "memory", "memory_states": states}
            ],
        }
    )


def response(payload):
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 100},
    }


def test_empty_reply_binds_existing_state_and_replaces_previous_choice():
    inventory = tuple(
        ScientificVariable.model_validate(variable(n)) for n in ("x", "z", "q")
    )
    bindings = {"memory": {"z"}}
    updated, decisions, errors = b.merge(brief(), inventory, reply(["q"]), bindings, {})
    assert updated == inventory and not decisions and not errors
    assert bindings == {"memory": {"q"}}


@pytest.mark.parametrize(
    "states,definition",
    [
        (["missing"], "differential"),
        (["z"], "algebraic"),
        (["x"], "differential"),
        (["u"], "differential"),
    ],
)
def test_bad_binding_preserves_previous_assignment(states, definition):
    inventory = tuple(
        ScientificVariable.model_validate(variable(n, definition))
        for n in ("x", "u", "z")
    )
    bindings = {"memory": {"old"}}
    _, _, errors = b.merge(brief(), inventory, reply(states), bindings, {})
    assert errors and bindings == {"memory": {"old"}}


def test_unknown_requirement_and_duplicate_assignments_rejected():
    raw = reply(["z"], [variable("z")]).model_dump(mode="json")
    raw["mechanism_bindings"][0]["requirement_id"] = "invented"
    bindings = {}
    inventory, _, errors = b.merge(
        brief(), (), b.BoundVariableReply.model_validate(raw), bindings, {}
    )
    assert inventory[0].name == "z" and errors and not bindings
    raw["mechanism_bindings"] *= 2
    with pytest.raises(ValueError, match="one binding"):
        b.BoundVariableReply.model_validate(raw)
    with pytest.raises(ValueError, match="unique"):
        reply(["z", "z"])


def test_fixed_target_role_rejected_before_it_can_be_committed():
    bindings = {}
    inventory, decisions, errors = b.merge(
        brief(),
        (),
        reply(["z"], [variable("x"), variable("z")]),
        bindings,
        {"x": "algebraic"},
    )
    assert {v.name for v in inventory} == {"z"}
    assert not decisions[0]["accepted"] and not errors
    assert bindings == {"memory": {"z"}}


def test_real_variable_repair_reuses_context_and_never_picks_first_state(tmp_path):
    calls = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        assert body["response_format"]["json_schema"]["name"] == "BoundVariableReply"
        if len(calls) == 1:
            # All variables are valid. Runtime must not assign z just because first.
            return response(
                {
                    "variables": [variable(n) for n in ("z", "q", "x")],
                    "mechanism_bindings": [],
                }
            )
        assert payload["public_brief"] == calls[0]["public_brief"]
        assert {v["name"] for v in payload["current_inventory"]} >= {"z", "q", "x"}
        assert payload["mechanism_binding_context"]["current_bindings"] == {}
        assert "runtime_diagnostics" in payload
        return response(reply(["q"]).model_dump(mode="json"))

    def run():
        client = StagedTopologyClient(
            settings=StagedModelSettings(),
            directory=tmp_path / "calls",
            namespace="new",
            seed=0,
            base_url="http://offline",
            transport=transport,
        )
        return run_staged_topology(
            brief(),
            ValidationContext(targets=("x",), external_inputs=("u",)),
            client,
            tmp_path / "variables",
            hybrid_variable_construction=True,
            explicit_mechanism_bindings=True,
            stop_after_inventory=True,
        )

    result = run()
    assert result["status"] == "variables_complete"
    assert result["memory_candidates"] == {"memory": ["q"]}
    assert len(calls) == 2
    assert run() == result and len(calls) == 2
    saved = json.loads((tmp_path / "variables/progress.json").read_text())
    assert saved["events"][0]["partial_acceptance"]
    assert saved["events"][0]["bindings_after"] == {}
    assert saved["events"][1]["explicit_bindings"][0]["memory_states"] == ["q"]


def test_process_repair_repeats_scientific_context_and_all_consumers(tmp_path):
    from tests.test_signed_processes import example

    public, inventory, process = example()
    calls = []

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        if "selected_lhs" not in payload:
            return response({"processes": [process]})
        calls.append(payload)
        name = payload["selected_lhs"]["name"]
        assert payload["selected_lhs"]["scientific_role"] == name
        if name == "outlet":
            source = "x" if "runtime_diagnostics" in payload else "q"
            return response(
                {
                    "terms": [
                        {
                            "sources": [source],
                            "outer_weight_sign": "positive",
                            "scientific_role": "outlet",
                        }
                    ],
                    "inventory_revision": None,
                }
            )
        return response({"terms": [], "inventory_revision": None})

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        directory=tmp_path / "calls",
        namespace="process-context",
        seed=0,
        base_url="http://offline",
        transport=transport,
    )
    result = run_staged_topology(
        public,
        ValidationContext(targets=("y",), fixed_covariates=("area",)),
        client,
        tmp_path / "topology",
        initial_inventory=inventory,
        hybrid_variable_construction=True,
        optional_process_review=True,
        bind_shared_processes=True,
        signed_shared_processes=True,
        complete_process_context=True,
    )
    assert result["complete_topology"], result
    outlet = [c for c in calls if c["selected_lhs"]["name"] == "outlet"]
    assert len(outlet) == 2
    for key in (
        "public_brief",
        "frozen_inventory",
        "current_equation_sketch",
        "committed_process_bindings",
    ):
        assert outlet[0][key] == outlet[1][key]
    bindings = outlet[1]["committed_process_bindings"]
    assert bindings[0]["signed_declaration"]["uses"] == process["uses"]
    assert "runtime_diagnostics" in outlet[1]


def test_no_memory_task_requests_an_empty_binding_list_consistently(tmp_path):
    public = brief().model_copy(update={"requirements": ()})

    def transport(url, body, timeout):
        assert body["response_format"]["json_schema"]["name"] == "BoundVariableReply"
        return response({"variables": [variable("x")], "mechanism_bindings": []})

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        directory=tmp_path / "calls",
        namespace="no-memory",
        seed=0,
        base_url="http://offline",
        transport=transport,
    )
    result = run_staged_topology(
        public,
        ValidationContext(targets=("x",), external_inputs=("u",)),
        client,
        tmp_path / "variables",
        hybrid_variable_construction=True,
        explicit_mechanism_bindings=True,
        stop_after_inventory=True,
    )
    assert result["status"] == "variables_complete"
    assert result["memory_candidates"] == {}
