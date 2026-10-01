"""Observation choices, scientific ownership and an atomic review before freezing."""

import json

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
)
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import variable_inventory_review as review
from autoformalism.search.staged_topology_runner import run_staged_topology
from tests.test_explicit_variable_bindings import brief, reply, response, variable


def public():
    raw = brief().model_dump(mode="json")
    raw["public_variables"].append({"name": "measured", "data_role": "auxiliary"})
    return PublicScientificBrief.model_validate(raw)


@pytest.mark.parametrize("choice", ["supplied", "unused", "differential", "algebraic"])
def test_explicit_observation_choices_and_alternative_state_coordinates(choice):
    # An algebraic output is permitted with a distinct differential memory.
    candidate, bindings = review.replace_inventory(
        public(),
        reply(
            ["z"],
            [variable("x", "algebraic"), variable("z"), variable("measured", choice)],
        ),
        {},
    )
    assert {v.name: v.definition for v in candidate} == {
        "u": "supplied",
        "x": "algebraic",
        "z": "differential",
        "measured": choice,
    }
    assert bindings == {"memory": {"z"}}


@pytest.mark.parametrize(
    "variables,states,targets,error",
    [
        ([variable("x"), variable("z")], ["z"], {}, "observation decisions missing"),
        (
            [variable("x", "supplied"), variable("z"), variable("measured", "unused")],
            ["z"],
            {},
            "target x",
        ),
        (
            [
                variable("x"),
                variable("z", "algebraic"),
                variable("measured", "supplied"),
            ],
            ["z"],
            {},
            "differential variables",
        ),
        (
            [variable("x"), variable("measured", "supplied")],
            ["x"],
            {},
            "drivers/targets",
        ),
        (
            [variable("x"), variable("z"), variable("measured", "unused")],
            ["z"],
            {"x": "algebraic"},
            "public target definitions",
        ),
        (
            [
                variable("x"),
                variable("z"),
                variable("measured", "unused"),
                variable("secret", "supplied"),
            ],
            ["z"],
            {},
            "internal secret",
        ),
        (
            [
                variable("x"),
                variable("z"),
                variable("measured", "unused"),
                variable("exp"),
            ],
            ["z"],
            {},
            "reserved variable",
        ),
        (
            [
                variable("x"),
                variable("z"),
                variable("measured", "unused"),
                variable("u", "unused"),
            ],
            ["z"],
            {},
            "required mechanism drivers",
        ),
    ],
)
def test_invalid_complete_decisions_rejected(variables, states, targets, error):
    with pytest.raises(ValueError, match=error):
        review.replace_inventory(public(), reply(states, variables), targets)


def test_bindings_cannot_be_silently_inherited():
    value = reply(["z"], [variable("x"), variable("z"), variable("measured", "unused")])
    value = value.model_copy(update={"mechanism_bindings": ()})
    with pytest.raises(ValueError, match=r"missing=.*memory"):
        review.replace_inventory(public(), value, {})


def run(tmp_path, transport, *, settings=None, can_start=None, enabled=True):
    client = StagedTopologyClient(
        settings=settings or StagedModelSettings(),
        directory=tmp_path / "calls",
        namespace="review-test",
        seed=0,
        base_url="http://offline",
        transport=transport,
        can_start=can_start or (lambda: True),
    )
    return run_staged_topology(
        public(),
        ValidationContext(
            targets=("x",), external_inputs=("u",), auxiliaries=("measured",)
        ),
        client,
        tmp_path / "variables",
        hybrid_variable_construction=True,
        explicit_mechanism_bindings=True,
        review_inventory=enabled,
        stop_after_inventory=True,
    )


def test_review_repeats_full_context_retypes_rebinds_and_resumes(tmp_path):
    calls = []

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        p = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        calls.append(p)
        if len(calls) == 1:
            assert p["observation_choices"][1]["current_definition"] == "not_declared"
            assert "does NOT imply" in body["messages"][0]["content"]
            return response(
                reply(["z"], [variable("x", "algebraic"), variable("z")]).model_dump(
                    mode="json"
                )
            )
        assert p["public_brief"] == calls[0]["public_brief"]
        assert {v["name"]: v["definition"] for v in p["current_inventory"]} == {
            "u": "supplied",
            "x": "algebraic",
            "z": "differential",
        }
        assert p["mechanism_binding_context"]["current_bindings"] == {"memory": ["z"]}
        assert "omitted bindings are retained" not in json.dumps(p)
        if len(calls) == 2:
            # Missing measured-channel decision; nothing is committed.
            return response(
                reply(
                    ["new_memory"], [variable("x"), variable("new_memory")]
                ).model_dump(mode="json")
            )
        assert "observation decisions missing" in p["runtime_diagnostics"]["error"]
        return response(
            reply(
                ["new_memory"],
                [
                    variable("x"),
                    variable("new_memory"),
                    variable("measured", "supplied"),
                ],
            ).model_dump(mode="json")
        )

    result = run(tmp_path, transport)
    assert result["status"] == "variables_complete"
    assert result["memory_candidates"] == {"memory": ["new_memory"]}
    assert {v["name"] for v in result["inventory"]} == {
        "x",
        "u",
        "new_memory",
        "measured",
    }
    decision = result["inventory_review"]
    assert decision["status"] == "accepted"
    assert {v["name"] for v in decision["changes"]} == {
        "x",
        "z",
        "new_memory",
        "measured",
    }
    assert decision["scientific_adequacy"] == "not_assessed"
    assert run(tmp_path, transport) == result and len(calls) == 3
    with pytest.raises(ValueError, match="evidence contract differs"):
        run(tmp_path, transport, enabled=False)


def test_failed_review_keeps_draft_and_cannot_complete(tmp_path):
    def transport(*_):
        return response(
            reply(["z"], [variable("x"), variable("z")]).model_dump(mode="json")
        )

    result = run(tmp_path, transport, settings=StagedModelSettings(attempts_per_step=1))
    assert result["status"] == "failed"
    assert result["inventory_review"]["status"] == "failed"
    assert result["memory_candidates"] == {"memory": ["z"]}
    assert {v["name"] for v in result["inventory"]} == {"u", "x", "z"}


def test_interrupted_review_resumes_without_repeating_paid_agenda_call(tmp_path):
    calls = []

    def transport(*_):
        calls.append(1)
        variables = [variable("x"), variable("z")]
        if len(calls) == 2:
            variables.append(variable("measured", "unused"))
        return response(reply(["z"], variables).model_dump(mode="json"))

    with pytest.raises(DeferredCall):
        run(tmp_path, transport, can_start=lambda: not calls)
    assert len(calls) == 1
    assert run(tmp_path, transport)["status"] == "variables_complete"
    assert len(calls) == 2


def test_review_shares_request_budget_and_does_not_reset_it(tmp_path):
    calls = []

    def transport(*_):
        calls.append(1)
        return response(
            reply(["z"], [variable("x"), variable("z")]).model_dump(mode="json")
        )

    settings = StagedModelSettings(maximum_requests=1)
    result = run(tmp_path, transport, settings=settings)
    assert result["status"] == "failed"
    assert "provider request budget exhausted" in result["error"]
    assert run(tmp_path, transport, settings=settings) == result
    assert len(calls) == 1
