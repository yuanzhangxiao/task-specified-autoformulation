"""Equation usage, advisory roles and a targeted agenda without self-review."""

import copy
import json

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import variable_confirmation as campaign
from autoformalism.schemas.staged_topology import ModelingLimits, ScientificVariable
from autoformalism.search import variable_equation_usage as usage
from autoformalism.search.staged_topology_runner import (
    _validate_memory_equation_obligations,
    run_staged_topology,
)
from autoformalism.staged_topology import public_structure_checks, validate_equation
from tests.test_explicit_variable_bindings import reply, response, variable
from tests.test_phase_c_construction_baseline import (
    construction_transport,
    fixture,
    tokenize,
)
from tests.test_staged_topology_contract import equation
from tests.test_variable_inventory_review import public


def run(root, transport, *, can_start=None, attempts=3, clarify=True, **kwargs):
    client = StagedTopologyClient(
        settings=StagedModelSettings(attempts_per_step=attempts),
        directory=root / "calls",
        namespace="usage",
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
        root / "variables",
        hybrid_variable_construction=True,
        explicit_mechanism_bindings=True,
        stop_after_inventory=True,
        clarify_equation_usage=clarify,
        **kwargs,
    )


def test_missing_choice_repaired_with_full_context_without_self_review(tmp_path):
    calls = []

    def transport(url, body, timeout):
        p = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(p)
        assert "NEITHER LHS NOR RHS" in body["messages"][0]["content"]
        assert "retains its first" not in body["messages"][0]["content"]
        assert p["mechanism_binding_context"]["allowed_requirement_ids"] == ["memory"]
        if len(calls) == 1:
            return response(
                reply(["z"], [variable("x"), variable("z")]).model_dump(mode="json")
            )
        assert p["public_brief"] == calls[0]["public_brief"]
        assert p["runtime_diagnostics"]["unresolved_obligations"] == [
            "observation_choice:measured"
        ]
        assert p["runtime_diagnostics"]["rejected_response"]["mechanism_bindings"]
        assert {v["name"] for v in p["current_inventory"]} == {"u", "x", "z"}
        # Wording is accepted without a semantic judgment or structural change.
        x = {**variable("x"), "scientific_role": "Revised description using delayed z"}
        return response(
            {
                "variables": [variable("measured", "supplied"), x],
                "mechanism_bindings": [],
            }
        )

    result = run(tmp_path, transport)
    assert result["status"] == "variables_complete", result
    assert len(calls) == 2 and "inventory_review" not in result
    assert next(v for v in result["inventory"] if v["name"] == "x")[
        "scientific_role"
    ].startswith("Revised")
    assert result["memory_candidates"] == {"memory": ["z"]}
    assert run(tmp_path, transport) == result and len(calls) == 2
    with pytest.raises(ValueError, match="evidence contract differs"):
        run(tmp_path, transport, clarify=False)


@pytest.mark.parametrize(
    "definition", ["supplied", "unused", "differential", "algebraic"]
)
def test_channel_choice_is_proposer_owned_and_no_semantic_prose_gate(
    tmp_path, definition
):
    def transport(*_):
        raw = reply(
            ["z"], [variable("x"), variable("z"), variable("measured", definition)]
        ).model_dump(mode="json")
        raw["variables"][0]["scientific_role"] = (
            "Possibly stale description of direct u"
        )
        return response(raw)

    result = run(tmp_path, transport)
    assert result["status"] == "variables_complete"
    assert (
        next(v for v in result["inventory"] if v["name"] == "measured")["definition"]
        == definition
    )
    assert not list((tmp_path / "calls").glob("*review*"))


def test_exhausted_missing_choice_does_not_become_completion(tmp_path):
    result = run(
        tmp_path,
        lambda *_: response(
            reply(["z"], [variable("x"), variable("z")]).model_dump(mode="json")
        ),
        attempts=1,
    )
    assert result["status"] == "failed"
    assert "observation_choice:measured" in result["error"]


def test_drain_resumes_existing_calls_without_paid_repeat(tmp_path):
    calls = []

    def transport(*_):
        calls.append(1)
        variables = [variable("x"), variable("z")]
        if len(calls) == 2:
            variables.append(variable("measured", "unused"))
        return response(reply(["z"], variables).model_dump(mode="json"))

    with pytest.raises(DeferredCall):
        run(tmp_path, transport, can_start=lambda: not calls)
    assert run(tmp_path, transport)["status"] == "variables_complete"
    assert len(calls) == 2


def test_no_memory_binding_list_and_incompatible_modes(tmp_path):
    context = usage.binding_context(
        public().model_copy(update={"requirements": ()}), {}, {}
    )
    assert context["allowed_requirement_ids"] == []
    assert "mechanism_bindings=[]" in context["binding_instruction"]
    with pytest.raises(ValueError, match="without self-review"):
        run(tmp_path, lambda *_: None, review_inventory=True)


def test_supplied_can_be_rhs_but_unused_cannot_and_wording_is_not_a_graph_test():
    inventory = tuple(
        ScientificVariable.model_validate(variable(n, d))
        for n, d in (
            ("u", "supplied"),
            ("measured", "supplied"),
            ("z", "differential"),
            ("x", "algebraic"),
        )
    )
    equations = (equation("z", ("u",)), equation("x", ("z", "measured"), "algebraic"))
    validate_equation(inventory, equations[:1], equations[1], ModelingLimits())
    assert all(c["passed"] for c in public_structure_checks(public(), equations))
    disconnected = (equations[0], equation("x", ("u", "measured"), "algebraic"))
    _validate_memory_equation_obligations(
        public(), inventory[-1], equations[:1], equations[1], {"memory": {"z"}}
    )
    with pytest.raises(ValueError, match="path through the selected"):
        _validate_memory_equation_obligations(
            public(),
            inventory[-1],
            disconnected[:1],
            disconnected[1],
            {"memory": {"z"}},
        )
    unused = tuple(
        v.model_copy(update={"definition": "unused"}) if v.name == "measured" else v
        for v in inventory
    )
    with pytest.raises(ValueError, match="unused sources"):
        validate_equation(unused, equations[:1], equations[1], ModelingLimits())
    with pytest.raises(ValueError, match="LHS"):
        validate_equation(inventory, (), equation("measured", ("u",)), ModelingLimits())
    reported = usage.usage_table([v.model_dump(mode="json") for v in inventory])
    measured = next(v for v in reported if v["name"] == "measured")
    assert measured["rhs_allowed"] and not measured["generated_lhs_required"]
    assert measured["actual_rhs_occurrences"] is None


def targeted_source(root):
    """Use synthetic briefs under the roster names, never real benchmark data."""
    source = fixture(root / "original")
    cell = next(iter(source["cells"].values()))
    source = {
        **source,
        "cells": {name: copy.deepcopy(cell) for name in campaign.TARGETED_CASES},
        "tasks": [
            {
                **source["tasks"][0],
                "benchmark_id": name,
                "task_id": f"case{i}_seed{seed}_{arm}",
                "arm": arm,
                "seed": seed,
            }
            for i, name in enumerate(campaign.TARGETED_CASES)
            for seed in (0, 1)
            for arm in ("full", "brief_only")
        ],
    }
    source.pop("artifact_sha256")
    return sealed_write(root / "source.json", source)


def test_targeted_plan_run_report_and_resume(tmp_path):
    source = targeted_source(tmp_path)
    root = tmp_path / "new"
    plan = campaign.freeze(tmp_path / "source.json", root, targeted_usage=True)
    assert plan == campaign.verify(root)
    assert len(plan["tasks"]) == 12 and len(plan["cells"]) == 3
    assert (
        not plan["config"]["review_inventory"]
        and plan["equation_usage_policy"] == usage.POLICY
    )
    assert not any(
        k in c
        for c in plan["cells"].values()
        for k in ("training", "validation", "independent_rules", "mechanism_spec")
    )
    calls = []
    transport = construction_transport(calls)
    for task in (plan["tasks"][0], plan["tasks"][1]):
        kwargs = {"transport": transport, "token_transport": tokenize}
        result = campaign.propose(root, plan, task, "http://offline", **kwargs)
        assert result["status"] == "variables_complete"
        assert campaign.propose(root, plan, task, "http://offline", **kwargs) == result
    assert len(calls) == 2
    for i, payload in enumerate(calls):
        assert ("training_observations" in payload["public_brief"]) == (i == 0)
    report = campaign.report(root, plan)
    assert report["status_counts"] == {"variables_complete": 2, "pending": 10}
    assert report["inventory_review_calls"] == 0 and not report["equations_available"]
    assert report["rows"][0]["equation_usage"]
    assert sealed_read(tmp_path / "source.json") == source
    with pytest.raises(ValueError, match="no final self-review"):
        campaign.freeze(
            tmp_path / "source.json", root, targeted_usage=True, review_inventory=True
        )
    with pytest.raises(ValueError):
        campaign.freeze(tmp_path / "source.json", root, targeted_usage=False)


def test_partial_roster_is_rejected_before_submission(tmp_path):
    fixture(tmp_path / "old")
    with pytest.raises(ValueError, match="all 12"):
        campaign.freeze(
            tmp_path / "old/plan.json", tmp_path / "new", targeted_usage=True
        )
