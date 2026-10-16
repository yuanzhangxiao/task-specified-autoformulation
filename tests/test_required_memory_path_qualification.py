"""Required delayed effects cannot be credited solely through their own driver."""

from copy import deepcopy

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import mechanism_realizations as core
from tests.test_construction_ledger import equation, process, variable


def public(*, required=True, drivers=("I",), targets=("y",)):
    """Two public targets and optional observed channels with explicit history."""
    return PublicScientificBrief.model_validate(
        {
            "scientific_context": "Represent the stated delayed history effect.",
            "public_variables": [
                {"name": "y", "data_role": "target"},
                {"name": "I", "data_role": "target"},
                {"name": "a", "data_role": "auxiliary"},
                {"name": "b", "data_role": "auxiliary"},
                {"name": "u", "data_role": "external_input"},
            ],
            "requirements": [
                {
                    "id": "history_effect",
                    "public_requirement": "Delayed action through listed drivers.",
                    "drivers": list(drivers),
                    "targets": list(targets),
                    "requires_dynamic_memory": required,
                }
            ],
        }
    )


def draft(*, variables=None, equations=None, processes=(), states=("X",), b=None):
    """Default example exposes X only by regenerating current I."""
    return core.apply_patch(
        b or public(),
        core.Draft(),
        core.DraftPatch.model_validate(
            {
                "variables": variables
                if variables is not None
                else [variable("y"), variable("I", "algebraic"), variable("X")],
                "equations": equations
                if equations is not None
                else [equation("y", "I"), equation("I", "X"), equation("X", "I", "u")],
                "processes": list(processes),
                "mechanism_realizations": [
                    {
                        "requirement_id": "history_effect",
                        "components": list(states),
                        "memory_states": list(states),
                    }
                ],
                "stage_complete": True,
            }
        ),
    )


def assess(d, b=None):
    b = b or public()
    return core.assess(
        b,
        ValidationContext(
            targets=("y", "I"), auxiliaries=("a", "b"), external_inputs=("u",)
        ),
        {},
        d,
    )


def qualified(d, b=None):
    return core.qualify_required_memory_paths(b or public(), d, assess(d, b))


def test_cyclic_return_through_current_driver_is_separate_actionable_failure():
    d = draft()
    previous = assess(d)
    saved = deepcopy(previous)
    assert previous["eligible"]
    assert previous["memory_binding_checks"][0]["status"] == "passed"
    result = core.qualify_required_memory_paths(public(), d, previous)
    assert previous == saved  # Historical assessment is never rewritten in place.
    assert "required_memory_path_qualification" not in previous
    assert not result["eligible"]
    issue = next(
        e
        for e in result["errors"]
        if e["code"] == "required_memory_path_revisits_driver"
    )
    assert issue["target"] == "y"
    assert issue["drivers"] == ["I"]
    assert issue["memories"] == ["X"]
    assert issue["memory_witnesses"] == [
        {
            "memory": "X",
            "ordinary_path": ["X", "I", "y"],
            "qualified_path": None,
            "intermediate_current_drivers": ["I"],
        }
    ]
    assert "does not identify a separate history effect" in issue["action"]
    assert "Preserve valid driver generation and feedback" in issue["repair_options"][2]
    # The old route exists. The extra predicate qualifies its scientific use.
    assert result["memory_binding_checks"] == previous["memory_binding_checks"]
    assert result["public_structure_checks"] == previous["public_structure_checks"]


def test_shared_process_witness_preserves_the_actual_disposal_route():
    d = draft(
        equations=[
            {"name": "y", "terms": []},
            equation("I", "X"),
            equation("X", "I", "u"),
        ],
        processes=[
            process(
                name="disposal",
                depends_on=["I"],
                kind="influence",
                uses=[{"target": "y", "sign": "negative", "conversion": None}],
            )
        ],
    )
    issue = next(
        e
        for e in qualified(d)["errors"]
        if e["code"] == "required_memory_path_revisits_driver"
    )
    assert issue["memory_witnesses"][0]["ordinary_path"] == ["X", "I", "disposal", "y"]


@pytest.mark.parametrize("readout", ["direct", "shared", "coupled"])
def test_actual_history_route_passes_with_driver_feedback_and_instantaneous_path(
    readout,
):
    variables = [variable("y"), variable("I", "algebraic"), variable("X")]
    equations = [equation("I", "X"), equation("X", "I", "u")]
    processes = []
    if readout == "direct":
        equations.append(equation("y", "I", "X"))
    elif readout == "shared":
        equations.append(equation("y", "I"))
        processes.append(
            process(
                name="history_action",
                depends_on=["X"],
                kind="influence",
                uses=[{"target": "y", "sign": "negative", "conversion": None}],
            )
        )
    else:
        variables.append(variable("Z"))
        equations = [
            equation("I", "X"),
            equation("X", "I", "Z", "u"),
            equation("Z", "X"),
            equation("y", "I", "Z"),
        ]
    result = qualified(
        draft(variables=variables, equations=equations, processes=processes)
    )
    assert result["eligible"], result["errors"]
    row = result["required_memory_path_qualification"]["checks"][0]
    assert row["status"] == "passed"
    path = row["target_checks"][0]["memory_witnesses"][0]["qualified_path"]
    assert path[0] == "X" and path[-1] == "y"
    assert "I" not in path[1:-1]


def test_delayed_self_feedback_can_end_at_its_own_driver():
    b = public(drivers=("I",), targets=("I",))
    d = draft(b=b)
    result = qualified(d, b)
    assert result["eligible"], result["errors"]
    row = result["required_memory_path_qualification"]["checks"][0]
    assert row["target_checks"][0]["memory_witnesses"][0]["qualified_path"] == [
        "X",
        "I",
    ]


def test_two_drivers_and_distinct_target_branches_keep_collective_memory_quantifier():
    b = public(drivers=("a", "b"), targets=("y", "I"))
    d = draft(
        b=b,
        states=("M1", "M2"),
        variables=[variable(n) for n in ("y", "I", "a", "b", "M1", "M2")],
        equations=[
            equation("a", "M1", "u"),
            equation("b", "M2", "u"),
            equation("M1", "a", "b"),
            equation("M2", "a", "b"),
            equation("y", "M1"),
            equation("I", "M2"),
        ],
    )
    result = qualified(d, b)
    assert result["eligible"], result["errors"]
    rows = result["required_memory_path_qualification"]["checks"][0]["target_checks"]
    assert [row["passed"] for row in rows] == [True, True]
    assert rows[0]["memory_witnesses"][1]["qualified_path"] is None
    assert rows[1]["memory_witnesses"][0]["qualified_path"] is None


def test_two_current_drivers_do_not_supply_extra_memory_by_returning_through_either():
    b = public(drivers=("I", "a"))
    d = draft(
        b=b,
        variables=[
            variable("y"),
            variable("I", "algebraic"),
            variable("a", "algebraic"),
            variable("X"),
        ],
        equations=[
            equation("I", "X"),
            equation("a", "X"),
            equation("X", "I", "a", "u"),
            equation("y", "I", "a"),
        ],
    )
    assert assess(d, b)["eligible"]
    result = qualified(d, b)
    assert not result["eligible"]
    assert result["required_memory_path_qualification"]["status"] == "failed"


def test_optional_memory_and_history_free_mechanisms_keep_previous_interpretation():
    b = public(required=False)
    d = draft(b=b)
    result = qualified(d, b)
    assert result["eligible"]
    assert result["required_memory_path_qualification"]["status"] == "not_applicable"
    assert result["errors"] == assess(d, b)["errors"]


def test_invalid_graph_and_missing_endpoints_are_not_guessed_failures():
    d = draft(equations=[equation("y", "I"), equation("I", "X")])
    result = qualified(d)
    assert result["required_memory_path_qualification"]["status"] == "unavailable"
    assert result["required_memory_path_qualification"]["checks"][0]["status"] == (
        "unavailable_graph"
    )
    assert not any(
        e["code"] == "required_memory_path_revisits_driver" for e in result["errors"]
    )
    b = public(drivers=())
    result = qualified(draft(b=b), b)
    assert result["required_memory_path_qualification"]["checks"][0]["status"] == (
        "unresolved_public_endpoints"
    )


def test_new_qualification_does_not_waive_missing_driver_to_memory_path():
    d = draft(equations=[equation("y", "X"), equation("I", "u"), equation("X", "u")])
    result = qualified(d)
    assert not result["eligible"]
    assert any(e["code"] == "memory_driver_path" for e in result["errors"])
    assert result["required_memory_path_qualification"]["status"] == "passed"


def test_ordinary_missing_target_path_keeps_existing_feedback_without_cycle_claim():
    d = draft(equations=[equation("y", "I"), equation("I", "u"), equation("X", "I")])
    result = qualified(d)
    assert not result["eligible"]
    assert any(e["code"] == "memory_target_path" for e in result["errors"])
    assert not any(
        e["code"] == "required_memory_path_revisits_driver" for e in result["errors"]
    )


def test_invalid_memory_type_is_left_to_declaration_feedback():
    d = draft(variables=[variable("y"), variable("I"), variable("X", "algebraic")])
    result = qualified(d)
    assert not result["eligible"]
    assert result["required_memory_path_qualification"]["checks"][0]["status"] == (
        "unavailable_memory_declaration"
    )
