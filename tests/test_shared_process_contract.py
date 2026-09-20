"""One shared function, per-suggestion admission, and faithful topology handoff."""

import json

import pytest

from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import shared_process_contract as shared
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.staged_topology_runner import run_staged_topology
from scripts import smoke_detention_process_pilot as smoke
from scripts.smoke_shared_process_contract import transport_for


@pytest.fixture
def inputs():
    brief = PublicScientificBrief(
        scientific_context="Two stores exchange material; y is observed.",
        public_variables=[
            {"name": "y", "data_role": "target"},
            {"name": "u", "data_role": "external_input"},
        ],
        requirements=[],
    )
    inventory = tuple(
        ScientificVariable(name=n, definition=d, scientific_role=n)
        for n, d in (("x", "differential"), ("y", "differential"), ("u", "supplied"))
    )
    p = {
        "name": "q",
        "depends_on": ["x"],
        "used_in_equations_for": ["x", "y"],
        "scientific_meaning": "transfer from x to y",
    }
    return brief, inventory, p


def test_invalid_outlet_does_not_discard_transfer(inputs):
    brief, inv, p = inputs
    bad = {**p, "name": "outlet", "used_in_equations_for": ["y", "u"]}
    result, audit = shared.admit(brief, inv, {"processes": [p, bad]})
    assert [v.name for v in result] == ["x", "y", "u", "q"]
    assert audit["status"] == "accepted_partial"
    assert audit["suggestions"] == [p]
    assert audit["original_suggestions"] == [p, bad]
    assert "u" in audit["rejected_suggestions"][0]["error"]


def test_local_outlet_and_empty_are_valid(inputs):
    brief, inv, p = inputs
    p["used_in_equations_for"] = ["y"]
    assert shared.admit(brief, inv, {"processes": [p]})[1]["status"] == "accepted"
    assert shared.admit(brief, inv, {"processes": []})[0] == inv


def test_reusing_process_keeps_structured_relationships(inputs):
    brief, inv, p = inputs
    inv += (
        ScientificVariable(name="q", definition="algebraic", scientific_role="flow"),
    )
    result, audit = shared.admit(brief, inv, {"processes": [p]})
    assert result == inv and audit["added_names"] == []
    assert audit["suggestions"] == [p]


def test_conflicts_and_malformed_suggestion_do_not_erase_independent_item(inputs):
    brief, inv, p = inputs
    other = {**p, "name": "r"}
    _, audit = shared.admit(brief, inv, {"processes": [p, p, {}, other]})
    assert audit["suggestions"] == [other]
    assert len(audit["rejected_suggestions"]) == 3


@pytest.mark.parametrize(
    "expression", ["q", "q/area", "gain*q/area", "q*(1+z*z)", "-q"]
)
def test_linear_conversion_preserves_signed_process(expression):
    shared.validate_function(expression, {"process": "q"})


@pytest.mark.parametrize(
    "expression", ["tanh(q)", "q*q", "q+1", "1/q", "k*x", "q-q", "q/(1+q)"]
)
def test_replacing_or_transforming_shared_law_is_rejected(expression):
    with pytest.raises(ValueError, match="reuse q once"):
        shared.validate_function(expression, {"process": "q"})


def test_signs_and_uses_are_preserved(inputs):
    _, inv, p = inputs
    reply = shared.UsesReply(
        uses=[
            {
                "target": name,
                "outer_weight_sign": sign,
                "conversion_sources": [],
                "scientific_role": "balance",
            }
            for name, sign in (("x", "negative"), ("y", "positive"))
        ]
    )
    binding = shared.validate_uses(p, reply, inv)
    definition = shared.definition(binding)
    assert len(definition.terms) == 1
    assert definition.terms[0].outer_weight_sign.value == "unrestricted"
    equation = EquationDefinition(
        name="x",
        definition="differential",
        terms=[
            {
                "sources": ["q"],
                "outer_weight_sign": "negative",
                "scientific_role": "loss",
            }
        ],
    )
    shared.validate_equation_uses([binding], equation)
    wrong = equation.model_copy(
        update={"terms": (equation.terms[0].model_copy(update={"sources": ("x",)}),)}
    )
    with pytest.raises(ValueError, match="preserve"):
        shared.validate_equation_uses([binding], wrong)
    with pytest.raises(ValueError, match="every declared"):
        shared.validate_uses(p, shared.UsesReply(uses=reply.uses[:1]), inv)
    # The conversion instruction must never be attached to P's defining law.
    process_term = {"lhs": "q", "sources": ["x"]}
    assert shared.decorate_function_term(process_term, [binding]) == process_term
    consumer_term = {"lhs": "x", "sources": ["q"]}
    assert shared.decorate_function_term(consumer_term, [binding])[
        "shared_process_use"
    ]["process"] == "q"


def construct(tmp_path, mode="add"):
    plan = smoke.fixture(tmp_path / "source", tmp_path / "legacy")
    pilot = smoke.pilot
    task = next(t for t in plan["tasks"] if t["case"] == "coupled" and t["review"])
    cell = plan["cells"]["coupled"]
    brief = pilot.PublicScientificBrief.model_validate(cell["brief"])
    context = pilot.ValidationContext.model_validate(cell["context"])
    calls = []
    client = pilot.make_client(
        tmp_path, plan, task, "offline", transport=transport_for(calls, mode)
    )
    topology = run_staged_topology(
        brief,
        context,
        client,
        tmp_path / "topology",
        hybrid_variable_construction=True,
        proposer_owns_unfixed_signs=True,
        shared_process_guidance=True,
        optional_process_review=True,
        bind_shared_processes=True,
    )
    assert topology["complete_topology"], topology
    return brief, context, client, topology, calls


@pytest.mark.parametrize(
    "granularity",
    ["atomic_interaction", "equation_batch", "equation_batch_atomic_repair"],
)
def test_single_law_reaches_compiled_equations_and_survives_resume(
    tmp_path, granularity
):
    brief, context, client, topology, calls = construct(tmp_path)
    assert topology["process_review"]["status"] == "accepted_partial"
    assert len(topology["shared_process_contract"]["bindings"]) == 1
    functions = run_staged_functions(
        brief,
        context,
        topology,
        client,
        tmp_path / "functions",
        generation_granularity=granularity,
        function_repair_policy="certified_outer_gain",
        initialization_policy="causal_training",
        shared_process_guidance=True,
    )
    assert functions["complete_model"], functions["error"]
    model = functions["candidate"]
    assert len(model["processes"]) == 1
    assert (
        len(model["parameters"]) == 2
    )  # one transfer coefficient, one local outlet coefficient
    assert all("q" in eq["rhs"] for eq in model["state_equations"])
    assert sum(eq["name"] == "q" for eq in topology["equations"]) == 1
    count = len(calls)
    again = run_staged_functions(
        brief,
        context,
        topology,
        client,
        tmp_path / "functions",
        generation_granularity=granularity,
        function_repair_policy="certified_outer_gain",
        initialization_policy="causal_training",
        shared_process_guidance=True,
    )
    assert again["candidate"] == functions["candidate"] and len(calls) == count


def test_nonlinear_consumer_is_locally_repaired_and_reconstruction_checks_it(tmp_path):
    brief, context, client, topology, _ = construct(tmp_path)
    original = client.transport

    def nonlinear(url, body, timeout):
        response = original(url, body, timeout)
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        if payload.get("selected_equation", {}).get("lhs") == "h_up":
            reply = json.loads(response["choices"][0]["message"]["content"])
            reply["functions"][1]["expression"] = "q*q/area_up"
            response["choices"][0]["message"]["content"] = json.dumps(reply)
        return response

    client.transport = nonlinear
    functions = run_staged_functions(
        brief,
        context,
        topology,
        client,
        tmp_path / "functions",
        generation_granularity="equation_batch_atomic_repair",
        function_repair_policy="certified_outer_gain",
        initialization_policy="causal_training",
        shared_process_guidance=True,
    )
    assert functions["complete_model"], functions["error"]
    repairs = [
        a for a in functions["batch_term_audits"] if a["atomic_repair_attempted"]
    ]
    assert len(repairs) == 1 and repairs[0]["atomic_repair_succeeded"]
    assert "reuse q once" in repairs[0]["batch_error"]
    from autoformalism.rebuttal.prefit_construction_audit import reconstruct

    cell = {
        "brief": brief.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
    }
    audit = reconstruct(cell, {"topology": topology, "functions": functions})
    assert audit["certificate"]["passed"]


def test_contract_cannot_be_lost_during_reconstruction(tmp_path):
    brief, context, client, topology, _ = construct(tmp_path)
    functions = run_staged_functions(
        brief,
        context,
        topology,
        client,
        tmp_path / "functions",
        generation_granularity="equation_batch_atomic_repair",
        function_repair_policy="certified_outer_gain",
        initialization_policy="causal_training",
        shared_process_guidance=True,
    )
    functions.pop("shared_process_contract")
    from autoformalism.rebuttal.prefit_construction_audit import reconstruct

    with pytest.raises(ValueError, match="shared-process contract differs"):
        reconstruct(
            {
                "brief": brief.model_dump(mode="json"),
                "context": context.model_dump(mode="json"),
            },
            {"topology": topology, "functions": functions},
        )
