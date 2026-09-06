"""Scientific-contract and compiler regressions for the first staged milestone."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
    strict_provider_schema,
    visible_response,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    EquationReply,
    PublicScientificBrief,
    PublicVariable,
    ScientificRequirement,
    ScientificVariable,
    VariableReply,
    equation_reply_model,
)
from autoformalism.staged_topology import (
    freeze_inventory,
    lower_topology,
    merge_variable_reply,
    public_structure_checks,
    validate_equation,
)


def brief() -> PublicScientificBrief:
    return PublicScientificBrief(
        scientific_context="Generate x using a causal delayed response to u.",
        public_variables=(
            PublicVariable(name="x", data_role="target"),
            PublicVariable(name="a", data_role="auxiliary"),
            PublicVariable(name="u", data_role="external_input"),
        ),
        requirements=(
            ScientificRequirement(
                id="delay",
                public_requirement="Delayed input response",
                targets=("x",),
                drivers=("u",),
                positive_requirements=("Dynamic memory",),
            ),
        ),
    )


def var(name: str, definition: str = "differential") -> ScientificVariable:
    return ScientificVariable(
        name=name, definition=definition, scientific_role=f"Role of {name}"
    )


def equation(
    name: str, sources: tuple[str, ...], definition: str = "differential"
) -> EquationDefinition:
    return EquationDefinition(
        name=name,
        definition=definition,
        terms=({"sources": sources, "outer_sign": "add", "scientific_role": "drive"},),
    )


def test_inventory_reuses_rephrased_roles_without_merging_similar_variables() -> None:
    inventory = (var("x"), var("fast"), var("slow"), var("u", "supplied"))
    reply = VariableReply(
        variables=(var("x").model_copy(update={"scientific_role": "new phrasing"}),)
    )
    assert merge_variable_reply(brief(), inventory, reply) == inventory
    assert freeze_inventory(brief(), inventory) == inventory


@pytest.mark.parametrize(
    "variable", [var("x", "supplied"), var("z", "supplied"), var("u"), var("exp")]
)
def test_inventory_rejects_invalid_public_or_internal_treatment(
    variable: ScientificVariable,
) -> None:
    with pytest.raises(ValueError):
        merge_variable_reply(brief(), (), VariableReply(variables=(variable,)))


def test_inventory_requires_explicit_mode_revision_and_all_targets() -> None:
    with pytest.raises(ValueError, match="inventory revision"):
        merge_variable_reply(
            brief(),
            (var("a", "algebraic"),),
            VariableReply(variables=(var("a", "differential"),)),
        )
    with pytest.raises(ValueError, match="missing generated targets"):
        freeze_inventory(brief(), (var("z"),))


def test_inventory_checks_required_drivers_and_allows_public_activation() -> None:
    with pytest.raises(ValueError, match="select required mechanism drivers"):
        freeze_inventory(brief(), (var("x"),))
    updated = merge_variable_reply(
        brief(),
        (var("x"), var("u", "unused")),
        VariableReply(variables=(var("u", "supplied"),)),
    )
    assert freeze_inventory(brief(), updated)[1].definition == "supplied"


def test_generated_differential_auxiliary_has_no_observation_reset() -> None:
    inventory = (var("x"), var("a"), var("u", "supplied"))
    topology, aliases = lower_topology(
        brief(),
        inventory,
        (equation("x", ("a",)), equation("a", ("u",))),
        ValidationContext(
            targets=("x",),
            auxiliaries=("a",),
            external_inputs=("u",),
            lagged_targets=("x",),
        ),
    )
    generated = next(item for item in topology.states if item.name == aliases["a"])
    assert generated.kind.value == "latent"
    assert topology.state_measurements == ()
    assert topology.external_symbols == ("u",)


def test_provider_schema_preserves_exact_active_sources_and_limits() -> None:
    model = equation_reply_model(("x", "u"), maximum_terms=2)
    schema = strict_provider_schema(model.model_json_schema())
    assert schema["$defs"]["SelectedEquationTerm"]["properties"]["sources"]["items"][
        "enum"
    ] == ["x", "u"]
    assert schema["properties"]["terms"]["maxItems"] == 2
    valid = {
        "terms": [
            {
                "sources": ["x", "u"],
                "outer_sign": "add",
                "scientific_role": "joint response",
            }
        ],
        "inventory_revision": None,
    }
    assert model.model_validate(valid).terms[0].sources == ("x", "u")
    bad = copy.deepcopy(valid)
    bad["terms"][0]["sources"] = ["new_unknown"]
    with pytest.raises(ValidationError):
        model.model_validate(bad)
    with pytest.raises(ValidationError):
        model.model_validate({**valid, "terms": valid["terms"] * 3})


def test_explicit_revision_is_an_alternative_to_an_equation() -> None:
    revision = {"variable": var("z").model_dump(), "reason": "a necessary memory"}
    assert (
        EquationReply(terms=(), inventory_revision=revision).inventory_revision
        is not None
    )
    with pytest.raises(ValidationError):
        EquationReply(terms=(), inventory_revision=None)
    with pytest.raises(ValidationError):
        EquationReply(terms=equation("x", ("u",)).terms, inventory_revision=revision)


def test_algebraic_cycles_fail_but_dynamic_feedback_is_legal() -> None:
    inventory = (var("x"), var("a", "algebraic"), var("b", "algebraic"))
    first = equation("a", ("b",), "algebraic")
    with pytest.raises(ValueError, match="algebraic cycle"):
        validate_equation(
            inventory, (first,), equation("b", ("a",), "algebraic"), brief().limits
        )
    validate_equation(
        inventory,
        (equation("x", ("a",)),),
        equation("a", ("x",), "algebraic"),
        brief().limits,
    )


def test_closed_compilation_preserves_grouped_sources_and_roles() -> None:
    inventory = (var("x"), var("z"), var("u", "supplied"))
    equations = (equation("x", ("z", "x")), equation("z", ("u", "z")))
    topology, aliases = lower_topology(
        brief(),
        inventory,
        equations,
        ValidationContext(targets=("x",), external_inputs=("u",)),
    )
    assert aliases == {}
    assert topology.interactions[0].sources == ("z", "x")
    assert topology.states[0].description == "Role of x"
    assert all(item["passed"] for item in public_structure_checks(brief(), equations))


def test_generated_algebraic_auxiliary_is_never_a_supplied_source() -> None:
    inventory = (var("x"), var("a", "algebraic"), var("u", "supplied"))
    topology, aliases = lower_topology(
        brief(),
        inventory,
        (equation("x", ("a",)), equation("a", ("u",), "algebraic")),
        ValidationContext(targets=("x",), auxiliaries=("a",), external_inputs=("u",)),
    )
    assert topology.external_symbols == ("u",)
    assert topology.interactions[0].sources == (aliases["a"],)
    assert topology.processes[0].name == aliases["a"]


def test_missing_equation_fails_and_source_free_path_is_not_a_mechanism() -> None:
    with pytest.raises(ValueError, match="exactly one equation"):
        lower_topology(
            brief(),
            (var("x"), var("z"), var("u", "supplied")),
            (equation("x", ("z",)),),
            ValidationContext(targets=("x",)),
        )
    checks = public_structure_checks(brief(), (equation("x", ()),))
    assert not any(item["passed"] for item in checks)


def test_physical_response_and_terminal_failures_are_cached(tmp_path: Path) -> None:
    calls = []

    def transport(url, body, timeout):
        calls.append(body)
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": "{broken"}}]
        }

    def client():
        return StagedTopologyClient(
            settings=StagedModelSettings(),
            base_url="http://localhost:8000",
            directory=tmp_path,
            namespace="test",
            seed=0,
            transport=transport,
        )

    kwargs = {
        "system": "s",
        "user": "u",
        "response_model": VariableReply,
        "step": "variables",
        "attempt": 0,
    }
    first = client().call(**kwargs)
    assert first["request"]["body"]["messages"][1]["content"] == "u"
    with pytest.raises(ValueError):
        visible_response(first)
    assert client().call(**kwargs) == first
    assert len(calls) == 1


def test_interrupted_call_is_uncertain_and_not_reissued(tmp_path: Path) -> None:
    def interrupted(*args):
        raise KeyboardInterrupt

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        base_url="http://localhost:8000",
        directory=tmp_path,
        namespace="test",
        seed=0,
        transport=interrupted,
    )
    kwargs = {
        "system": "s",
        "user": "u",
        "response_model": VariableReply,
        "step": "variables",
        "attempt": 0,
    }
    with pytest.raises(KeyboardInterrupt):
        client.call(**kwargs)
    record = client.call(**kwargs)
    assert record["status"] == "uncertain"
    assert (
        json.loads(next(tmp_path.glob("*.json")).read_text())["status"] == "uncertain"
    )


def test_shutdown_guard_starts_no_call(tmp_path: Path) -> None:
    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        base_url="http://localhost:8000",
        directory=tmp_path,
        namespace="test",
        seed=0,
        can_start=lambda: False,
    )
    with pytest.raises(DeferredCall):
        client.call(
            system="s", user="u", response_model=VariableReply, step="v", attempt=0
        )
    assert list(tmp_path.iterdir()) == []
