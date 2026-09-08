"""Function contracts, topology binding, causal initialization and safe compilation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoformalism.construction import finalize_functional_draft
from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.rebuttal.staged_topology_campaign import diagnostic_task
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_functions import (
    EquationFunctionBatchReply,
    InteractionFunctionReply,
    LatentInitialReply,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    ModelingLimits,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.staged_functions import (
    apply_equation_function_reply,
    apply_function_reply,
    apply_initial_reply,
    bind_function_reply,
    derive_interaction_function_obligation,
    repair_certified_outer_gain_role,
)
from autoformalism.staged_topology import lower_topology
from autoformalism.staging import topology_commitment_sha256


def fixture(generated_auxiliary: bool = False):
    """Use a joint-source toy topology without any benchmark-derived expression."""
    task = diagnostic_task(
        "generated_auxiliary" if generated_auxiliary else "driven_memory",
        ModelingLimits(),
    )
    brief = PublicScientificBrief.model_validate(task["brief"])
    context = ValidationContext.model_validate(task["context"])
    inventory = tuple(
        ScientificVariable.model_validate(item) for item in task["initial_inventory"]
    )
    hidden = "a" if generated_auxiliary else "z"
    equations = (
        EquationDefinition(
            name="x",
            definition="differential",
            terms=(
                {
                    "sources": ["x", hidden],
                    "outer_sign": "add",
                    "scientific_role": "joint response and relaxation",
                },
            ),
        ),
        EquationDefinition(
            name=hidden,
            definition="algebraic" if generated_auxiliary else "differential",
            terms=(
                {
                    "sources": ["u"] if generated_auxiliary else ["u", "z"],
                    "outer_sign": "add",
                    "scientific_role": "input response",
                },
            ),
        ),
    )
    topology, aliases = lower_topology(brief, inventory, equations, context)
    source = {
        "complete_topology": True,
        "inventory": [v.model_dump(mode="json") for v in inventory],
        "equations": [e.model_dump(mode="json") for e in equations],
        "topology": topology.model_dump(mode="json"),
    }
    return brief, context, source, topology, aliases


def reply(expression, **roles):
    return InteractionFunctionReply(
        expression=expression,
        parameters=tuple({"name": name, "role": role} for name, role in roles.items()),
    )


def test_joint_sources_compile_with_separate_causal_initial() -> None:
    _, context, _, topology, aliases = fixture()
    initial = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    draft = apply_function_reply(
        topology,
        initial,
        "term_0_0",
        reply("gain*z-rate*x", gain="scale", rate="rate"),
        context,
        aliases,
    )
    assert initial.interaction_functions == ()
    draft = apply_function_reply(
        topology,
        draft,
        "term_1_0",
        reply("(u-z)/tau", tau="time_constant"),
        context,
        aliases,
    )
    with pytest.raises(ValueError, match="missing_initials"):
        finalize_functional_draft(topology, draft, context)
    draft = apply_initial_reply(
        topology,
        draft,
        "z",
        LatentInitialReply(initial={"fixed_value": 0.0}),
        context,
        aliases,
    )
    candidate = finalize_functional_draft(topology, draft, context).candidate
    assert len(candidate.parameters) == 3
    assert candidate.state_equations[0].rhs


@pytest.mark.parametrize(
    "expression,roles,pattern",
    [
        ("gain*z", {"gain": "scale"}, "source mismatch"),
        ("gain*z-rate*x+u", {"gain": "scale", "rate": "rate"}, "source mismatch"),
        ("gain*z-rate*x", {"gain": "coefficient", "rate": "rate"}, "SIGNED_WEIGHT"),
        (
            "gain*z-rate*x",
            {"gain": "scale", "rate": "rate", "extra": "scale"},
            "UNUSED_LOCAL_PARAMETER",
        ),
        ("x*z", {"x": "scale"}, "collides"),
        ("z.__class__ + x", {}, "UNSUPPORTED_SYNTAX"),
        ("z**17 + x", {}, "POWER"),
    ],
)
def test_bad_function_cannot_mutate_parent(expression, roles, pattern) -> None:
    _, context, _, topology, aliases = fixture()
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    with pytest.raises((ValueError, ModelValidationError), match=pattern):
        apply_function_reply(
            topology, parent, "term_0_0", reply(expression, **roles), context, aliases
        )
    assert parent.interaction_functions == ()


def test_shared_parameter_role_conflict_is_rejected() -> None:
    _, context, _, topology, aliases = fixture()
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    parent = apply_function_reply(
        topology, parent, "term_0_0", reply("k*(z-x)", k="scale"), context, aliases
    )
    with pytest.raises(ValueError, match="PARAMETER_ROLE_CONFLICT"):
        apply_function_reply(
            topology, parent, "term_1_0", reply("k*(u-z)", k="rate"), context, aliases
        )


def test_runtime_obligation_rejects_linear_law_and_localizes_parameters() -> None:
    _, context, _, topology, aliases = fixture()
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    nonlinear = derive_interaction_function_obligation(
        "nonlinear joint response",
        parameter_identity_policy="interaction_local",
    )
    with pytest.raises(ValueError, match="NONLINEAR_SOURCE_DEPENDENCE_REQUIRED"):
        bind_function_reply(
            topology,
            parent,
            "term_0_0",
            reply("k*(z-x)", k="scale"),
            context,
            aliases,
            nonlinear,
        )
    first, normalized = bind_function_reply(
        topology,
        parent,
        "term_0_0",
        reply("k*(z**2/(1+z**2)-x)", k="scale"),
        context,
        aliases,
        nonlinear,
    )
    assert normalized.parameters[0].name != "k"
    assert normalized.parameters[0].name in normalized.expression
    second, normalized_second = bind_function_reply(
        topology,
        first,
        "term_1_0",
        reply("k*(u-z)", k="scale"),
        context,
        aliases,
        derive_interaction_function_obligation(
            "input response",
            parameter_identity_policy="interaction_local",
        ),
    )
    assert normalized_second.parameters[0].name != normalized.parameters[0].name
    assert (
        len(
            {
                parameter.name
                for function in second.interaction_functions
                for parameter in function.parameters
            }
        )
        == 2
    )


@pytest.mark.parametrize(
    "expression,roles,sources,expected",
    [
        ("k*x", {"k": "coefficient"}, {"x"}, True),
        ("k*(z-x)", {"k": "coefficient"}, {"x", "z"}, True),
        ("k*x+z", {"k": "coefficient"}, {"x", "z"}, False),
        ("exp(k*x)", {"k": "coefficient"}, {"x"}, False),
        (
            "k*x/tau",
            {"k": "coefficient", "tau": "time_constant"},
            {"x"},
            False,
        ),
        ("k*k*x", {"k": "coefficient"}, {"x"}, False),
        (
            "k*q*x",
            {"k": "coefficient", "q": "coefficient"},
            {"x"},
            False,
        ),
        ("2*k", {"k": "coefficient"}, {"x"}, False),
    ],
)
def test_certified_outer_gain_role_repair_is_conservative(
    expression, roles, sources, expected
) -> None:
    original = reply(expression, **roles)
    repaired, records = repair_certified_outer_gain_role(original, sources)
    assert bool(records) is expected
    if expected:
        assert repaired.parameters[0].role.value == "nonnegative_coefficient"
        assert records[0].parameter == "k"
    else:
        assert repaired == original


def test_namespaced_compatibility_diagnostic_uses_local_parameter_name() -> None:
    _, context, _, topology, aliases = fixture()
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    with pytest.raises(ValueError) as caught:
        bind_function_reply(
            topology,
            parent,
            "term_0_0",
            reply("k*z-rate*x", k="coefficient", rate="rate"),
            context,
            aliases,
            derive_interaction_function_obligation(
                "joint response",
                parameter_identity_policy="interaction_local",
            ),
        )
    assert "['k']" in str(caught.value)
    assert "k_term_0_0" not in str(caught.value)


def test_function_batch_is_bound_in_order_and_fails_atomically() -> None:
    _, context, _, topology, aliases = fixture()
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    valid = EquationFunctionBatchReply(
        functions=(
            reply("gain*z-rate*x", gain="scale", rate="rate"),
            reply("(u-z)/tau", tau="time_constant"),
        )
    )
    updated = apply_equation_function_reply(
        topology,
        parent,
        ("term_0_0", "term_1_0"),
        valid,
        context,
        aliases,
    )
    assert len(updated.interaction_functions) == 2
    assert parent.interaction_functions == ()


def test_function_batch_preserves_consistent_shared_parameter_identity() -> None:
    _, context, _, topology, aliases = fixture()
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    updated = apply_equation_function_reply(
        topology,
        parent,
        ("term_0_0", "term_1_0"),
        EquationFunctionBatchReply(
            functions=(
                reply("k*(z-x)", k="scale"),
                reply("k*(u-z)", k="scale"),
            )
        ),
        context,
        aliases,
    )
    assert len(updated.interaction_functions) == 2
    assert {
        parameter.role
        for function in updated.interaction_functions
        for parameter in function.parameters
        if parameter.name == "k"
    } == {"scale"}

    invalid = EquationFunctionBatchReply(
        functions=(
            reply("gain*z-rate*x", gain="scale", rate="rate"),
            reply("z", tau="time_constant"),
        )
    )
    with pytest.raises(ValueError, match=r"UNUSED_LOCAL_PARAMETER|source mismatch"):
        apply_equation_function_reply(
            topology,
            parent,
            ("term_0_0", "term_1_0"),
            invalid,
            context,
            aliases,
        )
    assert parent.interaction_functions == ()


def test_equation_batch_rejects_wrong_function_count() -> None:
    _, context, _, topology, aliases = fixture()
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    with pytest.raises(ValueError, match="function count mismatch"):
        apply_equation_function_reply(
            topology,
            parent,
            ("term_0_0", "term_1_0"),
            EquationFunctionBatchReply(functions=(reply("z-x"),)),
            context,
            aliases,
        )


def test_generated_auxiliary_names_are_mapped_only_after_public_source_validation() -> (
    None
):
    _, context, _, topology, aliases = fixture(True)
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    draft = apply_function_reply(
        topology,
        parent,
        "term_0_0",
        reply("gain*a-rate*x", gain="scale", rate="rate"),
        context,
        aliases,
    )
    assert aliases["a"] in draft.interaction_functions[0].expression
    with pytest.raises(ValueError, match="source mismatch"):
        apply_function_reply(
            topology, parent, "term_0_0", reply(f"{aliases['a']}-x"), context, aliases
        )
    draft = apply_function_reply(
        topology, draft, "term_1_0", reply("u"), context, aliases
    )
    assert (
        finalize_functional_draft(topology, draft, context).candidate.processes[0].name
        == aliases["a"]
    )


@pytest.mark.parametrize("expression", ["x", "z", "unknown", "k"])
def test_initial_cannot_read_targets_states_or_fitted_parameters(expression) -> None:
    _, context, _, topology, aliases = fixture()
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    with pytest.raises(ValueError, match="unavailable initialization"):
        apply_initial_reply(
            topology,
            parent,
            "z",
            LatentInitialReply(initial={"expression": expression}),
            context,
            aliases,
        )


def test_initial_accepts_supplied_information_and_rejects_generated_auxiliary() -> None:
    _, context, _, topology, aliases = fixture()
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    updated = apply_initial_reply(
        topology,
        parent,
        "z",
        LatentInitialReply(initial={"expression": "u"}),
        context,
        aliases,
    )
    assert updated.latent_initials[0].initial.expression == "u"
    context = context.model_copy(update={"auxiliaries": ("a",)})
    with pytest.raises(ValueError, match="unavailable initialization"):
        apply_initial_reply(
            topology,
            parent,
            "z",
            LatentInitialReply(initial={"expression": "a"}),
            context,
            {"a": "af_internal_aux_0"},
        )


@pytest.mark.parametrize(
    "extra", [{"scope": "global"}, {"bounds": {"lower": 0, "upper": 1}}, {"value": 1}]
)
def test_parameter_reply_rejects_runtime_owned_fields(extra) -> None:
    with pytest.raises(ValidationError):
        InteractionFunctionReply(
            expression="k*x", parameters=[{"name": "k", "role": "scale", **extra}]
        )


def test_initial_schema_rejects_two_modes_or_nonfinite_values() -> None:
    for initial in (
        {"fixed_value": 0, "expression": "u"},
        {},
        {"fixed_value": float("inf")},
    ):
        with pytest.raises(ValidationError):
            LatentInitialReply(initial=initial)
