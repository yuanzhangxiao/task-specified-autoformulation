"""Scientific sign decisions, derived gain domains, and legacy replay."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from autoformalism.construction import finalize_functional_draft
from autoformalism.expressions import RestrictedParser, ValidationContext
from autoformalism.rebuttal.staged_topology_campaign import diagnostic_task
from autoformalism.schemas.candidate import ParameterDomain, ParameterRole
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged import InteractionPolarity
from autoformalism.schemas.staged_functions import (
    InteractionFunctionReply,
    LatentInitialReply,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    ModelingLimits,
    PublicScientificBrief,
    ScientificVariable,
    equation_reply_model,
)
from autoformalism.staged_functions import (
    apply_function_reply,
    apply_initial_reply,
    normalize_outer_weight_reply,
)
from autoformalism.staged_topology import lower_topology
from autoformalism.staging import topology_commitment_sha256


def _reply(expression: str, **roles: str) -> InteractionFunctionReply:
    return InteractionFunctionReply(
        expression=expression,
        parameters=tuple(
            {"name": name, "role": role} for name, role in roles.items()
        ),
    )


def _normalize(
    expression: str, polarity: InteractionPolarity, **roles: str
) -> tuple[InteractionFunctionReply, tuple[object, ...]]:
    reply = _reply(expression, **roles)
    tree = RestrictedParser().parse(expression, location="test").tree
    return normalize_outer_weight_reply(reply, polarity, tree)


def test_new_provider_term_requires_an_explicit_three_way_sign_decision() -> None:
    model = equation_reply_model(("x",), maximum_terms=1)
    with pytest.raises(ValidationError, match="outer_weight_sign"):
        model.model_validate(
            {
                "terms": [{"sources": ["x"], "scientific_role": "response"}],
                "inventory_revision": None,
            }
        )
    schema = model.model_json_schema()
    sign_schema = schema["$defs"]["SelectedEquationTerm"]["properties"][
        "outer_weight_sign"
    ]
    assert sign_schema["$ref"].endswith("/$defs/OuterWeightSign")
    assert schema["$defs"]["OuterWeightSign"]["enum"] == [
        "positive",
        "negative",
        "unrestricted",
    ]


@pytest.mark.parametrize(
    ("polarity", "expected_sign"),
    [
        (InteractionPolarity.POSITIVE, "positive"),
        (InteractionPolarity.NEGATIVE, "negative"),
    ],
)
def test_fixed_sign_derives_only_the_identified_outer_gain_domain(
    polarity: InteractionPolarity, expected_sign: str
) -> None:
    normalized, records = _normalize(
        "gain*(z-threshold*x)",
        polarity,
        gain="coefficient",
        threshold="coefficient",
    )
    assert {item.name: item.role for item in normalized.parameters} == {
        "gain": ParameterRole.NONNEGATIVE_COEFFICIENT,
        "threshold": ParameterRole.COEFFICIENT,
    }
    assert len(records) == 1
    assert records[0].parameter == "gain"
    assert records[0].outer_weight_sign.value == expected_sign


def test_fixed_sign_does_not_claim_the_complete_law_is_nonnegative() -> None:
    normalized, _ = _normalize(
        "gain*tanh(z)", InteractionPolarity.POSITIVE, gain="coefficient"
    )
    assert normalized.parameters[0].role is ParameterRole.NONNEGATIVE_COEFFICIENT


def test_unrestricted_gain_and_baseline_remain_signed() -> None:
    gain, gain_records = _normalize(
        "gain*z", InteractionPolarity.UNRESTRICTED, gain="coefficient"
    )
    offset, offset_records = _normalize(
        "baseline", InteractionPolarity.UNRESTRICTED, baseline="offset"
    )
    assert gain.parameters[0].role is ParameterRole.COEFFICIENT
    assert offset.parameters[0].role is ParameterRole.OFFSET
    assert gain_records == offset_records == ()


def test_unrestricted_repairs_old_generic_nonnegative_outer_weight_to_real() -> None:
    normalized, records = _normalize(
        "baseline",
        InteractionPolarity.UNRESTRICTED,
        baseline="nonnegative_coefficient",
    )
    assert normalized.parameters[0].role is ParameterRole.COEFFICIENT
    assert records[0].requested_role is ParameterRole.NONNEGATIVE_COEFFICIENT
    assert records[0].effective_role is ParameterRole.COEFFICIENT
    assert records[0].outer_weight_sign.value == "unrestricted"


def test_unrestricted_does_not_override_an_intrinsically_positive_scale() -> None:
    normalized, records = _normalize(
        "scale*x", InteractionPolarity.UNRESTRICTED, scale="scale"
    )
    assert normalized.parameters[0].role is ParameterRole.SCALE
    assert records == ()


def test_semantic_positive_roles_remain_positive_without_sign_rewriting() -> None:
    normalized, records = _normalize(
        "scale*(z-x)/tau",
        InteractionPolarity.POSITIVE,
        scale="scale",
        tau="time_constant",
    )
    assert records == ()
    assert {item.name: item.role.domain for item in normalized.parameters} == {
        "scale": ParameterDomain.POSITIVE,
        "tau": ParameterDomain.POSITIVE,
    }


@pytest.mark.parametrize(
    ("expression", "roles", "code"),
    [
        (
            "-gain*z",
            {"gain": "coefficient"},
            "DUPLICATED_WHOLE_OUTER_SIGN",
        ),
        (
            "(-gain)*(-z)",
            {"gain": "coefficient"},
            "DUPLICATED_WHOLE_OUTER_SIGN",
        ),
        (
            "-(-gain)*z",
            {"gain": "coefficient"},
            "DUPLICATED_WHOLE_OUTER_SIGN",
        ),
        (
            "left*right*z",
            {"left": "coefficient", "right": "coefficient"},
            "AMBIGUOUS_OUTER_WEIGHT_PARAMETERS",
        ),
        (
            "z/gain",
            {"gain": "coefficient"},
            "SIGNED_PARAMETER_CONTROLS_OUTER_SIGN_IN_DENOMINATOR",
        ),
    ],
)
def test_uncertain_whole_sign_cases_request_repair_without_ast_rewriting(
    expression: str, roles: dict[str, str], code: str
) -> None:
    with pytest.raises(ValueError, match=code):
        _normalize(expression, InteractionPolarity.NEGATIVE, **roles)


def test_legacy_equation_replays_byte_equivalent_and_keeps_old_domain_rule() -> None:
    payload = {
        "name": "x",
        "definition": "differential",
        "terms": [
            {
                "sources": ["x"],
                "outer_sign": "subtract",
                "scientific_role": "legacy relaxation",
            }
        ],
    }
    equation = EquationDefinition.model_validate_json(json.dumps(payload))
    assert equation.model_dump(mode="json") == payload


def test_shared_fixed_sign_gain_is_consistent_and_assembles_once() -> None:
    task = diagnostic_task("driven_memory", ModelingLimits())
    brief = PublicScientificBrief.model_validate(task["brief"])
    context = ValidationContext.model_validate(task["context"])
    inventory = tuple(
        ScientificVariable.model_validate(item) for item in task["initial_inventory"]
    )
    equations = (
        EquationDefinition(
            name="x",
            definition="differential",
            terms=(
                {
                    "sources": ["z"],
                    "outer_weight_sign": "positive",
                    "scientific_role": "positive drive",
                },
                {
                    "sources": ["x"],
                    "outer_weight_sign": "negative",
                    "scientific_role": "clearance",
                },
            ),
        ),
        EquationDefinition(
            name="z",
            definition="differential",
            terms=(
                {
                    "sources": ["u"],
                    "outer_weight_sign": "positive",
                    "scientific_role": "input drive",
                },
                {
                    "sources": ["z"],
                    "outer_weight_sign": "negative",
                    "scientific_role": "memory clearance",
                },
            ),
        ),
    )
    topology, aliases = lower_topology(brief, inventory, equations, context)
    draft = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    for interaction_id, expression in (
        ("term_0_0", "gain*z"),
        ("term_0_1", "gain*x"),
        ("term_1_0", "u"),
        ("term_1_1", "z/tau"),
    ):
        roles = (
            {"gain": "coefficient"}
            if "gain" in expression
            else ({"tau": "time_constant"} if "tau" in expression else {})
        )
        draft = apply_function_reply(
            topology,
            draft,
            interaction_id,
            _reply(expression, **roles),
            context,
            aliases,
        )
    draft = apply_initial_reply(
        topology,
        draft,
        "z",
        LatentInitialReply(initial={"fixed_value": 0.0}),
        context,
        aliases,
    )
    candidate = finalize_functional_draft(topology, draft, context).candidate
    parameters = {item.name: item for item in candidate.parameters}
    assert parameters["gain"].domain is ParameterDomain.NONNEGATIVE
    assert parameters["tau"].domain is ParameterDomain.POSITIVE
    assert candidate.state_equations[0].rhs == "(gain * z) - (gain * x)"


def test_fixed_and_unrestricted_shared_gain_roles_conflict() -> None:
    fixed, _ = _normalize(
        "shared*x", InteractionPolarity.POSITIVE, shared="coefficient"
    )
    unrestricted, _ = _normalize(
        "shared*x", InteractionPolarity.UNRESTRICTED, shared="coefficient"
    )
    assert fixed.parameters[0].role is ParameterRole.NONNEGATIVE_COEFFICIENT
    assert unrestricted.parameters[0].role is ParameterRole.COEFFICIENT


def test_shell_level_unary_plus_is_transparent() -> None:
    normalized, derivations = _normalize(
        "+(gain*x)", InteractionPolarity.POSITIVE, gain="coefficient"
    )
    assert normalized.parameters[0].role is ParameterRole.NONNEGATIVE_COEFFICIENT
    assert derivations[0].parameter == "gain"
