"""Tests for the compact proposer contract and canonical enrichment."""

import pytest
from pydantic import ValidationError

from autoformalism.schemas import (
    ProposerCandidate,
    ProposerCandidateV2,
    enrich_proposal,
    enrich_proposal_v2,
)


def proposal_payload() -> dict[str, object]:
    return {
        "candidate_id": "compact_1",
        "states": [
            {
                "name": "x",
                "kind": "latent",
                "rhs": "-k * x + u",
                "initial_expression": "u",
                "constraints": [{"kind": "nonnegative"}],
            }
        ],
        "observations": [{"channel": "target", "expression": "x"}],
        "parameters": [
            {"name": "k", "bounds": {"lower": 0.0, "upper": 2.0}}
        ],
    }


def test_compact_proposal_round_trip_and_enrichment() -> None:
    proposal = ProposerCandidate.model_validate(proposal_payload())
    restored = ProposerCandidate.model_validate_json(proposal.model_dump_json())

    assert restored == proposal
    candidate = enrich_proposal(restored)
    assert candidate.state_equations[0].state == "x"
    assert candidate.state_equations[0].rhs == "-k * x + u"
    assert candidate.initial_conditions[0].expression == "u"
    assert candidate.constraints[0].subject == "x"
    assert candidate.constraints[0].kind.value == "nonnegative"
    assert candidate.parameters[0].bounds is None
    assert candidate.parameters[0].initialization_range is None


def test_state_without_rhs_cannot_be_proposed() -> None:
    payload = proposal_payload()
    del payload["states"][0]["rhs"]  # type: ignore[index]

    with pytest.raises(ValidationError):
        ProposerCandidate.model_validate(payload)


def test_state_rejects_ambiguous_initialization() -> None:
    payload = proposal_payload()
    payload["states"][0]["initial_value"] = 0.0  # type: ignore[index]

    with pytest.raises(ValidationError, match="only one initialization mode"):
        ProposerCandidate.model_validate(payload)


def test_top_level_or_orphan_constraint_is_not_in_proposer_contract() -> None:
    payload = proposal_payload()
    payload["constraints"] = [
        {"subject": "invented_name", "kind": "nonnegative"}
    ]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProposerCandidate.model_validate(payload)


def v2_payload() -> dict[str, object]:
    return {
        "schema_version": "2",
        "candidate_id": "minimal_ode",
        "states": [
            {
                "name": "Gp",
                "kind": "observed",
                "observed_channel": "Gp",
                "rhs": "meal_effect + EGP - Uii - E",
                "constraints": [{"kind": "nonnegative"}],
                "mechanisms": ["glucose_balance"],
            },
            {
                "name": "meal_effect",
                "kind": "latent",
                "rhs": "meal_event_g - k_abs * meal_effect",
                "initial": {"fixed_value": 0.0},
                "mechanisms": ["meal_appearance"],
            },
        ],
        "algebraics": [],
        "parameters": [
            {"name": "k_abs", "bounds": {"lower": 0.001, "upper": 2.0}}
        ],
    }


def test_v2_round_trip_and_enrichment() -> None:
    proposal = ProposerCandidateV2.model_validate(v2_payload())
    restored = ProposerCandidateV2.model_validate_json(proposal.model_dump_json())

    candidate = enrich_proposal_v2(restored, ("Gp",))

    assert restored == proposal
    assert {item.state for item in candidate.state_equations} == {
        "Gp",
        "meal_effect",
    }
    assert candidate.observation_mappings[0].channel == "Gp"
    initials = {item.state: item for item in candidate.initial_conditions}
    assert initials["Gp"].expression == "Gp"
    assert initials["meal_effect"].fixed_value == 0.0
    assert candidate.constraints[0].subject == "Gp"
    assert candidate.constraints[0].source.value == "proposer"
    assert candidate.constraints[0].enforcement.value == "soft"
    assert candidate.states[0].mechanisms == ("glucose_balance",)
    assert candidate.states[1].mechanisms == ("meal_appearance",)
    assert candidate.parameters[0].bounds is None
    assert candidate.parameters[0].initialization_range is None


def test_proposer_parameter_schema_omits_numeric_ranges() -> None:
    schema = ProposerCandidateV2.model_json_schema(mode="validation")
    parameter = schema["$defs"]["ProposedParameter"]

    assert set(parameter["properties"]) == {"name", "role", "scope"}
    assert parameter["required"] == ["name"]


@pytest.mark.parametrize(
    ("kind", "updates", "message"),
    [
        ("latent", {"observed_channel": "Gp"}, "must omit observed_channel"),
        ("latent", {"initial": None}, "requires initial"),
    ],
)
def test_v2_state_rules(
    kind: str,
    updates: dict[str, object],
    message: str,
) -> None:
    payload = v2_payload()
    state = payload["states"][0]  # type: ignore[index]
    state["kind"] = kind
    if kind == "latent":
        state["observed_channel"] = None
        state["initial"] = {"fixed_value": 0.0}
    state.update(updates)

    with pytest.raises(ValidationError, match=message):
        ProposerCandidateV2.model_validate(payload)


def test_v2_discards_redundant_observed_initialization() -> None:
    payload = v2_payload()
    payload["states"][0]["initial"] = {"fixed_value": 5.0}  # type: ignore[index]

    proposal = ProposerCandidateV2.model_validate(payload)

    assert proposal.states[0].kind.value == "observed"
    assert proposal.states[0].initial is None


def test_v2_infers_same_named_observed_channel() -> None:
    payload = v2_payload()
    payload["states"][0].pop("observed_channel")  # type: ignore[index]

    proposal = ProposerCandidateV2.model_validate(payload)

    assert proposal.states[0].observed_channel == "Gp"


def test_v2_infers_target_from_observed_channel() -> None:
    candidate = enrich_proposal_v2(
        ProposerCandidateV2.model_validate(v2_payload()),
        ("Gp",),
    )

    assert candidate.observation_mappings[0].channel == "Gp"
    assert candidate.observation_mappings[0].expression == "Gp"


def test_v2_rejects_missing_or_ambiguous_contextual_target_match() -> None:
    proposal = ProposerCandidateV2.model_validate(v2_payload())
    with pytest.raises(ValueError, match=r"matches=\[\]"):
        enrich_proposal_v2(proposal, ("unknown_target",))

    ambiguous = v2_payload()
    ambiguous["states"][0]["name"] = "plasma_pool"  # type: ignore[index]
    ambiguous["algebraics"] = [{"name": "Gp", "expression": "plasma_pool"}]
    proposal = ProposerCandidateV2.model_validate(ambiguous)
    with pytest.raises(ValueError, match=r"matches=.*Gp.*plasma_pool"):
        enrich_proposal_v2(proposal, ("Gp",))
