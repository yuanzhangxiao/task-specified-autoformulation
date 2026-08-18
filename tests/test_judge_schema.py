"""Judge result schema tests."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from autoformalism.schemas import (
    ComparativeJudgeResult,
    JudgeResult,
    ScientificJudgeResult,
    parse_judge_assessment,
)


@pytest.fixture
def judge_payload() -> dict[str, Any]:
    """Return a complete valid judge response."""
    return {
        "schema_version": "1",
        "hard_red_flags": [
            {
                "code": "target_leakage",
                "description": "The target appears as an exogenous forcing.",
                "evidence": "The target equation contains measured_target(t).",
            }
        ],
        "category_scores": {
            "task_output_coverage": 0.5,
            "mechanism_state_adequacy": 0.5,
            "mathematical_completeness": 0.5,
            "data_causal_consistency": 0.25,
            "constraint_compliance": 0.5,
            "parsimony_interpretability": 0.8,
        },
        "aggregate_score": 0.45,
        "missing_requirements": [
            "A generated causal input-response mechanism is missing."
        ],
        "actionable_edits": [
            {
                "target": "target_state",
                "instruction": "Replace target forcing with a generated state.",
                "priority": "required",
            }
        ],
    }


def test_judge_json_round_trip(judge_payload: dict[str, Any]) -> None:
    result = JudgeResult.model_validate(judge_payload)

    restored = JudgeResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.category_scores["data_causal_consistency"].score == 0.25
    assert restored.numeric_category_scores["data_causal_consistency"] == 0.25
    assert restored.actionable_edits[0].priority.value == "required"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aggregate_score", -0.01),
        ("aggregate_score", 1.01),
        ("aggregate_score", float("nan")),
    ],
)
def test_judge_rejects_invalid_aggregate_score(
    judge_payload: dict[str, Any],
    field: str,
    value: float,
) -> None:
    payload = copy.deepcopy(judge_payload)
    payload[field] = value

    with pytest.raises(ValidationError):
        JudgeResult.model_validate(payload)


def test_judge_rejects_invalid_or_empty_category_scores(
    judge_payload: dict[str, Any],
) -> None:
    invalid = copy.deepcopy(judge_payload)
    invalid["category_scores"]["data_causal_consistency"] = 2.0
    with pytest.raises(ValidationError):
        JudgeResult.model_validate(invalid)

    empty = copy.deepcopy(judge_payload)
    empty["category_scores"] = {}
    with pytest.raises(ValidationError, match="Field required"):
        JudgeResult.model_validate(empty)


def test_judge_rejects_extra_fields(judge_payload: dict[str, Any]) -> None:
    payload = copy.deepcopy(judge_payload)
    payload["fit_error"] = 0.01

    with pytest.raises(ValidationError, match="extra_forbidden"):
        JudgeResult.model_validate(payload)


def test_judge_accepts_score_justification_objects(
    judge_payload: dict[str, Any],
) -> None:
    judge_payload["category_scores"]["data_causal_consistency"] = {
        "score": 0.4,
        "justification": "Uses only causal history.",
    }

    result = JudgeResult.model_validate(judge_payload)

    assert result.category_scores["data_causal_consistency"].score == 0.4
    assert result.category_scores["data_causal_consistency"].justification == (
        "Uses only causal history."
    )


def test_hard_flag_description_defaults_to_evidence_only_contract(
    judge_payload: dict[str, Any],
) -> None:
    del judge_payload["hard_red_flags"][0]["description"]

    result = JudgeResult.model_validate(judge_payload)

    assert result.hard_red_flags[0].description == "unspecified"


def test_judge_schema_is_openai_structured_output_compatible() -> None:
    """The wire schema must avoid unsupported arbitrary-key map keywords."""
    schema = JudgeResult.model_json_schema(mode="validation")
    encoded = json.dumps(schema)

    assert '"propertyNames"' not in encoded
    category_schema = schema["$defs"]["CategoryScores"]
    assert set(category_schema["required"]) == {
        "task_output_coverage",
        "mechanism_state_adequacy",
        "mathematical_completeness",
        "data_causal_consistency",
        "constraint_compliance",
        "parsimony_interpretability",
    }
    assert category_schema["additionalProperties"] is False


@pytest.fixture
def scientific_judge_payload() -> dict[str, Any]:
    """Return a complete scientific-only v2 response."""
    return {
        "schema_version": "2",
        "hard_red_flags": [],
        "category_scores": {
            "mechanistic_coherence": 0.9,
            "source_sink_balance_semantics": 0.8,
            "dynamic_plausibility": 0.7,
            "mechanism_coupling_task_sufficiency": 0.6,
            "nonredundancy_accounting": 0.5,
            "latent_state_complexity_justification": 0.4,
        },
        "missing_requirements": [],
        "actionable_edits": [],
    }


def test_scientific_judge_round_trip_and_runtime_aggregate(
    scientific_judge_payload: dict[str, Any],
) -> None:
    result = ScientificJudgeResult.model_validate(scientific_judge_payload)
    restored = ScientificJudgeResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.aggregate_score == pytest.approx(0.69)
    assert "aggregate_score" not in restored.model_dump()


def test_scientific_judge_rejects_provider_aggregate(
    scientific_judge_payload: dict[str, Any],
) -> None:
    scientific_judge_payload["aggregate_score"] = 1.0

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ScientificJudgeResult.model_validate(scientific_judge_payload)


def test_scientific_judge_schema_has_only_scientific_categories() -> None:
    schema = ScientificJudgeResult.model_json_schema(mode="validation")
    categories = schema["$defs"]["ScientificCategoryScores"]

    assert set(categories["required"]) == {
        "mechanistic_coherence",
        "source_sink_balance_semantics",
        "dynamic_plausibility",
        "mechanism_coupling_task_sufficiency",
        "nonredundancy_accounting",
        "latent_state_complexity_justification",
    }
    assert "aggregate_score" not in schema["properties"]
    assert categories["additionalProperties"] is False


def test_checkpoint_parser_accepts_v1_and_v2(
    judge_payload: dict[str, Any],
    scientific_judge_payload: dict[str, Any],
) -> None:
    assert isinstance(parse_judge_assessment(judge_payload), JudgeResult)
    assert isinstance(
        parse_judge_assessment(scientific_judge_payload), ScientificJudgeResult
    )


def _comparative_payload(verdict: str = "candidate_a") -> dict[str, Any]:
    answer = {"verdict": verdict, "evidence": "A cited equation differs."}
    return {
        "schema_version": "comparative-1",
        "answers": {
            "claimed_mechanisms_represented": answer,
            "task_inputs_connected_to_targets": answer,
            "claimed_processes_connected_to_balances": answer,
            "source_terms_have_consistent_signs": answer,
            "sink_terms_have_consistent_signs": answer,
            "fluxes_not_duplicated": answer,
            "components_not_disconnected": answer,
            "mechanisms_not_conflicting": answer,
            "latent_states_have_incoming_pathways": answer,
            "latent_states_have_outgoing_influence": answer,
            "latent_accumulators_have_relaxation_or_justification": answer,
            "claimed_decay_opposes_accumulated_quantity": answer,
            "claimed_delay_has_drive_and_relaxation": answer,
            "claimed_saturation_is_structurally_bounded": answer,
        },
    }


def test_comparative_judge_computes_atomic_preference_and_uncertainty() -> None:
    payload = _comparative_payload()
    payload["answers"]["fluxes_not_duplicated"] = {
        "verdict": "candidate_b",
        "evidence": "B avoids a duplicated term.",
    }
    payload["answers"]["latent_states_have_incoming_pathways"] = {
        "verdict": "tie",
        "evidence": "Neither candidate adds a latent state.",
    }
    payload["answers"]["components_not_disconnected"] = {
        "verdict": "indeterminate",
        "evidence": "The public task does not distinguish these roles.",
    }

    payload["answers"]["claimed_delay_has_drive_and_relaxation"] = {
        "verdict": "not_applicable",
        "evidence": "Neither candidate claims a delay state.",
    }
    result = ComparativeJudgeResult.model_validate(payload)

    assert result.numeric_preference == pytest.approx(10.5 / 12.0)
    assert result.indeterminate_rate == pytest.approx(1.0 / 14.0)
    assert result.not_applicable_rate == pytest.approx(1.0 / 14.0)
    assert "numeric_preference" not in result.model_dump()


def test_comparative_judge_requires_every_atomic_question() -> None:
    payload = _comparative_payload()
    del payload["answers"]["fluxes_not_duplicated"]

    with pytest.raises(ValidationError, match="Field required"):
        ComparativeJudgeResult.model_validate(payload)
