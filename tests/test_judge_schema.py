"""Judge result schema tests."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from autoformalism.schemas import JudgeResult


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
