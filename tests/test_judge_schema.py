"""Judge result schema tests."""

from __future__ import annotations

import copy
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
            "causality": 0.25,
            "task_compliance": 0.5,
            "parsimony": 0.8,
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
    assert restored.category_scores["causality"] == 0.25
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
    invalid["category_scores"]["causality"] = 2.0
    with pytest.raises(ValidationError):
        JudgeResult.model_validate(invalid)

    empty = copy.deepcopy(judge_payload)
    empty["category_scores"] = {}
    with pytest.raises(ValidationError, match="at least 1"):
        JudgeResult.model_validate(empty)


def test_judge_rejects_extra_fields(judge_payload: dict[str, Any]) -> None:
    payload = copy.deepcopy(judge_payload)
    payload["fit_error"] = 0.01

    with pytest.raises(ValidationError, match="extra_forbidden"):
        JudgeResult.model_validate(payload)

