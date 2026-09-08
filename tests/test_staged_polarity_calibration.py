"""Frozen, cached tests for topology-stage interaction-polarity calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.llm.staged_topology import StagedTopologyClient
from autoformalism.rebuttal.staged_polarity_calibration import (
    audit_polarity_reply,
    freeze_polarity_calibration,
    polarity_calibration_source,
    run_polarity_calibration,
)


def _response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 100},
    }


def _valid_reply() -> dict[str, object]:
    return {
        "terms": [
            {
                "sources": ["u"],
                "outer_weight_sign": "positive",
                "scientific_role": "activating input drive",
            },
            {
                "sources": ["x"],
                "outer_weight_sign": "negative",
                "scientific_role": "stabilizing self-relaxation",
            },
            {
                "sources": ["q"],
                "outer_weight_sign": "unrestricted",
                "scientific_role": "direction-unspecified context effect",
            },
            {
                "sources": [],
                "outer_weight_sign": "unrestricted",
                "scientific_role": "unknown signed baseline",
            },
        ],
        "inventory_revision": None,
    }


def _config(path: Path, *, attempts: int = 2) -> Path:
    payload = {
        "protocol": "scientific-staged-polarity-calibration-1",
        "purpose": "test",
        "platform": "aces-h100x1",
        "serving_image_sha256": "a" * 64,
        "model_settings": {
            "model": "openai/gpt-oss-20b",
            "model_revision": "revision",
            "reasoning_effort": "low",
            "temperature": 0.2,
            "max_output_tokens": 1024,
            "timeout_seconds": 1,
            "maximum_requests": 4,
            "maximum_total_tokens": 65536,
            "attempts_per_step": attempts,
        },
        "served_context_tokens": 16384,
        "seeds": [0],
        "gates": {
            "minimum_response_success": 1.0,
            "minimum_exact_source_coverage": 1.0,
            "minimum_polarity_accuracy": 1.0,
            "minimum_fixed_polarity_accuracy": 1.0,
            "minimum_unrestricted_accuracy": 1.0,
        },
        "wall_seconds": 60,
        "shutdown_margin_seconds": 30,
    }
    path.write_text(json.dumps(payload))
    return path


def test_source_separates_public_evidence_from_evaluator_labels() -> None:
    source = polarity_calibration_source()
    brief = json.dumps(source["brief"])
    assert "explicitly activating" in brief
    assert "stabilizing self-relaxation" in brief
    assert "direction is unspecified" in brief
    assert "expected_outer_weight_sign" not in brief
    assert len(source["expected_obligations"]) == 4


def test_audit_matches_source_sets_without_using_role_wording() -> None:
    source = polarity_calibration_source()
    terms = list(reversed(_valid_reply()["terms"]))
    for term in terms:
        term["scientific_role"] = "different concise wording"
    audit = audit_polarity_reply(terms, source["expected_obligations"])
    assert audit["exact_source_coverage"]
    assert audit["correct_polarity_count"] == 4
    assert audit["calibration_pass"]


def test_duplicate_or_extra_source_sets_fail_exact_coverage() -> None:
    source = polarity_calibration_source()
    terms = list(_valid_reply()["terms"])
    terms.append(
        {
            "sources": ["u", "q"],
            "outer_weight_sign": "positive",
            "scientific_role": "extra",
        }
    )
    terms.append(dict(terms[0]))
    audit = audit_polarity_reply(terms, source["expected_obligations"])
    assert not audit["exact_source_coverage"]
    assert audit["extra_source_sets"] == [["q", "u"]]
    assert audit["duplicate_source_sets"] == [["u"]]
    assert not audit["calibration_pass"]


def test_schema_valid_scientific_mistake_is_scored_not_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    freeze_polarity_calibration(
        _config(tmp_path / "config.json", attempts=2), plan_path
    )
    calls: list[dict[str, object]] = []
    wrong = _valid_reply()
    wrong["terms"][2]["outer_weight_sign"] = "positive"

    def transport(
        url: str, body: dict[str, object], timeout: float
    ) -> dict[str, object]:
        del url, timeout
        calls.append(body)
        return _response(wrong)

    import autoformalism.rebuttal.staged_polarity_calibration as campaign

    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(**kwargs, transport=transport),
    )
    summary = run_polarity_calibration(
        plan_path, tmp_path / "results", "http://localhost:8000"
    )
    assert summary["status"] == "complete"
    assert summary["response_success"] == 1.0
    assert summary["unrestricted_accuracy"] == 0.5
    assert summary["overall_result"] == "fail"
    assert len(calls) == 1


def test_campaign_repairs_contract_error_then_caches_complete_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    plan = freeze_polarity_calibration(_config(tmp_path / "config.json"), plan_path)
    assert len(plan["tasks"]) == 1
    assert not plan["expected_labels_disclosed_to_provider"]
    assert not plan["parameter_fitting_performed"]
    calls: list[dict[str, object]] = []

    def transport(
        url: str, body: dict[str, object], timeout: float
    ) -> dict[str, object]:
        del url, timeout
        messages = body["messages"]
        assert isinstance(messages, list)
        public_request = messages[1]["content"]
        assert "expected_outer_weight_sign" not in public_request
        calls.append(body)
        if len(calls) == 1:
            invalid = _valid_reply()
            del invalid["terms"][0]["outer_weight_sign"]
            return _response(invalid)
        assert "runtime_diagnostics" in public_request
        return _response(_valid_reply())

    import autoformalism.rebuttal.staged_polarity_calibration as campaign

    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(**kwargs, transport=transport),
    )
    output = tmp_path / "results"
    first = run_polarity_calibration(plan_path, output, "http://localhost:8000")
    assert first["status"] == "complete"
    assert first["overall_result"] == "pass"
    assert first["polarity_accuracy"] == 1.0
    assert first["fixed_polarity_accuracy"] == 1.0
    assert first["unrestricted_accuracy"] == 1.0
    assert first["physical_requests"] == 2
    result = json.loads((output / "polarity_seed0/result.json").read_text())
    assert result["events"] == [
        {
            "accepted": False,
            "attempt": 0,
            "error": result["events"][0]["error"],
        },
        {"accepted": True, "attempt": 1, "error": None},
    ]
    second = run_polarity_calibration(plan_path, output, "http://localhost:8000")
    assert second == first
    assert len(calls) == 2


def test_freeze_refuses_to_overwrite_a_different_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        freeze_polarity_calibration(_config(tmp_path / "config.json"), plan_path)
