"""Offline complete-suite checks and real transport/schema orchestration."""

import csv
import json
from copy import deepcopy
from unittest.mock import patch

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm import jetstream
from autoformalism.llm.exceptions import LLMProviderError
from autoformalism.rebuttal import judge_jetstream_calibration as campaign
from autoformalism.rebuttal import judge_jetstream_calibration_report as reporting
from autoformalism.rebuttal import repair_scientific_judge as production
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedAbsoluteLabel,
    HybridCalibrationLabels,
    mutation_label_contract,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from scripts.build_hybrid_judge_consensus_validation_pairs import (
    build_consensus_validation_pairs,
)
from tests.test_hybrid_consensus_validation import _candidate
from tests.test_hybrid_symmetric_aggregation import _fixture
from tests.test_judge_jetstream import TOKEN, Endpoint
from tests.test_judge_sign_recheck import mock_client


@pytest.fixture
def bundle(tmp_path):
    context = ValidationContext(
        targets=("Gp", "U"),
        auxiliaries=("EGP",),
        external_inputs=("meal_event_g", "insulin_input"),
    )
    pairs = build_consensus_validation_pairs(
        [(name, _candidate(name), "phase_b_test", "easy") for name in ("k", "q")],
        contexts={("phase_b_test", "easy"): context},
    )
    labels = []
    for pair in pairs:
        contract = mutation_label_contract(pair.mutation_type)
        labels.append(
            HybridCalibrationLabels(
                pair_id=pair.pair_id,
                overall_preference=contract.overall_preference,
                absolute_labels=tuple(
                    ExpectedAbsoluteLabel(
                        **item.model_dump(),
                        label_source="mutation_contract:" + pair.mutation_type,
                    )
                    for item in contract.absolute
                ),
                comparative_labels=(),
            ).model_dump(mode="json")
        )
    failures = [
        {
            "pair_id": p.pair_id,
            "judge_model": campaign.MODEL,
            "repetition": rep,
            "order": order,
        }
        for p in pairs
        for rep in range(5)
        for order in campaign.ORDERS
    ]
    return {
        "protocol": campaign.PROTOCOL,
        "source_config": json.loads(campaign.CONFIG.read_text()),
        "pairs": [p.model_dump(mode="json") for p in pairs],
        "labels": labels,
        "contexts": {
            "phase_b_test/easy": {
                "public_prompt": "Predict Gp and U using the supplied public inputs.",
                "context": context.model_dump(mode="json"),
            }
        },
        "historical_rows": [],
        "historical_failures": failures,
        "source_files": {},
        "test_data_opened": False,
    }


def _freeze(tmp_path, bundle):
    source = tmp_path / "source.json"
    sealed_write(source, bundle)
    root = tmp_path / "calibration"
    plan = campaign.freeze(source, root)
    assert campaign.freeze(source, root) == plan
    return root, plan


def _request(bundle):
    pair = bundle["pairs"][0]
    return {
        "parent": pair["valid_candidate"],
        "candidate": pair["adversarial_candidate"],
        "public_prompt": bundle["contexts"]["phase_b_test/easy"]["public_prompt"],
    }


def test_export_reuses_cases_and_reads_only_original_public_context(
    tmp_path, bundle, monkeypatch
):
    source = tmp_path / "historical"
    (source / "gpt-oss-120b").mkdir(parents=True)
    for key, name in (
        ("pairs", "pairs.jsonl"),
        ("labels", "hybrid_labels.jsonl"),
        ("historical_failures", "gpt-oss-120b/hybrid_judge_failures.jsonl"),
    ):
        (source / name).write_text("".join(json.dumps(r) + "\n" for r in bundle[key]))
    (source / "protocol_config.json").write_text(json.dumps(bundle["source_config"]))
    (source / "consensus_validation_pairs_manifest.json").write_text(
        json.dumps(
            {
                "selected_baseline_count": 2,
                "pair_count": 14,
                "selected_pair_ids": [p["pair_id"] for p in bundle["pairs"]],
            }
        )
    )
    names = (
        "pairs.jsonl",
        "hybrid_labels.jsonl",
        "protocol_config.json",
        "consensus_validation_pairs_manifest.json",
    )
    (source / "frozen_inputs.sha256").write_text(
        "".join(f"{campaign._hash(source / name)}  {source / name}\n" for name in names)
    )
    with (source / "gpt-oss-120b/hybrid_judge_scores.csv").open("w") as stream:
        csv.writer(stream).writerow(["pair_id", "judge_model", "order", "repetition"])
    calls = []
    entry = bundle["contexts"]["phase_b_test/easy"]

    def context(root, pair):
        calls.append((root, pair.benchmark_id))
        return entry["public_prompt"], ValidationContext.model_validate(
            entry["context"]
        )

    monkeypatch.setattr(campaign, "_task_context", context)
    value = campaign.export(source, tmp_path / "public", tmp_path / "export.json")
    assert len(calls) == 1
    assert value["pairs"] == bundle["pairs"]
    assert value["labels"] == bundle["labels"]
    assert len(value["source_files"]) == 7
    assert not value["test_data_opened"]
    (source / "pairs.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="historically frozen input differs"):
        campaign.export(source, tmp_path / "public", tmp_path / "other.json")


@pytest.mark.parametrize(
    "damage", ["pairs", "labels", "historical", "config", "unlabeled"]
)
def test_incomplete_or_altered_inputs_rejected(tmp_path, bundle, damage):
    if damage in ("pairs", "labels"):
        bundle[damage].pop()
    elif damage == "historical":
        bundle["historical_failures"].pop()
    elif damage == "config":
        bundle["source_config"]["protocol"]["repetitions"] = 1
    else:
        bundle["labels"][-1]["overall_preference"] = "baseline"
    with pytest.raises(ValueError):
        _freeze(tmp_path, bundle)


def test_production_stages_blind_labels_seed_and_resume(tmp_path, bundle, monkeypatch):
    root, plan = _freeze(tmp_path, bundle)
    schedule = list(campaign.units(bundle))
    endpoint = Endpoint(_request(bundle))
    # Real API adapter, cache, validation and both orientations for one seed.
    monkeypatch.setattr(campaign, "units", lambda _: iter(schedule[:2]))
    with patch.object(jetstream.urllib.request, "build_opener", return_value=endpoint):
        campaign.run(root, lambda: TOKEN, lambda _: None)
        campaign.run(
            root, lambda: pytest.fail("must not read credentials"), lambda _: None
        )
    assert len(endpoint.requests) == 4
    request = production.review_request(
        CandidateModel.model_validate(bundle["pairs"][0]["valid_candidate"]),
        CandidateModel.model_validate(bundle["pairs"][0]["adversarial_candidate"]),
        ValidationContext.model_validate(
            bundle["contexts"]["phase_b_test/easy"]["context"]
        ),
        _request(bundle)["public_prompt"],
        0,
        "0" * 40,
        sign_policy=campaign.SIGN_POLICY,
    )
    mock = mock_client(request)
    production.perform_review(
        request, tmp_path / "production", "http://unused", client_factory=lambda _: mock
    )
    for payload, call in zip(endpoint.requests, mock.calls, strict=True):
        assert payload["messages"] == [
            {"role": "system", "content": call["system_prompt"]},
            {"role": "user", "content": call["user_prompt"]},
        ]
    for payload in endpoint.requests:
        text = json.dumps(payload["messages"])
        assert "mutation_contract" not in text
        assert "algebraic_reordering_equivalent" not in text
        assert "overall_preference" not in text
        assert bundle["pairs"][0]["pair_id"] not in text
        assert payload["seed"] == 10000
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 6144
    result = sealed_read(root / "orientations" / schedule[0][0] / "result.json")
    assert result["status"] == "reviewed"
    # Full planned denominator, even when only the first pair was attempted.
    monkeypatch.setattr(campaign, "units", lambda _: iter(schedule))
    summary = reporting.report(root)
    assert summary["status_counts"] == {"reviewed": 2, "pending": 138}
    assert summary["known_case_gate_passed"] is None
    assert summary["current"]["aggregation"]["paired_response_coverage"] == 1 / 70
    assert summary["cost"]["physical_requests_started"] == 4
    assert summary["cost"]["observed_total_tokens"] == 480
    assert summary["cost"]["usage_complete"]
    assert not summary["fresh_holdout_calibration_established"]
    assert plan["maximum_physical_requests"] == 2800
    assert "distinct_seed_attempts" not in plan["scientific_protocol"]
    assert TOKEN not in "".join(p.read_text() for p in root.rglob("*.json"))


def test_interrupted_unit_is_terminal_remaining_units_resume(
    tmp_path, bundle, monkeypatch
):
    root, plan = _freeze(tmp_path, bundle)
    schedule = list(campaign.units(bundle))[:2]
    monkeypatch.setattr(campaign, "units", lambda _: iter(schedule))
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign, "_assess", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run(root, lambda: TOKEN, lambda _: None)
    first = root / "orientations" / schedule[0][0]
    assert sealed_read(first / "result.json")["status"] == "interrupted"
    # Simulate a process kill on the second orientation, with unknown HTTP usage.
    second = root / "orientations" / schedule[1][0]
    sealed_write(
        second / "started.json",
        {"identity": {"plan": plan["artifact_sha256"], "unit": schedule[1][0]}},
    )
    call = second / "calls/call-000"
    call.mkdir(parents=True)
    (call / "started.json").write_text("{}")
    campaign.run(root, lambda: pytest.fail("no new requests"), lambda _: None)
    assert count == 1
    assert sealed_read(second / "result.json")["status"] == "interrupted"
    cost = reporting._cost(root)
    assert cost["usage_missing_events"] == 1
    assert not cost["usage_complete"]


def test_provider_failure_stops_before_consuming_remaining_suite(
    tmp_path, bundle, monkeypatch
):
    root, _ = _freeze(tmp_path, bundle)

    def unavailable(*args):
        raise LLMProviderError("synthetic provider failure", retryable=False)

    monkeypatch.setattr(campaign, "_assess", unavailable)
    campaign.run(root, lambda: TOKEN, lambda _: None)
    summary = reporting.report(root)
    assert summary["status_counts"] == {"unavailable": 1, "pending": 139}
    assert summary["known_case_gate_passed"] is None
    assert summary["current"]["absolute_coverage"]["missing_units"] > 0


def test_false_passes_and_missing_are_separate(bundle):
    labels = {
        r["pair_id"]: HybridCalibrationLabels.model_validate(r)
        for r in bundle["labels"]
    }
    label = next(
        item
        for item in labels.values()
        if any(u.mutated.value == "fail" for u in item.absolute_labels)
    )
    unit = next(u for u in label.absolute_labels if u.mutated.value == "fail")
    value = reporting._accuracy(
        {
            "trials": [
                {
                    "pair_id": label.pair_id,
                    "repetition": 0,
                    "consensus_absolute_assessments": [
                        {
                            "criterion": unit.criterion.value,
                            "subject_id": unit.subject_id,
                            "candidate_a": {"verdict": "pass"},
                            "candidate_b": {"verdict": "pass"},
                        }
                    ],
                }
            ]
        },
        labels,
    )
    assert value["known_defects_answered_pass"] == 1
    assert value["known_defects_missing"] > 0
    assert value["known_defects_detected"] == 0


def test_plan_drift_and_result_cross_link_rejected(tmp_path, bundle, monkeypatch):
    root, _ = _freeze(tmp_path, bundle)
    monkeypatch.setattr(campaign.jetstream, "POLICY", "changed")
    with pytest.raises(ValueError, match="identity differs"):
        campaign.verify(root)


def test_full_schedule_reports_all_trials_and_resume_does_no_work(
    tmp_path, bundle, monkeypatch
):
    root, _ = _freeze(tmp_path, bundle)
    templates = _fixture()[0]
    calls = []

    def assess(raw, entry, repetition, order, work, transport):
        calls.append((raw["pair_id"], repetition, order))
        row = deepcopy(next(r for r in templates if r["order"] == order))
        row.update(
            pair_id=raw["pair_id"],
            mutation_type=raw["mutation_type"],
            repetition=str(repetition),
            judge_model=campaign.MODEL,
        )
        return row

    monkeypatch.setattr(campaign, "_assess", assess)
    campaign.run(root, lambda: TOKEN, lambda _: None)
    summary = reporting.report(root)
    assert len(calls) == 140 and len(set(calls)) == 140
    assert summary["status"] == "complete"
    assert summary["status_counts"] == {"reviewed": 140}
    assert isinstance(summary["known_case_gate_passed"], bool)
    assert summary["current"]["aggregation"]["paired_response_coverage"] == 1
    assert summary["current"]["gate_assessment"]["unlabeled_tradeoff_trials"] == 10
    campaign.run(root, lambda: pytest.fail("no key on resume"), lambda _: None)
    assert len(calls) == 140
    # Results cannot be silently associated with another run identity.
    name = next(campaign.units(bundle))[0]
    path = root / "orientations" / name / "result.json"
    wrong = sealed_read(path)
    wrong.pop("artifact_sha256")
    wrong["identity"]["plan"] = "another plan"
    path.unlink()
    sealed_write(path, wrong)
    with pytest.raises(ValueError, match="another plan"):
        reporting.report(root)
