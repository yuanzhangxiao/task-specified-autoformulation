"""Saved-request provenance, paired rejudging, no new fitting, deterministic resume."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.judging import extract_public_requirements, semantic_absolute_units
from autoformalism.judging.hybrid import FACTOR_SIGN_POLICY, build_atomic_evidence_plan
from autoformalism.llm import MockLLMClient
from autoformalism.rebuttal import judge_sign_recheck as campaign
from autoformalism.rebuttal import repair_scientific_judge as judge
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import (
    AtomicJudgeResult,
    CandidateModel,
    HybridJudgeResult,
    RelativeCriterion,
)
from autoformalism.staged_topology import content_hash
from tests.test_search_hybrid_pair import _candidate


def source_fixture(root: Path) -> dict:
    """Two self-pairs, reused by child stages, only one has faulty evidence."""
    old = sealed_write(
        root / "plan.json",
        {
            "protocol": "general-advisory-critic-1",
            "test_data_opened": False,
            "judge_revision": "a" * 40,
            "serving_image_sha256": "b" * 64,
            "rows": [
                {"task": {"task_id": task}} for task in ("affected", "unaffected")
            ],
            "training": "must not enter imported plan or judge payload",
        },
    )
    for task, rhs in (("affected", "-2*Gp"), ("unaffected", "-(2*Gp)")):
        model = _candidate(task, rhs)
        request = judge.review_request(
            model,
            model,
            ValidationContext(targets=("Gp",)),
            "Public task: predict Gp. Gp has a loss term.",
            0,
            "a" * 40,
        )
        key = content_hash(request)
        sealed_write(root / "judge_cache" / key / "request.json", {"request": request})
        review = {"request_sha256": key, "status": "reviewed", "findings": []}
        receipt = sealed_write(
            root / "judge_cache" / key / "receipt.json",
            {"request": request, "review": review},
        )
        for stage in ("parent", "child"):
            sealed_write(
                root / "results" / task / f"{stage}_review.json",
                {
                    "identity": old["artifact_sha256"],
                    "request_sha256": key,
                    "receipt_sha256": receipt["artifact_sha256"],
                    "review": review,
                    "candidate_sha256": campaign.model_hash(model),
                    "reused_parent_review": stage == "child",
                },
            )
    return old


def mock_client(request):
    """Fixed role labels let the deterministic polarity check be exercised."""
    models = [
        CandidateModel.model_validate(request[k]) for k in ("parent", "candidate")
    ]
    atomic = []
    for a, b in (models, models[::-1]):
        plan = build_atomic_evidence_plan(a, b, sign_policy=FACTOR_SIGN_POLICY)
        atomic.append(
            AtomicJudgeResult.model_validate(
                {
                    "signed_occurrence_assessments": [
                        {
                            "occurrence_id": o.occurrence_id,
                            "expected_direction": "negative_contribution",
                            "evidence": "Synthetic sink.",
                        }
                        for o in plan.occurrences
                    ],
                    "repeated_contribution_assessments": [],
                }
            )
        )
    hybrid = HybridJudgeResult.model_validate(
        {
            "absolute_assessments": [
                {
                    "criterion": c.value,
                    "subject_id": s,
                    "candidate_a": {"verdict": "pass", "evidence": "Synthetic."},
                    "candidate_b": {"verdict": "pass", "evidence": "Synthetic."},
                }
                for c, s in semantic_absolute_units(
                    extract_public_requirements(request["public_prompt"]),
                    include_role_consistency=False,
                )
            ],
            "comparative_assessments": [
                {"criterion": c.value, "verdict": "tie", "evidence": "Identical."}
                for c in RelativeCriterion
            ],
        }
    )
    return MockLLMClient(atomic_responses=atomic, hybrid_responses=[hybrid, hybrid])


def smoke(tmp_path: Path) -> dict:
    """Exercise real import, atomic/comparative orchestration, report and resume."""
    source, root = tmp_path / "source", tmp_path / "recheck"
    source_fixture(source)
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    plan = campaign.freeze(source, root)
    assert campaign.freeze(source, root) == plan
    assert "must not enter" not in json.dumps(plan)
    assert len(plan["placements"]) == 4 and len(plan["reviews"]) == 2
    assert sum(r["affected"] for r in plan["reviews"]) == 1
    request = next(r["request"] for r in plan["reviews"] if r["affected"])
    client = mock_client(request)
    with patch.object(judge, "create_llm_client", return_value=client):
        results = [campaign.run_one(root, i, "http://unused") for i in range(2)]
        assert {r["status"] for r in results} == {"reviewed", "unchanged_evidence"}
        assert [campaign.run_one(root, i, "http://unused") for i in range(2)] == results
    assert len(client.calls) == 4
    for call in client.calls:
        payload = json.loads(call["user_prompt"])
        assert "training" not in payload and "validation" not in payload
        if call["role"] == "atomic_evidence_judge":
            assert "actual_polarity" not in call["user_prompt"]
            assert "atomic-evidence-plan-2-outer-factors" in call["user_prompt"]
        else:
            for facts in payload["deterministic_structural_facts"].values():
                assert FACTOR_SIGN_POLICY in facts["schema_version"]
                assert (
                    facts["algebraic_expressions"]["equation:Gp"][
                        "top_level_additive_terms"
                    ][0]["polarity"]
                    == "negative"
                )
    summary = campaign.report(root)
    assert summary["affected_unique_reviews"] == 1
    assert (
        summary["model_changes"]
        == summary["optimizer_calls"]
        == summary["selection_changes"]
        == 0
    )
    assert before == {str(p): p.read_bytes() for p in source.rglob("*.json")}
    assert (root / "SUMMARY.md").exists()
    return {
        "status_counts": summary["status_counts"],
        "mock_calls": len(client.calls),
        "historical_files_unchanged": True,
    }


def test_offline_smoke(tmp_path):
    smoke(tmp_path)


@pytest.mark.parametrize(
    "fault", ["request", "receipt", "stage", "missing", "extra_key"]
)
def test_source_corruption_is_not_rejudged(tmp_path, fault):
    source_fixture(tmp_path)
    path = tmp_path / "results/affected/parent_review.json"
    stage = sealed_read(path)
    cache = tmp_path / "judge_cache" / stage["request_sha256"]
    if fault == "missing":
        path.unlink()
        with pytest.raises(FileNotFoundError):
            campaign.snapshot(tmp_path)
        return
    if fault in {"request", "extra_key"}:
        path = cache / "request.json"
        value = sealed_read(path)
        if fault == "extra_key":
            value["request"]["validation_nmse"] = 0.1
        else:
            value["request"]["seed"] = 9
    elif fault == "receipt":
        path = cache / "receipt.json"
        value = sealed_read(path)
        value["review"]["status"] = "unavailable"
    else:
        value = stage
        value["candidate_sha256"] = "0" * 64
    path.unlink()
    sealed_write(path, {k: v for k, v in value.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError):
        campaign.snapshot(tmp_path)


def test_new_protocol_cannot_reuse_old_review_or_marker(tmp_path):
    source_fixture(tmp_path / "source")
    row = campaign.snapshot(tmp_path / "source")["reviews"][0]
    request = campaign.corrected_request(row["request"])
    assert content_hash(request) != row["historical_request_sha256"]
    for key in campaign.REQUEST_KEYS - {"protocol"}:
        assert request[key] == row["request"][key]
    tmp_path.joinpath("review.json").write_text(json.dumps(row["historical_review"]))
    with pytest.raises(ValueError, match="identity"):
        judge.perform_review(request, tmp_path, "http://unused")
    tmp_path.joinpath("review.json").unlink()
    tmp_path.joinpath("review_started.json").write_text(
        json.dumps({"request_sha256": row["historical_request_sha256"]})
    )
    with pytest.raises(ValueError, match="identity"):
        judge.perform_review(request, tmp_path, "http://unused")


def test_interruption_and_partial_costs_do_not_reset_budget(tmp_path):
    source, root = tmp_path / "source", tmp_path / "root"
    source_fixture(source)
    plan = campaign.freeze(source, root)
    row = plan["reviews"][0]
    cache = root / "reviews" / row["historical_request_sha256"] / "judge"
    request = campaign.corrected_request(row["request"])
    cache.mkdir(parents=True)
    campaign.public._write(
        cache / "review_started.json", {"request_sha256": content_hash(request)}
    )
    cache.joinpath("events.jsonl").write_text(
        json.dumps(
            {
                "event": "llm_response",
                "request_hash": "synthetic",
                "provider_attempts": 1,
                "usage": {"total_tokens": 123},
            }
        )
        + "\n"
    )
    assert campaign.report(root)["cost"]["observed_total_tokens"] == 123
    with patch.object(
        judge, "create_llm_client", side_effect=AssertionError("no retry")
    ):
        result = campaign.run_one(root, 0, "http://unused")
        assert result["status"] == "interrupted"
        assert campaign.run_one(root, 0, "http://unused") == result
    assert campaign.report(root)["cost"]["observed_total_tokens"] == 123


def test_resume_detects_changed_source_or_code(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "root"
    source_fixture(source)
    campaign.freeze(source, root)
    with pytest.raises(ValueError, match="separate"):
        campaign.freeze(source, source / "nested")
    monkeypatch.setattr(campaign, "launcher_hash", lambda: "changed")
    with pytest.raises(ValueError, match="identity"):
        campaign.verify(root)


def test_submission_is_durable_and_has_no_fits(tmp_path, monkeypatch):
    from scripts import submit_judge_sign_recheck as submitter

    source, root = tmp_path / "source", tmp_path / "root"
    source_fixture(source)
    monkeypatch.setattr(submitter, "source_commit", lambda _: "c" * 40)
    monkeypatch.setenv("AF_COMMIT", "c" * 40)
    monkeypatch.setenv("AF_PYTHON", __file__)
    monkeypatch.setenv("AF_VLLM_IMAGE", __file__)
    monkeypatch.setenv("AF_HF_HOME", str(tmp_path / "hf"))
    calls = []

    def queue(*args):
        calls.append(args)
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", queue)
    result = submitter.submit(source, root)
    assert list(result["jobs"]) == ["prepare", "review", "report"]
    assert result["maximum_new_paired_reviews"] == 1
    assert "--gres=gpu:h100:2" in calls[1][2]
    assert submitter.submit(source, root) == result and len(calls) == 3
    root.joinpath("submission_manifest.json").unlink()
    with pytest.raises(ValueError, match="partial submission"):
        submitter.submit(source, root)
    assert len(calls) == 3


def test_closed_request_cannot_smuggle_numerical_context(tmp_path):
    source_fixture(tmp_path)
    request = copy.deepcopy(campaign.snapshot(tmp_path)["reviews"][0]["request"])
    request["fit"] = {"normalized_mse": 0.001}
    with pytest.raises(ValueError, match="closed"):
        campaign.corrected_request(request)
