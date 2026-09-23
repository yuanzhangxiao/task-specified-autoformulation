"""Cross-runtime provenance, identical judge requests, durable Delta submission."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autoformalism.rebuttal import judge_sign_delta as delta
from autoformalism.rebuttal import judge_sign_recheck as original
from autoformalism.rebuttal import repair_scientific_judge as judge
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from tests.test_judge_sign_recheck import mock_client, source_fixture


def exported_plan(tmp_path: Path) -> Path:
    """Represent a transported plan whose original runtime and paths are unavailable."""
    source = tmp_path / "m4"
    source_fixture(source)
    value = original.freeze(source, tmp_path / "aces")
    value = {k: v for k, v in value.items() if k != "artifact_sha256"}
    value.update(source="/missing/aces/m4", runtime={"python": "3.11.5"})
    path = tmp_path / "export.json"
    sealed_write(path, value)
    return path


def smoke(tmp_path: Path) -> dict:
    """Exercise real portable import, both judge stages and zero-call resume offline."""
    source = exported_plan(tmp_path)
    before = source.read_bytes()
    root = tmp_path / "delta"
    plan = delta.freeze(source, root, "c" * 64)
    assert delta.freeze(source, root, "c" * 64) == plan
    assert delta.verify(root) == plan
    assert plan["execution"]["same_image_bytes"] is False
    assert plan["execution"]["origin_runtime"] == {"python": "3.11.5"}
    assert (
        sealed_read(root / "origin_plan.json")["artifact_sha256"]
        == plan["origin_plan_sha256"]
    )
    row = plan["reviews"][0]
    client = mock_client(original.corrected_request(row["request"]))
    with patch.object(judge, "create_llm_client", return_value=client):
        results = [delta.run_one(root, i, "http://unused") for i in range(2)]
        assert [r["status"] for r in results] == ["reviewed", "unchanged_evidence"]
        assert [delta.run_one(root, i, "http://unused") for i in range(2)] == results
    assert len(client.calls) == 4
    request = sealed_read(
        root / "reviews" / row["historical_request_sha256"] / "judge/request.json"
    )["request"]
    assert request == original.corrected_request(row["request"])
    result = delta.report(root)
    assert result["status_counts"] == {"reviewed": 1, "unchanged_evidence": 1}
    # MockLLMClient has no provider usage log; do not invent measured costs.
    assert result["cost"]["usage_complete"] is False
    assert all(result[k] == 0 for k in delta.ZERO_FIELDS)
    assert source.read_bytes() == before
    return {
        k: result[k]
        for k in (
            "status_counts",
            "cost",
            "model_changes",
            "selection_changes",
        )
    }


def test_portable_review_and_resume(tmp_path):
    smoke(tmp_path)


@pytest.mark.parametrize("change", ["evidence", "hash", "model", "placements", "test"])
def test_rejects_inconsistent_import_even_if_resealed(tmp_path, change):
    source = exported_plan(tmp_path)
    value = json.loads(source.read_text())
    value.pop("artifact_sha256")
    if change == "evidence":
        value["reviews"][0]["affected"] = False
    elif change == "hash":
        value["reviews"][0]["historical_request_sha256"] = "d" * 64
    elif change == "model":
        value["reviews"][0]["request"]["model_revision"] = "e" * 40
    elif change == "placements":
        value["placements"] = value["placements"][:1]
    else:
        value["test_data_opened"] = True
    bad = tmp_path / "bad.json"
    sealed_write(bad, value)
    with pytest.raises(ValueError):
        delta.freeze(bad, tmp_path / "delta", "c" * 64)


def test_tamper_runtime_image_and_source_detected(tmp_path, monkeypatch):
    source, root = exported_plan(tmp_path), tmp_path / "delta"
    delta.freeze(source, root, "c" * 64)
    with pytest.raises(ValueError, match="frozen artifact"):
        delta.freeze(source, root, "d" * 64)
    monkeypatch.setattr(original.public, "_runtime", lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="identity"):
        delta.verify(root)


def test_missing_output_reports_pending_not_zero_usage_complete(tmp_path):
    source, root = exported_plan(tmp_path), tmp_path / "delta"
    delta.freeze(source, root, "c" * 64)
    result = delta.report(root)
    assert result["status_counts"] == {"pending": 1, "unchanged_evidence": 1}
    assert result["cost"]["physical_requests"] == 0
    assert result["cost"]["usage_complete"] is False


def test_interrupted_request_is_not_restarted(tmp_path):
    source, root = exported_plan(tmp_path), tmp_path / "delta"
    plan = delta.freeze(source, root, "c" * 64)
    key = plan["reviews"][0]["historical_request_sha256"]
    cache = root / "reviews" / key / "judge"
    cache.mkdir(parents=True)
    request = original.corrected_request(plan["reviews"][0]["request"])
    (cache / "review_started.json").write_text(
        json.dumps({"request_sha256": original.content_hash(request)})
    )
    (cache / "events.jsonl").write_text(
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
    with patch.object(
        judge, "create_llm_client", side_effect=AssertionError("no retry")
    ):
        result = delta.run_one(root, 0, "http://unused")
        assert result["status"] == "interrupted"
        assert delta.run_one(root, 0, "http://unused") == result
    assert delta.report(root)["cost"]["observed_total_tokens"] == 123
    assert delta.report(root)["cost"]["physical_requests"] == 1


@pytest.mark.parametrize("affected", [True, False])
def test_delta_submission_idempotency_and_resources(tmp_path, monkeypatch, affected):
    from scripts import submit_judge_sign_delta as submitter

    source = exported_plan(tmp_path)
    if not affected:
        value = copy.deepcopy(sealed_read(source))
        value.pop("artifact_sha256")
        value["reviews"] = value["reviews"][1:]
        key = value["reviews"][0]["historical_request_sha256"]
        value["placements"] = [
            p for p in value["placements"] if p["request_sha256"] == key
        ]
        source = tmp_path / "unaffected.json"
        sealed_write(source, value)
    root = tmp_path / "delta"
    monkeypatch.setattr(submitter, "source_commit", lambda _: "c" * 40)
    monkeypatch.setenv("AF_COMMIT", "c" * 40)
    monkeypatch.setenv("AF_PYTHON", __file__)
    monkeypatch.setenv("AF_VLLM_IMAGE", __file__)
    monkeypatch.setenv("AF_HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("AF_GPU_ACCOUNT", raising=False)
    monkeypatch.delenv("AF_CPU_ACCOUNT", raising=False)
    calls = []

    def queue(*args):
        calls.append(args)
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", queue)
    result = submitter.submit(source, root)
    assert len(calls) == (3 if affected else 2)
    assert "--account=bibo-delta-cpu" in calls[0][2]
    if affected:
        assert "--gpus-per-node=4" in calls[1][2]
        assert "--partition=gpuA40x4" in calls[1][2]
        assert "--account=bibo-delta-gpu" in calls[1][2]
        assert "--dependency=afterok:101" in calls[1][2]
    assert result["maximum_new_paired_reviews"] == int(affected)
    count = len(calls)
    assert submitter.submit(source, root) == result and len(calls) == count
    root.joinpath("submission_manifest.json").unlink()
    with pytest.raises(ValueError, match="partial submission"):
        submitter.submit(source, root)
    assert len(calls) == count
