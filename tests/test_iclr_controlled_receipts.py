"""Portable resource/provenance evidence for the frozen controlled search."""

from __future__ import annotations

import json

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_controlled_receipts as receipts
from scripts import smoke_iclr_controlled_search as smoke


def test_frozen_search_exports_once_and_validates_the_copied_initial_fit(tmp_path):
    trial = tmp_path / "fixture"
    smoke.run(trial)
    root = trial / "search"
    out = tmp_path / "portable"
    report = receipts.export(root, out)
    rows = report["records"]
    assert report == receipts.export(root, out)
    assert len(rows["shared_initials"]) == 1
    shared = rows["shared_initials"][0]
    assert shared["status"] == "verified_shared_model" and shared["parameters_sha256"]
    assert len(rows["visits"]) == 4 and len(rows["proposer_requests"]) == 14
    assert len([r for r in rows["fit_evaluations"] if r["stage"] == "search"]) == 3
    assert all(r["actual_residual_calls"] is not None for r in rows["fit_evaluations"])
    assert any(r["optimizer_nfev"] is not None for r in rows["fit_evaluations"])
    assert len(rows["final_public_checks"]) == 4
    assert all(
        not r["scientific_correctness_certified"] for r in rows["final_public_checks"]
    )
    assert not rows["critic_findings"]
    path = root / "results/block0000_fixed_gamma/round_00/fit/result.json"
    receipt = json.loads(path.read_text())
    receipt["result"]["actual_residual_calls"] += 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="shared fit"):
        receipts.export(root, tmp_path / "tampered")


def test_no_private_or_pilot_evaluation_and_terminal_failures(tmp_path):
    root = tmp_path / "unfinished"
    sealed_write(root / "plan.json", {"protocol": "iclr-controlled-search-1"})
    with pytest.raises(FileNotFoundError):
        receipts.export(root, tmp_path / "report")
    row = {"status": "no_retained_model", "source_sha256": "receipt"}
    task = {"task_id": "one", "cell": "case", "seed": 1, "treatment": "no_spec"}
    got = receipts.public_check({}, task, row, None, "full")
    assert got["endpoint_status"] == "no_retained_model" and "requirements" not in got


def test_critic_receipt_must_match_parent_and_preserves_bounded_findings(tmp_path):
    finding = {
        "issue": "insulin pulse conflicts with the proposed path",
        "support": "x" * 8000,
    }
    review = sealed_write(
        tmp_path / "critic.json",
        {
            "identity": "plan",
            "task": {"task_id": "one"},
            "round": 1,
            "proposal_sha256": "prop",
            "status": "complete",
            "request_sha256": "req",
            "review": {"request_sha256": "req", "findings": [finding]},
        },
    )
    (row,) = receipts.review_rows(
        review, {"task": "one"}, "plan", "proposal_sha256", "prop"
    )
    assert row["truncated"] and len(row["finding_json_text"]) == receipts.MAX_QUOTE
    assert row["finding_sha256"] == receipts.HASH(finding)
    with pytest.raises(ValueError, match="source"):
        receipts.review_rows(
            sealed_read(tmp_path / "critic.json"),
            {"task": "one"},
            "plan",
            "proposal_sha256",
            "different",
        )
