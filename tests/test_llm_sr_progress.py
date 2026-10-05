"""Watching an LLM-SR campaign from the files its tasks write, without a model."""

from __future__ import annotations

import json
import os
from pathlib import Path

from autoformalism.rebuttal.llm_sr_progress import campaign_progress

START = 1_800_000_000.0


def _campaign(root: Path) -> Path:
    rows = [
        {"index": i, "benchmark_id": "cell", "repetition": i, "searched_targets": ["G"]}
        for i in range(3)
    ]
    plan = {"plan": {"budget": {"declared": 100}}, "rows": rows}
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return root


def _running_task(root: Path) -> None:
    samples = root / "results" / "0" / "llmsr-G" / "samples"
    samples.mkdir(parents=True)
    scores = [-5.0, None, -1.0, -2.0, float("nan"), -0.5, -3.0, -4.0, -0.75, -6.0]
    for order, score in enumerate(scores):
        path = samples / f"samples_{order}.json"
        payload = {"sample_order": order, "function": "f", "score": score}
        path.write_text(json.dumps(payload), encoding="utf-8")
        # Ten samples spread over one hour.
        stamp = START + order * 400.0
        os.utime(path, (stamp, stamp))
    calls = [
        {"event": "llm_response", "usage": {"total_tokens": 100},
         "finish_reasons": ["stop", "length", "stop", "stop"]},
        {"event": "llm_failure", "usage": {}, "reason": "URLError"},
        {"event": "llm_response", "usage": {"total_tokens": 200}, "cache_hit": True},
    ]
    log = root / "results" / "0" / "llm_calls.jsonl"
    log.write_text("".join(json.dumps(c) + "\n" for c in calls) + "{trunc", "utf-8")


def test_progress_reports_pace_and_status_without_a_model(tmp_path):
    root = _campaign(tmp_path / "campaign")
    _running_task(root)
    (root / "results" / "0.interrupted-1").mkdir()
    (root / "results" / "1").mkdir()
    (root / "results" / "1" / "result.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )

    report = campaign_progress(root, now=START + 3600.0 + 120.0)
    running, finished, waiting = report["tasks"]

    assert running["status"] == "running_or_interrupted"
    assert running["samples_logged"] == 10
    assert running["samples_that_scored"] == 8  # None and NaN never scored
    assert running["best_llm_sr_score"] == -0.5
    assert (running["requests"], running["failed_requests"]) == (2, 1)
    assert running["tokens"] == 300  # a torn final line is skipped
    assert running["replies_cut_off"] == 1  # stopped at the token limit
    assert running["cache_hits"] == 1  # replayed by a gateway, not generated
    assert running["samples_per_hour"] == 10.0
    assert running["hours_to_finish_at_this_pace"] == 9.0
    assert running["minutes_since_last_sample"] == 2.0
    assert running["interrupted_attempts"] == ["0.interrupted-1"]

    assert finished["status"] == "complete"
    assert waiting["status"] == "not_started"
    assert "interrupted_attempts" not in finished


def test_a_task_with_one_sample_has_no_pace_yet(tmp_path):
    root = _campaign(tmp_path / "campaign")
    samples = root / "results" / "0" / "llmsr-G" / "samples"
    samples.mkdir(parents=True)
    (samples / "samples_0.json").write_text('{"score": -1.0}', encoding="utf-8")
    (running, _, _) = campaign_progress(root)["tasks"]
    assert running["samples_logged"] == 1
    assert running["samples_per_hour"] is None
    assert running["hours_to_finish_at_this_pace"] is None
    assert running["requests"] == 0
