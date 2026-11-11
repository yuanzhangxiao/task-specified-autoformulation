"""How far an LLM-SR campaign has got, read from what its tasks write as they go.

LLM-SR's profiler writes one file per evaluated sample, and the transport
appends one call-log line per request, so a running search can be watched
without calling a model or opening anything but the campaign directory. The
score shown is LLM-SR's own, a training-data fit; the model the campaign
reports is chosen and scored only when the search ends.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path


def _finite_score(path: Path) -> float | None:
    """The score one sample file records, if it evaluated."""
    try:
        score = json.loads(path.read_text(encoding="utf-8")).get("score")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(score, (int, float)) and math.isfinite(score):
        return float(score)
    return None


def _calls(log: Path) -> dict[str, int]:
    """Requests answered and failed, tokens, replies cut off, answers replayed.

    A reply cut off stopped at the token limit, before the model finished it.
    A cache hit is an answer a gateway replayed instead of generating.
    """
    counts = {
        "requests": 0,
        "failed_requests": 0,
        "tokens": 0,
        "replies_cut_off": 0,
        "cache_hits": 0,
    }
    if not log.exists():
        return counts
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "llm_response":
            counts["requests"] += 1
            counts["tokens"] += int((event.get("usage") or {}).get("total_tokens") or 0)
            reasons = event.get("finish_reasons") or []
            counts["replies_cut_off"] += sum(reason == "length" for reason in reasons)
            counts["cache_hits"] += bool(event.get("cache_hit"))
        elif event.get("event") == "llm_failure":
            counts["failed_requests"] += 1
    return counts


def task_progress(directory: Path, *, planned_samples: int, now: float) -> dict:
    """Samples evaluated, best score, calls and pace of one unfinished task."""
    samples = sorted(directory.glob("llmsr-*/samples/samples_*.json"))
    scores = [score for path in samples if (score := _finite_score(path)) is not None]
    times = sorted(path.stat().st_mtime for path in samples)
    hours = (times[-1] - times[0]) / 3600 if len(times) > 1 else 0.0
    pace = len(samples) / hours if hours > 0 else None
    remaining = max(planned_samples - len(samples), 0)
    return {
        "samples_logged": len(samples),
        "planned_samples": planned_samples,
        "samples_that_scored": len(scores),
        "best_llm_sr_score": max(scores) if scores else None,
        **_calls(directory / "llm_calls.jsonl"),
        "samples_per_hour": None if pace is None else round(pace, 1),
        "hours_to_finish_at_this_pace": (
            None if pace is None else round(remaining / pace, 1)
        ),
        "minutes_since_last_sample": (
            None if not times else round((now - times[-1]) / 60, 1)
        ),
    }


def campaign_progress(root: Path, *, now: float | None = None) -> dict:
    """One line per task: its sealed status, or its progress so far."""
    now = time.time() if now is None else now
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    per_target = int(plan["plan"]["budget"]["declared"])
    tasks = []
    for row in plan["rows"]:
        directory = root / "results" / str(row["index"])
        entry = {
            "index": row["index"],
            "benchmark_id": row["benchmark_id"],
            "repetition": row["repetition"],
        }
        result = directory / "result.json"
        if result.exists():
            sealed = json.loads(result.read_text(encoding="utf-8"))
            entry["status"] = sealed.get("status")
        elif directory.is_dir() and any(directory.iterdir()):
            entry["status"] = "running_or_interrupted"
            entry.update(
                task_progress(
                    directory,
                    planned_samples=per_target * len(row["searched_targets"]),
                    now=now,
                )
            )
        else:
            entry["status"] = "not_started"
        pattern = f"{row['index']}.interrupted-*"
        interrupted = sorted(path.name for path in (root / "results").glob(pattern))
        if interrupted:
            entry["interrupted_attempts"] = interrupted
        tasks.append(entry)
    return {"root": str(root), "tasks": tasks}
