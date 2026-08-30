#!/usr/bin/env python3
"""Compare paired search arms at their initial proposer response only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.search_integration_ablation import (
    build_search_integration_tasks,
    load_search_integration_plan,
)


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _without_candidate_identity(value: object) -> object:
    if not isinstance(value, dict):
        return value
    return {
        key: item
        for key, item in value.items()
        if key not in {"candidate_id", "parent_candidate_id", "change_summary"}
    }


def _initial_response(run: Path) -> dict[str, Any] | None:
    for event in _read_events(run / "proposer_events.jsonl"):
        if event.get("event") == "llm_response" and event.get("role") == "proposer":
            return event
    return None


def _round_zero(run: Path) -> dict[str, object]:
    payload = _read_object(run / "checkpoints" / "round_0000.json")
    if payload is None:
        return {
            "checkpoint_present": False,
            "valid": None,
            "structural_hash": None,
        }
    record = payload.get("record")
    structural_hash = (
        record.get("structural_hash") if isinstance(record, dict) else None
    )
    return {
        "checkpoint_present": True,
        "valid": payload.get("valid"),
        "structural_hash": structural_hash,
    }


def compare_initial_search_candidates(
    plan_path: Path,
    search_root: Path,
) -> dict[str, object]:
    """Return pairwise initial-response equality for every benchmark and seed."""
    plan = load_search_integration_plan(plan_path)
    tasks = build_search_integration_tasks(plan)
    grouped: dict[tuple[str, str, int], dict[str, object]] = {}
    root = search_root.expanduser().resolve()
    for task in tasks:
        key = (task.benchmark_id, task.tier, task.repetition)
        grouped.setdefault(key, {})[task.arm_id] = task

    rows: list[dict[str, object]] = []
    for (benchmark_id, tier, repetition), by_arm in sorted(grouped.items()):
        if set(by_arm) != {"paired_question_consensus", "no_judge"}:
            raise ValueError(
                "expected exactly the paired-question-consensus and no-judge arms"
            )
        arm_records: dict[str, dict[str, object]] = {}
        for arm_id in ("paired_question_consensus", "no_judge"):
            run = (
                root
                / "searches"
                / arm_id
                / "runs"
                / f"{benchmark_id}_{tier}_seed{repetition}"
            )
            event = _initial_response(run)
            parsed = None if event is None else event.get("parsed_response")
            arm_records[arm_id] = {
                "run": str(run),
                "response_present": event is not None,
                "full_request_hash": (
                    None if event is None else event.get("request_hash")
                ),
                "cache_hit": None if event is None else event.get("cache_hit"),
                "parsed_response_sha256": (
                    None if parsed is None else _canonical_sha256(parsed)
                ),
                "identity_insensitive_candidate_sha256": (
                    None
                    if parsed is None
                    else _canonical_sha256(_without_candidate_identity(parsed))
                ),
                "round_zero": _round_zero(run),
            }

        judge = arm_records["paired_question_consensus"]
        no_judge = arm_records["no_judge"]
        response_covered = bool(
            judge["response_present"] and no_judge["response_present"]
        )
        exact_equal = response_covered and (
            judge["parsed_response_sha256"]
            == no_judge["parsed_response_sha256"]
        )
        content_equal = response_covered and (
            judge["identity_insensitive_candidate_sha256"]
            == no_judge["identity_insensitive_candidate_sha256"]
        )
        judge_round = judge["round_zero"]
        no_judge_round = no_judge["round_zero"]
        assert isinstance(judge_round, dict) and isinstance(no_judge_round, dict)
        structural_covered = (
            judge_round.get("structural_hash") is not None
            and no_judge_round.get("structural_hash") is not None
        )
        structural_equal = structural_covered and (
            judge_round["structural_hash"] == no_judge_round["structural_hash"]
        )
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "tier": tier,
                "repetition": repetition,
                "response_covered": response_covered,
                "full_request_hash_equal": response_covered
                and judge["full_request_hash"] == no_judge["full_request_hash"],
                "exact_parsed_response_equal": exact_equal,
                "identity_insensitive_candidate_equal": content_equal,
                "round_zero_structural_hash_covered": structural_covered,
                "round_zero_structural_hash_equal": structural_equal,
                "arms": arm_records,
            }
        )

    covered = [row for row in rows if row["response_covered"]]
    structural = [
        row for row in rows if row["round_zero_structural_hash_covered"]
    ]
    return {
        "schema_version": "phase-b-search-initial-comparability-1",
        "scope": "pairwise_within_benchmark_tier_and_repetition",
        "test_data_opened": False,
        "planned_pair_count": len(rows),
        "response_covered_pair_count": len(covered),
        "full_request_hash_match_count": sum(
            bool(row["full_request_hash_equal"]) for row in covered
        ),
        "exact_parsed_response_match_count": sum(
            bool(row["exact_parsed_response_equal"]) for row in covered
        ),
        "identity_insensitive_candidate_match_count": sum(
            bool(row["identity_insensitive_candidate_equal"]) for row in covered
        ),
        "round_zero_structural_covered_pair_count": len(structural),
        "round_zero_structural_match_count": sum(
            bool(row["round_zero_structural_hash_equal"]) for row in structural
        ),
        "all_pairwise_exact_initial_responses_match": (
            len(covered) == len(rows)
            and all(bool(row["exact_parsed_response_equal"]) for row in rows)
        ),
        "all_pairwise_candidate_content_matches": (
            len(covered) == len(rows)
            and all(
                bool(row["identity_insensitive_candidate_equal"])
                for row in rows
            )
        ),
        "pairs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_initial_search_candidates(args.plan, args.search_root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
