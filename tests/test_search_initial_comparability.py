"""Tests for pairwise initial search-candidate comparability diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.rebuttal.search_integration_ablation import (
    build_search_integration_tasks,
    load_search_integration_plan,
)
from scripts.check_phase_b_search_initial_comparability import (
    compare_initial_search_candidates,
)

CONFIG = Path("configs/phase_b_search_integration_ablation_v1.json")


def _candidate(identifier: str, expression: str = "-k * x") -> dict[str, object]:
    return {
        "schema_version": "2",
        "candidate_id": identifier,
        "parent_candidate_id": None,
        "change_summary": f"candidate {identifier}",
        "states": [
            {
                "name": "x",
                "kind": "observed",
                "observed_channel": "x",
                "dynamics": expression,
            }
        ],
        "algebraics": [],
        "parameters": [],
    }


def _write_run(
    root: Path,
    *,
    arm: str,
    benchmark_id: str,
    tier: str,
    repetition: int,
    candidate: dict[str, object],
    request_hash: str,
    structural_hash: str,
) -> None:
    run = (
        root
        / "searches"
        / arm
        / "runs"
        / f"{benchmark_id}_{tier}_seed{repetition}"
    )
    (run / "checkpoints").mkdir(parents=True)
    (run / "proposer_events.jsonl").write_text(
        json.dumps(
            {
                "event": "llm_response",
                "role": "proposer",
                "request_hash": request_hash,
                "cache_hit": arm == "no_judge",
                "parsed_response": candidate,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "checkpoints" / "round_0000.json").write_text(
        json.dumps(
            {
                "stage": "complete",
                "valid": True,
                "record": {"structural_hash": structural_hash},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_initial_comparability_is_pairwise_not_across_seeds(tmp_path: Path) -> None:
    tasks = build_search_integration_tasks(load_search_integration_plan(CONFIG))
    for task in tasks:
        identifier = f"candidate_seed_{task.repetition}"
        _write_run(
            tmp_path,
            arm=task.arm_id,
            benchmark_id=task.benchmark_id,
            tier=task.tier,
            repetition=task.repetition,
            candidate=_candidate(identifier),
            request_hash=f"transport-specific-{task.arm_id}",
            structural_hash=f"structure-seed-{task.repetition}",
        )

    report = compare_initial_search_candidates(CONFIG, tmp_path)

    assert report["planned_pair_count"] == 6
    assert report["full_request_hash_match_count"] == 0
    assert report["exact_parsed_response_match_count"] == 6
    assert report["identity_insensitive_candidate_match_count"] == 6
    assert report["round_zero_structural_match_count"] == 6
    assert report["all_pairwise_exact_initial_responses_match"] is True


def test_initial_comparability_detects_one_content_difference(tmp_path: Path) -> None:
    tasks = build_search_integration_tasks(load_search_integration_plan(CONFIG))
    changed_key = (
        tasks[0].benchmark_id,
        tasks[0].tier,
        tasks[0].repetition,
    )
    for task in tasks:
        key = (task.benchmark_id, task.tier, task.repetition)
        expression = (
            "-k * x + u"
            if task.arm_id == "no_judge" and key == changed_key
            else "-k * x"
        )
        _write_run(
            tmp_path,
            arm=task.arm_id,
            benchmark_id=task.benchmark_id,
            tier=task.tier,
            repetition=task.repetition,
            candidate=_candidate(f"{task.arm_id}-{task.repetition}", expression),
            request_hash="same" if key != changed_key else task.arm_id,
            structural_hash=(
                "changed"
                if task.arm_id == "no_judge" and key == changed_key
                else f"structure-{key}"
            ),
        )

    report = compare_initial_search_candidates(CONFIG, tmp_path)

    assert report["exact_parsed_response_match_count"] == 0
    assert report["identity_insensitive_candidate_match_count"] == 5
    assert report["round_zero_structural_match_count"] == 5
    assert report["all_pairwise_candidate_content_matches"] is False
