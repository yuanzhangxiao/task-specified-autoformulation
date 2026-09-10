"""Offline replay for multiround parent-role preservation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from autoformalism.rebuttal.fitter_recovery import recovery_candidate
from autoformalism.rebuttal.staged_multiround_role_replay import (
    replay_multiround_role_repairs,
)
from autoformalism.staged_topology import content_hash


def test_replay_uses_cached_responses_without_calls_or_fitting(tmp_path: Path) -> None:
    root = tmp_path / "source"
    candidate_path = root / "frozen" / "candidates" / "candidate_000.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(recovery_candidate().model_dump_json())
    task_id = "phase_b_anonymous_system_task_canonical_opaque_hard_seed0_multiround"
    task = {
        "task_id": task_id,
        "benchmark_id": "phase_b_anonymous_system_task_canonical_opaque_hard",
        "tier": "hard",
        "seed": 0,
        "candidate_path": str(candidate_path.relative_to(root)),
        "candidate_file_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
    }
    plan = {
        "schema_version": "scientific-staged-multiround-feedback-plan-1",
        "tasks": [task],
    }
    plan["plan_sha256"] = content_hash(plan)
    (root / "plan.json").write_text(json.dumps(plan))
    results = root / "results"
    results.mkdir()
    (results / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "validation_used_for_parameter_fitting": False,
                "scientific_judge_called": False,
                "test_data_opened": False,
                "private_reference_opened": False,
                "automatic_winner_defined": False,
            }
        )
    )
    call_root = results / task_id / "calls"
    call_root.mkdir(parents=True)
    response = {
        "revisions": [
            {
                "component": "f",
                "expression": "k*sigmoid(v01) - f/tau_f",
                "parameters": [
                    {"name": "k", "role": "positive_shape"},
                    {"name": "tau_f", "role": "coefficient"},
                ],
            }
        ]
    }
    request = {
        "selected_components": [
            {"component": "f", "required_nonparameter_sources": ["f", "v01"]}
        ]
    }
    record = {
        "request_hash": "cached_request",
        "step": "round_1_function_revision",
        "attempt": 0,
        "status": "responded",
        "request": {
            "body": {
                "messages": [
                    {"role": "system", "content": "repair"},
                    {"role": "user", "content": json.dumps(request)},
                ]
            }
        },
        "raw_response": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(response)},
                }
            ]
        },
    }
    (call_root / "cached_request.json").write_text(json.dumps(record))

    report = replay_multiround_role_repairs(root, root / "role_replay.json")

    assert report["status"] == "pass"
    assert report["stored_response_count"] == 1
    assert report["accepted_after_role_policy_count"] == 1
    assert report["responses_with_role_derivation_count"] == 1
    assert report["new_llm_calls_made"] is False
    assert report["parameter_fitting_performed"] is False
