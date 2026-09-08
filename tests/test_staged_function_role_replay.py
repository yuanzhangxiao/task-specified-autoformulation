"""Offline replay of certified function-role repairs."""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.rebuttal.staged_function_role_replay import (
    replay_hybrid_outer_gain_repairs,
)


def test_replay_separates_role_scaffolding_from_function_repair(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    results = root / "results"
    task = results / "opaque_seed0"
    task.mkdir(parents=True)
    selected = {
        "selected_term": {"sources": ["x"]},
        "expression": "k*x",
        "parameters": [{"name": "k_runtime", "role": "nonnegative_coefficient"}],
    }
    mismatch_selected = {
        "selected_term": {"sources": ["z"]},
        "expression": "z",
        "parameters": [],
    }
    terminal = {
        "task_id": "opaque_seed0",
        "seed": 0,
        "result": {
            "batch_term_audits": [
                {
                    "interaction_id": "term_0_0",
                    "lhs": "x",
                    "batch_error": (
                        "SIGNED_WEIGHT_WITH_TOPOLOGY_POLARITY: topology owns "
                        "the outer sign"
                    ),
                    "batch_function": {
                        "expression": "k*x",
                        "parameters": [{"name": "k", "role": "coefficient"}],
                    },
                },
                {
                    "interaction_id": "term_1_0",
                    "lhs": "z",
                    "batch_error": "source mismatch: missing=['z'], extra=['x']",
                    "batch_function": {"expression": "x", "parameters": []},
                },
            ],
            "accepted_functions": [selected, mismatch_selected],
            "events": [
                {"step": "atomic_repair_term_0_0"},
                {"step": "atomic_repair_term_1_0"},
            ],
        },
    }
    (task / "terminal.json").write_text(json.dumps(terminal))
    summary = {
        "status": "complete",
        "terminal_results": 1,
        "atomic_repair_activation_count": 2,
        "physical_requests": 5,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    (results / "summary.json").write_text(json.dumps(summary))
    report = replay_hybrid_outer_gain_repairs(root, root / "role_replay.json")
    assert report["status"] == "pass"
    assert report["certified_deterministic_role_repair_count"] == 1
    assert report["projected_llm_atomic_repair_count"] == 1
    assert report["projected_saved_provider_attempts"] == 1
    assert report["projected_physical_requests"] == 4
    assert not report["new_llm_calls_made"]
