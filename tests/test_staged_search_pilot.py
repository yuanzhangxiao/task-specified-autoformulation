"""Tests for the frozen public-only staged-search pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.rebuttal.staged_search_pilot import (
    build_staged_search_tasks,
    freeze_staged_search_pilot,
    load_staged_search_plan,
    summarize_staged_search_pilot,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_plan_is_two_public_cells_by_three_seeds() -> None:
    plan = load_staged_search_plan(
        Path("configs/phase_b_staged_search_pilot_v1.json")
    )
    tasks = build_staged_search_tasks(plan)

    assert len(tasks) == 6
    assert [item.task_index for item in tasks] == list(range(6))
    assert [item.repetition for item in tasks] == [0, 1, 2, 0, 1, 2]
    assert plan.proposer_construction_mode == "staged_v2"
    assert plan.proposal_policy == "incumbent_refinement_v1"
    assert plan.apply_postfit_pruning is False
    assert plan.test_data_opened is False
    assert plan.private_reference_opened is False


def test_freeze_validates_inputs_and_writes_hash_ledger(tmp_path: Path) -> None:
    prompt = tmp_path / "public" / "phase_b_v1" / "cell" / "proposer_prompt.txt"
    target = tmp_path / "targets" / "specs" / "cell.json"
    mechanism = tmp_path / "mechanisms" / "specs" / "cell.json"
    for path, content in (
        (prompt, "public prompt\n"),
        (target, "{}\n"),
        (mechanism, "{}\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    payload = json.loads(
        Path("configs/phase_b_staged_search_pilot_v1.json").read_text()
    )
    payload["cells"] = [
        {
            "benchmark_id": "cell",
            "tier": "easy",
            "public_prompt_sha256": _sha(prompt),
            "public_target_contract_sha256": _sha(target),
            "public_mechanism_spec_sha256": _sha(mechanism),
        }
    ]
    payload["repetitions"] = [0]
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    manifest = freeze_staged_search_pilot(
        config,
        tmp_path / "frozen",
        public_data_root=tmp_path / "public",
        target_contract_root=tmp_path / "targets",
        mechanism_spec_root=tmp_path / "mechanisms",
    )

    assert manifest["task_count"] == 1
    assert manifest["test_data_opened"] is False
    assert (tmp_path / "frozen" / "plan.json.sha256").is_file()
    assert (tmp_path / "frozen" / "task_plan.jsonl.sha256").is_file()
    assert (tmp_path / "frozen" / "freeze_manifest.json.sha256").is_file()


def test_freeze_rejects_public_prompt_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing public prompt"):
        freeze_staged_search_pilot(
            Path("configs/phase_b_staged_search_pilot_v1.json"),
            tmp_path / "frozen",
            public_data_root=tmp_path,
            target_contract_root=tmp_path,
            mechanism_spec_root=tmp_path,
        )


def test_summary_reports_stage_specific_revisions_and_transport(
    tmp_path: Path,
) -> None:
    source = json.loads(
        Path("configs/phase_b_staged_search_pilot_v1.json").read_text()
    )
    source["cells"] = source["cells"][:1]
    source["repetitions"] = [0]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(source), encoding="utf-8")
    plan = load_staged_search_plan(plan_path)
    task_path = tmp_path / "task_plan.jsonl"
    task_path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in build_staged_search_tasks(plan)
        ),
        encoding="utf-8",
    )
    run = (
        tmp_path
        / "search"
        / "runs"
        / "phase_b_dalla_man_t2_canonical_named_easy_easy_seed0"
    )
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "selection_validation_normalized_mse": 0.25,
                "selected_candidate": {
                    "states": [
                        {"name": "x", "kind": "observed"},
                        {"name": "z", "kind": "latent"},
                    ],
                    "processes": [{"name": "y"}],
                    "parameters": [{"name": "gain"}],
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "search_process_time.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 10.0,
                "user_cpu_seconds": 3.0,
                "system_cpu_seconds": 1.0,
                "max_rss_kib": 1024,
            }
        ),
        encoding="utf-8",
    )
    calls = {
        "topology_call": {
            "logical_calls": 1,
            "provider_attempts": 1,
            "input_tokens": 100,
            "output_tokens": 20,
        },
        "functional_call": {
            "logical_calls": 1,
            "provider_attempts": 2,
            "input_tokens": 200,
            "output_tokens": 40,
        },
    }
    (checkpoints / "round_000.json").write_text(
        json.dumps(
            {
                "valid": True,
                "public_target_evaluation": {"passed": True},
                "public_mechanism_evaluation": {
                    "graph_mechanism_compliance": 1.0,
                    "mechanism_annotation_compliance": 0.5,
                },
                "fit_attempts": [{"success": False}, {"success": True}],
                "fit": {
                    "success": True,
                    "training_metrics": {"normalized_mse": 0.2},
                    "validation_metrics": {"normalized_mse": 0.25},
                },
                "staged_proposal": {
                    "revision_decision": "initial_topology_and_functions",
                    **calls,
                },
            }
        ),
        encoding="utf-8",
    )
    (checkpoints / "round_001.json").write_text(
        json.dumps(
            {
                "valid": False,
                "staged_proposal": {
                    "revision_decision": "function_only_revision",
                    "topology_call": None,
                    "functional_call": calls["functional_call"],
                },
            }
        ),
        encoding="utf-8",
    )

    report = summarize_staged_search_pilot(
        plan_path,
        task_path,
        tmp_path / "search",
    )

    assert report["status"] == "complete"
    assert report["total_valid_round_count"] == 1
    assert report["total_topology_revision_count"] == 1
    assert report["total_function_only_revision_count"] == 1
    assert report["total_proposer_logical_calls"] == 3
    assert report["total_proposer_provider_attempts"] == 5
    assert report["public_target_pass_rate"] == 1.0
    assert report["mean_graph_mechanism_compliance"] == 1.0
    assert report["mean_mechanism_annotation_compliance"] == 0.5
    assert report["total_fit_successful_round_count"] == 1
    assert report["total_fit_retry_activation_count"] == 1
    assert report["total_process_wall_seconds"] == 10.0
    assert report["process_time_status_counts"] == {"json": 1}
    assert report["rows"][0]["proposer_accounting_source"] == (
        "checkpoint_receipts"
    )
    assert report["rows"][0]["selected_latent_state_count"] == 1
    assert report["median_validation_normalized_mse"] == 0.25


def test_summary_counts_failed_proposer_attempts_from_event_log(
    tmp_path: Path,
) -> None:
    source = json.loads(
        Path("configs/phase_b_staged_search_pilot_v1.json").read_text()
    )
    source["cells"] = source["cells"][:1]
    source["repetitions"] = [0]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(source), encoding="utf-8")
    plan = load_staged_search_plan(plan_path)
    task_path = tmp_path / "task_plan.jsonl"
    task_path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in build_staged_search_tasks(plan)
        ),
        encoding="utf-8",
    )
    run = (
        tmp_path
        / "search"
        / "runs"
        / "phase_b_dalla_man_t2_canonical_named_easy_easy_seed0"
    )
    run.mkdir(parents=True)
    events = [
        {
            "event": "llm_failure",
            "role": "staged_topology_proposer_v3",
            "request_hash": "a",
            "attempt": 1,
            "raw_response": {
                "usage": {"prompt_tokens": 10, "completion_tokens": 20}
            },
        },
        {
            "event": "llm_response",
            "role": "staged_topology_proposer_v3",
            "request_hash": "a",
            "provider_attempts": 2,
            "usage": {"input_tokens": 11, "output_tokens": 21},
        },
        {
            "event": "llm_failure",
            "role": "staged_function_proposer_v2",
            "request_hash": "b",
            "attempt": 1,
            "raw_response": {
                "usage": {"prompt_tokens": 12, "completion_tokens": 22}
            },
        },
        {
            "event": "llm_failure",
            "role": "staged_function_proposer_v2",
            "request_hash": "b",
            "attempt": 2,
            "provider_attempts": 1,
            "usage": {"input_tokens": 13, "output_tokens": 23},
        },
    ]
    (run / "proposer_events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events) + "truncated",
        encoding="utf-8",
    )

    report = summarize_staged_search_pilot(
        plan_path,
        task_path,
        tmp_path / "search",
    )

    row = report["rows"][0]
    assert row["proposer_accounting_source"] == "event_log"
    assert row["proposer_event_log_invalid_line_count"] == 1
    assert row["proposer_logical_calls"] == 2
    assert row["proposer_terminal_failed_logical_calls"] == 1
    assert row["proposer_failed_attempts"] == 3
    assert row["proposer_provider_attempts"] == 4
    assert row["proposer_input_tokens"] == 46
    assert row["proposer_output_tokens"] == 86
    assert row["proposer_token_usage_observed"] is True
    assert report["total_proposer_terminal_failed_logical_calls"] == 1
    assert report["total_proposer_failed_attempts"] == 3


@pytest.mark.parametrize(
    ("timing_text", "expected_status", "expected_wall_seconds"),
    [
        ("", "empty", None),
        ("not timing data\n", "invalid", None),
        (
            "schema_version=portable-child-process-timing-1\n"
            "elapsed_seconds=12.5\n"
            "user_cpu_seconds=3.0\n"
            "system_cpu_seconds=1.0\n"
            "max_rss_kib=2048\n"
            "exit_code=1\n",
            "legacy_key_value",
            12.5,
        ),
    ],
)
def test_summary_tolerates_non_json_process_timing(
    tmp_path: Path,
    timing_text: str,
    expected_status: str,
    expected_wall_seconds: float | None,
) -> None:
    source = json.loads(
        Path("configs/phase_b_staged_search_pilot_v1.json").read_text()
    )
    source["cells"] = source["cells"][:1]
    source["repetitions"] = [0]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(source), encoding="utf-8")
    plan = load_staged_search_plan(plan_path)
    task_path = tmp_path / "task_plan.jsonl"
    task_path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in build_staged_search_tasks(plan)
        ),
        encoding="utf-8",
    )
    run = (
        tmp_path
        / "search"
        / "runs"
        / "phase_b_dalla_man_t2_canonical_named_easy_easy_seed0"
    )
    run.mkdir(parents=True)
    (run / "search_process_time.json").write_text(timing_text, encoding="utf-8")

    report = summarize_staged_search_pilot(
        plan_path,
        task_path,
        tmp_path / "search",
    )

    assert report["status"] == "incomplete"
    assert report["process_time_status_counts"] == {expected_status: 1}
    assert report["rows"][0]["process_wall_seconds"] == expected_wall_seconds


@pytest.mark.parametrize(
    "path",
    [
        "scripts/hpc/run_phase_b_staged_search_task.sh",
        "scripts/hpc/phase_b_staged_search_pilot_aces.slurm",
        "scripts/hpc/phase_b_staged_search_pilot_delta.slurm",
        "scripts/hpc/phase_b_staged_search_summary.slurm",
        "scripts/hpc/submit_phase_b_staged_search_pilot_aces.sh",
        "scripts/hpc/submit_phase_b_staged_search_pilot_delta.sh",
    ],
)
def test_cluster_scripts_are_valid_bash(path: str) -> None:
    subprocess.run(["bash", "-n", path], check=True)


def test_worker_preserves_test_and_private_boundaries() -> None:
    worker = Path("scripts/hpc/run_phase_b_staged_search_task.sh").read_text()

    assert "--development-only" in worker
    assert "--proposer-construction-mode staged_v2" in worker
    assert "--disable-postfit-pruning" in worker
    assert "--public-target-contract" in worker
    assert "--public-mechanism-spec" in worker
    assert "private" not in worker.lower()
    assert "test-data" not in worker.lower()
