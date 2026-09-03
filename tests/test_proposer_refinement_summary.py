"""Tests for the public-only proposer refinement analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.schemas import CandidateModel
from scripts.summarize_phase_b_proposer_refinement_pilot import collect_report


def _candidate() -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "candidate",
            "parent_candidate_id": None,
            "states": [
                {
                    "name": "target",
                    "kind": "observed",
                    "mechanisms": [],
                }
            ],
            "state_equations": [
                {"state": "target", "rhs": "input_u - target"}
            ],
            "observation_mappings": [
                {"channel": "target", "expression": "target"}
            ],
            "initial_conditions": [
                {"state": "target", "scope": "global", "fixed_value": 0.0}
            ],
        }
    )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = json.loads(
        Path("configs/phase_b_proposer_refinement_pilot_v1.json").read_text()
    )
    source["cells"] = [
        {
            "benchmark_id": "cell",
            "tier": "easy",
            "public_prompt_sha256": "0" * 64,
            "public_target_contract_sha256": "1" * 64,
            "public_mechanism_spec_sha256": "2" * 64,
        }
    ]
    source["repetitions"] = [0]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(source), encoding="utf-8")
    target_root = tmp_path / "targets"
    mechanism_root = tmp_path / "mechanisms"
    (target_root / "specs").mkdir(parents=True)
    (mechanism_root / "specs").mkdir(parents=True)
    (target_root / "specs" / "cell.json").write_text(
        json.dumps(
            {
                "benchmark_id": "cell",
                "tier": "easy",
                "public_prompt_sha256": "0" * 64,
                "targets": [
                    {
                        "target_channel": "target",
                        "public_requirement": "Generate target.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (mechanism_root / "specs" / "cell.json").write_text(
        json.dumps(
            {
                "benchmark_id": "cell",
                "tier": "easy",
                "required_mechanisms": [
                    {
                        "id": "input_response",
                        "required_drivers": ["input_u"],
                        "required_targets": ["target"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate = _candidate()
    for arm_id, cache_hit in (
        ("rich_exploratory", False),
        ("rich_incumbent_refinement", True),
    ):
        run = (
            tmp_path
            / "searches"
            / arm_id
            / "runs"
            / "cell_easy_seed0"
        )
        (run / "checkpoints").mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "selection_hash": "structural-hash",
                    "selected_candidate": candidate.model_dump(mode="json"),
                    "selection_validation_normalized_mse": 0.25,
                }
            ),
            encoding="utf-8",
        )
        (run / "checkpoints" / "round_000.json").write_text(
            json.dumps(
                {
                    "round_index": 0,
                    "valid": True,
                    "fit_attempts": [{}],
                    "record": {
                        "structural_hash": "structural-hash",
                        "pruned_fit": {
                            "training_metrics": {"normalized_mse": 0.2}
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "proposer_events.jsonl").write_text(
            json.dumps(
                {
                    "event": "llm_response",
                    "role": "proposer",
                    "request_hash": "same-request",
                    "cache_hit": cache_hit,
                    "logical_calls": 1,
                    "provider_attempts": 1,
                    "latency_ms": 100.0,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run / "task_runtime.json").write_text(
            json.dumps(
                {"allocated_cpu_core_hours": 1.0, "allocated_gpu_hours": 0.5}
            ),
            encoding="utf-8",
        )
    return plan, tmp_path, target_root, mechanism_root


def test_refinement_summary_enforces_round_zero_and_reports_public_metrics(
    tmp_path: Path,
) -> None:
    plan, search, targets, mechanisms = _write_fixture(tmp_path)

    report = collect_report(
        plan,
        search,
        target_contract_root=targets,
        mechanism_spec_root=mechanisms,
    )

    assert report["matched_round_zero_count"] == 1
    assert report["source_completion_count"] == 2
    assert report["status"] == "complete"
    assert all(item["public_target_pass"] for item in report["tasks"])
    assert all(
        item["public_mechanism_compliance"] == 1.0
        for item in report["tasks"]
    )
    assert all(
        item["public_graph_mechanism_compliance"] == 1.0
        for item in report["tasks"]
    )
    assert all(
        item["public_mechanism_annotation_compliance"] == 0.0
        for item in report["tasks"]
    )
    assert report["groups"][0]["mean_public_graph_mechanism_compliance"] == 1.0
    assert (
        report["groups"][0]["mean_public_mechanism_annotation_compliance"]
        == 0.0
    )
    assert report["groups"][0]["proposer_provider_attempts"] == 1
    assert report["groups"][1]["proposer_provider_attempts"] == 0


def test_refinement_summary_rejects_nonmatched_round_zero(tmp_path: Path) -> None:
    plan, search, targets, mechanisms = _write_fixture(tmp_path)
    event = (
        search
        / "searches"
        / "rich_incumbent_refinement"
        / "runs"
        / "cell_easy_seed0"
        / "proposer_events.jsonl"
    )
    payload = json.loads(event.read_text(encoding="utf-8"))
    payload["request_hash"] = "different-request"
    event.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="round-zero proposer request differs"):
        collect_report(
            plan,
            search,
            target_contract_root=targets,
            mechanism_spec_root=mechanisms,
        )


def test_refinement_summary_marks_missing_sources_incomplete(tmp_path: Path) -> None:
    plan, search, targets, mechanisms = _write_fixture(tmp_path)
    for summary in search.glob("searches/*/runs/*/summary.json"):
        summary.unlink()

    report = collect_report(
        plan,
        search,
        target_contract_root=targets,
        mechanism_spec_root=mechanisms,
    )

    assert report["status"] == "incomplete"
    assert report["source_completion_count"] == 0
