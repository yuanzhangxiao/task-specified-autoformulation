"""Tests for the full fitted raw-agent baseline matrix and summary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from autoformalism.data import BenchmarkRegistry
from scripts.resolve_raw_data_agent_matrix_task import resolve_task
from scripts.summarize_raw_data_agent_full_evaluation import aggregate, build_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "raw_data_agent_fitted_model_full_v1.json"
AGENT_SLURM = (
    ROOT / "scripts" / "hpc" / "phase_b_raw_data_agent_fitted_model_full.slurm"
)
AUDIT_SLURM = (
    ROOT / "scripts" / "hpc" / "phase_b_raw_agent_scientific_audit_full_120b.slurm"
)


def test_full_matrix_is_exactly_the_registered_phase_b_suite() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registered = {
        item
        for item in BenchmarkRegistry().identifiers()
        if item.startswith("phase_b_")
    }
    configured = {item["benchmark_id"] for item in config["benchmarks"]}

    assert configured == registered
    assert len(configured) == 40
    assert config["repetitions"] == 3
    assert config["output_contract"] == "fitted_model"
    assert config["evaluation"]["parameter_refit_applied"] is False
    assert config["test_data_opened"] is False


def test_matrix_resolver_has_benchmark_major_repetition_order() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    first = resolve_task(config, 0)
    third = resolve_task(config, 2)
    fourth = resolve_task(config, 3)
    last = resolve_task(config, 119)

    assert first[2] == 0
    assert third[:2] == first[:2] and third[2] == 2
    assert fourth[0] == config["benchmarks"][1]["benchmark_id"]
    assert last[0] == config["benchmarks"][-1]["benchmark_id"]
    assert last[2] == 2


def test_full_matrix_launchers_are_syntactically_valid_and_frozen() -> None:
    agent = AGENT_SLURM.read_text(encoding="utf-8")
    audit = AUDIT_SLURM.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-119%12" in agent
    assert "--output-contract fitted_model" in agent
    assert "#SBATCH --array=0-3" in audit
    assert "AF_SHARD_COUNT:=4" in audit
    assert "AF_REPAIR_MISSING_ATOMIC_UNITS:=true" in audit
    subprocess.run(["bash", "-n", str(AGENT_SLURM)], check=True)
    subprocess.run(["bash", "-n", str(AUDIT_SLURM)], check=True)


def test_combined_summary_keeps_numerical_and_scientific_results_separate(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run_0"
    run.mkdir()
    (run / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "candidate_id": "candidate",
                "parameter_refit_applied": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "evaluation.json").write_text(
        json.dumps(
            {
                "training_metrics": {
                    "normalized_mse": 0.2,
                    "failed_trajectories": [],
                    "soft_constraint_violations": {},
                },
                "validation_metrics": {
                    "normalized_mse": 0.3,
                    "failed_trajectories": [],
                    "soft_constraint_violations": {},
                },
            }
        ),
        encoding="utf-8",
    )
    rows = build_rows(
        expected_keys=(("cell", "easy", 0),),
        run_index={("cell", "easy", 0): run},
        audit_index={
            "run_0": {
                "status": "complete",
                "task_compliance": "pass",
                "scientific_absolute_assessments": [
                    {"verdict": "pass"},
                    {"verdict": "fail"},
                ],
            }
        },
        tool_index={
            "run_0": {
                "raw_processed_code_interpreter_calls": "12",
                "limit_exceeded": "False",
            }
        },
    )
    summary = aggregate(rows)

    assert rows[0]["validation_normalized_mse"] == 0.3
    assert rows[0]["task_compliance"] == "pass"
    assert rows[0]["scientific_fail"] == 1
    assert rows[0]["scientific_all_applicable_pass"] is False
    assert summary["numerical_completion_rate"] == 1.0
    assert summary["scientific_audit_coverage"] == 1.0
    assert summary["scientific_absolute_verdict_counts"]["fail"] == 1
    assert summary["scientific_accuracy_claimed"] is False
