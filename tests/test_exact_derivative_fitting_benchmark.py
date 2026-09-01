"""Tests for the synthetic exact-derivative fitting benchmark."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.run_exact_derivative_fitting_benchmark import run_benchmark

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "exact_derivative_fitting_benchmark_v1.json"
JOB = ROOT / "scripts" / "hpc" / "exact_derivative_fitting_benchmark.slurm"


def test_exact_derivative_benchmark_recovers_all_parameters(tmp_path: Path) -> None:
    summary = run_benchmark(CONFIG, tmp_path)
    groups = {item["backend"]: item for item in summary["groups"]}

    assert summary["derivative_provenance"] == "exact"
    assert summary["test_evaluated_only_after_parameter_freeze"] is True
    assert summary["weighted_overall_score_defined"] is False
    assert groups["exact_derivative_linear_ridge"]["success_rate"] == 1.0
    assert (
        groups["exact_derivative_linear_ridge"]
        ["maximum_absolute_parameter_error_mean"]
        < 1e-8
    )
    assert groups["exact_derivative_linear_ridge"]["test_nmse_mean"] < 1e-12
    assert groups["bounded_nonlinear"]["success_rate"] == 1.0
    rows = [
        json.loads(line)
        for line in (tmp_path / "benchmark_rows.jsonl").read_text().splitlines()
    ]
    exact = [
        item for item in rows if item["backend"] == "exact_derivative_linear_ridge"
    ]
    assert {tuple(item["diagnostic_backends"]) for item in exact} == {
        ("exact_derivative_linear_ridge",)
    }
    assert {item["function_evaluations"] for item in exact} == {1}
    assert (tmp_path / "fitted_parameter_freeze.jsonl").is_file()

    repeated = run_benchmark(CONFIG, tmp_path)
    assert repeated["fitted_parameter_freeze_sha256"] == (
        summary["fitted_parameter_freeze_sha256"]
    )


def test_exact_derivative_job_is_cpu_only_and_syntactically_valid() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "--gres" not in text
    assert "--gpus" not in text
    assert "OPENBLAS_NUM_THREADS=1" in text
    subprocess.run(["bash", "-n", str(JOB)], check=True)
