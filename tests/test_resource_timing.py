"""Tests for the portable child-process resource wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("exit_code", [0, 7])
def test_resource_wrapper_records_usage_and_preserves_exit_code(
    tmp_path: Path,
    exit_code: int,
) -> None:
    output = tmp_path / "process_time.txt"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_with_resource_timing.py",
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            f"raise SystemExit({exit_code})",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == exit_code
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "portable-child-process-timing-1"
    assert payload["exit_code"] == exit_code
    assert payload["elapsed_seconds"] >= 0.0
    values = dict(
        line.split("=", 1)
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert int(values["exit_code"]) == exit_code
    assert float(values["user_cpu_seconds"]) >= 0.0


def test_resource_wrapper_can_write_atomic_json(tmp_path: Path) -> None:
    output = tmp_path / "process_time.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_with_resource_timing.py",
            "--output",
            str(output),
            "--output-format",
            "json",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "portable-child-process-timing-1"
    assert payload["exit_code"] == 0
