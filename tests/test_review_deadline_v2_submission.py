"""Versioned campaign launcher guards and shared server dispatch."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/hpc/submit_review_deadline_v2_aces.sh"


def _launch(tmp_path, *, config_protocol="review-deadline-2", old_protocol=None):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"protocol": config_protocol}))
    root = tmp_path / "output"
    root.mkdir(exist_ok=True)
    if old_protocol:
        (root / "plan.json").write_text(
            json.dumps({"config": {"protocol": old_protocol}})
        )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env={
            **os.environ,
            "AF_REPO_ROOT": str(ROOT),
            "AF_CONFIG": str(config),
            "AF_OUTPUT_ROOT": str(root),
        },
        capture_output=True,
        text=True,
    )
    return result, root


def test_v2_launcher_rejects_historical_config_before_scheduler(tmp_path):
    result, _ = _launch(tmp_path, config_protocol="review-deadline-1")
    assert result.returncode == 2
    assert "requires protocol review-deadline-2" in result.stderr


def test_v2_launcher_preserves_historical_plan(tmp_path):
    result, root = _launch(tmp_path, old_protocol="review-deadline-1")
    assert result.returncode == 2
    assert "Historical output is preserved" in result.stderr
    assert json.loads((root / "plan.json").read_text())["config"]["protocol"] == (
        "review-deadline-1"
    )


def test_v2_launcher_rejects_nonempty_unplanned_directory(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    (root / "keep.txt").write_text("existing result")
    result, _ = _launch(tmp_path)
    assert result.returncode == 2
    assert "nonempty without a plan" in result.stderr
    assert (root / "keep.txt").read_text() == "existing result"


def test_shared_server_dispatches_v2_to_review_worker(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"protocol": "review-deadline-2", "platform": "aces-h100x1"})
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/hpc/run_staged_topology_server.sh"),
            "--check-config",
            str(config),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "review_deadline.py"


def test_all_public_assets_checked_before_creating_output(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"protocol": "review-deadline-2", "public_cells": ["first", "last"]})
    )
    public_root = tmp_path / "public"
    first = public_root / "phase_b_v1/first"
    first.mkdir(parents=True)
    for name in ("manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv"):
        (first / name).write_text("fixture")
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env={
            **os.environ,
            "AF_REPO_ROOT": str(ROOT),
            "AF_CONFIG": str(config),
            "AF_OUTPUT_ROOT": str(output),
            "AF_PUBLIC_ROOT": str(public_root),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.count("Missing public development file:") == 4
    assert not output.exists()


def test_corrected_launcher_and_worker_shell_syntax():
    for path in (LAUNCHER, ROOT / "scripts/hpc/run_review_deadline_aces.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
