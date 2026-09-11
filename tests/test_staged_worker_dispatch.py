"""Exercise the actual shell dispatcher before allocating cluster resources."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/hpc/run_staged_topology_server.sh"
CONFIGS = sorted((ROOT / "configs").glob("staged_multiround_feedback_v*.json"))
pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="jq required")


def _clean_environment():
    """Prove check-only dispatch does not require cluster runtime variables."""
    return {
        key: value for key, value in os.environ.items() if not key.startswith("AF_")
    }


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.stem)
def test_every_committed_multiround_config_has_a_shell_worker(config):
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--check-config", str(config)],
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "staged_multiround_feedback_campaign.py"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "protocol",
            "scientific-staged-multiround-feedback-999",
            "unsupported frozen worker protocol",
        ),
        ("platform", "unknown-gpu", "unsupported frozen platform"),
    ],
)
def test_unknown_worker_contract_is_not_silently_accepted(
    tmp_path, field, value, message
):
    config = {
        "protocol": "scientific-staged-multiround-feedback-6",
        "platform": "aces-h100x1",
    }
    config[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--check-config", str(path)],
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_submission_rejects_unsupported_protocol_before_inputs_or_scheduler(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"protocol": "unsupported", "platform": "aces-h100x1"})
    )
    env = _clean_environment()
    env.update(
        AF_REPO_ROOT=str(ROOT),
        AF_PYTHON=sys.executable,
        AF_CONFIG=str(config),
        AF_SOURCE_RESCUE_ROOT=str(tmp_path / "missing-source"),
        AF_OUTPUT_ROOT=str(tmp_path / "not-created"),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/hpc/submit_staged_multiround_feedback_aces.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "unsupported frozen worker protocol" in result.stderr
    assert "missing source" not in result.stderr
    assert not (tmp_path / "not-created").exists()


def test_v6_runtime_dispatch_reaches_frozen_platform_check(tmp_path):
    """Exercise the production (non-check-only) branch without a GPU or scheduler."""
    (tmp_path / "plan.json").write_text(
        json.dumps(
            {
                "config": {
                    "protocol": "scientific-staged-multiround-feedback-6",
                    "platform": "invalid",
                }
            }
        )
    )
    env = _clean_environment()
    env.update(
        {
            key: str(tmp_path)
            for key in (
                "AF_REPO_ROOT",
                "AF_PYTHON",
                "AF_OUTPUT_ROOT",
                "AF_VLLM_IMAGE",
                "AF_HF_HOME",
                "AF_COMPUTE_CACHE_ROOT",
                "AF_IPC_TMP_ROOT",
            )
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "unsupported frozen platform" in result.stderr
    assert "unsupported frozen worker protocol" not in result.stderr
