"""Shard report checks without resubmitting completed scientific work."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from scripts import recover_prefix_noise_sharded as recovery
from tests.test_recover_prefix_noise_report import (
    campaign as report_campaign,
)
from tests.test_recover_prefix_noise_report import (
    preparation_campaign,
)

assert preparation_campaign and report_campaign


@pytest.fixture
def campaign(report_campaign, monkeypatch, tmp_path):
    c = report_campaign
    recovery.prior.recover(c.root)
    helper = tmp_path / "helpers"
    helper.mkdir()
    for name in recovery.FILES:
        (helper / name).write_text("fixed-helper\n")
    monkeypatch.setattr(recovery, "__file__", str(helper / recovery.FILES[0]))
    c.helper = helper
    c.retry_state = "TIMEOUT"
    c.retry_queue = ""
    c.failure_stage = None
    c.fail_mode = "timeout"
    c.sbatch.clear()
    c.commands.clear()
    c.envs.clear()
    original = recovery.subprocess.run

    def run(argv, **kwargs):
        if argv[0] == "sacct" and argv[argv.index("-j") + 1] == "900":
            return SimpleNamespace(stdout=(
                "900|prefix-noise-reliability-report-retry|"
                f"{c.retry_state}|0:0\n"
            ))
        if argv[0] == "squeue" and c.retry_queue:
            return SimpleNamespace(stdout=c.retry_queue)
        if argv[0] != "sbatch":
            return original(argv, **kwargs)
        stage = argv[-1]
        assert stage in ("check", "reduce")
        assert recovery.previous.read(
            c.root / recovery.RECOVERY / f"{stage}.intent.json"
        ) == {"argv": argv}
        c.sbatch.append(argv)
        c.envs.append(kwargs["env"])
        if c.failure_stage == stage:
            if c.fail_mode == "timeout":
                raise subprocess.TimeoutExpired(argv, 45)
            return SimpleNamespace(returncode=0, stdout="", stderr="not confirmed")
        return SimpleNamespace(
            returncode=0, stdout="1001\n" if stage == "check" else "1002\n", stderr=""
        )

    monkeypatch.setattr(recovery.subprocess, "run", run)
    return c


def snapshot(root):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_shards_preserve_prior_artifacts_and_submit_no_fitting(campaign):
    c = campaign
    before = snapshot(c.root)
    result = recovery.recover(c.root)
    assert result["jobs"] == {"check": "1001", "reduce": "1002"}
    assert not result["fitting_submitted"]
    assert result["array_tasks"] == 36
    assert result["gpus"] == result["live_llm_calls"] == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == v for p, v in before.items())
    check, reduce = c.sbatch
    assert "--array=0-35%2" in check
    assert not any(a.startswith("--dependency") for a in check)
    assert "--dependency=afterok:1001" in reduce
    assert all("--time=01:00:00" in argv for argv in c.sbatch)
    assert all(not any(a.startswith(("--gres", "--gpus")) for a in argv)
               for argv in c.sbatch)
    assert all(e["AF_REPO_ROOT"] == str(c.repo) for e in c.envs)
    assert all(e["AF_REPORT_HELPER_ROOT"] == str(c.helper) for e in c.envs)


def test_repeated_submission_reuses_confirmed_jobs(campaign):
    c = campaign
    result = recovery.recover(c.root)
    before = snapshot(c.root)
    c.retry_state = "must not query old status"
    assert recovery.recover(c.root) == result
    assert len(c.sbatch) == 2
    assert snapshot(c.root) == before


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "FAILED", "COMPLETED"])
def test_only_timed_out_retry_qualifies(campaign, state):
    campaign.retry_state = state
    with pytest.raises(ValueError, match="confirmed TIMEOUT"):
        recovery.recover(campaign.root)
    assert not campaign.sbatch


def test_queued_retry_blocks_submission(campaign):
    campaign.retry_queue = "900\n"
    with pytest.raises(ValueError, match="still queued"):
        recovery.recover(campaign.root)
    assert not campaign.sbatch


@pytest.mark.parametrize("name", ["identity.json", "report.accepted.json",
                                 "report.intent.json", "manifest.json"])
def test_missing_prior_confirmation_cannot_trigger_old_retry(campaign, name):
    (campaign.root / recovery.prior.RECOVERY / name).unlink()
    with pytest.raises(ValueError, match="missing confirmed"):
        recovery.recover(campaign.root)
    assert not campaign.sbatch


@pytest.mark.parametrize("stage", ["check", "reduce"])
@pytest.mark.parametrize("mode", ["timeout", "empty"])
def test_uncertain_submission_is_never_repeated(campaign, stage, mode):
    c = campaign
    c.failure_stage, c.fail_mode = stage, mode
    with pytest.raises(ValueError, match=r"unknown|unconfirmed"):
        recovery.recover(c.root)
    attempts = len(c.sbatch)
    with pytest.raises(ValueError, match="Unconfirmed"):
        recovery.recover(c.root)
    assert len(c.sbatch) == attempts


def test_changed_helper_is_rejected_without_new_submission(campaign):
    c = campaign
    recovery.recover(c.root)
    (c.helper / recovery.FILES[-1]).write_text("changed")
    with pytest.raises(ValueError, match="identity differs"):
        recovery.recover(c.root)
    assert len(c.sbatch) == 2
