"""Observational operations tools preserve uncertainty and never delete sources."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import component_scheduler_audit as scheduler
from scripts import inventory_research_storage as storage

HEADER = "JobID|JobName|State|ExitCode|Elapsed\n"


def audit():
    return {
        "artifact_sha256": "fixture",
        "submissions": [{"wave": "w1", "jobs": {"critic": "100", "fit": "200"}}],
        "workers": [{"worker": "a", "stage": "critic", "job_id": "101"}],
        "lineages": [{"task": "t1", "frontier": {"action": "critic", "round": 6}}],
    }


def test_array_counts_ignore_steps_and_parent_and_keep_missing_submission():
    text = HEADER + (
        "100|critic|COMPLETED|0:0|05:00:00\n"
        "100_0|critic|FAILED|1:0|00:01:00\n"
        "100_0.batch|batch|FAILED|1:0|00:01:00\n"
        "100_0.extern|extern|COMPLETED|0:0|00:01:00\n"
        "100_1|critic|COMPLETED|0:0|05:00:00\n"
        "900|other|RUNNING|0:0|00:03:00\n"
    )
    result = scheduler.reconcile(audit(), text)
    assert result["observed_tasks"] == 2
    assert result["state_counts"] == {"COMPLETED": 1, "FAILED": 1}
    assert result["unobserved_submissions"][0]["submission_id"] == "200"
    assert result["worker_match_counts"] == {"unresolved": 1}
    assert len(result["non_success_steps"]) == 1
    assert result["unmatched_rows"] == 1
    assert result["remaining_work"][0]["action"] == "critic"
    assert result["resume_authorized"] is False


def test_raw_ids_match_workers_without_arithmetic_guess():
    text = HEADER.rstrip() + "|JobIDRaw\n100_0|critic|FAILED|1:0|00:01:00|101\n"
    result = scheduler.reconcile(audit(), text)
    assert result["worker_match_counts"] == {"exact": 1}
    assert result["workers"][0]["scheduler_rows"][0]["JobID"] == "100_0"


def test_active_unknown_and_conflicting_states_are_not_terminal_certification():
    text = HEADER + (
        "100_0|critic|RUNNING|0:0|00:01:00\n"
        "100_1|critic|OUT_OF_ME+|1:0|00:02:00\n"
        "100_2|critic|CANCELLED by 5|0:15|00:03:00\n"
        "100_0|critic|COMPLETED|0:0|00:04:00\n"
    )
    result = scheduler.reconcile(audit(), text)
    assert result["observed_active_tasks"] == ["100_0"]
    assert result["unknown_state_tasks"] == ["100_1"]
    assert result["conflicting_job_ids"] == ["100_0"]


@pytest.mark.parametrize(
    "text", ["", "JobID|State\n1|COMPLETED\n", HEADER + "10+|x|RUNNING|0:0|1\n"]
)
def test_scheduler_invalid_input_fails_explicitly(text):
    with pytest.raises(ValueError):
        scheduler.parse(text)


def test_empty_scheduler_is_unknown_and_never_complete():
    result = scheduler.reconcile(audit(), HEADER)
    assert result["observed_tasks"] == 0
    assert len(result["unobserved_submissions"]) == 2
    assert result["worker_match_counts"] == {"unresolved": 1}


def test_storage_metadata_only_protection_hardlinks_and_symlinks(tmp_path, monkeypatch):
    root = tmp_path / "data"
    cache = root / "__pycache__"
    cache.mkdir(parents=True)
    protected = root / "campaign"
    protected.mkdir()
    original = cache / "a.pyc"
    original.write_bytes(b"private_payload" * 300)
    (cache / "b.pyc").hardlink_to(original)
    (protected / "test.csv").write_text("SECRET")
    (root / "linked_directory").symlink_to(tmp_path, target_is_directory=True)
    (root / "dangling").symlink_to(tmp_path / "missing")

    def forbidden(*args, **kwargs):
        raise AssertionError("inventory must not read file contents")

    monkeypatch.setattr(Path, "open", forbidden)
    result = storage.inventory([root], [protected], 100, 2)["roots"][0]
    assert result["complete"]
    assert result["totals"]["files"] == 3
    assert result["totals"]["symlinks"] == 2
    assert result["totals"]["apparent_file_bytes"] == original.stat().st_size * 2 + 6
    cached = next(b for b in result["buckets"] if b["path"] == "__pycache__")
    assert (
        cached["allocated_bytes_unique_inodes"]
        == (original.stat().st_blocks + cache.stat().st_blocks) * 512
    )
    saved = next(b for b in result["buckets"] if b["path"] == "campaign")
    assert saved["metadata_policy_counts"] == {"protected": 2}
    assert all(not b["deletion_eligible"] for b in result["buckets"])
    assert original.exists()


def test_storage_budget_and_errors_stay_incomplete(tmp_path, monkeypatch):
    for i in range(4):
        (tmp_path / str(i)).touch()
    result = storage.inventory([tmp_path], [], 2, 1)["roots"][0]
    assert result["truncated"] and not result["complete"]
    assert result["entries_visited"] == 2

    def denied(*args, **kwargs):
        raise PermissionError("cannot enumerate")

    monkeypatch.setattr(storage.os, "scandir", denied)
    result = storage.inventory([tmp_path], [], 20, 1)["roots"][0]
    assert not result["complete"]
    assert result["issue_counts"] == {"directory_unreadable": 1}


def test_storage_roots_cannot_overlap_or_be_symlinks(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        storage.inventory([tmp_path, child], [], 20, 1)
    alias = tmp_path / "alias"
    alias.symlink_to(child, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        storage.inventory([alias], [], 20, 1)


def test_storage_cli_preserves_sources_and_refuses_output_inside_root(tmp_path):
    script = (
        Path(__file__).resolve().parents[1] / "scripts/inventory_research_storage.py"
    )
    root = tmp_path / "source"
    root.mkdir()
    (root / "result.json").write_text("not opened")
    output = tmp_path / "report"
    command = [sys.executable, "-I", str(script), "--root", str(root), "--output"]
    subprocess.run([*command, str(output)], check=True, capture_output=True)
    value = json.loads((output / "inventory.json").read_text())
    assert value["content_files_opened"] == value["files_deleted"] == 0
    assert (root / "result.json").read_text() == "not opened"
    for bad in (output, root / "output"):
        assert subprocess.run([*command, str(bad)], capture_output=True).returncode != 0
