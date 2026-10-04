"""Quota recovery must preserve authoritative records and the frozen runtime."""

import errno
import fcntl
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from scripts import recover_phase_c_storage as storage
from tests.test_construction_comparison import (
    campaign,
    source_fixture,
    tokenize,
    transport,
)


def prepared(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "comparison"
    plan = campaign.freeze(tmp_path / "new", root)
    task, calls = plan["tasks"][0], []
    result = campaign.propose(
        root,
        plan,
        task,
        "http://offline",
        transport=transport(calls),
        token_transport=tokenize,
    )
    assert result["status"] == "topology_complete"
    directory = campaign.baseline.location(root, task)
    return root, plan, directory, calls


def authoritative(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
        and (
            p.name in {"plan.json", "proposal.json", "before_repair.json"}
            or "calls" in p.parts
            or "events" in p.parts
        )
    }


def test_recovery_reclaims_only_views_preserves_results_and_uses_no_llm(tmp_path):
    root, plan, directory, calls = prepared(tmp_path)
    before, n = authoritative(root), len(calls)
    renderer = campaign.construction_trace.render
    audit = storage.inventory(root, campaign, plan)
    assert audit["status_counts"] == {"topology_complete": 1, "unstarted": 95}
    assert audit["reclaimable_bytes"] > 0
    result = storage.recover(root, campaign, plan)
    assert result["status_counts"] == {"topology_complete": 1, "pending": 95}
    assert result["removed_derived_files"] == 2
    assert authoritative(root) == before and len(calls) == n
    assert campaign.construction_trace.render is renderer
    assert campaign.verify(root) == plan
    now = storage.inventory(root, campaign, plan)
    assert now["derived_bytes"] < audit["derived_bytes"] / 5
    index = json.loads((directory / "trace.json").read_text())
    assert index["policy"] == storage.POLICY
    assert len(index["calls"]) == n
    for row in index["calls"]:
        assert (directory / row["call_file"]).is_file()
        assert row["event_files"]
        assert "raw_response" not in row and "request" not in row
    package = tmp_path / "inspection.tar.gz"
    storage.export(root, package)
    with tarfile.open(package) as archive:
        names = archive.getnames()
        assert "plan.json" in names and "SUMMARY.md" in names
        assert str((directory / "TRACE.html").relative_to(root)) in names
        assert (
            sum(
                Path(name).parent.name == "calls" and name.endswith(".json")
                for name in names
            )
            == n
        )
    again = storage.recover(root, campaign, plan)
    assert again["status_counts"] == result["status_counts"]
    assert authoritative(root) == before


def test_missing_call_or_corrupt_checkpoint_preserves_derived_evidence(tmp_path):
    root, plan, directory, _ = prepared(tmp_path)
    next((directory / "calls").glob("*.json")).write_text("{broken")
    before = {name: (directory / name).read_bytes() for name in storage.DERIVED}
    result = storage.recover(root, campaign, plan)
    assert result["status"] == "checkpoint_errors_require_inspection"
    assert result["removed_derived_files"] == 0
    assert before == {name: (directory / name).read_bytes() for name in storage.DERIVED}
    assert not (root / "summary.json").exists()


def test_derived_enospc_does_not_damage_primary_or_leave_partial_view(
    tmp_path, monkeypatch
):
    root, plan, directory, _ = prepared(tmp_path)
    before = authoritative(root)
    write = Path.write_text

    def no_space(path, *args, **kwargs):
        if ".recovery-" in path.name:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", no_space)
    renderer = campaign.construction_trace.render
    with pytest.raises(OSError, match="Disk quota"):
        storage.recover(root, campaign, plan)
    assert authoritative(root) == before
    assert campaign.construction_trace.render is renderer
    assert not list(directory.glob(".*.recovery-*.tmp"))


def test_active_construction_refused_before_cleanup(tmp_path):
    root, plan, directory, _ = prepared(tmp_path)
    path = directory / "construction-lock/.lock"
    with path.open("r") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            storage.recover(root, campaign, plan)
    assert all((directory / name).is_file() for name in storage.DERIVED)


def test_cleanup_rejects_unapproved_paths_and_symlinks(tmp_path):
    root = tmp_path.resolve()
    original = root / "proposal.json"
    original.write_text("{}")
    audit = {
        "rows": [
            {"task": "x", "evidence_valid": True, "derived_files": ["proposal.json"]}
        ]
    }
    with pytest.raises(ValueError, match="approved"):
        storage.reclaim(root, audit)
    assert original.read_text() == "{}"
    link = root / "TRACE.html"
    link.symlink_to(original)
    with pytest.raises(ValueError, match="linked"):
        storage.safe_file(link, root)


def test_export_failure_removes_only_its_temporary_file(tmp_path, monkeypatch):
    root, plan, _, _ = prepared(tmp_path)
    storage.recover(root, campaign, plan)
    destination = tmp_path / "inspection.tar.gz"
    destination.write_bytes(b"previous archive")

    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left")

    monkeypatch.setattr(tarfile.TarFile, "add", fail)
    with pytest.raises(OSError):
        storage.export(root, destination)
    assert destination.read_bytes() == b"previous archive"
    assert not list(tmp_path.glob("inspection.tar.gz.*.tmp"))


def test_archive_cannot_replace_authoritative_evidence(tmp_path):
    root, _, directory, _ = prepared(tmp_path)
    before = authoritative(root)
    for destination in (root / "plan.json", directory / "calls/archive.tar.gz"):
        with pytest.raises(ValueError, match="archive destination"):
            storage.export(root, destination)
    assert authoritative(root) == before


def test_orphan_views_are_not_assumed_reconstructable(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "comparison"
    plan = campaign.freeze(tmp_path / "new", root)
    directory = campaign.baseline.location(root, plan["tasks"][0])
    directory.mkdir(parents=True)
    (directory / "TRACE.html").write_text("only surviving evidence")
    result = storage.recover(root, campaign, plan)
    assert result["status"] == "checkpoint_errors_require_inspection"
    assert (directory / "TRACE.html").read_text() == "only surviving evidence"
