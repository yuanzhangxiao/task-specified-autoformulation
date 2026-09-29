"""Only finished compiler-cache trees may be archived and retired."""

import json
import shutil
import tarfile

import pytest

from scripts import archive_finished_component_caches as cache


@pytest.fixture
def paths(tmp_path):
    root = tmp_path / "component-runtime-cache"
    for job in ("123", "456"):
        for name in cache.CACHE_KINDS:
            (root / job / name).mkdir(parents=True)
        (root / job / "triton/kernel.bin").write_bytes(b"a compiled kernel\x00")
        (root / job / "vllm/config.json").write_text('{"compiled":true}')
    protected = tmp_path / "final-components-v1/judge_cache"
    protected.mkdir(parents=True)
    (protected / "response.json").write_text("scientific response must survive")
    return root, tmp_path / "archives", protected


def completed(_jobs):
    pass


def test_archive_roundtrip_and_idempotent_removal(paths, tmp_path):
    root, destination, protected = paths
    before = cache.snapshot(root, "123")
    planned = cache.run(root, destination, ["123"], apply=False, guard=completed)
    assert planned["entries"] == {"123": len(before)}
    assert not destination.exists()
    result = cache.run(root, destination, ["123"], apply=True, guard=completed)
    assert result["entries_removed"] == len(before)
    assert not (root / "123").exists() and (root / "456").is_dir()
    assert (
        protected / "response.json"
    ).read_text() == "scientific response must survive"
    restored = tmp_path / "restored"
    restored.mkdir()
    with tarfile.open(destination / "123.tar") as archive:
        archive.extractall(restored, filter="data")
    assert cache.snapshot(restored, "123") == before
    again = cache.run(root, destination, ["123"], apply=True, guard=completed)
    assert again["entries_removed"] == 0


@pytest.mark.parametrize("failure", ["symlink", "hardlink", "unexpected", "fifo"])
def test_unsafe_entries_preserve_sources(paths, failure):
    root, destination, _ = paths
    if failure == "symlink":
        (root / "123/triton/linked").symlink_to(root / "456", target_is_directory=True)
    elif failure == "hardlink":
        (root / "123/triton/copy").hardlink_to(root / "123/triton/kernel.bin")
    elif failure == "unexpected":
        (root / "123/judge_cache").mkdir()
    else:
        cache.os.mkfifo(root / "123/triton/pipe")
    with pytest.raises(ValueError):
        cache.run(root, destination, ["123"], apply=True, guard=completed)
    assert (root / "123/triton/kernel.bin").exists()


def test_corrupted_archive_prevents_removal(paths):
    root, destination, _ = paths
    destination.mkdir()
    cache.prepare_archive(root, destination, "123")
    archive = destination / "123.tar"
    with archive.open("r+b") as stream:
        stream.seek(0)
        stream.write(b"bad header")
    with pytest.raises((ValueError, tarfile.TarError)):
        cache.run(root, destination, ["123"], apply=True, guard=completed)
    assert (root / "123/triton/kernel.bin").exists()


def test_changed_cache_and_missing_receipt_preserve_sources(paths):
    root, destination, _ = paths
    destination.mkdir()
    cache.prepare_archive(root, destination, "123")
    (root / "123/triton/kernel.bin").write_text("new kernel")
    with pytest.raises(ValueError, match="differs"):
        cache.run(root, destination, ["123"], apply=True, guard=completed)
    (destination / "123.json").unlink()
    with pytest.raises(ValueError, match="differs"):
        cache.run(root, destination, ["123"], apply=True, guard=completed)
    assert (root / "123/triton/kernel.bin").read_text() == "new kernel"


def test_orphan_archive_recovered_only_against_intact_source(paths):
    root, destination, _ = paths
    destination.mkdir()
    cache.prepare_archive(root, destination, "123")
    (destination / "123.json").unlink()
    result = cache.run(root, destination, ["123"], apply=True, guard=completed)
    assert result["entries_removed"] > 0
    assert (destination / "123.json").exists()


def test_partial_removal_resumes_from_verified_archive(paths):
    root, destination, _ = paths
    destination.mkdir()
    cache.prepare_archive(root, destination, "123")
    shutil.rmtree(root / "123/triton")
    assert (
        cache.run(root, destination, ["123"], apply=True, guard=completed)[
            "entries_removed"
        ]
        > 0
    )


def test_second_guard_failure_prevents_all_removal(paths):
    root, destination, _ = paths
    calls = []

    def guard(jobs):
        calls.append(jobs)
        if len(calls) == 2:
            raise ValueError("active worker appeared")

    with pytest.raises(ValueError, match="active worker"):
        cache.run(root, destination, ["123", "456"], apply=True, guard=guard)
    assert (root / "123").exists() and (root / "456").exists()
    assert json.loads((destination / "123.json").read_text())["archive_sha256"]


@pytest.mark.parametrize("problem", ["running", "missing", "wrong_name", "queued"])
def test_scheduler_requires_positive_exact_raw_id_evidence(monkeypatch, problem):
    def output(argv, **_kwargs):
        if argv[0] == "squeue":
            return "component-critic\n" if problem == "queued" else ""
        header = "JobID|JobIDRaw|JobName|State|ExitCode\n"
        row = "100_9|123|component-propose|COMPLETED|0:0\n"
        if problem == "running":
            row = row.replace("COMPLETED", "RUNNING")
        if problem == "wrong_name":
            row = row.replace("component-propose", "other-job")
        return header + ("" if problem == "missing" else row)

    monkeypatch.setattr(cache.subprocess, "check_output", output)
    with pytest.raises(ValueError):
        cache.guard_jobs(["123"])


def test_scheduler_matches_raw_id_not_array_suffix(monkeypatch):
    def output(argv, **_kwargs):
        return (
            ""
            if argv[0] == "squeue"
            else (
                "JobID|JobIDRaw|JobName|State|ExitCode\n"
                "100_9|123|component-propose|COMPLETED|0:0\n"
                "100_9.batch|123.batch|batch|COMPLETED|0:0\n"
            )
        )

    monkeypatch.setattr(cache.subprocess, "check_output", output)
    cache.guard_jobs(["123"])


def test_root_alias_overlap_and_traversal_rejected(paths, tmp_path):
    root, destination, _ = paths
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    for r, d, jobs in (
        (alias, destination, ["123"]),
        (root, root / "archive", ["123"]),
        (root, destination, ["../outside"]),
    ):
        with pytest.raises(ValueError):
            cache.run(r, d, jobs, apply=True, guard=completed)
    assert (root / "123/triton/kernel.bin").exists()
