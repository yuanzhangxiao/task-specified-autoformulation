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
    assert json.loads((destination / "123.json").read_text())["protocol"] == (
        "finished-component-cache-archive-1"
    )
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


@pytest.fixture
def phase_paths(paths):
    root, destination, protected = paths
    phase_root = root.with_name("phase-c-runtime-cache")
    root.rename(phase_root)
    return phase_root, destination, protected


def phase_accounting(state="FAILED", owner=None):
    """Include allocation and step rows to exercise exact allocation matching."""
    owner = str(cache.os.getuid()) if owner is None else owner
    return (
        "JobID|JobIDRaw|JobName|State|ExitCode|UID\n"
        f"100_9|123|phasec-confirmation-propose|{state}|1:0|{owner}\n"
        f"100_9.batch|123.batch|batch|FAILED|1:0|{owner}\n"
    )


@pytest.mark.parametrize("state", [*sorted(cache.TERMINAL_STATES), "CANCELLED by 99"])
def test_phase_c_terminal_states_allowed(monkeypatch, state):
    calls = []

    def output(argv, **_kwargs):
        calls.append(argv)
        return "" if argv[0] == "squeue" else phase_accounting(state)

    monkeypatch.setattr(cache.subprocess, "check_output", output)
    cache.guard_phase_c_jobs(["123"])
    assert calls[0] == [
        "squeue",
        "--noheader",
        "--user",
        str(cache.os.getuid()),
        "--format=%i",
    ]
    assert calls[1][1:3] == ["-j", "123"]


@pytest.mark.parametrize(
    "problem",
    [
        "active_unrelated_job",
        "missing",
        "step_only",
        "duplicate",
        "wrong_owner",
        "RUNNING",
        "PENDING",
        "COMPLETING",
        "REQUEUED",
        "SPECIAL_EXIT",
        "SUSPENDED",
        "CANCELLED+",
        "CANCELLED by unknown",
    ],
)
def test_phase_c_scheduler_rejects_unconfirmed_jobs(phase_paths, monkeypatch, problem):
    root, destination, _ = phase_paths

    def output(argv, **_kwargs):
        if argv[0] == "squeue":
            return "999\n" if problem == "active_unrelated_job" else ""
        if problem == "missing":
            return "JobID|JobIDRaw|JobName|State|ExitCode|UID\n"
        if problem == "step_only":
            return phase_accounting().replace("100_9|123|", "100_9.batch|123|")
        if problem == "duplicate":
            result = phase_accounting()
            return result + result.splitlines()[1] + "\n"
        if problem == "wrong_owner":
            return phase_accounting(owner=str(cache.os.getuid() + 1))
        return phase_accounting(problem)

    monkeypatch.setattr(cache.subprocess, "check_output", output)
    with pytest.raises(ValueError):
        cache.run(root, destination, ["123"], apply=True, profile="phase-c")
    assert (root / "123/triton/kernel.bin").is_file()
    assert not destination.exists()


def test_phase_c_failed_job_archive_roundtrip(phase_paths, monkeypatch):
    root, destination, protected = phase_paths
    calls = []

    def output(argv, **_kwargs):
        calls.append(argv[0])
        return "" if argv[0] == "squeue" else phase_accounting()

    monkeypatch.setattr(cache.subprocess, "check_output", output)
    before = cache.snapshot(root, "123")
    planned = cache.run(root, destination, ["123"], apply=False, profile="phase-c")
    assert planned["entries"] == {"123": len(before)}
    assert not destination.exists()
    calls.clear()
    result = cache.run(root, destination, ["123"], apply=True, profile="phase-c")
    assert calls == ["squeue", "sacct", "squeue", "sacct"]
    assert result["entries_removed"] == len(before)
    receipt = json.loads((destination / "123.json").read_text())
    assert receipt["protocol"] == "finished-phase-c-cache-archive-1"
    assert (
        cache.verify_archive(destination / "123.tar", before)
        == (receipt["archive_sha256"])
    )
    assert not (root / "123").exists() and (root / "456").is_dir()
    assert (protected / "response.json").is_file()
    assert (
        cache.run(root, destination, ["123"], apply=True, profile="phase-c")[
            "entries_removed"
        ]
        == 0
    )


def test_phase_c_queue_rechecked_before_removal(phase_paths, monkeypatch):
    root, destination, _ = phase_paths
    queue_calls = 0

    def output(argv, **_kwargs):
        nonlocal queue_calls
        if argv[0] == "squeue":
            queue_calls += 1
            return "" if queue_calls == 1 else "999\n"
        return phase_accounting()

    monkeypatch.setattr(cache.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="queued/running"):
        cache.run(root, destination, ["123"], apply=True, profile="phase-c")
    assert (root / "123/triton/kernel.bin").is_file()
    assert (destination / "123.json").is_file()


def test_phase_c_rejects_wrong_root_and_default_profile(phase_paths):
    root, destination, _ = phase_paths
    with pytest.raises(ValueError, match="component-runtime-cache"):
        cache.run(root, destination, ["123"], apply=False, guard=completed)
    for name in ("component-runtime-cache", "runtime-cache", "results"):
        wrong_root = root.with_name(name)
        wrong_root.mkdir()
        with pytest.raises(ValueError, match="phase-c-runtime-cache"):
            cache.run(
                wrong_root,
                destination,
                ["123"],
                apply=False,
                profile="phase-c",
                guard=completed,
            )
    assert not destination.exists()


def test_phase_c_checksum_tampering_preserves_originals(phase_paths):
    root, destination, _ = phase_paths
    destination.mkdir()
    cache.prepare_archive(root, destination, "123", profile="phase-c")
    archive_path = destination / "123.tar"
    with tarfile.open(archive_path) as archive:
        position = archive.getmember("123/triton/kernel.bin").offset_data
    with archive_path.open("r+b") as stream:
        stream.seek(position)
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        cache.run(
            root,
            destination,
            ["123"],
            apply=True,
            profile="phase-c",
            guard=completed,
        )
    assert (root / "123/triton/kernel.bin").is_file()


def test_phase_c_receipt_profile_must_match(phase_paths):
    root, destination, _ = phase_paths
    destination.mkdir()
    cache.prepare_archive(root, destination, "123")
    with pytest.raises(ValueError, match="identity differs"):
        cache.run(
            root,
            destination,
            ["123"],
            apply=True,
            profile="phase-c",
            guard=completed,
        )
    assert (root / "123/triton/kernel.bin").is_file()


def test_cli_phase_c_is_explicit_and_defaults_to_read_only(monkeypatch, capsys):
    monkeypatch.setattr(
        cache.sys,
        "argv",
        [
            "archive_finished_component_caches.py",
            "--profile",
            "phase-c",
            "--root",
            "/cache/phase-c-runtime-cache",
            "--archive-root",
            "/archive",
            "--jobs",
            "123",
        ],
    )
    observed = {}

    def run(root, destination, jobs, **kwargs):
        observed.update(root=root, destination=destination, jobs=jobs, **kwargs)
        return {"apply": kwargs["apply"]}

    monkeypatch.setattr(cache, "run", run)
    cache.main()
    assert observed["profile"] == "phase-c"
    assert observed["apply"] is False
    assert observed["jobs"] == ["123"]
    assert json.loads(capsys.readouterr().out) == {"apply": False}
