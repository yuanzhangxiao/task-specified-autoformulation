"""Storage integrity and disjoint scheduler ownership for the combined ACES array."""

import json
import tarfile

import pytest

from scripts import iclr_stress_remainder as jobs
from scripts import iclr_stress_solver_archive as archive


def roots(tmp_path):
    root, mirror = tmp_path / "campaign", tmp_path / "mirror"
    names = {"results/block0000_full/round_00/fit/backend"}
    (name,) = names
    work = root / name
    work.mkdir(parents=True)
    (work / "iterations.json").write_bytes(b'{"n":12}\n')
    (work / "stages").mkdir()
    (work / "stages/first").write_bytes(b"first")
    fit = {"identity": "qualified"}
    (work.parent / "result.json").write_text(
        json.dumps({"sha256": archive.public.content_sha256(fit), "result": fit})
    )
    return root, mirror, names, work


def test_compaction_restores_every_byte_and_keeps_the_fit_receipt(tmp_path):
    root, mirror, names, work = roots(tmp_path)
    bind = {"block": 0, "plan": "test"}
    before = (work.parent / "result.json").read_bytes()
    info = archive.archive_paths(root, names, "block0000", mirror, bind)
    assert info["entries"] == 4 and not work.exists()
    assert (work.parent / "result.json").read_bytes() == before
    copy = mirror / info["archive"]
    assert (
        archive.digest(copy)
        == archive.digest(root / archive.STORE / info["archive"])
        == info["sha256"]
    )
    with tarfile.open(copy) as t:
        assert t.extractfile(next(iter(names)) + "/stages/first").read() == b"first"
    # Simulate a partially retired or restored tree. Altered members are never unlinked.
    work.mkdir()
    (work / "iterations.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from archive"):
        archive.archive_paths(root, names, "block0000", mirror, bind)
    assert (work / "iterations.json").read_bytes() == b"changed"
    (work / "iterations.json").write_bytes(b'{"n":12}\n')
    assert archive.archive_paths(root, names, "block0000", mirror, bind) == info
    assert not work.exists()


def test_backup_mismatch_and_link_cannot_retire_files(tmp_path):
    root, mirror, names, work = roots(tmp_path)
    link = work / "link"
    link.symlink_to(work.parent / "result.json")
    with pytest.raises(ValueError, match="link or special"):
        archive.archive_paths(root, names, "block0000", mirror, {})
    link.unlink()
    (mirror / "block0000.tar.gz").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="second archive"):
        archive.archive_paths(root, names, "block0000", mirror, {})
    assert (work / "stages/first").exists()


def roster(tmp_path):
    for i in range(1, 9):
        p = tmp_path / "submissions" / f"wave{i}"
        p.mkdir(parents=True)
        (p / "array.id").write_text(jobs.PREVIOUS if i == 8 else str(i))
        (p / "array.intent.json").write_text(
            json.dumps({"argv": ["sbatch", f"--array={3 * i - 3}-{3 * i - 1}%2"]})
        )


def test_previous_wave_coverage_refuses_overlapping_roster_and_uncertain_submit(
    tmp_path,
):
    roster(tmp_path)
    jobs.previous_waves(tmp_path)
    assert jobs.array_blocks("24-56%2") == set(range(24, 57))
    folder = tmp_path / "submissions" / "another"
    folder.mkdir()
    (folder / "array.intent.json").write_text("{}")
    with pytest.raises(ValueError, match="reconcile"):
        jobs.previous_waves(tmp_path)
    (folder / "array.intent.json").unlink()
    (tmp_path / "submissions/wave8/array.intent.json").write_text(
        json.dumps({"argv": ["--array=21,22,24%2"]})
    )
    with pytest.raises(ValueError, match="prior Slurm"):
        jobs.previous_waves(tmp_path)


def test_one_dependency_two_gpus_total_and_idempotent_submission(monkeypatch, tmp_path):
    root = tmp_path / "stress"
    ops = tmp_path / "ops"
    mirror = tmp_path / "mirror"
    roster(root)
    plan = {
        "blocks": [{"tasks": [{"task_id": f"block{i:04d}_full"}]} for i in range(57)]
    }
    monkeypatch.setattr(jobs, "roster", lambda root: plan)
    monkeypatch.setattr(jobs.critic, "checked_authorization", lambda *args: None)
    monkeypatch.setattr(
        jobs,
        "capacity",
        lambda *args: {"free_group_entries": 10, "estimated_entries_with_headroom": 9},
    )
    monkeypatch.setattr(jobs, "digest", lambda path: jobs.HELPER)
    monkeypatch.setattr(
        jobs, "wave8_dependency", lambda: ["--dependency=afterany:2356070"]
    )
    monkeypatch.setattr(
        jobs.runpy,
        "run_path",
        lambda path: {
            "checked_critic_key": str,
            "read_critic_key": lambda: "test",
            "verify_critic_access": lambda key: None,
        },
    )
    for name in (
        "AF_JETSTREAM_API_KEY",
        "AF_SEARCH_ROOT",
        "AF_REMAINDER_OPS",
        "AF_SOLVER_BACKUP",
    ):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def submit(folder, key, options, worker, stage, index):
        calls.append(options)
        (folder / "array.intent.json").write_text(
            json.dumps({"argv": ["sbatch", *options, str(worker), stage, str(index)]})
        )
        (folder / "array.id").write_text("321")
        return "321"

    monkeypatch.setattr(jobs, "submit_job", submit)
    res = jobs.run(root, ops, mirror, "test", "submit")
    assert res["job"] == "321"
    assert "--array=24-56%2" in calls[0] and "--dependency=afterany:2356070" in calls[0]
    assert jobs.run(root, ops, mirror, "test", "submit") == {"already_submitted": "321"}
    assert len(calls) == 1


def test_dependency_for_queued_completed_and_unconfirmed_parent(monkeypatch):
    monkeypatch.setenv("USER", "operator")
    replies = iter(
        [
            "2356070\n2356070\n",
            "",
            "2356070_21|COMPLETED\n2356070_22|TIMEOUT\n2356070_23|CANCELLED by 123\n",
            "",
            "",
        ]
    )
    monkeypatch.setattr(jobs.subprocess, "check_output", lambda *a, **kw: next(replies))
    assert jobs.wave8_dependency() == ["--dependency=afterany:2356070"]
    assert jobs.wave8_dependency() == []
    with pytest.raises(ValueError, match="accounting is incomplete"):
        jobs.wave8_dependency()


def test_worker_compacts_finished_parent_once_and_keeps_partial_for_recovery(
    tmp_path, monkeypatch
):
    store = tmp_path / jobs.STORE
    store.mkdir()
    mirror = tmp_path / "mirror"
    calls = []

    def compact(root, backup, i):
        calls.append(i)
        assert root == tmp_path and backup == mirror
        return {"block": i, "status": "pending" if i == 22 else "complete"}

    free = [4, 20]

    def capacity(root, backup, *, after_wave8):
        assert after_wave8
        return {
            "free_group_entries": free.pop(0),
            "estimated_entries_with_headroom": 12,
        }

    monkeypatch.setattr(jobs, "compact", compact)
    monkeypatch.setattr(jobs, "capacity", capacity)
    with pytest.raises(ValueError, match="insufficient free entries"):
        jobs.prepare_workers(tmp_path, mirror)
    assert not (store / "wave8-closeout.json").exists()
    assert jobs.prepare_workers(tmp_path, mirror)["blocks"][1]["status"] == "pending"
    assert jobs.prepare_workers(tmp_path, mirror) == {"wave8_checked": True}
    assert calls == [21, 22, 23] * 2
    with pytest.raises(ValueError, match="wrong wave-eight"):
        jobs.prepare_workers(tmp_path, tmp_path / "another")


def test_queue_capacity_and_worker_capacity_account_for_distinct_peaks(
    tmp_path, monkeypatch
):
    store, mirror = tmp_path / jobs.STORE, tmp_path / "backup"
    store.mkdir()
    mirror.mkdir()
    for i in range(21):
        name = f"block{i:04d}"
        arc = name + ".tar.gz"
        (store / arc).touch()
        backup = mirror / arc
        backup.touch()
        (tmp_path / "results" / name).mkdir(parents=True)
        (store / (name + ".closed.json")).write_text(
            json.dumps(
                {
                    "binding": {
                        "plan": jobs.PLAN,
                        "block": i,
                        "endpoints": [{"source_id": name}],
                    },
                    "entries": 100,
                    "archive": arc,
                    "backup": str(backup),
                }
            )
        )
    monkeypatch.setattr(jobs, "group_free", lambda: 500)
    before = jobs.capacity(tmp_path, mirror)
    after = jobs.capacity(tmp_path, mirror, after_wave8=True)
    # Persistent allowance 7 per pair, two live workers at 101, plus three
    # already compacted wave-eight blocks in the initial reservation only.
    assert before["estimated_entries_with_headroom"] == 7 * 36 + 202
    assert after["estimated_entries_with_headroom"] == 7 * 33 + 202
    assert before["reserved_for_wave8_growth"] == 3 * 107
    assert before["free_group_entries"] == 500
