"""Report snapshots retain original M47 semantics and reject stale dependencies."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import report_prefix_noise_sharded as report
from tests.test_prefix_noise_reliability import completed

q = report.campaign
completed = completed


@pytest.fixture(scope="module")
def finished(completed, tmp_path_factory):
    """Use one real qualified fit and explicit unavailable terminal peers."""
    root = tmp_path_factory.mktemp("sharded_original") / "campaign"
    shutil.copytree(completed[0], root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(q.base.execute, "fit_training", lambda *a, **kw: None)
        for index in range(1, 12):
            q.run_task(root, index)
    q.report(root)
    return root


@pytest.fixture
def campaign(finished, tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    shutil.copytree(finished, root)
    monkeypatch.setattr(report, "source_identity", lambda: {"bundle": "frozen"})

    def forbidden(*args, **kwargs):
        pytest.fail("report performed numerical work")

    monkeypatch.setattr(q.base.execute, "fit_training", forbidden)
    monkeypatch.setattr(q.base.execute, "infer_prefix", forbidden)
    monkeypatch.setattr(q.base.verification, "forecast", forbidden)
    monkeypatch.setattr(q.support, "predictive_support", forbidden)
    return root


def snapshot(root: Path) -> dict:
    """Include modification times so same-byte rewriting is also detected."""
    return {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and report.OUTPUT not in p.parts
    }


def checkpoint(root: Path, index: int = 0) -> Path:
    task = q.read_seal(root / "plan.json")["tasks"][index]
    return root / report.OUTPUT / "checkpoints" / f"{task['task_id']}.json"


def all_checked(root: Path) -> None:
    for index in range(12):
        report.check(root, index)


def reseal(path: Path, value: dict) -> None:
    path.unlink()
    q.seal(path, value)


def test_check_calls_original_once_resume_skips_and_original_bytes_unchanged(
    campaign, monkeypatch
):
    before = snapshot(campaign)
    original = q._completed
    calls = []

    def tracked(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(q, "_completed", tracked)
    first = report.check(campaign, 0)
    saved = (checkpoint(campaign).read_bytes(), checkpoint(campaign).stat().st_mtime_ns)
    assert report.check(campaign, 0) == first
    assert len(calls) == 1
    assert snapshot(campaign) == before
    assert saved == (
        checkpoint(campaign).read_bytes(),
        checkpoint(campaign).stat().st_mtime_ns,
    )
    events = [
        json.loads(line)
        for line in (campaign / report.OUTPUT / "check-000.progress.jsonl")
        .read_text()
        .splitlines()
    ]
    assert any(e["step"] == "original_completed" for e in events)
    assert all(
        not e["included_in_fit_cost"] for e in events if "reporting_cpu_seconds" in e
    )


def test_reduce_exactly_matches_original_and_restores_resolver(campaign, monkeypatch):
    expected = q.report(campaign)
    all_checked(campaign)
    before = snapshot(campaign)

    def forbidden(*args):
        pytest.fail("reduce repeated semantic reconstruction")

    monkeypatch.setattr(q, "_completed", forbidden)
    assert report.reduce(campaign) == expected
    assert json.loads((campaign / "summary.json").read_text()) == expected
    assert q._completed is forbidden
    after = snapshot(campaign)
    assert {p: v for p, v in after.items() if p.name != "summary.json"} == {
        p: v for p, v in before.items() if p.name != "summary.json"
    }


@pytest.mark.parametrize(
    "change", ["receipt", "progress", "selections", "result", "added"]
)
@pytest.mark.parametrize("mode", ["resume", "reduce"])
def test_changed_original_dependencies_rejected(campaign, change, mode):
    if mode == "reduce":
        all_checked(campaign)
    else:
        report.check(campaign, 0)
    folder = (
        campaign
        / "results"
        / q.read_seal(campaign / "plan.json")["tasks"][0]["task_id"]
    )
    if change == "receipt":
        path = next((folder / "operations").glob("*/result.json"))
    elif change == "progress":
        path = next((folder / "operations").glob("*/progress.json"))
    elif change == "added":
        path = folder / "operations" / "extra.json"
    else:
        path = folder / f"{change}.json"
    path.write_text("{}\n")
    summary = (campaign / "summary.json").read_bytes()
    with pytest.raises((ValueError, KeyError)):
        report.reduce(campaign) if mode == "reduce" else report.check(campaign, 0)
    assert (campaign / "summary.json").read_bytes() == summary


@pytest.mark.parametrize(
    "change", ["source", "runtime", "plan", "inputs", "seal", "identity"]
)
def test_checkpoint_binding_rejects_changes(campaign, monkeypatch, change):
    report.check(campaign, 0)
    if change == "source":
        monkeypatch.setattr(report, "source_identity", lambda: {"bundle": "changed"})
    elif change == "runtime":
        monkeypatch.setattr(q.public, "_runtime", lambda: {"changed": True})
    elif change in ("plan", "inputs"):
        path = campaign / f"{change}.json"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path = checkpoint(campaign)
        value = q.read_seal(path)
        value["identity"]["index"] = 1
        if change == "identity":
            reseal(path, value)
        else:
            envelope = json.loads(path.read_text())
            envelope["value"] = value
            path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError):
        report.check(campaign, 0)


@pytest.mark.parametrize("change", ["boolean_to_integer", "null_to_nonfinite"])
@pytest.mark.parametrize("mode", ["resume", "reduce"])
def test_cached_result_preserves_exact_json_types_and_finiteness(
    campaign, change, mode
):
    """A valid envelope must not weaken the original strict result comparison."""
    if mode == "reduce":
        all_checked(campaign)
    else:
        report.check(campaign, 0)
    path = checkpoint(campaign)
    value = q.read_seal(path)
    if change == "boolean_to_integer":
        assert value["result"]["test_data_opened"] is False
        value["result"]["test_data_opened"] = 0
    else:
        choice = value["result"]["selections"]["noise_aware"]
        assert choice["confidence_probability"] is None
        choice["confidence_probability"] = float("nan")
    # The historical seal normalizes nonfinite values to null when hashing;
    # hand-written JSON can retain NaN, which strict snapshot reuse must reject.
    path.write_text(json.dumps({
        "value": value,
        "sha256": q.public.content_sha256(value),
    }))
    before = snapshot(campaign)
    with pytest.raises(ValueError):
        report.reduce(campaign) if mode == "reduce" else report.check(campaign, 0)
    assert snapshot(campaign) == before


@pytest.mark.parametrize("change", ["missing", "extra", "bad_seal", "copied"])
def test_incomplete_or_corrupt_checkpoint_set_preserves_summary(campaign, change):
    all_checked(campaign)
    path = checkpoint(campaign)
    if change == "missing":
        path.unlink()
    elif change == "extra":
        shutil.copyfile(path, path.with_name("extra.json"))
    elif change == "copied":
        shutil.copyfile(path, checkpoint(campaign, 1))
    else:
        value = json.loads(path.read_text())
        value["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    before = snapshot(campaign)
    with pytest.raises(ValueError):
        report.reduce(campaign)
    assert snapshot(campaign) == before


def test_original_check_input_mutation_cannot_publish_checkpoint(campaign, monkeypatch):
    original = q._completed

    def changed(*args):
        value = original(*args)
        (args[0] / "operations" / "new.json").write_text("{}")
        return value

    monkeypatch.setattr(q, "_completed", changed)
    with pytest.raises(ValueError, match="changed during validation"):
        report.check(campaign, 0)
    assert not checkpoint(campaign).exists()


@pytest.mark.parametrize("task_id", ["../outside", "/outside", "a/b", "..", ""])
def test_task_path_traversal_rejected(campaign, monkeypatch, task_id):
    plan, data = q.verify(campaign, runtime=False)
    plan = deepcopy(plan)
    plan["tasks"][0]["task_id"] = task_id
    monkeypatch.setattr(q, "verify", lambda *a, **kw: (plan, data))
    with pytest.raises(ValueError, match="unsafe task"):
        report.check(campaign, 0)


def test_reducer_guards_roster_before_atomic_write_and_restores(campaign, monkeypatch):
    all_checked(campaign)
    original_write, original_completed = q.public._write, q._completed
    before = snapshot(campaign)

    def incomplete(root):
        q.public._write(root / "summary.json", {"wrong": True})

    monkeypatch.setattr(q, "report", incomplete)
    with pytest.raises(ValueError, match="full task roster"):
        report.reduce(campaign)
    assert q.public._write is original_write and q._completed is original_completed
    assert snapshot(campaign) == before


def test_source_inventory_rejects_changed_code(tmp_path, monkeypatch):
    path = Path(q.__file__).resolve()
    repo = path.parents[3]
    monkeypatch.setenv("AF_REPO_ROOT", str(tmp_path))
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("0" * 64 + "  source.py\n")
    (tmp_path / "source.py").write_text("changed")
    with pytest.raises(ValueError, match="source inventory differs"):
        report.source_identity()
    manifest.write_text(report.file_digest(tmp_path / "source.py") + "  source.py\n")
    with pytest.raises(ValueError, match="outside frozen source"):
        report.source_identity()
    assert path.is_relative_to(repo)
