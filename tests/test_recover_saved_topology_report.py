"""Report recovery preserves identity, generation behavior, and worker exclusion."""

import hashlib
import json
import tarfile
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import recover_saved_topology_report as recovery


@pytest.mark.parametrize("values", [[], ["clear"], ["a", "a", "b"]])
def test_counter_is_unchanged_without_pending_values(values):
    assert recovery.report_counter(values) == Counter(values)


def test_counter_preserves_pending_counts_and_existing_label():
    result = recovery.report_counter([None, "clear", None, "not_assessed"])
    assert result == {"not_assessed": 3, "clear": 1}
    assert json.loads(json.dumps(dict(result), sort_keys=True)) == result
    assert recovery.report_counter({None: 2, "clear": 3}) == {
        "not_assessed": 2,
        "clear": 3,
    }


def test_override_exists_only_during_report_and_restores_on_failure():
    campaign = SimpleNamespace(Counter=Counter)

    def report(fail=False):
        assert campaign.Counter is recovery.report_counter
        if fail:
            raise RuntimeError("report error")
        return campaign.Counter([None, "clear"])

    campaign.report = report
    with recovery.corrected_report(campaign):
        assert campaign.Counter is Counter
        assert campaign.report() == {"not_assessed": 1, "clear": 1}
        assert campaign.Counter is Counter
        with pytest.raises(RuntimeError, match="report error"):
            campaign.report(True)
        assert campaign.Counter is Counter
    assert campaign.report is report
    assert campaign.Counter is Counter


@pytest.fixture
def existing_campaign(tmp_path, monkeypatch):
    bundle, root, work = (tmp_path / name for name in ("bundle", "campaign", "work"))
    for path in (bundle / "inputs/saved-repair", root, work):
        path.mkdir(parents=True)
    (bundle / recovery.REPORT_MODULE).parent.mkdir(parents=True)
    (bundle / recovery.REPORT_MODULE).write_text("original source\n")
    (bundle / "MANIFEST.sha256").write_text("fixture manifest\n")
    plan = {
        "artifact_sha256": recovery.PLAN_IDENTITY,
        "source_identity": recovery.SOURCE_IDENTITY,
        "protocol": "saved-topology-repair-comparison-1",
        "test_data_opened": False,
    }
    for path in (bundle / "inputs/saved-repair/plan.json", root / "plan.json"):
        path.write_text(json.dumps(plan))
    execution = {**plan, "plan_sha256": recovery.PLAN_IDENTITY}
    (root / "execution.json").write_text(json.dumps(execution))
    campaign = SimpleNamespace(
        Counter=Counter,
        source_identity=lambda: recovery.SOURCE_IDENTITY,
        verify=lambda path: json.loads((path / "plan.json").read_text()),
        sealed_read=lambda path: json.loads(path.read_text()),
    )

    def sealed_write(path, value):
        sealed = {**value, "artifact_sha256": "fixture seal"}
        if path.exists():
            assert json.loads(path.read_text()) == sealed
        else:
            path.write_text(json.dumps(sealed))
        return sealed

    def report(root, plan):
        value = {"scientific_status_counts": dict(campaign.Counter([None, "clear"]))}
        (root / "summary.json").write_text(json.dumps(value, sort_keys=True))
        return value

    @contextmanager
    def legacy_lock(directory):
        directory.mkdir(exist_ok=True)
        with recovery.worker_lock(directory / ".lock"):
            yield

    campaign.sealed_write, campaign.report = sealed_write, report
    cli = SimpleNamespace(_lock=legacy_lock)
    legacy = SimpleNamespace(campaign=campaign, cli=cli)
    monkeypatch.setattr(recovery, "load_legacy", lambda path: legacy)
    return bundle, root, work, legacy


@pytest.mark.parametrize("field", ["artifact_sha256", "source_identity"])
def test_input_identity_mismatch_is_rejected(existing_campaign, field):
    bundle, root, _, legacy = existing_campaign
    path = bundle / "inputs/saved-repair/plan.json"
    value = json.loads(path.read_text())
    value[field] = "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="pinned saved-repair identity"):
        recovery.verify_campaign(legacy, bundle, root)
    assert not (root / recovery.RECEIPT).exists()


def test_existing_root_and_execution_must_match(existing_campaign):
    bundle, root, _, legacy = existing_campaign
    path = root / "plan.json"
    original = path.read_text()
    path.write_text(json.dumps({**json.loads(original), "extra": "edited"}))
    with pytest.raises(ValueError, match="existing campaign plan differs"):
        recovery.verify_campaign(legacy, bundle, root)
    path.write_text(original)
    execution = root / "execution.json"
    value = json.loads(execution.read_text())
    value["source_identity"] = "different"
    execution.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="execution identity differs"):
        recovery.verify_campaign(legacy, bundle, root)


def test_manifest_hash_and_listed_files_checked_before_import(tmp_path, monkeypatch):
    source = tmp_path / recovery.REPORT_MODULE
    source.parent.mkdir(parents=True)
    source.write_text("original")
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(f"{recovery.sha256(source)}  {recovery.REPORT_MODULE}\n")
    monkeypatch.setattr(recovery, "MANIFEST_SHA256", recovery.sha256(manifest))
    monkeypatch.setattr(recovery, "REPORT_SHA256", recovery.sha256(source))
    recovery.verify_bundle(tmp_path)
    source.write_text("edited")
    with pytest.raises(ValueError, match="bundle file differs"):
        recovery.verify_bundle(tmp_path)
    manifest.write_text("substituted manifest")
    with pytest.raises(ValueError, match="bundle manifest differs"):
        recovery.verify_bundle(tmp_path)


@pytest.mark.parametrize("command", ["report", "inspect"])
def test_reporting_cli_never_contacts_server_and_preserves_originals(
    existing_campaign, monkeypatch, capsys, command
):
    bundle, root, work, legacy = existing_campaign
    originals = {path: path.read_bytes() for path in root.glob("*.json")}
    monkeypatch.setattr(
        recovery.subprocess, "run", lambda *a, **k: pytest.fail("Docker/network used")
    )
    monkeypatch.setattr(
        recovery, "refresh_server", lambda *a: pytest.fail("server used")
    )
    assert (
        recovery.main(
            [
                command,
                "--bundle",
                str(bundle),
                "--root",
                str(root),
                "--work",
                str(work),
            ]
        )
        == 0
    )
    value = json.loads(capsys.readouterr().out)
    summary = value["summary"] if command == "inspect" else value
    assert summary["scientific_status_counts"] == {"not_assessed": 1, "clear": 1}
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert legacy.campaign.Counter is Counter
    assert not (root / "exit-code.txt").exists()
    assert not (work / "server.json").exists()
    if command == "inspect":
        with tarfile.open(value["archive"]) as archive:
            names = archive.getnames()
            assert {
                "plan.json",
                "execution.json",
                recovery.RECEIPT,
                "recovery.log",
            } <= (set(names))
            assert not any("inspect.json" in name for name in names)


@pytest.mark.parametrize("lock_name", ["shared", "worker", "campaign"])
def test_busy_worker_prevents_receipts_and_markers(existing_campaign, lock_name):
    bundle, root, work, _ = existing_campaign
    paths = {
        "shared": work / "worker.lock",
        "worker": root / "worker.lock",
        "campaign": root / "campaign-lock/.lock",
    }
    paths[lock_name].parent.mkdir(exist_ok=True)
    with (
        recovery.worker_lock(paths[lock_name]),
        pytest.raises(RuntimeError, match="worker is still running"),
    ):
        recovery.execute("run", bundle, root, work)
    assert not (root / recovery.RECEIPT).exists()
    assert not (root / "exit-code.txt").exists()
    assert not (root / "recovery.log").exists()


def test_run_reuses_legacy_cli_under_all_locks(existing_campaign, monkeypatch):
    bundle, root, work, legacy = existing_campaign
    monkeypatch.setattr(recovery, "refresh_server", lambda *a: None)
    monkeypatch.setattr(
        recovery.shutil, "disk_usage", lambda p: SimpleNamespace(free=3 * 1024**3)
    )
    monkeypatch.setattr(
        recovery.os, "statvfs", lambda p: SimpleNamespace(f_files=20000, f_favail=15000)
    )
    original = legacy.campaign.report

    def run(inputs, given_root, given_work, requests):
        assert (inputs, given_root, given_work, requests) == (
            bundle / "inputs/saved-repair",
            root,
            work,
            bundle / "probe-requests.json",
        )
        assert legacy.campaign.Counter is Counter
        for path in (
            work / "worker.lock",
            root / "worker.lock",
            root / "campaign-lock/.lock",
        ):
            with (
                pytest.raises(RuntimeError, match="worker is still running"),
                recovery.worker_lock(path),
            ):
                pytest.fail("missing worker exclusion")
        return legacy.campaign.report(root, {})

    legacy.cli.run = run
    value = recovery.execute("run", bundle, root, work)
    assert value["scientific_status_counts"]["not_assessed"] == 1
    assert (root / "exit-code.txt").read_text() == "0\n"
    assert legacy.campaign.report is original


def test_failed_run_leaves_controlled_exit_and_restores_report(
    existing_campaign, monkeypatch
):
    bundle, root, work, legacy = existing_campaign
    original = legacy.campaign.report

    def fail(*args):
        raise ValueError("server is stopped")

    monkeypatch.setattr(recovery, "refresh_server", fail)
    with pytest.raises(ValueError, match="server is stopped"):
        recovery.execute("run", bundle, root, work)
    assert (root / "exit-code.txt").read_text() == "1\n"
    assert "server is stopped" in (root / "recovery.log").read_text()
    assert legacy.campaign.report is original
    assert legacy.campaign.Counter is Counter


def test_server_refresh_checks_owner_before_persisting_raw_inspection(
    existing_campaign, monkeypatch
):
    _, root, work, legacy = existing_campaign
    inspection = {
        "Config": {"Labels": {"autoformalism.role": "construction-compatibility"}},
        "State": {"Running": True},
        "sensitive_unused_field": "never persist this",
    }
    monkeypatch.setattr(
        recovery.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps([inspection])),
    )
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *a, **k: "H100\n")
    saved_inspections = []

    def server_record(value, path):
        saved_inspections.append(deepcopy(value))
        assert path == work
        return {"validated": True}

    legacy.server = SimpleNamespace(server_record=server_record)
    (work / "probe").mkdir()
    (work / "probe/plan.json").write_text("{}")
    (work / "weights.json").write_text("{}")
    recovery.refresh_server(legacy, work, root)
    assert saved_inspections == [inspection]
    assert json.loads((root / "runtime/server.json").read_text()) == {
        "validated": True,
        "gpu_driver": "H100",
    }
    for path in root.rglob("*.json"):
        assert "sensitive_unused_field" not in path.read_text()
    before = hashlib.sha256((work / "server.json").read_bytes()).hexdigest()
    inspection["Config"]["Labels"]["autoformalism.role"] = "unowned"
    with pytest.raises(ValueError, match="expected owned model server"):
        recovery.refresh_server(legacy, work, root)
    assert recovery.sha256(work / "server.json") == before
