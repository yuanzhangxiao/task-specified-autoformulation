"""Saved-reply counterfactuals authenticate history without adopting it."""

import hashlib
import json
import tarfile
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import replay_structural_revision as replay
from scripts import structural_revision_campaign as campaign
from tests.test_construction_feedback_revision import (  # noqa: F401
    prepared,
    prepared_v2,
    prepared_v3,
)
from tests.test_construction_initialization_graph_cli import reply as response
from tests.test_construction_initialization_graph_cli import tokenizer
from tests.test_structural_revision_campaign import (  # noqa: F401
    revision,
    structural_source,
)


@pytest.fixture
def returned(structural_source, tmp_path, monkeypatch):  # noqa: F811
    """Create genuine frozen calls/receipts using a bounded synthetic source."""
    source, _, _ = structural_source
    monkeypatch.setattr(
        campaign.sources,
        "SOURCE_ARCHIVE_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    root = tmp_path / "returned"
    campaign.prepare(source, root, "20b")
    campaign.run(
        root,
        "http://127.0.0.1",
        transport=lambda *args: response(revision()),
        token_transport=tokenizer,
    )
    archive = tmp_path / "returned.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
    monkeypatch.setitem(
        replay.SOURCE_PINS,
        "20b",
        {
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "plan_sha256": sealed_read(root / "plan.json")["artifact_sha256"],
        },
    )
    return archive


def forbidden(*args, **kwargs):
    pytest.fail("offline replay must not generate, fit, simulate, or redo cached work")


def test_saved_replies_checkpoint_resume_without_generation_or_adoption(
    returned, tmp_path, monkeypatch
):
    from autoformalism.fitting import prefix_initialization

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    monkeypatch.setattr(campaign.ConstructionClient, "call", forbidden)
    archive_before = returned.read_bytes()
    root = tmp_path / "replay"
    result = replay.replay({"20b": returned}, root)
    assert result["arms"]["20b"]["saved_calls_replayed"] == 18
    assert result["arms"]["20b"]["gate_eligible_episodes"] == 18
    assert result["llm_calls"] == result["optimizer_calls"] == 0
    assert result["fitting_exports"] == 0
    assert result["test_data_opened"] is False
    assert not any(r["adopted"] or r["fitting_exported"] for r in result["rows"])
    assert not list(root.rglob("fit_handoff.json"))
    monkeypatch.setattr(replay, "assess_reply", forbidden)
    assert replay.replay({"20b": returned}, root) == result
    assert returned.read_bytes() == archive_before
    assert (root / "REPORT.html").exists()


def test_pinned_source_mismatch_and_symlink_fail_before_reading(tmp_path):
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"untrusted bytes")
    with pytest.raises(ValueError, match="archive identity"):
        replay.load_source(archive, "20b")
    link = tmp_path / "link.tar.gz"
    link.symlink_to(archive)
    with pytest.raises(ValueError, match="bounded regular"):
        replay.load_source(link, "20b")


def test_plan_drift_prevents_replay_resume(returned, tmp_path, monkeypatch):
    root = tmp_path / "replay"
    replay.replay({"20b": returned}, root)
    monkeypatch.setattr(replay, "runtime_source_hash", lambda: "different-runtime")
    monkeypatch.setattr(replay, "assess_reply", forbidden)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        replay.replay({"20b": returned}, root)


def test_tampered_checkpoint_fails_closed(returned, tmp_path):
    root = tmp_path / "replay"
    replay.replay({"20b": returned}, root)
    row = next((root / "rows").glob("*.json"))
    value = json.loads(row.read_text())
    value["outcome"] = "tampered"
    row.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifact digest differs"):
        replay.replay({"20b": returned}, root)


def test_output_cannot_cover_source_storage(returned):
    with pytest.raises(ValueError, match="separate"):
        replay.replay({"20b": returned}, returned.parent)


def test_original_plan_and_controls_are_rechecked(returned, monkeypatch):
    real_reader = replay._Reader

    class ChangedReader(real_reader):
        def json(self, name, seal=None):
            value = deepcopy(super().json(name, seal))
            if name.startswith("controls/"):
                value["control_unchanged"] = False
            return value

    monkeypatch.setattr(replay, "_Reader", ChangedReader)
    with pytest.raises(ValueError, match="parent control"):
        replay.load_source(returned, "20b")


def test_audit_probe_preserves_production_schema_error(structural_source):  # noqa: F811
    parent = campaign._parent(structural_source[1], "parent_0")
    raw = revision(new_states=[{"name": "missing_equation_and_initializer"}])
    result = replay.assess_reply(
        parent, {"record": {"status": "responded", "raw_response": response(raw)}}
    )
    assert result["outcome"] == "REVISION_SCHEMA"
