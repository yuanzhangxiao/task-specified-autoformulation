"""Historical replies remain immutable while bookkeeping is audited offline."""

import json

import pytest

from scripts import replay_construction_search as cli
from tests.test_construction_feedback_revision import (  # noqa: F401
    prepared,
    prepared_v2,
    prepared_v3,
)
from tests.test_replay_structural_revision import returned  # noqa: F401
from tests.test_structural_revision_campaign import structural_source  # noqa: F401


def forbidden(*args, **kwargs):
    pytest.fail("No new provider/optimizer/replay assessment calls allowed")


def test_replay_resume_keeps_archives_and_no_live_policy_claim(
    returned, tmp_path, monkeypatch  # noqa: F811
):
    root = tmp_path / "audit"
    before = returned.read_bytes()
    result = cli.replay({"20b": returned}, root)
    assert result["arms"]["20b"]["saved_replies"] == 18
    assert result["arms"]["20b"]["eligible_replies"] == 18
    assert (
        result["llm_calls"]
        == result["optimizer_calls"]
        == result["fitting_exports"]
        == 0
    )
    assert not result["test_data_opened"]
    monkeypatch.setattr(cli.engine, "assess_candidate", forbidden)
    assert cli.replay({"20b": returned}, root) == result
    assert returned.read_bytes() == before
    assert (root / "REPORT.html").exists()
    path = next((root / "rows").glob("*.json"))
    row = json.loads(path.read_text())
    row["candidate"]["status"] = "invented"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="digest"):
        cli.replay({"20b": returned}, root)


def test_replay_requires_separate_output(returned):  # noqa: F811
    with pytest.raises(ValueError, match="separate"):
        cli.replay({"20b": returned}, returned.parent)
