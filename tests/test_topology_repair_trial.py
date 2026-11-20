"""Check native graph replay and offline/live score equivalence for the saved case."""

from pathlib import Path

import pytest

from autoformalism.research import topology_repair_trial as trial
from autoformalism.research.topology_repair_case import prepare_case
from autoformalism.search.topology_repair_generator import TopologyEdit


def test_memory_revision_preserves_parent_edges_and_ownership():
    parent = {
        "targets": {"level": {"kind": "dynamic", "sources": ["outlet"]}},
        "additional_components": [],
    }
    edit = TopologyEdit(kind="via_new_memory", source="rain", target="level")
    after = trial._edited_parent(parent, edit, "storage")
    assert after["targets"]["level"]["sources"] == ["outlet", "storage"]
    assert after["additional_components"][0]["sources"] == ["rain", "storage"]
    assert parent["targets"]["level"]["sources"] == ["outlet"]
    assert not parent["additional_components"]


def test_seed102_live_scorer_matches_enumeration_and_rejects_wrong_target():
    archive = Path("output/t2-fresh-20b-results-20261119/returned")
    if not archive.exists():
        pytest.skip("Requires the saved public training inspection")
    case = prepare_case(archive)
    rows = trial.offline_portfolio(case)
    assert len(rows) == len({r["edit_sha256"] for r in rows}) == 20
    for source in ("Gp", "U", "Uii", "I"):
        row = next(
            r
            for r in rows
            if r["edit"] == {"kind": "add_dependency", "source": source, "target": "I"}
        )
        reply = trial.RepairReply(edit=row["edit"], reason="Check the proposed edge.")
        assert trial.assess_reply(case, reply) == row
    wrong = trial.RepairReply(
        edit=TopologyEdit(kind="add_dependency", source="U", target="Gp"),
        reason="Changes a different equation.",
    )
    assert trial.assess_reply(case, wrong)["status"] == "invalid"
