"""Verify offline replay provenance and separation from its archived input."""

import json

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.staged_topology import content_hash
from scripts import replay_topology_admission as cli
from scripts.construction_functions import file_hash
from tests.test_preproposal_complete_graphs import inputs


def test_replay_rejects_modified_provider_receipt_and_preserves_input(
    tmp_path, monkeypatch
):
    case, frozen = inputs()
    pool = cli.generate_pool(case, frozen, max_candidates=1)
    candidate = pool["candidates"][0]
    root = tmp_path / "original"
    intake = sealed_write(
        root / "intake.json",
        {
            "selection": {"cases": [case.model_dump(mode="json")]},
            "frozen_cases": {case.case_id: frozen},
            "graph_pools": {case.case_id: pool},
        },
    )
    plan = sealed_write(
        root / "plan.json", {"intake_sha256": intake["artifact_sha256"]}
    )
    exported = {
        "graphs": [
            {"case_id": case.case_id, "structure_sha256": candidate["structure_sha256"]}
        ]
    }
    (root / "admitted-graphs.json").write_text(json.dumps(exported))
    slot = case.case_id + "--seed-100--graph-first"
    directory = root / "steps" / slot
    physical = directory / "calls/abc.json"
    physical.parent.mkdir(parents=True)
    physical.write_text('{"status":"responded"}')
    reply = sealed_write(
        directory / "reply.json",
        {
            "value": candidate["graph"],
            "status": "visible",
            "request_hash": "abc",
            "physical_receipt_sha256": file_hash(physical),
        },
    )
    saved = sealed_write(
        root / "results" / f"{slot}.json",
        {
            "slot": slot,
            "plan_sha256": plan["artifact_sha256"],
            "reply_sha256": reply["artifact_sha256"],
            "assessment": {"status": "topology_screen_passed", "checked": 0},
        },
    )
    monkeypatch.setattr(cli, "PLAN", plan["artifact_sha256"])
    monkeypatch.setattr(cli, "EXPORT", content_hash(exported))
    monkeypatch.setattr(
        cli, "ASSESSMENTS", content_hash({slot: saved["artifact_sha256"]})
    )
    out = tmp_path / "analysis.json"
    summary = cli.save(root, out)
    assert summary["original_export_recheck"] == {"topology_screen_passed": 1}
    assert summary["regenerated_pool_sizes"] == {case.case_id: 1}
    assert json.loads(out.read_text())["provider_calls"] == 0
    with pytest.raises(ValueError, match="outside the archived inspection"):
        cli.save(root, root / "plan.json")
    assert json.loads((root / "plan.json").read_text()) == plan

    physical.write_text("{}")
    with pytest.raises(ValueError, match="Archived request and assessment differ"):
        cli.save(root, out)
