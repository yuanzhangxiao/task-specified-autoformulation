"""Audited historical contexts, raw reply integrity and deterministic CPU resume."""

import json

import pytest

from autoformalism.rebuttal import basin_repair_audit as audit
from autoformalism.rebuttal import basin_repair_pilot as old
from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import smoke_basin_repair_pilot as smoke


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("repair-audit")
    source, gate = smoke.fixture(root / "history")
    repair_root = root / "repair"
    plan = old.freeze(source, gate, repair_root)
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    calls = []
    proposal = old.run_one(
        repair_root,
        plan,
        task,
        smoke.client(repair_root, plan, task, calls, accept=False),
    )
    assert proposal["status"] == "unconfirmed_trial"
    # A second task reproduces the accept+edits error at every call. The saved
    # context must remain the parent even though v2 accepts each individual patch.
    other = next(t for t in plan["tasks"] if t["case"] == "independent" and t != task)
    client = smoke.client(repair_root, plan, other, [])

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        assert "outlet_rate" not in str(payload["assembled_model"])
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "hypothesis": "Public inflow and crest.",
                                "accept_displayed": True,
                                "equations": [
                                    {
                                        "component": "h_down",
                                        "expression": (
                                            "inflow_down/area_down-"
                                            "new_rate*max(0,h_down-crest_down)"
                                        ),
                                    }
                                ],
                                "new_parameters": [
                                    {"name": "new_rate", "role": "rate"}
                                ],
                            }
                        )
                    },
                }
            ],
            "usage": {"total_tokens": 100},
        }

    client.transport = transport
    p = old.run_one(repair_root, plan, other, client)
    assert len(p["events"]) == 3
    assert all("cannot accompany" in e["error"] for e in p["events"])
    return root, repair_root, plan, task, other


def test_audit_does_not_splice_counterfactual_contexts_and_is_resumable(saved):
    root, source, plan, task, other = saved
    before = audit._files(source, plan)
    output = root / "audit"
    result = audit.audit(source, output)
    assert result["status_counts"] == {"audited": 2, "unavailable": 14}
    assert (
        result["llm_calls"]
        == result["optimizer_calls"]
        == result["solver_rollouts"]
        == 0
    )
    assert result["newly_eligible_saved_drafts"] == [task["task_id"]]
    assert result["newly_mechanically_valid_attempts"] == 3
    assert not result["unexpected_acceptance_regressions"]
    row = next(r for r in result["rows"] if r["task"] == other["task_id"])
    assert {a["classification"] for a in row["attempts"]} == {"eligible_for_fit"}
    assert len({a["context_sha256"] for a in row["attempts"]}) == 1
    assert row["saved_final_draft"]["status"] == "unchanged_parent"
    assert audit.audit(source, output) == result
    assert audit._files(source, plan) == before
    assert sealed_read(output / "summary.json")["identity"] == result["identity"]


def test_source_drift_rejects_resume(saved):
    root, source, plan, _, _ = saved
    output = root / "drift-audit"
    audit.audit(source, output)
    path = source / "plan.json"
    content = path.read_bytes()
    try:
        path.write_bytes(content + b"\n")  # same sealed JSON, different frozen bytes
        with pytest.raises(ValueError):
            audit.audit(source, output)
    finally:
        path.write_bytes(content)
    assert audit._files(source, plan)["plan.json"]


def test_raw_call_tampering_is_not_reinterpreted_as_a_proposal_failure(saved):
    root, source, _, task, _ = saved
    proposal = sealed_read(source / "results" / task["task_id"] / "proposal.json")
    path = (
        source
        / "results"
        / task["task_id"]
        / "calls"
        / f"{proposal['events'][0]['request_hash']}.json"
    )
    content = path.read_bytes()
    record = json.loads(content)
    record["request"]["body"]["messages"][1]["content"] = "{}"
    try:
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="identity differs"):
            audit.audit(source, root / "tampered")
    finally:
        path.write_bytes(content)


def test_output_must_be_separate(saved):
    _, source, _, _, _ = saved
    with pytest.raises(ValueError, match="separate"):
        audit.audit(source, source / "new")
