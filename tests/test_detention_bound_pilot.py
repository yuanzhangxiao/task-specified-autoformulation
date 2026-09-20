"""Matched saved inventories, historical replay and new shared-law construction."""

import json
import subprocess

import pytest

from autoformalism.rebuttal import detention_process_pilot as pilot
from scripts import smoke_detention_process_pilot as smoke
from scripts.smoke_shared_process_contract import transport_for


def fixture(tmp_path):
    source, parent, root = (tmp_path / n for n in ("source", "parent", "bound"))
    plan = smoke.fixture(source, parent)
    for task in plan["tasks"]:
        if not task["review"]:
            continue
        mode = "independent" if task["case"] == "independent" else "add"
        result = pilot.construct(
            parent,
            plan,
            task,
            pilot.make_client(
                parent, plan, task, "offline", transport=smoke.transport_for([], mode)
            ),
        )
        assert result["status"] == "constructed"
    config = pilot.REPO / "configs/detention_process_pilot_v2.json"
    return source, parent, root, config


def test_matched_inventory_and_checkpoint_identity(tmp_path):
    source, parent, root, config = fixture(tmp_path)
    plan = pilot.freeze(source, root, config, parent)
    assert len(plan["paired_inputs"]["pairs"]) == 8
    assert "PRIVATE_ORACLE_POISON" not in json.dumps(plan)
    assert pilot.freeze(source, root, config, parent) == plan
    task = next(t for t in plan["tasks"] if t["case"] == "coupled" and t["review"])
    calls = []
    result = pilot.construct(
        root,
        plan,
        task,
        pilot.make_client(root, plan, task, "offline", transport=transport_for(calls)),
    )
    assert result["status"] == "constructed", result
    assert not any("agenda_item" in c["payload"] for c in calls)
    assert result["shared_process_contract"]["bindings"][0]["proposal"]["name"] == "q"
    pair = next(
        t
        for t in plan["tasks"]
        if t["task_id"] == task["task_id"].replace("_on", "_off")
    )
    control_calls = []
    control = pilot.construct(
        root,
        plan,
        pair,
        pilot.make_client(
            root,
            plan,
            pair,
            "offline",
            transport=smoke.transport_for(control_calls, "empty"),
        ),
    )
    assert control["status"] == "constructed", control
    assert (
        calls[0]["payload"]["inventory"]
        == control_calls[0]["payload"]["frozen_inventory"]
    )
    pilot.report(root)
    assert (root / "SAVED_REPLY_AUDIT.json").exists()
    key = plan["paired_inputs"]["pairs"][task["task_id"]]["parent_request_hash"]
    path = parent / "results" / task["task_id"] / "calls" / f"{key}.json"
    raw = json.loads(path.read_text())
    raw["request"]["body"]["messages"][1]["content"] = "{}"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="request differs"):
        pilot.freeze(source, root, config, parent)


def test_new_protocol_requires_parent_and_server_mapping(tmp_path):
    source, _, root, config = fixture(tmp_path)
    with pytest.raises(ValueError, match="--parent"):
        pilot.freeze(source, root, config)
    result = subprocess.run(
        [
            "bash",
            str(pilot.REPO / "scripts/hpc/run_staged_topology_server.sh"),
            "--check-config",
            str(config),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "detention_process_pilot.py"


def test_process_topology_failure_falls_back_to_original_inventory(tmp_path):
    # Every semantic contract can be abandoned explicitly as an optional attempt;
    # the caller's unchanged inventory remains available within the same budget.
    source, parent, root, config = fixture(tmp_path)
    plan = pilot.freeze(source, root, config, parent)
    task = next(t for t in plan["tasks"] if t["case"] == "coupled" and t["review"])
    calls = []
    good = transport_for(calls)

    def bad_uses(url, body, timeout):
        response = good(url, body, timeout)
        if '"stage": "process_uses"' in body["messages"][1]["content"]:
            response["choices"][0]["message"]["content"] = '{"uses": []}'
        return response

    result = pilot.construct(
        root,
        plan,
        task,
        pilot.make_client(root, plan, task, "offline", transport=bad_uses),
    )
    assert result["status"] == "constructed", result
    assert result["fallback_used"]
    assert result["shared_process_contract"] is None
