"""Small live roster, honest delivery reporting, storage failure and cached resume."""

import errno
import json
import subprocess
import sys

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import construction_comparison as campaign
from scripts import phase_c_construction_comparison as cli
from scripts import submit_phase_c_baseline as submitter
from tests.test_construction_comparison import source_fixture, transport
from tests.test_phase_c_construction_baseline import tokenize


def prepare(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "live"
    return root, campaign.freeze(tmp_path / "new", root, study="live_confirmation")


def test_live_roster_includes_all_cases_once_per_policy_and_no_imported_models(
    tmp_path,
):
    root, plan = prepare(tmp_path)
    assert campaign.verify(root) == plan
    assert len(plan["tasks"]) == 24
    assert {t["seed"] for t in plan["tasks"]} == {0}
    assert {t["arm"] for t in plan["tasks"]} == {"full"}
    assert len({t["benchmark_id"] for t in plan["tasks"][:8]}) == 8
    assert sum("detention" in t["benchmark_id"] for t in plan["tasks"]) == 6
    assert plan["config"]["wall_seconds"] == 10800
    assert "inputs" not in plan
    assert not plan["automatic_followup"] and not plan["test_data_opened"]
    for cell in plan["cells"].values():
        assert (
            not {"inventory", "proposal", "results", "independent_rules"} & cell.keys()
        )


def test_basin_confirmation_is_six_fresh_matched_constructions_and_resumes(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "basins"
    plan = campaign.freeze(tmp_path / "new", root, study="basin_confirmation")
    assert campaign.verify(root) == plan
    assert len(plan["tasks"]) == 6
    assert set(plan["cells"]) == set(campaign.BASIN_CASES)
    assert {t["benchmark_id"] for t in plan["tasks"][:2]} == set(campaign.BASIN_CASES)
    assert {t["seed"] for t in plan["tasks"]} == {0}
    assert {t["arm"] for t in plan["tasks"]} == {"full"}
    calls = []
    base_transport = transport(calls)

    def basin_transport(url, body, timeout):
        reply = base_transport(url, body, timeout)
        message = reply["choices"][0]["message"]
        edits = json.loads(message["content"])
        for equation in edits.get("equations", []):
            if equation["name"] == "h_down":
                equation["terms"][0]["sources"].append("h_down")
        message["content"] = json.dumps(edits)
        return reply

    for task in plan["tasks"]:
        args = {"transport": basin_transport, "token_transport": tokenize}
        result = campaign.propose(root, plan, task, "http://offline", **args)
        assert result["status"] == "topology_complete"
        count = len(calls)
        assert campaign.propose(root, plan, task, "http://offline", **args) == result
        assert len(calls) == count
    summary = campaign.report(root, plan)
    assert summary["status_counts"] == {"topology_complete": 6}
    assert summary["all_tasks_terminal"]
    assert "Basin confirmation: both cases" in (root / "SUMMARY.md").read_text()


@pytest.mark.parametrize("change", ["non_basin", "extra_cell", "missing_policy"])
def test_basin_roster_cannot_silently_expand_or_drop_a_schedule(tmp_path, change):
    source_fixture(tmp_path)
    root = tmp_path / "basins"
    plan = campaign.freeze(tmp_path / "new", root, study="basin_confirmation")
    if change == "non_basin":
        plan["tasks"][0]["benchmark_id"] = "unplanned_case"
    elif change == "extra_cell":
        plan["cells"]["unplanned_case"] = next(iter(plan["cells"].values()))
    else:
        plan["tasks"].pop()
    plan.pop("artifact_sha256")
    (root / "plan.json").unlink()
    sealed_write(root / "plan.json", plan)
    with pytest.raises(ValueError):
        campaign.verify(root)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "seed", "arm", "study", "flag", "wall"]
)
def test_live_plan_refuses_roster_or_scope_changes(tmp_path, change):
    root, plan = prepare(tmp_path)
    if change == "missing":
        plan["tasks"].pop()
    elif change == "duplicate":
        plan["tasks"][0] = plan["tasks"][1]
    elif change in {"seed", "arm"}:
        plan["tasks"][0][change] = 1 if change == "seed" else "brief_only"
    elif change == "study":
        plan["study"] = "comparison"
    elif change == "wall":
        plan["config"]["wall_seconds"] = 21600
    else:
        plan["tasks"][0]["scientific_verifier"] = False
    plan.pop("artifact_sha256")
    (root / "plan.json").unlink()
    sealed_write(root / "plan.json", plan)
    with pytest.raises(ValueError):
        campaign.verify(root)


def test_three_live_policies_reuse_runner_and_cached_responses(tmp_path):
    root, plan = prepare(tmp_path)
    calls = []
    first_case = plan["tasks"][0]["benchmark_id"]
    tasks = [t for t in plan["tasks"] if t["benchmark_id"] == first_case]
    for task in tasks:
        args = {"transport": transport(calls), "token_transport": tokenize}
        result = campaign.propose(root, plan, task, "http://offline", **args)
        assert result["status"] == "topology_complete"
        count = len(calls)
        assert campaign.propose(root, plan, task, "http://offline", **args) == result
        assert len(calls) == count
    summary = campaign.report(root, plan)
    assert summary["status_counts"] == {"topology_complete": 3, "pending": 21}
    assert not summary["all_tasks_terminal"]
    assert summary["planned_tasks"] == 24
    assert len(summary["groups"]) == 3
    assert summary["groups"]["separate:full"]["equation_stage_reached"] == 1
    assert summary["delivery"]["length_limited_responses"] == 0
    assert "not a strategy ranking" in (root / "SUMMARY.md").read_text()


def test_delivery_diagnostics_distinguish_truncation_and_unavailable_response():
    def response(reason, content):
        return {
            "raw_response": {
                "choices": [{"finish_reason": reason, "message": {"content": content}}]
            }
        }

    counts = campaign.delivery_counts(
        [
            response("length", "{" + " " * 100),
            response("length", '{"x":'),
            response("stop", "{}"),
            {"raw_response": {"choices": None}},
        ]
    )
    assert counts["length_limited_responses"] == 2
    assert counts["whitespace_heavy_length_responses"] == 1
    assert counts["finish_reason_counts"] == {"length": 2, "stop": 1, "unavailable": 1}


def test_storage_probe_is_removed_on_success_and_failure(tmp_path, monkeypatch):
    marker = tmp_path / "existing.json"
    marker.write_text("preserve")
    result = campaign.check_storage(tmp_path, probe_bytes=8192)
    assert result == {"write_probe_bytes": 8192, "space_reserved": False}

    def quota(_):
        raise OSError(errno.EDQUOT, "quota")

    monkeypatch.setattr(campaign.os, "fsync", quota)
    with pytest.raises(OSError, match="quota"):
        campaign.check_storage(tmp_path, probe_bytes=8192)
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == "preserve"


def test_quota_failure_prevents_scheduler_submission(tmp_path, monkeypatch):
    root, _ = prepare(tmp_path)

    def quota(_):
        raise OSError(errno.EDQUOT, "quota")

    monkeypatch.setattr(campaign, "check_storage", quota)
    monkeypatch.setattr(submitter, "support", lambda _: pytest.fail("must not submit"))
    with pytest.raises(OSError, match="quota"):
        submitter.submit(root, "live-1", compare_construction=True)
    assert not (root / "submissions").exists()


def test_worker_quota_failure_prevents_new_requests(tmp_path, monkeypatch):
    root, _ = prepare(tmp_path)

    def quota(_):
        raise OSError(errno.EDQUOT, "quota")

    monkeypatch.setattr(campaign, "check_storage", quota)
    monkeypatch.setattr(
        campaign, "propose", lambda *a, **kw: pytest.fail("must not call provider")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["comparison", "run", "--root", str(root), "--base-url", "http://offline"],
    )
    with pytest.raises(OSError, match="quota"):
        cli.main()


def test_live_cli_preparation_and_frozen_study(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "cli"
    script = "scripts/phase_c_construction_comparison.py"
    command = [sys.executable, script]
    result = subprocess.run(
        [
            *command,
            "prepare",
            "--source",
            str(tmp_path / "new"),
            "--root",
            str(root),
            "--study",
            "live_confirmation",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["tasks"] == 24
    subprocess.run(
        [*command, "verify", "--root", str(root)], check=True, capture_output=True
    )
    bad = subprocess.run(
        [*command, "verify", "--root", str(root), "--study", "comparison"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0 and "frozen" in bad.stderr
