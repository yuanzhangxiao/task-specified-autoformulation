"""Read-only census preserves incomplete denominators and consumed work."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import audit_component_campaign as audit
from scripts.component_audit_io import Snapshot, digest

CELL = "phase_b_dalla_man_t1_canonical_named_easy"
REPO = Path(__file__).resolve().parents[1]


def write(path, value, *, sealed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**value, "artifact_sha256": digest(value)} if sealed else value
    path.write_text(json.dumps(payload))
    return payload


def campaign(root, *, critic=False, rounds=2):
    task = {
        "task_id": "cell00_seed0_full_c1v1s1" if critic else "cell00_seed0_full_c0v1s1",
        "cell": CELL,
        "arm": "full",
        "seed": 0,
        "block": 0,
        "critic": critic,
        "scientific_verifier": True,
        "shared_processes": True,
    }
    plan = write(
        root / "plan.json",
        {
            "protocol": audit.CAMPAIGN,
            "tasks": [task],
            "config": {"rounds": rounds},
            "cells": {CELL: {"training": "PRIVATE_PROJECTION_SENTINEL"}},
        },
        sealed=True,
    )
    # These must never be opened or copied by the audit.
    for name in (
        "summary.json",
        "component_summary.json",
        "test.csv",
        "evaluation/report.json",
        "judge_cache/calls/request.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SECRET_SENTINEL this is not json")
    return plan, task


def visit(root, plan, task, index=0, parent=None, *, terminal=True):
    base = root / "results" / task["task_id"] / f"round_{index:02d}"
    proposal = write(
        base / "proposal.json",
        {
            "task": task,
            "round": index,
            "parent_sha256": (parent or {}).get("artifact_sha256"),
            "status": "constructed" if index == 0 else "committed",
            "attempts": [],
            "fallback_used": False,
        },
        sealed=True,
    )
    if task["critic"]:
        write(
            root / "critic_authorization.json",
            {
                "identity": plan["artifact_sha256"],
                "known_case_gate_passed": True,
            },
            sealed=True,
        )
        write(
            base / "critic.json",
            {
                "identity": plan["artifact_sha256"],
                "task": task,
                "round": index,
                "proposal_sha256": proposal["artifact_sha256"],
                "status": "reviewed",
            },
            sealed=True,
        )
    request = {"candidate": "MODEL_SENTINEL"}
    metrics = {"available": True, "failed_trajectories": [], "normalized_mse": 0.2}
    fit = {
        "status": "complete",
        "parameters": {"k": 1},
        "training": metrics,
        "validation": metrics,
        "request_sha256": digest(request, compact=True),
    }
    fit_receipt = {"result": fit, "sha256": digest(fit, compact=True)}
    write(base / "worker_started.json", {"task": task}, sealed=True)
    write(base / "fit/started.json", {"started": True})
    write(base / "fit/result.json", fit_receipt)
    selected = {
        "fit": fit,
        "request": request,
        "origin_task": task["task_id"],
        "origin_round": index,
        "fit_result_sha256": digest(fit_receipt, compact=True),
    }
    result = {
        "task": task,
        "round": index,
        "status": "complete",
        "selected": selected,
        "trial": selected,
        "proposal_sha256": proposal["artifact_sha256"],
        "parent_sha256": (parent or {}).get("artifact_sha256"),
    }
    return write(base / "result.json", result, sealed=True) if terminal else None


def prune(root, plan, task, last):
    base = root / "pruning" / task["task_id"]
    write(
        base / "parent.json",
        {
            "identity": plan["artifact_sha256"],
            "task": task,
            "source_result_sha256": last["artifact_sha256"],
            "parent_sha256": digest(last["selected"], compact=True),
        },
        sealed=True,
    )
    choice = write(
        base / "choice.json",
        {
            "identity": plan["artifact_sha256"],
            "status": "no_legal_removal",
            "request": None,
        },
        sealed=True,
    )
    fit = last["selected"]["fit"]
    write(
        base / "control/result.json",
        {"result": fit, "sha256": digest(fit, compact=True)},
    )
    return write(
        base / "result.json",
        {
            "identity": plan["artifact_sha256"],
            "task": task,
            "status": "complete",
            "choice": choice,
            "fits": {"control": fit, "pruned": None},
            "selection": {"selected": "parent", "fit": fit},
        },
        sealed=True,
    )


def test_complete_denominator_missing_frontier_and_no_payload_export(tmp_path):
    plan, task = campaign(tmp_path)
    visit(tmp_path, plan, task)
    value = audit.collect(tmp_path)
    assert len(value["rounds"]) == 2
    assert value["rounds"][0]["selected"]["finite_complete_fit"]
    assert value["lineages"][0]["frontier"] == {"round": 1, "action": "propose"}
    assert value["issues"] == []
    assert "SENTINEL" not in json.dumps(value)
    assert all("test.csv" not in f["path"] for f in value["files"])
    assert value == audit.collect(tmp_path)
    summary = json.loads(audit.report(value)["summary.json"])
    assert summary["round_status_counts"] == {"complete": 1, "missing": 1}
    assert summary["endpoint_status_counts"] == {"missing": 1}


def test_started_or_saved_fit_without_round_is_not_fresh_retry(tmp_path):
    plan, task = campaign(tmp_path)
    visit(tmp_path, plan, task, terminal=False)
    value = audit.collect(tmp_path)
    assert value["lineages"][0]["frontier"]["action"] == "inspect_consumed_fit_attempt"
    assert value["rounds"][0]["fit"]["status"] == "complete"
    assert value["rounds"][0]["status"] == "missing"
    assert value["rounds"][1]["proposal_status"] is None


def test_sealed_interruption_remains_consumed_but_next_visit_can_continue(tmp_path):
    plan, task = campaign(tmp_path)
    result = visit(tmp_path, plan, task)
    value = {
        k: v
        for k, v in result.items()
        if k not in {"artifact_sha256", "parent_sha256", "proposal_sha256"}
    }
    value.update(status="worker_interrupted", selected=None, trial=None)
    write(
        tmp_path / "results" / task["task_id"] / "round_00/result.json",
        value,
        sealed=True,
    )
    observed = audit.collect(tmp_path)
    assert observed["issues"] == []
    assert observed["rounds"][0]["status"] == "worker_interrupted"
    assert observed["lineages"][0]["frontier"]["action"] == "propose"


def test_final_round_pruning_and_final_critic_are_distinct(tmp_path):
    plan, task = campaign(tmp_path, critic=True, rounds=1)
    last = visit(tmp_path, plan, task)
    assert audit.collect(tmp_path)["lineages"][0]["frontier"]["action"] == "prune"
    pruning = prune(tmp_path, plan, task, last)
    value = audit.collect(tmp_path)
    row = value["lineages"][0]
    assert value["issues"] == []
    assert row["frontier"]["action"] == "final_critic"
    assert [e["status"] for e in row["pruning"]["endpoints"]] == [
        "missing",
        "complete",
        "complete",
    ]
    write(
        tmp_path / "pruning" / task["task_id"] / "critic.json",
        {
            "identity": plan["artifact_sha256"],
            "pruning_sha256": pruning["artifact_sha256"],
            "status": "interrupted",
        },
        sealed=True,
    )
    row = audit.collect(tmp_path)["lineages"][0]
    assert row["frontier"]["action"] == "recorded_endpoint_complete"
    assert row["pruning"]["critic_status"] == "interrupted"
    # Terminal receipt coverage is not advice availability or scientific success.
    assert row["pruning"]["endpoints"][0]["status"] == "complete"


def test_stage_precedence_partial_acceptance_and_redaction(tmp_path):
    plan, task = campaign(tmp_path)
    visit(tmp_path, plan, task)
    base = tmp_path / "results" / task["task_id"] / "round_00"
    bad_event = {
        "step": "equation_functions_0",
        "attempt": 0,
        "accepted": False,
        "error": "dependency differs SECRET_SENTINEL",
        "raw": "SECRET_SENTINEL",
    }
    result = {
        "status": "complete",
        "events": [
            bad_event,
            {"step": "equation_functions_0", "attempt": 1, "accepted": True},
        ],
    }
    write(base / "process/function_stage.json", {"result": result}, sealed=True)
    write(base / "process/functions/progress.json", {"events": [bad_event]})
    write(
        base / "variables/result.json",
        {
            "status": "variables_complete",
            "events": [
                {
                    "step": "variable_inventory",
                    "attempt": 0,
                    "accepted": False,
                    "partial_acceptance": True,
                }
            ],
        },
    )
    value = audit.collect(tmp_path)
    assert len(value["attempts"]) == 3
    assert sum(a["partial_acceptance"] for a in value["attempts"]) == 1
    assert any(a["category"] == "dependency_contract" for a in value["attempts"])
    assert "SECRET_SENTINEL" not in json.dumps(value)
    assert not any(
        f["path"]
        == f"results/{task['task_id']}/round_00/process/functions/progress.json"
        for f in value["files"]
    )


@pytest.mark.parametrize("kind", ["seal", "task", "predecessor", "fit_hash", "symlink"])
def test_bad_artifacts_never_disappear_silently(tmp_path, kind):
    plan, task = campaign(tmp_path)
    result = visit(tmp_path, plan, task)
    base = tmp_path / "results" / task["task_id"] / "round_00"
    if kind == "fit_hash":
        path = base / "fit/result.json"
        value = json.loads(path.read_text())
        value["sha256"] = "bad"
        write(path, value)
    elif kind == "symlink":
        (base / "result.json").unlink()
        (base / "result.json").symlink_to(tmp_path / "test.csv")
    else:
        value = {k: v for k, v in result.items() if k != "artifact_sha256"}
        if kind == "seal":
            write(base / "result.json", {**value, "artifact_sha256": "bad"})
        else:
            value["task" if kind == "task" else "parent_sha256"] = "bad"
            write(base / "result.json", value, sealed=True)
    observed = audit.collect(tmp_path)
    assert observed["issues"]
    assert json.loads(audit.report(observed)["summary.json"])[
        "integrity_review_required"
    ]


def test_snapshot_notices_missing_and_read_files_changing(tmp_path):
    reader = Snapshot(tmp_path)
    assert reader.read("later.json") is None
    write(tmp_path / "later.json", {})
    write(tmp_path / "read.json", {"a": 1})
    assert reader.exists("read.json")
    assert reader.read("read.json") == {"a": 1}
    assert (
        reader.issues == []
    )  # A presence check followed by reading is not a mutation.
    write(tmp_path / "read.json", {"a": 200})
    reader.finish()
    assert {i["code"] for i in reader.issues} == {
        "appeared_during_collection",
        "changed_during_collection",
    }


def test_portable_cli_no_scientific_python_and_idempotent_report(tmp_path):
    root, output, portable = (tmp_path / n for n in ("campaign", "output", "portable"))
    plan, task = campaign(root)
    visit(root, plan, task)
    command = [sys.executable, "-S", str(REPO / "scripts/audit_component_campaign.py")]
    subprocess.run(
        [*command, "collect", "--root", str(root), "--output", str(output)], check=True
    )
    for _ in range(2):
        subprocess.run(
            [
                *command,
                "report",
                "--input",
                str(output / "audit.json"),
                "--output",
                str(portable),
            ],
            check=True,
        )
    assert (output / "SUMMARY.md").read_bytes() == (
        portable / "SUMMARY.md"
    ).read_bytes()
    assert (output / "audit.json").read_bytes() == (
        portable / "audit.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="fresh output"):
        audit.write_outputs(portable, {"SUMMARY.md": "different"})
    tampered = json.loads((output / "audit.json").read_text())
    tampered["rounds"][0]["status"] = "forged"
    with pytest.raises(ValueError, match="seal"):
        audit.report(tampered)
    assert (
        subprocess.run(
            [*command, "collect", "--root", str(root), "--output", str(root / "bad")],
            capture_output=True,
        ).returncode
        != 0
    )


def test_invalid_frozen_matrix_and_freeze_marker(tmp_path):
    plan, _ = campaign(tmp_path)
    (tmp_path / "evaluation_freeze.json").write_text("not read")
    assert (
        audit.collect(tmp_path)["lineages"][0]["frontier"]["action"]
        == "frozen_no_search_actions"
    )
    payload = {k: v for k, v in plan.items() if k != "artifact_sha256"}
    duplicate = deepcopy(payload)
    duplicate["tasks"] *= 2
    write(tmp_path / "plan.json", duplicate, sealed=True)
    with pytest.raises(ValueError, match="duplicate"):
        audit.collect(tmp_path)
    payload["tasks"][0]["task_id"] = "../test"
    write(tmp_path / "plan.json", payload, sealed=True)
    with pytest.raises(ValueError, match="identifier"):
        audit.collect(tmp_path)


def test_submission_exit_zero_without_job_id_is_unconfirmed(tmp_path):
    plan, _ = campaign(tmp_path)
    base = tmp_path / "submissions/wave1"
    write(base / "identity.json", {"plan_sha256": plan["artifact_sha256"]})
    write(
        base / "fit.reply.json",
        {
            "returncode": 0,
            "stdout": "\n",
            "stderr": "QOSMaxSubmitJobPerUserLimit SECRET_SENTINEL",
        },
    )
    write(base / "propose.reply.json", {"returncode": 0, "stdout": "1234;aces"})
    value = audit.collect(tmp_path)
    assert value["submissions"] == [
        {"wave": "wave1", "jobs": {"propose": "1234"}, "unconfirmed_stages": ["fit"]}
    ]
    assert audit.report(value)["job_ids.txt"] == "1234\n"
    assert "SECRET_SENTINEL" not in json.dumps(value)


def test_invalid_authorization_and_review_only_process_checkpoint(tmp_path):
    plan, task = campaign(tmp_path, critic=True)
    write(
        tmp_path / "critic_authorization.json",
        {
            "identity": plan["artifact_sha256"],
            "known_case_gate_passed": False,
        },
        sealed=True,
    )
    source = (
        tmp_path
        / "results"
        / task["task_id"]
        / "round_00/process/topology/process_review.json"
    )
    review = {"status": "empty", "suggestions": [], "error": None}
    write(source, {**review, "sha256": digest(review)})
    value = audit.collect(tmp_path)
    assert value["issues"]
    assert value["lineages"][0]["frontier"]["action"] == "inspect_critic_authorization"
    assert value["stages"][0]["stage"] == "process_proposal"
    assert value["stages"][0]["status"] == "empty"
