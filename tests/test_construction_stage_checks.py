"""Stage-local process feedback without inferred scientific corrections."""

import json

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_shared_report
from autoformalism.search import construction_schedules as schedules
from autoformalism.search import construction_stage_checks as checks
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    patch,
    process,
    variable,
)
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


def check(draft):
    return checks.shared_checklist(brief(), draft)


def test_valid_transfer_needs_no_ordinary_topology_and_forward_drivers_are_deferred():
    d = apply(variables=[variable("x"), variable("y")], processes=[process()])
    assert check(d)["status"] == "ready"
    assert check(d)["shared_declarations"] == 1
    d = apply(d, processes=[process(depends_on=["later"])])
    assert check(d)["status"] == "ready"
    assert check(d)["processes"][0]["unresolved_references"] == ["later"]
    d = apply(
        d,
        processes=[
            process(
                uses=[
                    {"target": "later", "sign": "positive"},
                    {"target": "y", "sign": "negative"},
                ]
            )
        ],
    )
    assert check(d)["status"] == "ready"  # Future variables may still be declared.


@pytest.mark.parametrize("target", ["a", "u", "area"])
def test_supplied_receivers_are_blocked_but_observed_auxiliary_can_be_modeled(target):
    d = apply(
        variables=[variable("y")],
        processes=[
            process(
                kind="influence",
                depends_on=["u"],
                uses=[{"target": target, "sign": "positive"}],
            )
        ],
    )
    result = check(d)
    assert result["items"][0]["code"] == "supplied_receiver"
    assert result["items"][0]["receiver"] == target
    if target == "a":
        assert "explicitly declare" in result["items"][0]["action"]
        d = apply(d, variables=[variable("a", "algebraic")])
        assert check(d)["status"] == "ready"
    else:
        assert "cannot be redeclared" in result["items"][0]["action"]


def test_same_receiver_with_opposite_signs_is_not_shared_or_a_valid_transfer():
    d = apply(
        variables=[variable("y")],
        processes=[
            process(
                uses=[
                    {"target": "y", "sign": "negative"},
                    {"target": "y", "sign": "positive"},
                ]
            )
        ],
    )
    result = check(d)
    assert [item["code"] for item in result["items"]] == [
        "duplicate_receiver",
        "transfer_signs",
    ]
    assert "revise kind to influence" in result["items"][1]["action"]
    assert result["shared_declarations"] == 0
    assert result["status"] == "incomplete"
    # Both diagnoses are available immediately; the proposer need not first
    # delete one use just to discover the single-receiver transfer error.
    assert len(d.processes[0].uses) == 2
    assert d.processes[0].kind == "transfer"
    repaired = apply(
        d,
        processes=[
            process(
                kind="influence",
                uses=[{"target": "y", "sign": "positive"}],
            )
        ],
    )
    assert check(repaired)["status"] == "ready"
    assert check(repaired)["local_declarations"] == 1


def test_transfer_sign_options_do_not_invent_a_receiver_or_change_kind():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[
            process(
                uses=[
                    {"target": "x", "sign": "positive"},
                    {"target": "y", "sign": "positive"},
                ]
            )
        ],
    )
    assert check(d)["items"][0]["code"] == "transfer_signs"
    assert d.processes[0].kind == "transfer"
    d = apply(
        d,
        processes=[
            process(
                kind="influence",
                uses=[
                    {"target": "y", "sign": "positive"},
                ],
            )
        ],
    )
    assert check(d)["status"] == "ready"
    assert check(d)["local_declarations"] == 1
    assert check(d)["shared_declarations"] == 0


def test_identifier_examples_and_empty_ordinary_entries_are_only_suggestions():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[equation("dy_dt", "u")],
    )
    view = checks.topology_editing(d)
    mismatch = view["possible_derivative_label_mismatches"][0]
    assert mismatch["possible_state_names"] == ["y"]
    assert mismatch["remove_old_label_if_renaming"] == {"remove_equations": ["dy_dt"]}
    assert [
        r["if_no_other_contributions"] for r in view["pending_process_consumers"]
    ] == [
        {"name": "x", "terms": []},
        {"name": "y", "terms": []},
    ]
    assert d.equations[0].name == "dy_dt"
    d = apply(d, variables=[variable("dy_dt")])
    assert checks.topology_editing(d)["possible_derivative_label_mismatches"] == []


def test_empty_shared_decision_is_valid_and_algebraic_driver_loops_are_not_guessed():
    assert check(apply(variables=[variable("y")]))["status"] == "ready"
    d = apply(
        variables=[variable("y", "algebraic")],
        processes=[
            process(
                kind="influence",
                depends_on=["y"],
                uses=[{"target": "y", "sign": "positive"}],
            )
        ],
    )
    assert check(d)["status"] == "ready"
    assert "algebraic cycles" in check(d)["deferred"]


def execute(root, send, can_start=lambda: True):
    client = ConstructionClient(
        settings=StagedModelSettings(),
        directory=root / "calls",
        namespace="stage-checks",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
        can_start=can_start,
    )
    b = brief()
    return schedules.run(
        b,
        context(),
        {},
        b.model_dump(mode="json"),
        client,
        root / "construction",
        "joint_adaptive",
        prompt_family="minimal",
        process_question="dedicated",
        bookkeeping_policy=checks.POLICY,
    )


def test_shared_stage_immediate_repair_then_specific_name_feedback_and_cached_resume(
    tmp_path,
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        if p["stage"] == "variables":
            edit = patch(variables=[variable("x"), variable("y")], stage_complete=True)
        elif p["stage"] == "shared_laws" and len(calls) == 2:
            edit = patch(
                processes=[
                    process(
                        uses=[
                            {"target": "x", "sign": "negative"},
                            {"target": "a", "sign": "positive"},
                        ]
                    )
                ],
                stage_complete=True,
            )
        elif p["stage"] == "shared_laws":
            assert (
                p["runtime_diagnostics"]["stage_completion"]["status"] == "incomplete"
            )
            assert (
                p["shared_process_checklist"]["items"][0]["code"] == "supplied_receiver"
            )
            # Only an explicit proposer choice changes the supplied auxiliary.
            edit = patch(variables=[variable("a", "algebraic")], stage_complete=True)
        elif p["stage"] == "equations":
            assert checks.IDENTIFIERS in p["stage_instructions"]
            assert {"name": "a", "terms": []} in [
                x["if_no_other_contributions"]
                for x in p["topology_editing"]["pending_process_consumers"]
            ]
            edit = patch(
                equations=[
                    equation("dy_dt", "a"),
                    equation("x", "u"),
                    {"name": "a", "terms": []},
                ],
                stage_complete=True,
            )
        else:
            assert p["topology_editing"]["possible_derivative_label_mismatches"][0][
                "possible_state_names"
            ] == ["y"]
            edit = patch(
                equations=[equation("y", "a")],
                remove_equations=["dy_dt"],
                stage_complete=True,
            )
        return response(edit.model_dump(mode="json"))

    with pytest.raises(DeferredCall):
        execute(tmp_path, send, can_start=lambda: len(calls) < 2)
    result = execute(tmp_path, send)
    assert result["status"] == "topology_complete"
    assert len(calls) == 5 and execute(tmp_path, send) == result and len(calls) == 5
    first = sealed_read(tmp_path / "construction/events/001.json")
    assert first["accepted"] and not first["stage_complete"]
    assert first["shared_process_checklist_after"]["status"] == "incomplete"
    last = sealed_read(tmp_path / "construction/events/002.json")
    assert last["stage_complete"] and last["stage_completion"]["status"] == "ready"


def test_shared_stage_exhaustion_keeps_partial_records_and_continues_topology(
    tmp_path, monkeypatch
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        if p["stage"] == "variables":
            edit = patch(variables=[variable("y")], stage_complete=True)
        elif p["stage"] == "shared_laws":
            edit = patch(
                processes=[
                    process(
                        depends_on=["u"],
                        uses=[
                            {"target": "y", "sign": "positive"},
                            {"target": "y", "sign": "negative"},
                        ],
                    )
                ],
                stage_complete=True,
            )
        else:
            assert p["shared_process_checklist"]["status"] == "incomplete"
            edit = patch(
                remove_processes=["p"],
                equations=[equation("y", "u")],
                stage_complete=True,
            )
        return response(edit.model_dump(mode="json"))

    directory = tmp_path / "results" / "task"
    result = execute(directory, send)
    assert result["status"] == "topology_complete"
    assert sum(c["stage"] == "shared_laws" for c in calls) == 3
    outcome = result["before_repair"]["stage_outcomes"][1]
    assert not outcome["completed"] and outcome["continued_to_equations"]
    assert "exhausted" in outcome["error"]
    sealed_write(directory / "proposal.json", result)
    monkeypatch.setattr(
        construction_shared_report, "visible_brief", lambda *args: brief()
    )
    report = construction_shared_report.report(
        tmp_path,
        {
            "artifact_sha256": "test",
            "tasks": [{"task_id": "task", "benchmark_id": "case"}],
            "cells": {"case": {}},
        },
    )
    row = report["rows"][0]
    assert report["completed_shared_stages"] == 0
    assert row["immediate_repair_calls"] == 2
    assert row["first_retained"]["status"] == "incomplete"
    assert row["last_shared_snapshot"]["status"] == "incomplete"
    assert row["before_global_repair"]["status"] == "ready"
    assert row["after_global_repair"]["processes"] == []
