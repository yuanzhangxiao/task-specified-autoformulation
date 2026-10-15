"""Conditional data advice is asked, recorded and replayed without a science gate."""

import json

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search import construction_evidence as core
from autoformalism.search import construction_schedules as schedules
from autoformalism.search.topology_compatibility import EvidenceContract, build_evidence
from tests.test_construction_ledger import brief, context, equation, variable
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


def complete_evidence():
    data = Trajectory(
        trajectory_id="train-0",
        time=np.array([0.0, 1.0, 2.0]),
        targets={"y": np.array([0.0, 1.0, 0.0])},
        external_inputs={"u": np.zeros(3)},
        auxiliaries={"a": np.zeros(3)},
        fixed_covariates={"area": 1.0},
        derivatives={},
    )
    return build_evidence(
        DatasetSplit(SplitName.TRAIN, (data,), "fixture"),
        context(),
        EvidenceContract(
            provenance="Synthetic explicit candidate-class hypotheses",
            dependencies_exhaustive=True,
            shared_parameters=True,
            open_loop=True,
            reset_impulse_free=True,
            uniqueness_assumed=True,
            interpolation={"u": "linear_complete_knots", "a": "linear_complete_knots"},
            initial_rules={"y": {"mode": "observed_initial"}},
            absolute_error_bounds={"y": 0, "u": 0, "a": 0, "area": 0},
        ),
    ).model_dump(mode="json")


def execute(root, send, evidence=None, *, policy=core.POLICY):
    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=root / "calls",
        namespace="evidence",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
    )
    return schedules.run(
        brief(),
        context(),
        {},
        brief().model_dump(mode="json"),
        client,
        root / "construction",
        "joint_adaptive",
        prompt_family="minimal",
        process_question="dedicated",
        bookkeeping_policy=policy,
        training_evidence=complete_evidence() if evidence is None else evidence,
    )


def test_qualified_conflict_asks_after_complete_topology_and_retains_approximation(
    tmp_path,
):
    calls = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        calls.append(packet)
        edit = {**packet["response_template"], "stage_complete": True}
        if packet["stage"] == "variables":
            edit["variables"] = [variable("y")]
        elif packet["stage"] == "equations":
            edit["equations"] = [equation("y", "u")]
        elif packet["stage"] == "repair":
            # The live presentation must expose the exact question IDs.
            def questions(value):
                if isinstance(value, dict):
                    if "question_id" in value and "witnesses" in value:
                        yield value
                    for child in value.values():
                        yield from questions(child)
                elif isinstance(value, list):
                    for child in value:
                        yield from questions(child)

            shown = {q["question_id"]: q for q in questions(packet)}
            assert shown
            assert all(q["assumptions"] for q in shown.values())
            edit["compatibility_decisions"] = [
                {
                    "question_id": key,
                    "disposition": "retain_approximation",
                    "explanation": "Keep this approximation for later fitting.",
                }
                for key in shown
            ]
        return response(edit)

    result = execute(tmp_path, send)
    assert [p["stage"] for p in calls] == [
        "variables",
        "shared_laws",
        "equations",
        "repair",
    ]
    assert result["before_repair"]["assessment"]["eligible"]
    assert result["status"] == "topology_complete"
    assert (
        result["assessment"]["training_compatibility"]["review_status"] == "addressed"
    )
    assert (
        result["assessment"]["training_compatibility"]["scientific_correctness"]
        == "unverified"
    )
    event = sealed_read(tmp_path / "construction/events/003.json")
    assert event["before"] == event["after"]
    assert event["compatibility_decisions"]
    count = len(calls)
    assert execute(tmp_path, send) == result
    assert len(calls) == count


def test_unknown_question_cannot_be_acknowledged():
    class Decision:
        def model_dump(self, **kwargs):
            return {"question_id": "not-shown", "disposition": "retain_approximation"}

    with pytest.raises(ValueError, match="displayed"):
        schedules.validate_advisory_decisions([Decision()], [])


def test_evidence_integrity_checked_before_any_request(tmp_path):
    value = complete_evidence()
    value["trajectories"][0]["targets"]["y"][1] += 1
    with pytest.raises(ValueError, match=r"digest|sha256"):
        execute(tmp_path, lambda *args: pytest.fail("no request expected"), value)


def test_answering_one_question_each_turn_is_useful_progress(tmp_path, monkeypatch):
    from autoformalism.research import construction_training

    def advice(draft, evidence):
        return {
            "evidence_sha256": evidence.sha256,
            "topology_sha256": "synthetic-workflow-fixture",
            "advisory_questions": [
                {"question_id": f"q{i}", "question": "Synthetic workflow question"}
                for i in range(4)
            ],
            "full_report": {},
        }

    monkeypatch.setattr(construction_training, "compatibility_feedback", advice)
    repaired = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        edit = {**packet["response_template"], "stage_complete": True}
        if packet["stage"] == "variables":
            edit["variables"] = [variable("y")]
        elif packet["stage"] == "equations":
            edit["equations"] = [equation("y", "u")]
        elif packet["stage"] == "repair":
            question = packet["compatibility_questions"][0]
            repaired.append(question["question_id"])
            edit["compatibility_decisions"] = [
                {
                    "question_id": question["question_id"],
                    "disposition": "dispute_assumption",
                    "explanation": "This fixture tests incremental advisory answers.",
                }
            ]
        return response(edit)

    result = execute(tmp_path, send)
    assert repaired == ["q0", "q1", "q2", "q3"]
    assert (
        result["assessment"]["training_compatibility"]["review_status"] == "addressed"
    )
    assert result["stop_reason"] is None
