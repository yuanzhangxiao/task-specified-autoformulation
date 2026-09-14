"""Matched inputs, bounded repair, provenance, controls, and deterministic resume."""

import copy
import json
import sys
from pathlib import Path

import pytest

from autoformalism.llm.staged_topology import DeferredCall, atomic_json
from autoformalism.rebuttal import prefit_feedback as feedback
from autoformalism.rebuttal.prefit_replay import Diagnosis, ReplayCase, sealed_read
from scripts.smoke_prefit_feedback import client_for, synthetic_plan


@pytest.fixture
def frozen(tmp_path):
    plan = synthetic_plan(tmp_path)
    return tmp_path / "campaign", plan


def test_matched_requests_differ_only_in_feedback(frozen):
    root, plan = frozen
    entry = next(e for e in plan["selected"] if e["cohort"] == "repair")
    tasks = [t for t in plan["tasks"] if t["case_id"] == entry["case"]["case_id"]]
    bodies = []
    for task in tasks:
        calls = []
        result = feedback.run_episode(
            root, plan, task, client_for(root, plan, task, calls)
        )
        assert result["final"]["valid"]
        assert len(calls) == 1
        body = copy.deepcopy(calls[0])
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        assert "runtime_diagnostics" not in payload["construction_context"]
        payload.pop("feedback")
        body["messages"][1]["content"] = payload
        bodies.append(body)
    assert bodies[0] == bodies[1]


def test_resume_costs_and_summary_keep_control_denominator_separate(frozen):
    root, plan = frozen
    calls = []
    for task in plan["tasks"]:
        result = feedback.run_episode(
            root, plan, task, client_for(root, plan, task, calls)
        )
        count = len(calls)
        assert (
            feedback.run_episode(root, plan, task, client_for(root, plan, task, calls))
            == result
        )
        assert len(calls) == count
    result = feedback.summarize(root)
    assert result["status"] == "complete"
    for arm in feedback.ARMS:
        assert result["arms"][arm]["repair"]["expected"] == 2
        assert result["arms"][arm]["repair"]["valid_final"] == 2
        assert result["arms"][arm]["valid_control"]["expected"] == 4
        assert result["arms"][arm]["valid_control"]["valid_control_changed"] == 0
    assert sum(r["physical_requests"] for r in result["rows"]) == len(calls)


def test_deferred_episode_is_pending_and_resumes(frozen):
    root, plan = frozen
    task = plan["tasks"][0]
    calls = []
    with pytest.raises(DeferredCall):
        feedback.run_episode(
            root,
            plan,
            task,
            client_for(root, plan, task, calls, can_start=lambda: False),
        )
    assert not calls
    assert feedback.summarize(root)["status"] == "partial"
    assert feedback.run_episode(root, plan, task, client_for(root, plan, task, calls))[
        "final"
    ]["valid"]


def test_exhausted_repairs_do_not_restart_or_disappear(frozen):
    root, plan = frozen
    task = next(t for t in plan["tasks"] if t["cohort"] == "repair")
    calls = []
    client = client_for(root, plan, task, calls)

    def failing(url, body, timeout):
        calls.append(body)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"expression":"m","parameters":[]}'},
                }
            ],
            "usage": {"total_tokens": 100},
        }

    client.transport = failing
    result = feedback.run_episode(root, plan, task, client)
    assert result["stop_reason"] == "attempts_exhausted"
    assert len(result["attempts"]) == len(calls) == 3
    assert (
        feedback.run_episode(root, plan, task, client_for(root, plan, task, calls))
        == result
    )
    row = next(
        r for r in feedback.summarize(root)["rows"] if r["task_id"] == task["task_id"]
    )
    assert not row["valid_final"] and row["repeated_error_evaluations"] == 3
    assert row["physical_requests"] == 3


def test_inflight_response_is_consumed_as_uncertain_not_resent(frozen):
    root, plan = frozen
    task = next(t for t in plan["tasks"] if t["cohort"] == "repair")
    calls = []
    client = client_for(root, plan, task, calls)
    client.transport = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        feedback.run_episode(root, plan, task, client)
    result = feedback.run_episode(root, plan, task, client_for(root, plan, task, calls))
    assert len(calls) == 1 and len(result["attempts"]) == 2
    assert (
        result["attempts"][0]["diagnosis"]["diagnostics"][0]["code"]
        == "PROVIDER_RESPONSE_ERROR"
    )
    row = next(
        r for r in feedback.summarize(root)["rows"] if r["task_id"] == task["task_id"]
    )
    assert row["physical_requests"] == 2 and row["unknown_usage_requests"] == 1


@pytest.mark.parametrize("missing", [True, False])
def test_referenced_response_cannot_be_lost_or_modified(frozen, missing):
    root, plan = frozen
    task = plan["tasks"][0]
    client = client_for(root, plan, task, [])
    feedback.run_episode(root, plan, task, client)
    path = next(client.directory.glob("*.json"))
    if missing:
        path.unlink()
    else:
        data = json.loads(path.read_text())
        data["raw_response"]["usage"] = {"total_tokens": 1}
        atomic_json(path, data)
    with pytest.raises(ValueError, match="missing or changed"):
        feedback.summarize(root)


def test_changed_validator_and_corpus_fail_closed(frozen, monkeypatch):
    root, plan = frozen
    monkeypatch.setattr(feedback, "runtime_source_hash", lambda: "changed")
    with pytest.raises(ValueError, match="runtime differs"):
        feedback.verify(root)
    corpus = sealed_read(root / "replay.json")
    monkeypatch.setattr(feedback, "diagnose", lambda *args: Diagnosis(valid=False))
    with pytest.raises(ValueError, match="validator differs"):
        feedback.select_cases(
            corpus, feedback.FeedbackConfig.model_validate(plan["config"])
        )


def test_no_validation_tables_fit_results_or_optimizer_calls(frozen, monkeypatch):
    root, plan = frozen
    original = Path.read_bytes

    def guard(path):
        assert path.suffix != ".csv" and path.name != "fit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)

    def forbidden(*args, **kwargs):
        pytest.fail("fitting or rollout was invoked by the local feedback experiment")

    for name, module in tuple(sys.modules.items()):
        if name.startswith("autoformalism.") and module is not None:
            for entrypoint in (
                "fit_candidate",
                "evaluate_fitted_candidate",
                "simulate_trajectory",
                "initialize_parameters_with_multiple_shooting",
            ):
                if hasattr(module, entrypoint):
                    monkeypatch.setattr(module, entrypoint, forbidden)
    task = plan["tasks"][0]
    assert feedback.run_episode(root, plan, task, client_for(root, plan, task, []))[
        "final"
    ]["valid"]
    assert feedback.verify(root) == plan


def test_structured_feedback_contains_facts_not_a_replacement_formula(frozen):
    _, plan = frozen
    entry = next(e for e in plan["selected"] if e["cohort"] == "repair")
    case = ReplayCase.model_validate(entry["case"])
    payload = feedback.payload_for(
        case,
        case.original_reply,
        Diagnosis.model_validate(entry["baseline"]),
        "structured_feedback",
    )
    assert payload["feedback"]["facts"]["required_sources"] == ["m", "u01"]
    assert payload["feedback"]["facts"]["topology_editable"] is False
    assert payload["feedback"]["scientific_status"] == "not_assessed"


def test_valid_control_can_acquire_a_violation_and_change_after_repair(frozen):
    root, plan = frozen
    entry = next(
        e
        for e in plan["selected"]
        if e["cohort"] == "valid_control" and e["case"]["kind"] == "initializer"
    )
    task = next(t for t in plan["tasks"] if t["case_id"] == entry["case"]["case_id"])
    client = client_for(root, plan, task, [])
    replies = iter(
        [
            {
                "initial": {
                    "mode": "causal_map",
                    "expression": "future",
                    "parameters": [],
                }
            },
            {
                "initial": {
                    "mode": "causal_map",
                    "expression": "2*v01",
                    "parameters": [],
                }
            },
        ]
    )

    def transport(*args):
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(next(replies)),
                    },
                }
            ],
            "usage": {"total_tokens": 100},
        }

    client.transport = transport
    assert feedback.run_episode(root, plan, task, client)["final"]["valid"]
    row = next(
        r for r in feedback.summarize(root)["rows"] if r["task_id"] == task["task_id"]
    )
    assert row["valid_control_changed"] and row["valid_final"]
    assert not row["first_attempt_valid"] and row["new_violation_evaluations"] == 1


def test_modified_inference_settings_require_a_separate_experiment(frozen):
    root, plan = frozen
    config = copy.deepcopy(plan["config"])
    config["model_settings"]["temperature"] = 0.7
    path = root.parent / "changed-config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="inference setting differs"):
        feedback.freeze(root / "replay.json", path, root.parent / "changed")


def test_replay_with_no_residual_failures_does_not_launch_live_calls(frozen):
    root, plan = frozen
    corpus = sealed_read(root / "replay.json")
    corpus["rows"] = [r for r in corpus["rows"] if r["after_normalization"]["valid"]]
    ids = {r["case_id"] for r in corpus["rows"]}
    corpus["cases"] = [c for c in corpus["cases"] if c["case_id"] in ids]
    with pytest.raises(ValueError, match="no residual local failures"):
        feedback.select_cases(
            corpus, feedback.FeedbackConfig.model_validate(plan["config"])
        )


def test_worker_drains_before_a_request_and_restores_signal_handlers(frozen):
    import signal

    root, _ = frozen
    before = signal.getsignal(signal.SIGTERM)
    report = feedback.run(root, "http://unused", wall_seconds=0)
    assert report["status"] == "partial"
    assert all(r["physical_requests"] == 0 for r in report["rows"])
    assert signal.getsignal(signal.SIGTERM) == before
