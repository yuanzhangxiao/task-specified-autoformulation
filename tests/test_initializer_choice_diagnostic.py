"""Isolated initializer candidates, exact probes, and charged deterministic resume."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.staged_topology import StagedModelSettings
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_training as training
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research import initialized_topology_repair as initialized
from autoformalism.research import initializer_choice_diagnostic as diagnostic
from autoformalism.research.initializer_choice_inputs import BUDGETS
from autoformalism.search import construction_initialized_repair as api
from autoformalism.search.topology_initialization import InitializerDependencyHandoff
from autoformalism.staged_topology import content_hash
from tests.test_construction_ledger import equation, variable
from tests.test_explicit_variable_bindings import response
from tests.test_initialization_evidence_feedback import sealed
from tests.test_phase_c_construction_baseline import tokenize
from tests.test_saved_topology_repair import setup as saved_setup  # noqa: F401


@pytest.fixture
def plan(saved_setup):  # noqa: F811
    """Two complete matched experiments contradict a shared-initial latent subsystem.

    Both targets start at zero and receive the same forcing, but y(1) differs.
    The retained topology y'=F(z,u), z'=G(z,u) and shared z(0) cannot explain
    that discrepancy under the explicit deterministic/unique-solution contract.
    The recorded covariate area differs, allowing a distinct initialization
    hypothesis without changing the equations or inventing a measurement.
    """
    parent, tasks, problem = saved_setup
    task = tasks[1]
    raw_evidence = problem.evidence.model_dump(mode="json")
    second = deepcopy(raw_evidence["trajectories"][0])
    second["trajectory_id"] = "train-1"
    second["targets"]["y"] = [0.0, 2.0, 0.0]
    second["covariates"]["area"] = 2.0
    raw_evidence["trajectories"].append(second)
    problem.evidence = sealed(raw_evidence)
    problem.highlights = training.initial_highlights(problem.evidence)
    draft = api.Draft.model_validate(
        {
            "variables": [variable("y"), variable("z")],
            "equations": [equation("y", "z", "u"), equation("z", "z", "u")],
        }
    )
    initial_contract = audit.information_contract(
        problem.evidence, "causal_runtime"
    ).model_dump(mode="json")
    initial_contract.update(
        protocol="recorded-t0-initial-information-2", measured_initial_bindings=[]
    )
    start = {
        "task": {},
        "draft": draft.model_dump(mode="json"),
        "assessment": problem.assess(draft),
        "diagnostics": problem.diagnose(draft),
        "initial_information_contract": initial_contract,
        "initializer_handoff": {
            "records": [
                {"state": "z", "rule": {"mode": "shared_parameter", "arguments": []}}
            ]
        },
    }
    start["initializer_handoff"] = InitializerDependencyHandoff.model_validate(
        start["initializer_handoff"]
    ).model_dump(mode="json")
    session = initialized.Session(problem, start, policy=api.POLICY_V6)
    check = problem.assess(draft, qualified_memory=True)
    report = session.diagnose(draft, session.baseline)
    state = session.state(draft, report, session.baseline, check["eligible"], {})
    assert check["eligible"]
    assert state["repair_status"] == "repair_required"
    settings = StagedModelSettings.model_validate(parent["settings"])
    body = diagnostic.request_body(
        settings,
        task["seed"],
        "repair_001",
        1,
        "Saved test request",
        "Saved test packet",
        api.response_model("repair"),
    )
    record = {
        "step": "repair_001",
        "attempt": 1,
        "request": {
            "body": json.loads(json.dumps(body, sort_keys=True)),
            "settings": settings.model_dump(mode="json"),
        },
    }
    case = {
        "task": task,
        "start": start,
        "cell": {},
        "editable_state": "z",
        "source_result": {"scientific_state": state, "cost": _cost([])},
        "latest_request_record": record,
    }
    # Exercise the actual projection/validation in the fixture, not a stubbed card.
    packet = diagnostic.choice_packet(case, problem, draft, session, state, None, 0, 0)
    assert packet["model_data_conflicts"]
    assert packet["compatibility_questions"]
    return {
        "protocol": "saved-initializer-choice-diagnostic-1",
        "artifact_sha256": "fixture",
        "parent_plan": parent,
        "budgets": dict(BUDGETS),
        "cases": {"saved": case},
        "completion_probe": {
            "case_id": "saved",
            "request": record["request"],
            "request_hash": content_hash(record["request"]),
            "repeats": 2,
        },
    }


def run_choice(root, plan, send, **kwargs):
    return diagnostic.run_choice(
        root,
        plan,
        "choice-saved",
        "saved",
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
        **kwargs,
    )


def decision(mode="shared_parameter", arguments=()):
    return {
        "decision": "propose_initializer",
        "rule": {"mode": mode, "arguments": list(arguments)},
        "explanation": "A shared initializer dependency hypothesis.",
    }


def test_choice_updates_only_diagnostic_initializer_and_retains_evidence(
    tmp_path, plan
):
    original = deepcopy(plan)
    result = run_choice(
        tmp_path, plan, lambda *args: response(decision("common_map", ["u"]))
    )
    event = sealed_read(tmp_path / "results/choice-saved/events/000.json")
    assert event["accepted_initializer"]
    assert event["initializer_before"] != event["initializer_after"]
    assert result["frozen_draft_sha256"] == content_hash(
        plan["cases"]["saved"]["start"]["draft"]
    )
    assert result["source_model_changes"] == 0
    assert not result["scientific_accuracy_established"]
    assert "tracked_conflicts" in event["scientific_state"]
    assert plan == original
    assert run_choice(tmp_path, plan, lambda *a: pytest.fail("cached")) == result


@pytest.mark.parametrize(
    "reply",
    [
        decision("observed_initial"),
        decision("common_map", ["future"]),
        {**decision(), "equations": []},
    ],
)
def test_illegal_choices_get_one_feedback_attempt_without_edits(tmp_path, plan, reply):
    sent = []

    def send(url, body, timeout):
        sent.append(json.loads(body["messages"][1]["content"]))
        return response(reply)

    result = run_choice(tmp_path, plan, send)
    assert result["cost"]["physical_requests"] == 2
    assert (
        result["candidate_initializer_handoff"]
        == plan["cases"]["saved"]["start"]["initializer_handoff"]
    )
    assert "error" in json.dumps(sent[1])
    assert all(
        not sealed_read(p)["accepted_initializer"]
        for p in (tmp_path / "results/choice-saved/events").glob("*.json")
    )


@pytest.mark.parametrize(
    "choice", ["topology_change_needed", "insufficient_information"]
)
def test_scope_decision_does_not_close_scientific_conflict(tmp_path, plan, choice):
    reply = {
        "decision": choice,
        "rule": None,
        "explanation": "Initializer-only scope is insufficient.",
    }
    result = run_choice(tmp_path, plan, lambda *a: response(reply))
    assert result["stop_reason"] == choice
    assert result["cost"]["physical_requests"] == 1
    assert result["scientific_status"] == "repair_required"
    assert (
        result["candidate_initializer_handoff"]
        == plan["cases"]["saved"]["start"]["initializer_handoff"]
    )


def test_incomplete_output_is_terminal_charged_and_not_retried(tmp_path, plan):
    def send(*args):
        raw = response(decision())
        raw["choices"][0]["finish_reason"] = "length"
        raw["choices"][0]["message"]["content"] = "{  "
        return raw

    result = run_choice(tmp_path, plan, send)
    assert result["status"] == "provider_unavailable"
    assert result["cost"]["physical_requests"] == 1
    assert result["cost"]["observed_tokens"] > 0
    assert run_choice(tmp_path, plan, lambda *a: pytest.fail("retry")) == result


def test_inflight_crash_resumes_as_uncertain_without_repeating_call(tmp_path, plan):
    def crash(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_choice(tmp_path, plan, crash)
    result = run_choice(tmp_path, plan, lambda *a: pytest.fail("retry inflight"))
    assert result["status"] == "provider_unavailable"
    assert result["cost"]["requests_with_unknown_usage"] == 1
    assert result["cost"]["budget_charge"] > 0


def test_probe_repeats_exact_body_and_does_not_gate_independent_choices(tmp_path, plan):
    sent = []

    def send(url, body, timeout):
        sent.append(body)
        if len(sent) <= 2:
            result = response({})
            result["choices"][0]["finish_reason"] = "length"
            return result
        return response(
            {
                "decision": "topology_change_needed",
                "rule": None,
                "explanation": "Additional topology is required.",
            }
        )

    result = diagnostic.run(
        tmp_path,
        plan,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
    )
    assert sent[0] == sent[1] == plan["completion_probe"]["request"]["body"]
    assert len(sent) == 3
    assert result["status"] == "complete"
    assert result["availability_status"] == "unavailable"
    assert result["status_counts"] == {"provider_unavailable": 2, "assessed": 1}
    assert (
        diagnostic.run(
            tmp_path,
            plan,
            base_url="http://offline",
            transport=lambda *a: pytest.fail("retry"),
            token_transport=tokenize,
        )
        == result
    )


def test_tampered_event_or_call_is_rejected_on_resume_and_report(tmp_path, plan):
    run_choice(tmp_path, plan, lambda *a: response(decision()))
    path = next((tmp_path / "results/choice-saved/calls").glob("*.json"))
    value = json.loads(path.read_text())
    value["raw_response"]["usage"]["total_tokens"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        diagnostic.report(tmp_path, plan)
    with pytest.raises(ValueError):
        run_choice(tmp_path, plan, lambda *a: pytest.fail("tampered cache"))


def test_deferred_work_has_no_generation_and_reports_pending(tmp_path, plan):
    summary = diagnostic.run(
        tmp_path,
        plan,
        base_url="http://offline",
        can_start=lambda: False,
        transport=lambda *a: pytest.fail("deadline"),
        token_transport=tokenize,
    )
    assert summary["status"] == "pending"
    assert summary["physical_requests"] == 0


def run_unit(root, plan, kind, send):
    if kind == "choice":
        return run_choice(root, plan, send)
    return diagnostic.run_probe(
        root,
        plan,
        "completion-0",
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
    )


@pytest.mark.parametrize("kind", ["choice", "probe"])
@pytest.mark.parametrize("keep_result", [True, False])
def test_missing_call_with_saved_event_fails_before_paid_transport(
    tmp_path, plan, kind, keep_result
):
    run_unit(
        tmp_path,
        plan,
        kind,
        lambda *a: response(
            {
                "decision": "topology_change_needed",
                "rule": None,
                "explanation": "A topology change is required.",
            }
        ),
    )
    unit = "choice-saved" if kind == "choice" else "completion-0"
    directory = tmp_path / "results" / unit
    if not keep_result:
        (directory / "result.json").unlink()
    next((directory / "calls").glob("*.json")).unlink()
    with pytest.raises(ValueError):
        run_unit(
            tmp_path,
            plan,
            kind,
            lambda *a: pytest.fail("missing call cannot be retried"),
        )
    with pytest.raises(ValueError):
        diagnostic.report(tmp_path, plan)


@pytest.mark.parametrize("kind", ["choice", "probe"])
def test_response_checkpoint_before_event_crash_replays_without_http(
    tmp_path, plan, monkeypatch, kind
):
    write = diagnostic.sealed_write

    def crash_before_event(path, payload):
        if path.parent.name == "events":
            raise KeyboardInterrupt("crash after response checkpoint")
        return write(path, payload)

    monkeypatch.setattr(diagnostic, "sealed_write", crash_before_event)
    with pytest.raises(KeyboardInterrupt):
        run_unit(
            tmp_path,
            plan,
            kind,
            lambda *a: response(
                {
                    "decision": "topology_change_needed",
                    "rule": None,
                    "explanation": "An initializer cannot repair the topology.",
                }
            ),
        )
    monkeypatch.setattr(diagnostic, "sealed_write", write)
    result = run_unit(
        tmp_path, plan, kind, lambda *a: pytest.fail("saved response must replay")
    )
    assert result["status"] == "assessed"
    assert result["cost"]["physical_requests"] == 1
    assert result["cost"]["requests_with_unknown_usage"] == 0


@pytest.mark.parametrize("kind", ["choice", "probe"])
def test_unlinked_different_request_refuses_fresh_generation(tmp_path, plan, kind):
    run_unit(
        tmp_path,
        plan,
        kind,
        lambda *a: response(
            {
                "decision": "topology_change_needed",
                "rule": None,
                "explanation": "A topology change is required.",
            }
        ),
    )
    unit = "choice-saved" if kind == "choice" else "completion-0"
    directory = tmp_path / "results" / unit
    (directory / "result.json").unlink()
    for path in (directory / "events").glob("*.json"):
        path.unlink()
    path = next((directory / "calls").glob("*.json"))
    record = json.loads(path.read_text())
    record["request"]["body"]["messages"][1]["content"] = "A different request."
    record["request_hash"] = content_hash(record["request"])
    path.unlink()
    path.with_name(f"{record['request_hash']}.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="orphan cached request"):
        run_unit(
            tmp_path,
            plan,
            kind,
            lambda *a: pytest.fail("mismatched orphan cannot authorize another call"),
        )


@pytest.mark.parametrize("kind", ["choice", "probe"])
def test_terminal_preflight_without_token_cache_does_not_retry(tmp_path, plan, kind):
    def unavailable(*args):
        raise ValueError("tokenizer unavailable")

    unit = "choice-saved" if kind == "choice" else "completion-0"

    def run(counter):
        common = {
            "base_url": "http://offline",
            "transport": lambda *a: pytest.fail("preflight cannot start generation"),
            "token_transport": counter,
        }
        if kind == "choice":
            return diagnostic.run_choice(tmp_path, plan, unit, "saved", **common)
        return diagnostic.run_probe(tmp_path, plan, unit, **common)

    first = run(unavailable)
    assert first["status"] == "preflight_blocked"
    assert first["cost"]["physical_requests"] == 0
    for path in (tmp_path / "results" / unit / "calls/tokenization").glob("*.json"):
        path.unlink()
    assert run(tokenize) == first


def test_renamed_event_is_rejected_before_any_generation(tmp_path, plan):
    reply = {
        "decision": "topology_change_needed",
        "rule": None,
        "explanation": "Requires a topology change.",
    }
    run_choice(tmp_path, plan, lambda *a: response(reply))
    event = tmp_path / "results/choice-saved/events/000.json"
    event.rename(event.with_name("099.json"))
    with pytest.raises(ValueError, match="event filenames"):
        run_choice(tmp_path, plan, lambda *a: pytest.fail("new generation"))
    with pytest.raises(ValueError, match="event filenames"):
        diagnostic.report(tmp_path, plan)
