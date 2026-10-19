"""Readable retests preserve completed history while bounding new initializer calls."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_training as training
from autoformalism.research import initialization_handoff_audit as handoff_audit
from autoformalism.research import initializer_choice_diagnostic as diagnostic
from autoformalism.research import initializer_choice_inputs as inputs
from autoformalism.research import saved_topology_repair as saved
from autoformalism.search import construction_initialized_repair as api
from autoformalism.staged_topology import content_hash
from tests.test_explicit_variable_bindings import response
from tests.test_initialized_topology_repair import packet_and_reply
from tests.test_initializer_choice_inputs import (  # noqa: F401
    bookkeeping_plan,
    choices,
    equality_plan,
    evidence_plan,
    live,
    prepared,
    saved_setup,
    source,
    v2,
)
from tests.test_phase_c_construction_baseline import tokenize


def unavailable_information():
    return {
        "decision": "insufficient_information",
        "rule": None,
        "explanation": (
            "A new initialization dependency needs scientific justification."
        ),
    }


@pytest.fixture
def v6_source(tmp_path, bookkeeping_plan):  # noqa: F811
    """Create V6 history from a real matched-initial/forcing witness."""
    parent, task = deepcopy(bookkeeping_plan)
    original = deepcopy(parent["starts"][task["source_task"]])
    cell = parent["cells"][task["benchmark_id"]]
    problem = saved.Problem(cell, original["task"])
    evidence = problem.evidence.model_dump(mode="json")
    second = deepcopy(evidence["trajectories"][0])
    second["trajectory_id"] = "train-1"
    second["targets"]["y"] = [0.0, 2.0, 0.0]
    second["covariates"]["area"] = 2.0
    evidence["trajectories"].append(second)
    evidence["sha256"] = content_hash(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    problem.evidence = training.load_evidence(evidence)
    problem.highlights = training.initial_highlights(problem.evidence)
    draft = api.Draft.model_validate(original["draft"])
    original["assessment"] = problem.assess(draft)
    original["diagnostics"] = problem.diagnose(draft)
    original["previous_conflicts"] = deepcopy(original["diagnostics"])
    contract = handoff_audit.information_contract(
        problem.evidence, "causal_runtime"
    ).model_dump(mode="json")
    contract.update(
        protocol="recorded-t0-initial-information-2", measured_initial_bindings=[]
    )
    original["initial_information_contract"] = contract
    original["predecessor_historical_cost"] = {"earlier_scope": "retained"}
    parent.pop("artifact_sha256", None)
    parent["source_identity"] = "archived-v6-source"
    parent["settings"]["max_output_tokens"] = 8192
    parent["tasks"] = [
        {
            **task,
            "task_id": f"case_{index}",
            "source_task": f"start_{index}",
            "seed": index,
        }
        for index in (1, 2)
    ]
    parent["starts"] = {f"start_{index}": deepcopy(original) for index in (1, 2)}
    root = tmp_path / "v6"
    sealed_write(root / "plan.json", parent)
    parent = sealed_read(root / "plan.json")
    results = {}
    for task in parent["tasks"]:
        seen = []

        def send(url, body, timeout, task=task, seen=seen):
            _, reply = packet_and_reply(body, complete=False)
            seen.append(reply)
            raw = response(reply)
            if task["seed"] == 2 and len(seen) == 2:
                raw["choices"][0]["finish_reason"] = "length"
                raw["choices"][0]["message"]["content"] = '{"partial":'
            return raw

        results[task["task_id"]] = saved.run_episode(
            root,
            parent,
            task,
            base_url="http://offline",
            transport=send,
            token_transport=tokenize,
        )
    assert results["case_1"]["event_count"] == 3
    assert results["case_2"]["status"] == "provider_unavailable"
    return root, parent, results


@pytest.fixture
def completed(tmp_path, v6_source):
    source_root, _, _ = v6_source
    root = tmp_path / "completed"
    plan = inputs.freeze(source_root, root)

    def send(url, body, timeout):
        if body["response_format"]["json_schema"]["name"] == "ChoiceReply":
            return response(unavailable_information())
        _, reply = packet_and_reply(body, complete=False)
        return response(reply)

    summary = diagnostic.run(
        root, plan, base_url="http://offline", transport=send, token_transport=tokenize
    )
    assert summary["status_counts"] == {"assessed": 4}
    return root, plan


def test_readable_freezer_preserves_all_completed_work_and_does_not_repeat_probes(
    tmp_path, completed
):
    source_root, original = completed
    before = {str(p): p.read_bytes() for p in source_root.rglob("*.json")}
    destination = tmp_path / "readable"
    plan = inputs.refine_completed(source_root, destination)
    assert inputs.verify(destination) == plan
    assert inputs.refine_completed(source_root, destination) == plan
    assert plan["cases"] == original["cases"]
    assert plan["prior_diagnostic"]["plan"] == original
    assert plan["prior_diagnostic"]["summary"]["physical_requests"] == 4
    assert plan["budgets"] == inputs.BUDGETS_V2
    assert plan["completion_probe"] is None
    assert all(name is not None for _, name in diagnostic.units(plan))
    assert {str(p): p.read_bytes() for p in source_root.rglob("*.json")} == before
    seen = []

    def send(url, body, timeout):
        seen.append(body)
        assert "PERMITTED INITIAL MEASUREMENTS" in body["messages"][1]["content"]
        assert body["response_format"]["json_schema"]["name"] == "ChoiceReply"
        return response(unavailable_information())

    summary = diagnostic.run(
        destination,
        plan,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
    )
    assert summary["physical_requests"] == 2
    assert summary["maximum_calls"] == 4
    assert summary["maximum_charged_tokens"] == 131072
    assert (
        diagnostic.run(
            destination,
            plan,
            base_url="http://offline",
            transport=lambda *a: pytest.fail("cached"),
            token_transport=tokenize,
        )
        == summary
    )
    assert len(seen) == 2


@pytest.mark.parametrize(
    "tamper",
    [
        "budget",
        "history",
        "source",
        "case",
        "tokenization_count_negative",
        "tokenization_count_bool",
        "tokenization_max_model_len",
        "tokenization_error",
    ],
)
def test_readable_verification_rejects_tampering(tmp_path, completed, tamper):
    source_root, _ = completed
    root = tmp_path / "readable"
    plan = deepcopy(inputs.refine_completed(source_root, root))
    if tamper == "budget":
        plan["budgets"]["maximum_calls"] = 6
    elif tamper == "history":
        plan["prior_diagnostic"]["summary"]["physical_requests"] = 0
    elif tamper == "source":
        plan["source_identity"] = "wrong-source"
    elif tamper == "case":
        next(iter(plan["cases"].values()))["editable_state"] = "other_state"
    else:
        unit, _ = diagnostic.units(plan)[0]
        record = plan["prior_diagnostic"]["units"][unit]["tokenization"][0]
        if tamper == "tokenization_count_negative":
            record["count"] = -1
        elif tamper == "tokenization_count_bool":
            record["count"] = True
        elif tamper == "tokenization_max_model_len":
            record["max_model_len"] = 123
        else:
            record["error"] = "unexpected"
    plan.pop("artifact_sha256")
    (root / "plan.json").unlink()
    sealed_write(root / "plan.json", plan)
    with pytest.raises(ValueError):
        inputs.verify(root)


def test_readable_freezer_rejects_missing_historical_calls(tmp_path, completed):
    source_root, plan = completed
    unit, _ = diagnostic.units(plan)[-1]
    next((source_root / "results" / unit / "calls").glob("*.json")).unlink()
    with pytest.raises(ValueError):
        inputs.refine_completed(source_root, tmp_path / "readable")


def test_preview_matches_first_actual_readable_request(tmp_path, completed):
    source_root, _ = completed
    root = tmp_path / "readable"
    plan = inputs.refine_completed(source_root, root)
    output = tmp_path / "preview"
    diagnostic.preview(plan, output)
    assert not (output / "completion-probe.json").exists()
    expected = json.loads((output / "choice-0.json").read_text())
    unit, name = diagnostic.units(plan)[0]
    seen = []
    diagnostic.run_choice(
        root,
        plan,
        unit,
        name,
        base_url="http://offline",
        transport=lambda url, body, timeout: seen.append(body)
        or response(unavailable_information()),
        token_transport=tokenize,
    )
    assert seen == [expected]
    event = sealed_read(root / "results" / unit / "events/000.json")
    assert event["initializer_before"] == event["initializer_after"]
