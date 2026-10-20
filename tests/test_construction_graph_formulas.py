"""Graph formulas preserve dependency ownership, M43 scope, and durable budgets."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from autoformalism.fitting.initialization_graph import InitializationGraph
from autoformalism.fitting.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.fitting.simulation import trajectory_initial_state
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
)
from autoformalism.search.construction_graph_formulas import (
    RHS_ONLY_POLICY,
    SYSTEM_PROMPT,
    GraphMapReply,
    GraphRootReply,
    _choice_rule,
    compile_graph_formula_result,
    construct_graph_formulas,
    render_graph_formula_system_prompt,
)
from autoformalism.search.construction_initialization_graph import (
    GraphDependencyHandoff,
    GraphDependencyRecord,
    build_contract,
)
from autoformalism.staged_topology import content_hash
from tests.test_initialization_graph import problem
from tests.test_staged_function_runner import response

MAP = {
    "expression": (
        "gain*state_1 + state_2 + state_3 + public_1 + public_2 + public_3 + public_4"
    ),
    "parameters": [{"name": "gain", "role": "coefficient"}],
}


def setup(tmp_path, replies=None, *, trajectory=True, prefix=False, known=False):
    """Prepare a frozen, dependency-bearing four-state construction fixture."""
    model, row = problem()
    candidate, context = model.validated.candidate, model.validated.context
    contract = build_contract(
        context=context,
        state_names=model.state_names,
        permitted_public_initials=("signal", "aux", "u", "dose", "t"),
        observed_initial_channels={"y": "signal"},
        known_initials=(
            {
                "a": {
                    "mode": "known",
                    "value": 2.0,
                    "justification": "public preparation",
                }
            }
            if known
            else {}
        ),
        trajectory_initialization="observation_prefix" if trajectory else "forbidden",
        prefix_policy=PrefixPolicy(duration=0.5, future_inputs=("u",))
        if prefix
        else None,
        provenance="test public boundary",
    )
    handoff = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": "a",
                    "rule": {"mode": "known"}
                    if known
                    else {"mode": "value", "scope": "shared_training"},
                    "justification": "common preparation",
                },
                {
                    "state": "b",
                    "rule": {
                        "mode": "value",
                        "scope": "trajectory" if trajectory else "shared_training",
                    },
                    "justification": "independent preparation",
                },
                {
                    "state": "m",
                    "rule": {
                        "mode": "map",
                        "state_initials": ["y", "b", "a"],
                        "public_initials": ["u", "aux", "t", "dose"],
                    },
                    "justification": "dependent preparation",
                },
            ]
        }
    )
    defaults = ([{"role": "coefficient"}] if not known else []) + [
        {"role": "nonnegative_coefficient"},
        MAP,
    ]
    requests = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        requests.append(payload)
        actual = defaults if replies is None else replies
        return response(actual[min(len(requests) - 1, len(actual) - 1)])

    client = StagedTopologyClient(
        settings=StagedModelSettings(attempts_per_step=3),
        base_url="http://unused",
        directory=tmp_path / "calls",
        namespace="graph-formulas",
        seed=0,
        transport=transport,
    )
    return candidate, context, contract, handoff, client, requests, row


def construct(fixture, tmp_path, **kwargs):
    """Call the public construction entry point without any observation arrays."""
    candidate, context, contract, handoff, client, _, _ = fixture
    return construct_graph_formulas(
        candidate, context, {}, contract, handoff, client, tmp_path / "result", **kwargs
    )


def test_graph_export_preserves_scopes_dependencies_and_exact_cached_resume(tmp_path):
    fixture = setup(tmp_path)
    candidate, context, contract, handoff, client, requests, row = fixture
    result = construct(fixture, tmp_path)
    graph = InitializationGraph.model_validate(result["graph"])
    exported = PrefixFitProblem.model_validate(result["prefix_fit_problem"])
    assert exported.graph == graph
    assert set(exported.parameter_guesses) == {"rate"}
    assert graph.rules["a"].initial.scope == "shared_training"
    assert graph.rules["b"].initial.scope == "trajectory"
    assert graph.rules["m"].initial.state_initials == {
        "state_1": "a",
        "state_2": "b",
        "state_3": "y",
    }
    assert graph.rules["m"].initial.public_initials == {
        "public_1": "aux",
        "public_2": "dose",
        "public_3": "t",
        "public_4": "u",
    }
    assert result["audit"]["trajectory_root_parameters"] == {"b": "init_b_value"}
    assert result["audit"]["legacy_shared_compatible"] is False
    assert result["fitting_handoff"] == "m43_prefix_initialization_only"
    assert result["fitting_policy_required"] is True
    assert result["window_policy_unresolved"] is True
    assert result["prefix_policy"] is None
    assert result["parameter_fitting_performed"] is False
    assert (
        result["candidate"]["state_equations"]
        == candidate.model_dump(mode="json")["state_equations"]
    )
    assert (
        result["candidate"]["observation_mappings"]
        == candidate.model_dump(mode="json")["observation_mappings"]
    )
    assert set(result["parameter_guesses"]) == {
        "rate",
        "init_a_value",
        "init_b_value",
        "init_m_gain",
    }
    assert "init_m_value" not in result["parameter_guesses"]
    assert requests[2]["dependency_aliases"]["state_initials"]["state_3"] == "y"
    assert "d(m)/dt = -rate*m" in requests[2]["equations"]
    assert "time" not in requests[2]
    compiled = compile_graph_formula_result(
        candidate, context, result, contract=contract, handoff=handoff
    )
    parameters = result["parameter_guesses"]
    initial = trajectory_initial_state(compiled, row, {}, parameters=parameters)
    np.testing.assert_allclose(initial, [1, 1.5, 24.5, 4])
    changed = replace(
        row,
        trajectory_id="unseen",
        targets={"signal": np.array([4.0, -1e9, 1e9])},
        auxiliaries={"aux": np.array([5.0, 1e9, 1e9])},
    )
    np.testing.assert_array_equal(
        initial, trajectory_initial_state(compiled, changed, {}, parameters=parameters)
    )
    assert len(client.records) == len(requests) == 3
    resumed = setup(tmp_path)
    assert construct(resumed, tmp_path) == result
    assert resumed[5] == []
    assert len(resumed[4].records) == 3
    assert sum(record["budget_charge"] for record in resumed[4].records) == 300


@pytest.mark.parametrize("trajectory,prefix", [(False, False), (True, True)])
def test_shared_graph_or_declared_window_needs_no_missing_policy(
    tmp_path, trajectory, prefix
):
    result = construct(setup(tmp_path, trajectory=trajectory, prefix=prefix), tmp_path)
    assert result["fitting_policy_required"] is False
    assert result["audit"]["legacy_shared_compatible"] is (not trajectory)


def test_runtime_known_value_and_observed_identity_never_call_provider(tmp_path):
    fixture = setup(tmp_path, known=True)
    result = construct(fixture, tmp_path)
    assert [request["selected_state"] for request in fixture[5]] == ["b", "m"]
    assert result["graph"]["rules"]["a"]["initial"] == {
        "mode": "known",
        "value": 2.0,
        "justification": "public preparation",
    }
    assert "y" not in result["graph"]["rules"]
    assert result["audit"]["observed_initial_channels"] == {"y": "signal"}
    assert "init_a_value" not in result["parameter_guesses"]


@pytest.mark.parametrize(
    "bad",
    [
        {"role": "coefficient", "guess": 5},
        {"role": "coefficient", "scope": "shared_training"},
        {"role": "coefficient", "mode": "known", "value": 0},
        {"role": "rate"},
    ],
)
def test_roots_accept_only_qualitative_role_without_scope_or_numbers(tmp_path, bad):
    fixture = setup(
        tmp_path,
        [bad, {"role": "coefficient"}, {"role": "nonnegative_coefficient"}, MAP],
    )
    result = construct(fixture, tmp_path)
    assert [event["accepted"] for event in result["attempts"]] == [
        False,
        True,
        True,
        True,
    ]
    assert fixture[5][1]["diagnostics"][0]["error"]
    assert result["graph"]["rules"]["b"]["initial"]["scope"] == "trajectory"
    assert set(GraphRootReply.model_json_schema()["properties"]) == {"role"}
    assert set(GraphMapReply.model_json_schema()["properties"]) == {
        "expression",
        "parameters",
    }


@pytest.mark.parametrize(
    "expression,parameters",
    [
        ("state_1 + state_2 + state_3 + public_1 + public_2 + public_3", []),
        (MAP["expression"] + " + rate", MAP["parameters"]),
        (MAP["expression"] + " + a", MAP["parameters"]),
        (MAP["expression"] + " + aux", MAP["parameters"]),
        (
            MAP["expression"],
            MAP["parameters"] + [{"name": "unused", "role": "coefficient"}],
        ),
        (
            MAP["expression"],
            MAP["parameters"] + [{"name": "state_1", "role": "coefficient"}],
        ),
        (MAP["expression"] + " + public_1[1]", MAP["parameters"]),
        (MAP["expression"] + " + public_1(t)", MAP["parameters"]),
        (MAP["expression"] + " + public_1.real", MAP["parameters"]),
        (MAP["expression"] + " + open('forbidden')", MAP["parameters"]),
    ],
)
def test_maps_require_exact_aliases_and_restricted_scalar_grammar(
    tmp_path, expression, parameters
):
    fixture = setup(
        tmp_path,
        [
            {"role": "coefficient"},
            {"role": "nonnegative_coefficient"},
            {"expression": expression, "parameters": parameters},
            MAP,
        ],
    )
    result = construct(fixture, tmp_path)
    assert [event["accepted"] for event in result["attempts"]] == [
        True,
        True,
        False,
        True,
    ]
    assert fixture[5][3]["diagnostics"][0]["error"]
    assert result["graph"]["rules"]["m"]["initial"]["expression"] == MAP["expression"]


def test_distinct_aliases_preserve_same_named_state_and_public_channel(tmp_path):
    fixture = setup(tmp_path)
    candidate, context, _, _, client, requests, _ = fixture
    contract = build_contract(
        context=context,
        state_names=("aux", "b", "m", "y"),
        permitted_public_initials=("aux", "signal"),
        observed_initial_channels={"y": "signal"},
        provenance="aliased public boundary",
    )
    handoff = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": "aux",
                    "rule": {"mode": "value", "scope": "shared_training"},
                    "justification": "root",
                },
                {
                    "state": "b",
                    "rule": {"mode": "value", "scope": "shared_training"},
                    "justification": "root",
                },
                {
                    "state": "m",
                    "rule": {
                        "mode": "map",
                        "state_initials": ["aux"],
                        "public_initials": ["aux"],
                    },
                    "justification": "distinct concepts",
                },
            ]
        }
    )
    replies = iter(
        [
            {"role": "coefficient"},
            {"role": "coefficient"},
            {"expression": "state_1 + public_1", "parameters": []},
        ]
    )
    client.transport = lambda url, body, timeout: (
        requests.append(json.loads(body["messages"][1]["content"].split("\n", 1)[1]))
        or response(next(replies))
    )
    result = construct_graph_formulas(
        candidate,
        context,
        {},
        contract,
        handoff,
        client,
        tmp_path / "result",
        state_aliases={"aux": "a"},
    )
    initial = result["graph"]["rules"]["m"]["initial"]
    assert initial["state_initials"] == {"state_1": "a"}
    assert initial["public_initials"] == {"public_1": "aux"}
    assert requests[-1]["dependency_aliases"] == {
        "state_initials": {"state_1": "a"},
        "public_initials": {"public_1": "aux"},
    }


@pytest.mark.parametrize(
    "field",
    [
        "graph",
        "parameter_guesses",
        "prefix_fit_problem",
        "candidate",
        "audit",
        "fitting_policy_required",
    ],
)
def test_resealed_artifact_tampering_fails_deterministic_lowering(tmp_path, field):
    fixture = setup(tmp_path)
    result = copy.deepcopy(construct(fixture, tmp_path))
    if field == "graph":
        result[field]["rules"]["b"]["initial"]["scope"] = "shared_training"
    elif field == "parameter_guesses":
        result[field]["init_b_value"] = 100
    elif field == "prefix_fit_problem":
        result[field]["parameter_guesses"]["rate"] = 100
    elif field == "candidate":
        result[field]["state_equations"][0]["rhs"] = "0"
    elif field == "audit":
        result[field]["trajectory_parameter_names"] = []
    else:
        result[field] = False
    result.pop("result_sha256")
    result["result_sha256"] = content_hash(result)
    with pytest.raises(ValueError, match="deterministic lowering"):
        compile_graph_formula_result(fixture[0], fixture[1], result)


def test_exhausted_attempts_and_budget_survive_a_new_client(tmp_path):
    invalid = [{"role": "coefficient", "guess": 50}]
    fixture = setup(tmp_path, invalid)
    with pytest.raises(ValueError, match="exhausted"):
        construct(fixture, tmp_path)
    assert len(fixture[5]) == 3
    resumed = setup(tmp_path, invalid)
    with pytest.raises(ValueError, match="exhausted"):
        construct(resumed, tmp_path)
    assert resumed[5] == []
    assert len(resumed[4].records) == 3


@pytest.mark.parametrize("formula_policy", ["legacy", RHS_ONLY_POLICY])
def test_deferred_request_preserves_accepted_sibling_and_consumed_cost(
    tmp_path, formula_policy
):
    fixture = setup(tmp_path)
    fixture[4].can_start = lambda: len(fixture[5]) < 1
    with pytest.raises(DeferredCall):
        construct(fixture, tmp_path, formula_policy=formula_policy)
    checkpoint = json.loads((tmp_path / "result/state.json").read_text())
    assert checkpoint["choices"] == {"a": {"role": "coefficient"}}
    resumed = setup(tmp_path, [{"role": "nonnegative_coefficient"}, MAP])
    result = construct(resumed, tmp_path, formula_policy=formula_policy)
    assert [item["selected_state"] for item in resumed[5]] == ["b", "m"]
    assert resumed[5][0]["accepted_formulas"] == checkpoint["choices"]
    assert result["choices"]["a"] == checkpoint["choices"]["a"]
    assert len(resumed[4].records) == 3


@pytest.mark.parametrize("formula_policy", ["legacy", RHS_ONLY_POLICY])
def test_interrupted_uncertain_call_never_silently_retries_or_resets_budget(
    tmp_path, formula_policy
):
    fixture = setup(tmp_path)

    def interrupt(url, body, timeout):
        raise KeyboardInterrupt

    fixture[4].transport = interrupt
    with pytest.raises(KeyboardInterrupt):
        construct(fixture, tmp_path, formula_policy=formula_policy)
    for _ in range(2):
        resumed = setup(tmp_path)
        with pytest.raises(ValueError, match="uncertain outcome remains charged"):
            construct(resumed, tmp_path, formula_policy=formula_policy)
        assert resumed[5] == []
        assert len(resumed[4].records) == 1
        assert resumed[4].records[0]["status"] == "uncertain"
        assert resumed[4].records[0]["budget_charge"] > 0
    checkpoint = json.loads((tmp_path / "result/state.json").read_text())
    assert checkpoint["status"] == "uncertain"
    assert len(checkpoint["attempts"]) == 1


@pytest.mark.parametrize(
    "changed", ["namespace", "brief", "handoff", "checkpoint", "runtime"]
)
def test_resume_rejects_changed_namespace_inputs_or_checkpoint(
    tmp_path, changed, monkeypatch
):
    fixture = setup(tmp_path)
    construct(fixture, tmp_path)
    candidate, context, contract, handoff, client, requests, _ = fixture
    brief = {}
    if changed == "namespace":
        client.namespace = "different"
    elif changed == "brief":
        brief = {"scientific_context": "changed preparation"}
    elif changed == "handoff":
        payload = handoff.model_dump(mode="json")
        payload["records"][0]["justification"] = "changed preparation"
        handoff = GraphDependencyHandoff.model_validate(payload)
    elif changed == "runtime":
        monkeypatch.setattr(
            "autoformalism.rebuttal.staged_topology_campaign.runtime_source_hash",
            lambda: "0" * 64,
        )
    else:
        path = tmp_path / "result/state.json"
        payload = json.loads(path.read_text())
        payload["attempts"] = []
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"identity or digest|scientific context"):
        construct_graph_formulas(
            candidate, context, brief, contract, handoff, client, tmp_path / "result"
        )
    assert len(requests) == 3


def test_exact_resume_accepts_changed_local_forwarding_port_without_calls(tmp_path):
    fixture = setup(tmp_path)
    fixture[4].base_url = "http://127.0.0.1:31001"
    result = construct(fixture, tmp_path)
    resumed = setup(tmp_path)
    resumed[4].base_url = "http://127.0.0.1:31002"
    assert construct(resumed, tmp_path) == result
    assert resumed[5] == []
    assert len(resumed[4].records) == 3
    assert {record["endpoint"] for record in resumed[4].records} == {
        "http://127.0.0.1:31001"
    }


def test_missing_observed_runtime_identity_fails_before_any_formula_call(tmp_path):
    fixture = setup(tmp_path)
    candidate, context, contract, handoff, client, requests, _ = fixture
    contract = contract.model_copy(update={"observed_initial_channels": {}})
    payload = handoff.model_dump(mode="json")
    payload["records"].append(
        {
            "state": "y",
            "rule": {"mode": "value", "scope": "shared_training"},
            "justification": "incorrect attempt to free a measured boundary",
        }
    )
    handoff = GraphDependencyHandoff.model_validate(payload)
    with pytest.raises(ValueError, match="observed identities differ"):
        construct_graph_formulas(
            candidate, context, {}, contract, handoff, client, tmp_path / "result"
        )
    assert requests == []


@pytest.mark.parametrize("formula_policy", ["legacy", RHS_ONLY_POLICY])
def test_construction_never_calls_numerical_fitting_or_prefix_inference(
    tmp_path, monkeypatch, formula_policy
):
    from autoformalism.fitting import prefix_initialization

    def forbidden(*args, **kwargs):
        raise AssertionError("construction must not perform numerical inference")

    monkeypatch.setattr(prefix_initialization, "fit_training", forbidden)
    monkeypatch.setattr(prefix_initialization, "infer_prefix", forbidden)
    result = construct(setup(tmp_path), tmp_path, formula_policy=formula_policy)
    assert result["parameter_fitting_performed"] is False


def test_base_runtime_parameter_ranges_determine_guesses(tmp_path):
    fixture = list(setup(tmp_path))
    candidate = fixture[0].model_dump(mode="json")
    candidate["parameters"][0].update(
        bounds={"lower": 5.0, "upper": 10.0},
        initialization_range={"lower": 6.0, "upper": 8.0},
    )
    fixture[0] = type(fixture[0]).model_validate(candidate)
    result = construct(fixture, tmp_path)
    assert result["prefix_fit_problem"]["parameter_guesses"] == {"rate": 7.0}
    assert result["parameter_guesses"]["rate"] == 7.0


def test_legacy_prompt_and_default_replay_remain_exact(tmp_path):
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == (
        "a9e5586213f946af9dffd764374e5b7a0aa25c491e1f246ec2031ad834147f93"
    )
    assert render_graph_formula_system_prompt() == SYSTEM_PROMPT
    assert render_graph_formula_system_prompt("legacy") == SYSTEM_PROMPT
    default = setup(tmp_path / "default")
    explicit = setup(tmp_path / "explicit")
    result = construct(default, tmp_path / "default")
    assert construct(explicit, tmp_path / "explicit", formula_policy="legacy") == result
    assert default[5] == explicit[5]
    assert "formula_policy" not in result
    assert all("formula_policy" not in request for request in default[5])


def test_rhs_policy_example_is_an_exact_valid_json_rhs(tmp_path):
    system = render_graph_formula_system_prompt(RHS_ONLY_POLICY)
    assert "z2_j(0) =" not in system
    example = json.loads(
        next(line for line in system.splitlines() if line.startswith("{"))
    )
    reply = GraphMapReply.model_validate(example)
    assert "=" not in reply.expression
    fixture = setup(tmp_path)
    record = GraphDependencyRecord.model_validate(
        {
            "state": "m",
            "rule": {"mode": "map", "state_initials": ["a"], "public_initials": ["u"]},
            "justification": "illustrative alias binding",
        }
    )
    rule = _choice_rule(record, example, fixture[2], None)
    assert rule["initial"]["expression"] == "a*state_1 + b*public_1"
    assert rule["initial"]["state_initials"] == {"state_1": "a"}
    assert rule["initial"]["public_initials"] == {"public_1": "u"}


def test_rhs_assignment_feedback_repairs_only_formula_and_preserves_graph(tmp_path):
    bad = {**MAP, "expression": "m(0) = " + MAP["expression"]}
    fixture = setup(
        tmp_path,
        [{"role": "coefficient"}, {"role": "nonnegative_coefficient"}, bad, MAP],
    )
    original = [value.model_dump(mode="json") for value in fixture[:4]]
    result = construct(fixture, tmp_path, formula_policy=RHS_ONLY_POLICY)
    first, retry = fixture[5][2:]
    diagnostic = retry["diagnostics"][0]
    feedback = diagnostic["repair_feedback"]
    assert not diagnostic["accepted"]
    assert "SYNTAX_ERROR" in diagnostic["error"]
    assert feedback["code"] == "INITIALIZER_RHS_ONLY"
    assert feedback["rejected_expression"] == bad["expression"]
    assert feedback["selected_state"] == "m"
    assert feedback["required_aliases"] == [
        "state_1",
        "state_2",
        "state_3",
        "public_1",
        "public_2",
        "public_3",
        "public_4",
    ]
    assert "Remove the equation left-hand side and '='" in feedback["instruction"]
    assert "Retain every required alias" in feedback["instruction"]
    for key in (
        "accepted_formulas",
        "initializer_graph",
        "selected_initializer_dependency",
    ):
        assert retry[key] == first[key]
    assert "m" not in retry["accepted_formulas"]
    assert result["choices"]["m"] == MAP
    assert result["formula_policy"] == RHS_ONLY_POLICY
    assert [value.model_dump(mode="json") for value in fixture[:4]] == original
    assert len(fixture[4].records) == 4
    raw = json.loads(
        (tmp_path / "calls" / (diagnostic["request_hash"] + ".json")).read_text()
    )
    assert raw["request"]["body"]["messages"][0]["content"] == (
        render_graph_formula_system_prompt(RHS_ONLY_POLICY)
    )
    assert json.loads(raw["raw_response"]["choices"][0]["message"]["content"]) == bad
    compile_graph_formula_result(fixture[0], fixture[1], result)


def test_rhs_policy_does_not_hide_missing_dependency_aliases(tmp_path):
    bad = {**MAP, "expression": MAP["expression"].replace(" + public_4", "")}
    fixture = setup(
        tmp_path,
        [{"role": "coefficient"}, {"role": "nonnegative_coefficient"}, bad, MAP],
    )
    result = construct(fixture, tmp_path, formula_policy=RHS_ONLY_POLICY)
    diagnostic = fixture[5][3]["diagnostics"][0]
    assert "missing=['public_4']" in diagnostic["error"]
    assert diagnostic["repair_feedback"]["code"] == "INITIALIZER_FORMULA_REJECTED"
    assert "public_4" in diagnostic["repair_feedback"]["required_aliases"]
    assert result["choices"]["m"] == MAP
    assert len(fixture[4].records) == 4


@pytest.mark.parametrize("formula_policy", ["legacy", RHS_ONLY_POLICY])
def test_assignment_exhaustion_preserves_limits_and_does_not_strip(
    tmp_path, formula_policy
):
    bad = {**MAP, "expression": "m(0) = " + MAP["expression"]}
    replies = [{"role": "coefficient"}, {"role": "nonnegative_coefficient"}, bad]
    fixture = setup(tmp_path, replies)
    with pytest.raises(ValueError, match="exhausted for m"):
        construct(fixture, tmp_path, formula_policy=formula_policy)
    path = tmp_path / "result/state.json"
    saved = path.read_bytes()
    checkpoint = json.loads(saved)
    assert len(fixture[5]) == 5
    assert set(checkpoint["choices"]) == {"a", "b"}
    assert len(checkpoint["attempts"]) == 5
    assert not any(event["accepted"] for event in checkpoint["attempts"][2:])
    if formula_policy == RHS_ONLY_POLICY:
        assert checkpoint["status"] == "failed"
        assert checkpoint["failed_state"] == "m"
        assert checkpoint["error"] == "bounded graph formula repair exhausted for m"
        assert checkpoint["formula_policy"] == RHS_ONLY_POLICY
    else:
        assert checkpoint["status"] == "running"
        assert "formula_policy" not in checkpoint
        assert all("repair_feedback" not in event for event in checkpoint["attempts"])
    resumed = setup(tmp_path, replies)
    with pytest.raises(ValueError, match="exhausted for m"):
        construct(resumed, tmp_path, formula_policy=formula_policy)
    assert resumed[5] == []
    assert len(resumed[4].records) == 5
    assert sum(record["budget_charge"] for record in resumed[4].records) == 500
    assert path.read_bytes() == saved


def test_rhs_complete_resume_is_exact_and_policy_change_cannot_reuse_checkpoint(
    tmp_path,
):
    fixture = setup(tmp_path)
    result = construct(fixture, tmp_path, formula_policy=RHS_ONLY_POLICY)
    saved = (tmp_path / "result/state.json").read_bytes()
    resumed = setup(tmp_path)
    assert construct(resumed, tmp_path, formula_policy=RHS_ONLY_POLICY) == result
    assert resumed[5] == []
    assert len(resumed[4].records) == 3
    assert (tmp_path / "result/state.json").read_bytes() == saved
    with pytest.raises(ValueError, match="identity or digest"):
        construct(resumed, tmp_path)
    assert resumed[5] == []


def test_unknown_formula_policy_is_rejected_before_calls_or_checkpoint(tmp_path):
    fixture = setup(tmp_path)
    with pytest.raises(ValueError, match="unknown initializer formula policy"):
        construct(fixture, tmp_path, formula_policy="unknown")
    assert fixture[5] == []
    assert not (tmp_path / "result/state.json").exists()
