"""Topology-owned initializer dependencies constrain later executable formulas."""

import copy
import json

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.staged_function_campaign import function_diagnostic
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    ModelingLimits,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.causal_initialization import (
    compile_initialization_result,
    construct_initializers,
)
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.topology_initialization import (
    InitialInformationContract,
    InitializerDependencyHandoff,
)
from autoformalism.staged_topology import lower_topology
from tests.test_causal_initialization_construction import MAP, setup
from tests.test_staged_function_runner import response


@pytest.fixture(autouse=True)
def stable_runtime_identity(monkeypatch):
    """Concurrent source edits must not alter a synthetic resume test mid-call."""
    monkeypatch.setattr(
        "autoformalism.rebuttal.staged_topology_campaign.runtime_source_hash",
        lambda: "synthetic-runtime",
    )


def contract(context):
    return InitialInformationContract(
        provenance="explicit synthetic runtime contract",
        scope="recorded_t0",
        shared_deterministic=True,
        complete=True,
        permitted_symbols=tuple(
            sorted(
                set(context.targets)
                | set(context.forcing_channels)
                | {context.time_symbol}
            )
        ),
    )


def handoff(state="m", mode="common_map", arguments=("v01",)):
    return InitializerDependencyHandoff.model_validate(
        {"records": [{"state": state, "rule": {"mode": mode, "arguments": arguments}}]}
    )


def build(tmp_path, candidate, context, client, choices, runtime=None):
    return construct_initializers(
        candidate,
        context,
        {},
        client,
        tmp_path / "result",
        initial_information_contract=runtime or contract(context),
        initializer_handoff=choices,
    )


def test_declared_map_is_compiled_and_exactly_resumed(tmp_path):
    candidate, context, client, requests, _ = setup(tmp_path)
    choices = handoff()
    result = build(tmp_path, candidate, context, client, choices)
    assert result["attempts"][0]["accepted"]
    assert requests[0]["allowed_initial_symbols"] == ["v01"]
    assert requests[0]["selected_initializer_dependency"] == {
        "mode": "common_map",
        "arguments": ["v01"],
    }
    assert result["initialization_dependency_contract"][
        "initializer_handoff"
    ] == choices.model_dump(mode="json")
    assert result["audit"]["observed_initial_channels"] == {"y": "v01"}
    assert build(tmp_path, candidate, context, client, choices) == result
    assert len(requests) == 1
    assert compile_initialization_result(candidate, context, result)


@pytest.mark.parametrize(
    "reply,code",
    [
        ({"initial": {"mode": "shared_value"}}, "INITIALIZER_DEPENDENCY_MODE"),
        (
            {"initial": {"mode": "causal_map", "expression": "u01"}},
            "INITIALIZER_DEPENDENCY_SYMBOLS",
        ),
        (
            {"initial": {"mode": "causal_map", "expression": "v01+u01"}},
            "INITIALIZER_DEPENDENCY_SYMBOLS",
        ),
        (
            {"initial": {"mode": "causal_map", "expression": "1"}},
            "INITIALIZER_DEPENDENCY_SYMBOLS",
        ),
    ],
)
def test_mode_and_dependency_edits_receive_specific_repair_feedback(
    tmp_path, reply, code
):
    candidate, context, client, requests, _ = setup(tmp_path, [reply, MAP])
    choices = handoff()
    original = choices.model_dump(mode="json")
    result = build(tmp_path, candidate, context, client, choices)
    assert [row["accepted"] for row in result["attempts"]] == [False, True]
    feedback = requests[1]["diagnostics"][0]["error"]
    assert code in feedback
    assert "explicitly upstream" in feedback
    assert choices.model_dump(mode="json") == original


def test_shared_parameter_cannot_be_silently_replaced_by_a_map(tmp_path):
    reply = {"initial": {"mode": "shared_value", "role": "nonnegative_coefficient"}}
    candidate, context, client, requests, _ = setup(tmp_path, [MAP, reply])
    result = build(
        tmp_path,
        candidate,
        context,
        client,
        handoff(mode="shared_parameter", arguments=()),
    )
    assert [row["accepted"] for row in result["attempts"]] == [False, True]
    assert requests[0]["allowed_initial_symbols"] == []
    assert result["plan"]["rules"]["m"]["initial"]["mode"] == "value"


@pytest.mark.parametrize(
    "kind", ["missing", "unknown", "stale", "algebraic", "observed", "hidden"]
)
def test_invalid_handoff_is_rejected_before_calls(tmp_path, kind):
    candidate, context, client, requests, _ = setup(tmp_path)
    choices = handoff()
    if kind == "missing":
        choices = InitializerDependencyHandoff()
    elif kind == "unknown":
        choices = handoff(mode="unknown", arguments=())
    elif kind in {"stale", "algebraic"}:
        choices = handoff(state="removed")
        if kind == "algebraic":
            raw = candidate.model_dump(mode="json")
            raw["processes"].append({"name": "removed", "expression": "u01"})
            candidate = type(candidate).model_validate(raw)
    elif kind == "observed":
        raw = choices.model_dump(mode="json")
        raw["records"].append({"state": "y", "rule": {"mode": "shared_parameter"}})
        choices = InitializerDependencyHandoff.model_validate(raw)
    else:
        choices = handoff(arguments=("future",))
    with pytest.raises(ValueError, match="INITIALIZER_DEPENDENCY"):
        build(tmp_path, candidate, context, client, choices)
    assert requests == []


def test_unknown_runtime_contract_cannot_authorize_formulas(tmp_path):
    candidate, context, client, requests, _ = setup(tmp_path)
    runtime = contract(context).model_copy(update={"scope": "unknown"})
    with pytest.raises(ValueError, match="INITIALIZER_DEPENDENCY_CONTRACT"):
        build(tmp_path, candidate, context, client, handoff(), runtime)
    assert requests == []


@pytest.mark.parametrize("change", ["contract", "handoff"])
def test_changed_handoff_or_runtime_cannot_resume(tmp_path, change):
    candidate, context, client, requests, _ = setup(tmp_path)
    runtime, choices = contract(context), handoff()
    build(tmp_path, candidate, context, client, choices, runtime)
    if change == "contract":
        runtime = runtime.model_copy(update={"provenance": "different preparation"})
    else:
        choices = handoff(arguments=("u01",))
    with pytest.raises(ValueError, match="checkpoint identity"):
        build(tmp_path, candidate, context, client, choices, runtime)
    assert len(requests) == 1


def test_artifact_recompilation_rechecks_dependencies(tmp_path):
    candidate, context, client, _, _ = setup(tmp_path)
    result = build(tmp_path, candidate, context, client, handoff())
    changed = copy.deepcopy(result)
    changed["plan"]["rules"]["m"]["initial"]["expression"] = "a+b*u01"
    with pytest.raises(ValueError, match="INITIALIZER_DEPENDENCY_SYMBOLS"):
        compile_initialization_result(candidate, context, changed)


def test_same_formula_under_changed_runtime_has_a_distinct_cached_request(tmp_path):
    candidate, context, client, requests, _ = setup(tmp_path)
    runtime = contract(context)
    first = build(tmp_path / "first", candidate, context, client, handoff(), runtime)
    runtime = runtime.model_copy(update={"provenance": "revised explicit contract"})
    second = build(tmp_path / "second", candidate, context, client, handoff(), runtime)
    assert len(requests) == 2
    assert first["attempts"][0]["request_hash"] != second["attempts"][0]["request_hash"]
    assert first["candidate"] == second["candidate"]


def runner_setup(tmp_path, *, alias=False):
    task = function_diagnostic("driven_memory", ModelingLimits())
    brief = PublicScientificBrief.model_validate(task["brief"])
    context = ValidationContext.model_validate(task["context"])
    source = task["source"]
    if alias:
        context = context.model_copy(update={"auxiliaries": ("z",)})
        topology, _ = lower_topology(
            brief,
            tuple(
                ScientificVariable.model_validate(row) for row in source["inventory"]
            ),
            tuple(
                EquationDefinition.model_validate(row) for row in source["equations"]
            ),
            context,
        )
        source["topology"] = topology.model_dump(mode="json")
    dependency = "z" if alias else "x"
    choices = handoff("z", arguments=(dependency,))
    runtime = contract(context)
    source["initializer_handoff"] = choices.model_dump(mode="json")
    source["initial_information_contract"] = runtime.model_dump(mode="json")
    requests = []

    def transport(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        requests.append(packet)
        if "selected_state" in packet:
            return response(
                {"initial": {"mode": "causal_map", "expression": dependency}}
            )
        expression = "z-x" if packet["selected_term"]["lhs"] == "x" else "u-z"
        return response({"expression": expression, "parameters": []})

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        base_url="http://unused",
        directory=tmp_path / "calls",
        namespace="handoff-functions",
        seed=0,
        transport=transport,
    )
    return (brief, context, source, client, tmp_path), requests


@pytest.mark.parametrize("alias", [False, True])
def test_runner_consumes_source_handoff_and_preserves_public_alias_arguments(
    tmp_path, alias
):
    args, requests = runner_setup(tmp_path, alias=alias)
    result = run_staged_functions(*args)
    assert result["complete_model"], result["error"]
    assert result["initialization_policy"] == "causal_training"
    assert len(requests) == 3
    assert all("initialization_dependency_contract" in packet for packet in requests)
    selected = requests[-1]
    assert selected["allowed_initial_symbols"] == (["z"] if alias else ["x"])
    state = selected["selected_state"]["name"]
    assert state.startswith("af_internal_aux_") if alias else state == "z"
    assert result["initialization"]["plan"]["rules"][state]["initial"][
        "expression"
    ] == ("z" if alias else "x")
    assert run_staged_functions(*args) == result
    assert len(requests) == 3
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert (
        progress["initialization_dependency_contract"]
        == result["initialization_dependency_contract"]
    )


def test_runner_rejects_explicit_source_conflict_without_calls(tmp_path):
    args, requests = runner_setup(tmp_path)
    with pytest.raises(ValueError, match="differ from source handoff"):
        run_staged_functions(*args, initializer_handoff=handoff("z", arguments=("u",)))
    assert requests == []


def test_runner_cannot_silently_ignore_explicit_handoff_in_legacy_mode(tmp_path):
    args, requests = runner_setup(tmp_path)
    runtime = InitialInformationContract.model_validate(
        args[2].pop("initial_information_contract")
    )
    choices = InitializerDependencyHandoff.model_validate(
        args[2].pop("initializer_handoff")
    )
    with pytest.raises(ValueError, match="requires causal_training"):
        run_staged_functions(
            *args, initial_information_contract=runtime, initializer_handoff=choices
        )
    assert requests == []
