"""Opt-in staged interactions export M43 scopes without altering historical paths."""

import copy
import json

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.fitting.prefix_initialization import PrefixFitProblem
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.staged_function_campaign import function_diagnostic
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    ModelingLimits,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    build_contract,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.staged_topology import lower_topology
from tests.test_staged_function_runner import response


def lower(source, brief, context):
    """Update only the toy's authentic compiled topology from its declarations."""
    topology, _ = lower_topology(
        brief,
        tuple(ScientificVariable.model_validate(item) for item in source["inventory"]),
        tuple(EquationDefinition.model_validate(item) for item in source["equations"]),
        context,
    )
    source["topology"] = topology.model_dump(mode="json")
    return topology


def fixture(tmp_path, *, auxiliary=False, dependent=False, invalid_formula=False):
    """Prepare explicit graph metadata over the repository's existing toy topology."""
    task = function_diagnostic(
        "generated_auxiliary" if auxiliary else "driven_memory", ModelingLimits()
    )
    brief = PublicScientificBrief.model_validate(task["brief"])
    context = ValidationContext.model_validate(task["context"])
    source = task["source"]
    if dependent:
        source["inventory"].append(
            {
                "name": "z2",
                "definition": "differential",
                "scientific_role": "Dependent memory coordinate",
            }
        )
        source["equations"][0]["terms"][0]["sources"] = ["x", "z2"]
        source["equations"].append(
            {
                "name": "z2",
                "definition": "differential",
                "terms": [
                    {
                        "sources": ["z", "z2"],
                        "outer_sign": "add",
                        "scientific_role": "Driven relaxation",
                    }
                ],
            }
        )
    lower(source, brief, context)
    declaration = Draft.model_validate(
        {
            "variables": [
                item
                for item in source["inventory"]
                if item["definition"] in {"differential", "algebraic"}
            ],
        }
    )
    contract = build_contract(
        declaration,
        context,
        scientific_context=brief.scientific_context,
        permitted_public_initials=tuple(
            sorted(
                set(context.targets) | context.forcing_channels | {context.time_symbol}
            )
        ),
        observed_initial_channels={"x": "x"},
        trajectory_initialization="observation_prefix",
        provenance="Reviewed public toy identities",
    )
    records = (
        []
        if auxiliary
        else [
            {
                "state": "z",
                "rule": {"mode": "value", "scope": "trajectory"},
                "justification": "Unmeasured independent preparation",
            }
        ]
    )
    if dependent:
        records.append(
            {
                "state": "z2",
                "rule": {
                    "mode": "map",
                    "state_initials": ["z"],
                    "public_initials": ["x"],
                },
                "justification": "Preparation depends on root and initial output",
            }
        )
    handoff = GraphDependencyHandoff.model_validate({"records": records})
    calls = []

    def transport(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(packet)
        if "selected_initializer_dependency" in packet:
            if packet["selected_initializer_dependency"]["rule"]["mode"] == "value":
                return response({"role": "nonnegative_coefficient"})
            return response(
                {
                    "expression": "unavailable"
                    if invalid_formula
                    else "gain*state_1+public_1",
                    "parameters": []
                    if invalid_formula
                    else [{"name": "gain", "role": "coefficient"}],
                }
            )
        if "selected_state" in packet:
            return response({"initial": {"fixed_value": 0}})
        lhs = packet["selected_term"]["lhs"]
        expression = {
            "x": "a-x" if auxiliary else "z2-x" if dependent else "z-x",
            "z": "u-z",
            "a": "u",
            "z2": "z-z2",
        }[lhs]
        return response({"expression": expression, "parameters": []})

    client = StagedTopologyClient(
        settings=StagedModelSettings(attempts_per_step=2),
        base_url="http://unused",
        directory=tmp_path / "calls",
        namespace="staged-graph",
        seed=0,
        transport=transport,
    )
    return brief, context, source, contract, handoff, client, calls


def run(parts, directory, **kwargs):
    brief, context, source, contract, handoff, client, _ = parts
    return run_staged_functions(
        brief,
        context,
        source,
        client,
        directory,
        **(
            {
                "initialization_policy": "initialization_graph",
                "initialization_graph_contract": contract,
                "initialization_graph_handoff": handoff,
            }
            | kwargs
        ),
    )


def test_staged_functions_export_exact_m43_problem_and_dependent_scopes(tmp_path):
    parts = fixture(tmp_path, dependent=True)
    result = run(parts, tmp_path / "result")
    assert result["status"] == "complete", result["error"]
    assert result["protocol"] == "scientific-staged-functions-3"
    assert result["candidate"] is None
    assert result["fitting_backend"] == "prefix-fit-problem-1"
    problem = PrefixFitProblem.model_validate(result["prefix_fit_problem"])
    root = problem.graph.rules["z"].initial
    dependent = problem.graph.rules["z2"].initial
    assert root.scope == "trajectory"
    assert dependent.state_initials == {"state_1": "z"}
    assert dependent.public_initials == {"public_1": "x"}
    assert dependent.expression == "gain*state_1+public_1"
    assert result["initialization"]["audit"]["trajectory_root_parameters"] == {
        "z": "init_z_value"
    }
    assert "init_z2_value" not in result["initialization"]["parameter_guesses"]
    assert "x" not in problem.graph.rules
    assert all(not name.startswith("init_") for name in problem.parameter_guesses)
    assert (
        problem.model_dump(mode="json")
        == result["initialization"]["prefix_fit_problem"]
    )
    assert result["initialization"]["window_policy_unresolved"] is True
    assert result["parameter_fitting_performed"] is False
    assert result["test_data_opened"] is False
    assert len(parts[-1]) == result["physical_requests"] == 5
    assert sum("selected_initializer_dependency" in call for call in parts[-1]) == 2


def test_retained_graph_metadata_opts_in_and_replays_without_extra_calls(tmp_path):
    parts = fixture(tmp_path)
    brief, context, source, contract, handoff, client, calls = parts
    source.update(
        initialization_graph_contract=contract.model_dump(mode="json"),
        initialization_graph_handoff=handoff.model_dump(mode="json"),
    )
    result = run_staged_functions(brief, context, source, client, tmp_path / "result")
    assert result["complete_model"]
    assert result["initialization_policy"] == "initialization_graph"
    replay = fixture(tmp_path)
    replay[2].update(
        initialization_graph_contract=contract.model_dump(mode="json"),
        initialization_graph_handoff=handoff.model_dump(mode="json"),
    )
    assert (
        run_staged_functions(
            replay[0], replay[1], replay[2], replay[5], tmp_path / "result"
        )
        == result
    )
    assert replay[-1] == []
    assert len(calls) == 3


def test_nonstate_compiler_aliases_are_filtered_from_graph_handoff(tmp_path):
    parts = fixture(tmp_path, auxiliary=True)
    result = run(parts, tmp_path / "result")
    assert result["complete_model"], result["error"]
    receipt = result["initialization"]["initialization_dependency_contract"]
    assert receipt["state_aliases"] == {"x": "x"}
    assert result["prefix_fit_problem"]["graph"]["rules"] == {}
    assert len(parts[-1]) == 2
    assert not any("selected_initializer_dependency" in call for call in parts[-1])


def test_legacy_path_keeps_original_candidate_and_initial_schema(tmp_path):
    parts = fixture(tmp_path)
    result = run_staged_functions(*parts[:3], parts[5], tmp_path / "result")
    assert result["complete_model"]
    assert result["protocol"] == "scientific-staged-functions-1"
    assert result["candidate"] is not None
    assert "prefix_fit_problem" not in result
    assert "initialization_policy" not in result
    assert sum("selected_state" in call for call in parts[-1]) == 1


@pytest.mark.parametrize(
    "case", ["explicit_old", "source_old", "old_policy", "missing", "wrong_policy"]
)
def test_graph_and_historical_metadata_are_mutually_exclusive_before_provider(
    tmp_path, case
):
    parts = fixture(tmp_path)
    kwargs = {}
    if case == "explicit_old":
        kwargs["initializer_handoff"] = object()
    elif case == "source_old":
        parts[2]["initial_information_contract"] = {}
    elif case == "old_policy":
        parts[2].update(
            initialization_graph_contract=parts[3].model_dump(mode="json"),
            initialization_graph_handoff=parts[4].model_dump(mode="json"),
        )
        kwargs["initialization_policy"] = "causal_training"
    elif case == "missing":
        kwargs["initialization_graph_handoff"] = None
    else:
        kwargs["initialization_policy"] = "legacy"
    with pytest.raises(ValueError):
        run(parts, tmp_path / "result", **kwargs)
    assert parts[-1] == []


@pytest.mark.parametrize(
    "case", ["declaration", "observed_channel", "cycle", "scope", "lagged", "lowered"]
)
def test_graph_preflight_rejects_invalid_contract_before_interaction_calls(
    tmp_path, case
):
    parts = list(fixture(tmp_path, auxiliary=case == "observed_channel"))
    brief, context, source, contract, handoff = parts[:5]
    if case == "declaration":
        next(item for item in source["inventory"] if item["name"] == "z")[
            "scientific_role"
        ] = "Another coordinate"
        lower(source, brief, context)
    elif case == "observed_channel":
        raw = contract.model_dump(mode="json")
        raw["observed_initial_channels"] = {"x": "a"}
        parts[3] = GraphConstructionContract.model_validate(raw)
    elif case == "cycle":
        raw = handoff.model_dump(mode="json")
        raw["records"][0]["rule"] = {"mode": "map", "state_initials": ["z"]}
        parts[4] = GraphDependencyHandoff.model_validate(raw)
    elif case == "scope":
        raw = contract.model_dump(mode="json")
        raw["trajectory_initialization"] = "forbidden"
        parts[3] = GraphConstructionContract.model_validate(raw)
    elif case == "lagged":
        parts[1] = context.model_copy(update={"lagged_targets": ("x",)})
    else:
        parts[1] = context.model_copy(update={"fitted_initialization": True})
    with pytest.raises(ValueError):
        run(parts, tmp_path / "result")
    assert parts[-1] == []


def test_explicit_graph_arguments_must_match_retained_source_metadata(tmp_path):
    parts = fixture(tmp_path)
    source = parts[2]
    source["initialization_graph_contract"] = parts[3].model_dump(mode="json")
    source["initialization_graph_handoff"] = copy.deepcopy(
        parts[4].model_dump(mode="json")
    )
    source["initialization_graph_handoff"]["records"][0]["justification"] = (
        "Other preparation"
    )
    with pytest.raises(ValueError, match="differ from source"):
        run(parts, tmp_path / "result")
    assert parts[-1] == []


def test_exhausted_formula_repair_never_exports_partial_or_legacy_candidate(tmp_path):
    parts = fixture(tmp_path, dependent=True, invalid_formula=True)
    result = run(parts, tmp_path / "result")
    assert result["status"] == "failed"
    assert not result["complete_model"]
    assert "exhausted" in result["error"]
    assert result["candidate"] is result["prefix_fit_problem"] is None
    assert result["initialization"] is None
