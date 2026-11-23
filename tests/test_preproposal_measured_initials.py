"""Measured physical coordinates and forecast covariates have separate permissions."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from autoformalism.fitting.initialization_graph import (
    InitializationGraph,
    apply_initialization_graph,
)
from autoformalism.fitting.simulation import trajectory_initial_state
from autoformalism.research import preproposal_cases as cases
from autoformalism.research import preproposal_complete_graphs as graphs
from autoformalism.research import preproposal_topology as topology
from autoformalism.search.construction_graph_formulas import (
    runtime_observed_initial_rules,
)
from autoformalism.search.construction_initialization_graph import build_contract
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.staged_topology import content_hash
from scripts.preproposal_graph_exports import _canonical
from scripts.preproposal_graph_portfolio import with_reviewed_initial_bindings
from tests.test_initialization_graph import graph_payload, problem
from tests.test_preproposal_topology import case_and_choice


def measured_case():
    """The declared store's initial physical value is measured once per event."""
    case, _, frozen, graph = case_and_choice()
    evidence = frozen["evidence"]
    evidence["context"]["fixed_covariates"] = ["survey", "area"]
    evidence["contract"]["absolute_error_bounds"].update(survey=0, area=0)
    for trajectory in evidence["trajectories"]:
        trajectory["covariates"].update(
            survey=trajectory["targets"]["upstream"][0], area=10
        )
    verified = TrainingCompatibilityEvidence.model_validate(evidence)
    evidence["sha256"] = content_hash(
        verified.model_dump(mode="json", exclude={"sha256"})
    )
    case = case.model_copy(update={"evidence_sha256": evidence["sha256"]})
    frozen.update(
        allow_topology_latents=True,
        check_all_measured=True,
        inventory=[v.model_dump() for v in case.inventory],
        required_histories=[],
        measured_initial_bindings={"store": "survey"},
    )
    frozen["initials"].append("survey")
    frozen["public_brief"]["public_variables"].extend(
        {"name": n, "data_role": "covariate"} for n in ("survey", "area")
    )
    graph["sources"]["level"].append("area")
    graph["latent_initializers"] = []
    return case, frozen, graph


def test_initial_binding_and_ongoing_geometry_in_requests_check_and_export():
    case, frozen, graph = measured_case()
    system, user = topology.topology_messages(case, None, frozen)
    packet = json.loads(user)
    assert packet["observed_initials"]["store"] == "survey"
    assert "store" not in packet["initializer_owners"]
    assert "area" in packet["available_rhs_names"]
    assert "area" not in packet["permitted_baseline_readings"]
    assert "exact initial value" in system
    checked = topology.assess_topology(case, None, graph, frozen)
    assert checked["status"] == "topology_screen_passed"
    canonical = _canonical(
        case,
        frozen,
        topology.reply_schema(case, frozen),
        graph,
        checked["structure_sha256"],
    )
    assert canonical["initializers"]["store"] == {
        "mode": "observed_initial",
        "channel": "survey",
    }
    # An older dependency-only map can implement this identity, even with
    # an additional permissible measurement. It must still export the identity.
    legacy = deepcopy(graph)
    legacy["latent_initializers"] = [
        {
            "state": "store",
            "rule": {"mode": "map", "public_initials": ["survey", "upstream"]},
            "justification": "Two permitted readings.",
        }
    ]
    again = topology.assess_topology(case, None, legacy, frozen)
    assert again["structure_sha256"] == checked["structure_sha256"]
    assert (
        _canonical(
            case,
            frozen,
            topology.reply_schema(case, frozen),
            legacy,
            again["structure_sha256"],
        )
        == canonical
    )
    # A different latent coordinate keeps its free initialization; names alone
    # cannot impose the measurement or create its structural identity.
    unbound = {**frozen, "measured_initial_bindings": {}}
    old = topology.assess_topology(case, None, legacy, unbound)
    assert old["structure_sha256"] != again["structure_sha256"]
    pool = graphs.generate_pool(case, frozen, max_candidates=2)
    assert len(pool["candidates"]) == 2
    assert all(not r["graph"]["latent_initializers"] for r in pool["candidates"])
    assert json.loads(graphs.graph_selection_messages(case, pool, 1)[1])[
        "measured_initial_bindings"
    ] == {"store": "survey"}


@pytest.mark.parametrize(
    "rule",
    [
        {"mode": "value", "scope": "shared_training"},
        {"mode": "map", "public_initials": ["upstream"]},
        {"mode": "map", "public_initials": ["survey", "unavailable"]},
    ],
)
def test_rejected_conflicting_measured_state_rule(rule):
    case, frozen, graph = measured_case()
    graph["latent_initializers"] = [
        {"state": "store", "rule": rule, "justification": "Test."}
    ]
    result = topology.assess_topology(case, None, graph, frozen)
    assert result["status"] == "invalid_topology"
    assert "store(0) = survey" in result["reason"]


def test_basin_binding_requires_the_reviewed_physical_state_declaration():
    _, _, _, receipt, evidence = cases.STAGED[cases.BASIN_CASE]
    frozen = {
        "case_id": cases.BASIN_CASE,
        "provenance": {"receipt_sha256": receipt},
        "evidence": {"sha256": evidence},
        "inventory_roles": {"h_up": "state of upstream basin depth"},
        "inventory": [{"name": "h_up", "kind": "differential"}],
    }
    assert with_reviewed_initial_bindings(frozen)["measured_initial_bindings"] == {
        "h_up": "initial_up"
    }
    other = {**frozen, "provenance": {"receipt_sha256": "other-declaration"}}
    assert with_reviewed_initial_bindings(other) is other
    with pytest.raises(ValueError, match="upstream depth declaration"):
        with_reviewed_initial_bindings(
            {**frozen, "inventory_roles": {"h_up": "volume"}}
        )


def test_fixed_initial_identity_lowers_to_parameter_free_map_per_trajectory():
    model, row = problem()
    contract = build_contract(
        context=model.validated.context,
        state_names=model.state_names,
        permitted_public_initials=["signal", "dose"],
        observed_initial_channels={"y": "signal", "a": "dose"},
        provenance="Reviewed state a is the quantity measured by dose at t0.",
    )
    fixed, provenance = runtime_observed_initial_rules(model, contract, None)
    assert not provenance[0]["independent_parameter"]
    graph = graph_payload(trajectory=False)
    graph["rules"].update(fixed)
    compiled, guesses, _ = apply_initialization_graph(
        model, InitializationGraph.model_validate(graph)
    )
    assert "init_a_value" not in guesses
    for dose in (2.0, 9.0):
        state = trajectory_initial_state(
            compiled,
            replace(row, fixed_covariates={"dose": dose}),
            {},
            parameters={"rate": 0.3, **guesses},
        )
        assert state[compiled.state_names.index("a")] == dose
