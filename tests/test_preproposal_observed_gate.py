"""Whole-graph proofs on measured differential/algebraic variables and auxiliaries."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.research.preproposal_observed_gate import (
    ObservedGraphGate,
    check_bound_memories,
)
from autoformalism.research.preproposal_topology import make_topology_assessor
from autoformalism.search.topology_compatibility import build_evidence
from tests.test_preproposal_topology import case_and_choice
from tests.test_topology_repair_generator import evidence, row


def init(*public):
    return {"public_initials": list(public), "trajectory_roots": []}


def test_shared_or_distinguishing_latent_initial_with_algebraic_readout():
    rows = [
        row("a", [1, 1, 1], [3, 3, 3], [0, 0, 0], [0, 0, 0]),
        row("b", [2, 2, 2], [5, 5, 5], [0, 0, 0], [0, 0, 0]),
    ]
    graph = {"level": ["store"], "store": ["store"]}
    gate = ObservedGraphGate(evidence(rows))
    states = {"store"}
    checked = gate.assess("level", graph, states, {"store": init()}, 0)
    assert checked["initial_conflicting_pairs"] == 1
    assert checked["conflicting_argument_groups"] == 3
    assert checked["status"] == "contradicted"
    assert (
        gate.assess("level", graph, states, {"store": init("upstream")}, 0)["status"]
        == "screened"
    )
    graph["level"].append("upstream")
    assert (
        gate.assess("level", graph, states, {"store": init()}, 0)["status"]
        == "screened"
    )


def test_readout_compares_later_times_within_one_trajectory_and_measured_states():
    rows = [row("a", [0, 1, 2], [0, 1, 0], [0, 1, 2], [0, 0, 0])]
    gate = ObservedGraphGate(evidence(rows))
    graph = {"level": ["upstream"], "upstream": ["rain"]}
    checked = gate.assess(
        "level", graph, {"upstream"}, {"upstream": init("upstream")}, 0
    )
    assert checked["kind"] == "algebraic" and checked["initial_conflicting_pairs"] == 0
    assert checked["conditioned_measured_arguments"] == ["upstream"]
    assert [s["time"] for s in checked["witness"]] == [0, 2]
    graph["level"].append("t")
    assert (
        gate.assess("level", graph, {"upstream"}, {"upstream": init()}, 0)["status"]
        == "screened"
    )


@pytest.mark.parametrize(
    "held, expected", [(True, "contradicted"), (False, "screened")]
)
def test_latent_uses_complete_forcing_prefix_before_observation(held, expected):
    rows = [
        row("a", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]),
        row("b", [0, 2, 2], [0, 0, 0], [0, 1, 1], [0, 0, 0]),
    ]
    mode = "previous_complete_knots" if held else "linear_complete_knots"
    gate = ObservedGraphGate(evidence(rows, interpolation=mode))
    checked = gate.assess(
        "level",
        {"level": ["store"], "store": ["rain"]},
        {"store"},
        {"store": init()},
        0,
    )
    assert checked["status"] == expected


def test_different_knot_grids_can_prove_equal_latent_prefix():
    rows = [
        row("a", [0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]),
        row("b", [0, 0, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], time=(0, 0.5, 2)),
    ]
    gate = ObservedGraphGate(evidence(rows))
    checked = gate.assess(
        "level", {"level": ["z"], "z": ["rain"]}, {"z"}, {"z": init()}, 0
    )
    assert checked["status"] == "contradicted"
    assert [w["time"] for w in checked["witness"]] == [2, 2]


def test_bounds_and_sensitivity_only_apply_to_selected_output():
    rows = [row("a", [0, 1, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0])]
    p = evidence(rows, absolute_error_bounds={"level": 1.0, "rain": 0.0})
    gate = ObservedGraphGate(p)
    assert (
        gate.assess("level", {"level": ["rain"]}, set(), {}, 0)["status"] == "screened"
    )
    assert (
        gate.assess("level", {"level": ["spill"]}, set(), {}, 0)["status"]
        == "unresolved"
    )
    exact = ObservedGraphGate(evidence(rows))
    assert (
        exact.assess("level", {"level": ["rain"]}, set(), {}, 1.0)["status"]
        == "contradicted"
    )


def test_full_gate_detects_constant_derivative_conflict_and_checks_modeled_auxiliary():
    case, _, frozen, graph = case_and_choice()
    rows = [
        row("a", [0, 1, -1], [0, 0, 0], [1, 1, 1], [0, 0, 0]),
        row("b", [0, 1, -1], [0, 0, 0], [1, 1, 1], [0, 0, 0]),
    ]
    base = evidence(rows)
    ctx = base.context.model_copy(update={"auxiliaries": ("height",)})
    contract = base.contract.model_copy(
        update={
            "absolute_error_bounds": base.contract.absolute_error_bounds
            | {"height": 0.0}
        }
    )
    measured = [
        replace(r, auxiliaries={"height": np.array([1.0, 3.0, 1.0])}) for r in rows
    ]
    packet = build_evidence(
        DatasetSplit(SplitName.TRAIN, tuple(measured), "train"), ctx, contract
    )
    frozen = frozen | {
        "evidence": packet.model_dump(mode="json"),
        "check_all_measured": True,
    }
    inventory = [
        *case.inventory,
        case.inventory[0].model_copy(update={"name": "height", "kind": "algebraic"}),
    ]
    case = case.model_copy(
        update={"inventory": tuple(inventory), "evidence_sha256": packet.sha256}
    )
    graph["sources"] = {
        "level": ["rain"],
        "upstream": ["store", "height"],
        "store": ["store"],
        "height": ["spill"],
    }
    checked = make_topology_assessor(case, frozen)(None, graph)
    assert checked["status"] == "training_conflict"
    checks = checked["training_checks"]
    assert set(checks) == {"level", "upstream", "height"}
    assert checks["level"]["constant_driver_increments"]["status"] == "contradicted"
    assert checks["height"]["status"] == "contradicted"


def test_binding_requires_memory_to_lie_on_the_driver_to_target_path():
    brief = SimpleNamespace(
        requirements=[SimpleNamespace(id="delayed", drivers=["I"], targets=["U"])]
    )
    binding = [{"requirement_id": "delayed", "memory_states": ["M"]}]
    graph = {"H": ["I", "M"], "M": ["H"], "U": ["H"]}
    with pytest.raises(ValueError, match="no qualified path"):
        check_bound_memories(graph, {"M"}, brief, binding)
    graph["U"].append("M")
    check_bound_memories(graph, {"M"}, brief, binding)
