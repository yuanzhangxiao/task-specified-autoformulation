"""Exact candidate-specific witnesses respect complete forcing and root scope."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.research.construction_revision_consistency import (
    TARGET_COMPARISON_POLICY,
    assess,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.topology_compatibility import EvidenceContract, build_evidence
from autoformalism.staged_topology import content_hash


def problem(*, rhs="-a*x+u", root="trajectory", process=False, transitive=False):
    """Observed x and a scoped hidden z, optionally behind an algebraic law."""
    states = ["x", "z", *(["w"] if transitive else [])]
    rule = (
        {
            "mode": "map",
            "expression": "b*reading",
            "parameters": [{"name": "b"}],
            "public_initials": {"reading": "v"},
        }
        if root == "map"
        else {"mode": "known", "value": 2, "justification": "known preparation"}
        if root == "known"
        else {"mode": "value", "guess": 2, "scope": root}
    )
    rules = {"z": {"initial": rule}}
    if transitive:
        rules["w"] = {
            "initial": {
                "mode": "map",
                "expression": "b*ancestor",
                "parameters": [{"name": "b"}],
                "state_initials": {"ancestor": "z"},
            }
        }
    return PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "fixture",
                "parent_candidate_id": None,
                "states": [{"name": s, "kind": "latent"} for s in states],
                "state_equations": [{"state": "x", "rhs": rhs}]
                + [{"state": s, "rhs": "-" + s} for s in states[1:]],
                "processes": [{"name": "q", "expression": "a*z+v"}] if process else [],
                "observation_mappings": [{"channel": "y", "expression": "x"}],
                "parameters": [{"name": "a", "scope": "global", "role": "rate"}],
                "initial_conditions": [
                    {"state": s, "scope": "global", "fixed_value": 0} for s in states
                ],
            },
            "context": {
                "targets": ["y"],
                "external_inputs": ["u"],
                "auxiliaries": ["v"],
            },
            "graph": {"rules": rules},
            "parameter_guesses": {"a": 1},
        }
    )


def row(name, values=(1, 3, 4), *, time=(0, 1, 2), u=None, v=None, targets=None):
    """Only synthetic public training values, with complete forcing knots."""
    return Trajectory(
        trajectory_id=name,
        time=np.asarray(time, float),
        targets={
            k: np.asarray(x, float) for k, x in (targets or {"y": values}).items()
        },
        external_inputs={"u": np.asarray(u or [0] * len(time), float)},
        auxiliaries={"v": np.asarray(v or [2] * len(time), float)},
        fixed_covariates={},
        derivatives={},
    )


def evidence(source, *rows, **metadata):
    """Seal explicit assumptions; historical initial_rules are intentionally wrong."""
    contract = EvidenceContract.model_validate(
        {
            "provenance": "synthetic public task/runtime fixture",
            "dependencies_exhaustive": True,
            "shared_parameters": True,
            "open_loop": True,
            "reset_impulse_free": True,
            "uniqueness_assumed": True,
            "interpolation": {
                "u": "linear_complete_knots",
                "v": "linear_complete_knots",
            },
            "absolute_error_bounds": dict.fromkeys(
                (*source.context.targets, "u", "v"), 0
            ),
            "initial_rules": {"z": {"mode": "shared_parameter"}},
            **metadata,
        }
    )
    return build_evidence(
        DatasetSplit(SplitName.TRAIN, rows, "synthetic"), source.context, contract
    ).model_dump(mode="json")


def closed(report):
    return next(
        c for c in report["checks"] if c["property"] == "closed_subsystem_equality"
    )


def test_scalar_conflict_ignores_unrelated_trajectory_root_and_is_deterministic():
    source = problem()
    data = evidence(source, row("a", (1, 1, 1)), row("b", (1, 3, 4)))
    original = source.model_dump(mode="json"), deepcopy(data)
    report = assess(source, data)
    assert report == assess(source, data)
    assert report["status"] == "conflict"
    check = closed(report)
    assert check["initialized_states"] == ["x"]
    assert check["trajectory_roots"] == []
    witness = check["witnesses"][0]
    assert witness["times"] == [2, 2]
    assert witness["observations"] == [1, 4]
    assert witness["pair_squared_error_lower_bound"] == 4.5
    assert witness["maximum_absolute_error_lower_bound"] == 1.5
    assert (source.model_dump(mode="json"), data) == original
    assert not report["numerical_rollout_performed"]
    assert not report["parameter_fitting_performed"]


@pytest.mark.parametrize("scale,difference", [(0, 5e-10), (1, 1e-8), (1e5, 1e-3)])
def test_small_target_discrepancies_defer_with_separate_numerical_policy(
    scale, difference
):
    source = problem()
    report = assess(
        source,
        evidence(
            source,
            row("a", (scale, scale, scale)),
            row("b", (scale, scale + difference, scale + difference)),
        ),
    )
    assert not report["conflicts"]
    assert closed(report)["status"] == "deferred"
    witness = closed(report)["numerically_unresolved"][0]
    assert witness["error_intervals_disjoint"]
    assert not witness["exceeds_target_numerical_guard"]
    assert report["target_comparison_policy"] == TARGET_COMPARISON_POLICY
    assert witness["absolute_error_bounds"] == [0, 0]


def test_large_conflict_survives_small_numerically_unresolved_pairs():
    source = problem()
    report = assess(
        source,
        evidence(
            source,
            row("a", (1, 1, 1)),
            row("b", (1, 1 + 1e-8, 1 + 1e-8)),
            row("c", (1, 3, 4)),
        ),
    )
    assert report["conflicts"]
    assert closed(report)["numerically_unresolved"]
    assert closed(report)["witnesses"][0]["exceeds_target_numerical_guard"]


def test_extreme_finite_observations_keep_witness_without_overflowing_error_floor():
    source = problem()
    report = assess(
        source,
        evidence(
            source,
            row("a", (1, -1e308, -1e308)),
            row("b", (1, 1e308, 1e308)),
        ),
    )
    witness = closed(report)["witnesses"][0]
    assert witness["pair_squared_error_lower_bound"] is None
    assert witness["maximum_absolute_error_lower_bound"] == 1e308


@pytest.mark.parametrize(
    "rhs,process,transitive",
    [
        ("-a*x+z", False, False),
        ("-a*x+q", True, False),
        ("-a*x+w", False, True),
    ],
)
def test_coupled_trajectory_root_defers_even_when_legacy_contract_says_shared(
    rhs,
    process,
    transitive,
):
    source = problem(rhs=rhs, process=process, transitive=transitive)
    report = assess(source, evidence(source, row("a", (1, 1, 1)), row("b")))
    assert not report["conflicts"]
    assert closed(report)["trajectory_roots"] == ["z"]
    assert closed(report)["status"] == "deferred"


@pytest.mark.parametrize("root", ["shared_training", "known", "map"])
def test_shared_known_and_matching_map_initials_allow_equality(root):
    source = problem(rhs="-a*x+w", root=root, transitive=True)
    assert assess(source, evidence(source, row("a", (1, 1, 1)), row("b")))["conflicts"]


def test_differing_public_initial_map_arguments_prevent_equality():
    source = problem(rhs="-a*x+z", root="map")
    report = assess(
        source, evidence(source, row("a", (1, 1, 1)), row("b", v=[3, 2, 2]))
    )
    assert not report["conflicts"]
    assert closed(report)["conditioning_distinctions"]["different_initial_information"]


def test_different_observed_initials_are_runtime_owned_not_candidate_fixed_zero():
    source = problem()
    report = assess(source, evidence(source, row("a", (1, 1, 1)), row("b", (2, 3, 4))))
    assert not report["conflicts"]
    assert closed(report)["conditioning_distinctions"]["different_initial_information"]


@pytest.mark.parametrize("change", [1.0, 1e-15])
def test_different_complete_driving_history_even_nearly_equal_prevents_proof(change):
    source = problem()
    report = assess(
        source, evidence(source, row("a", (1, 1, 1)), row("b", u=[0, change, 0]))
    )
    assert not report["conflicts"]


def test_future_forcing_difference_does_not_erase_an_earlier_witness():
    source = problem()
    report = assess(
        source, evidence(source, row("a", (1, 1, 1)), row("b", u=[0, 0, 2]))
    )
    assert closed(report)["witnesses"][0]["times"] == [1, 1]


def test_exact_interpolation_accepts_different_complete_knot_grids():
    source = problem()
    report = assess(
        source,
        evidence(
            source,
            row("a", (1, 1, 1), u=[0, 0.5, 1]),
            row("b", (1, 4), time=(0, 2), u=[0, 1]),
        ),
    )
    assert closed(report)["witnesses"][0]["times"] == [2, 2]


def test_offset_knots_and_absolute_time_origins_do_not_match():
    source = problem(rhs="-a*x+u+t")
    report = assess(
        source, evidence(source, row("a", (1, 1, 1)), row("b", time=(1, 2, 3)))
    )
    assert not report["conflicts"]
    assert closed(report)["conditioning_distinctions"][
        "different_absolute_time_origins"
    ]
    data = evidence(
        source,
        row("a", (1, 1, 1), u=[0, 1, 0]),
        row("b", time=(0, 0.5, 2), u=[0, 1, 0]),
    )
    assert not assess(source, data)["conflicts"]


def test_algebraic_closure_detects_relevant_auxiliary_history():
    source = problem(rhs="-a*x+q", root="shared_training", process=True)
    report = assess(
        source, evidence(source, row("a", (1, 1, 1)), row("b", v=[2, 3, 2]))
    )
    assert closed(report)["forcing_channels"] == ["v"]
    assert not report["conflicts"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"uniqueness_assumed": None},
        {"shared_parameters": False},
        {"interpolation": {"u": "unknown", "v": "linear_complete_knots"}},
        {"absolute_error_bounds": {"y": 0, "u": 0.01, "v": 0}},
        {"absolute_error_bounds": {"u": 0, "v": 0}},
    ],
)
def test_unknown_or_inexact_conditioning_metadata_defers(metadata):
    source = problem()
    report = assess(source, evidence(source, row("a", (1, 1, 1)), row("b"), **metadata))
    assert not report["conflicts"]
    assert report["deferred"]


def test_initial_algebraic_readout_uses_only_initial_not_dynamical_ancestry():
    raw = problem(rhs="-a*x+z").model_dump(mode="json")
    raw["base_candidate"]["observation_mappings"] = [
        {"channel": "seen", "expression": "x"},
        {"channel": "y", "expression": "a*x+v"},
    ]
    raw["context"]["targets"] = ["seen", "y"]
    source = PrefixFitProblem.model_validate(raw)
    data = evidence(
        source,
        row("a", targets={"seen": [1, 1, 1], "y": [3, 3, 3]}),
        row("b", targets={"seen": [1, 1, 1], "y": [7, 7, 7]}),
        absolute_error_bounds={"seen": 0, "y": 0.5, "u": 0, "v": 0},
    )
    report = assess(source, data)
    assert len(report["conflicts"]) == 1
    check = report["conflicts"][0]
    assert check["property"] == "initial_readout_equality"
    assert check["target"] == "y"
    assert check["witnesses"][0]["pair_squared_excess_error_lower_bound"] == 4.5
    assert check["trajectory_roots"] == []


def test_overlapping_error_intervals_are_not_a_conflict():
    raw = problem().model_dump(mode="json")
    raw["base_candidate"]["observation_mappings"].append(
        {"channel": "other", "expression": "v"}
    )
    raw["context"]["targets"].append("other")
    source = PrefixFitProblem.model_validate(raw)
    data = evidence(
        source,
        row("a", targets={"y": [1, 1, 1], "other": [2, 2, 2]}),
        row("b", targets={"y": [1, 1, 1], "other": [3, 3, 3]}),
        absolute_error_bounds={"y": 0, "other": 1, "u": 0, "v": 0},
    )
    assert not assess(source, data)["conflicts"]


def test_initial_readout_time_argument_prevents_offset_equality():
    raw = problem().model_dump(mode="json")
    raw["base_candidate"]["observation_mappings"].append(
        {"channel": "other", "expression": "v+t"}
    )
    raw["context"]["targets"].append("other")
    source = PrefixFitProblem.model_validate(raw)
    data = evidence(
        source,
        row("a", targets={"y": [1, 1, 1], "other": [2, 2, 2]}),
        row("b", time=(1, 2, 3), targets={"y": [1, 1, 1], "other": [3, 3, 3]}),
    )
    assert not assess(source, data)["conflicts"]


def test_malformed_unsealed_or_nontraining_evidence_is_rejected():
    source = problem()
    original = evidence(source, row("a"), row("b"))
    for field, value in [("sha256", "0" * 64), ("split", "test")]:
        data = deepcopy(original)
        data[field] = value
        with pytest.raises(ValueError):
            assess(source, data)
    data = deepcopy(original)
    data["contract"]["made_up_tolerance"] = 1
    with pytest.raises(ValueError):
        assess(source, data)
    data = deepcopy(original)
    data["context"]["time_symbol"] = "time"
    data["sha256"] = content_hash({k: v for k, v in data.items() if k != "sha256"})
    with pytest.raises(ValueError, match="context differs"):
        assess(source, data)


def test_single_flat_sampled_trajectory_does_not_claim_equilibrium():
    source = problem(rhs="a+u")
    assert not assess(source, evidence(source, row("a", (1, 1, 1))))["conflicts"]
