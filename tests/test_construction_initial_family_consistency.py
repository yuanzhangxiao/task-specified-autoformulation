"""Exact shared-initial family witnesses never substitute or fit hidden initials."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.research.construction_initial_family_consistency import (
    LIMITS,
    _Budget,
    _null_relations,
    _Unsupported,
    assess,
    prompt_view,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.topology_compatibility import EvidenceContract, build_evidence
from autoformalism.staged_topology import content_hash


def problem(*, expression="a*first+b*second+c*time", root=False, shared=False):
    """Observed dynamic states and a latent map feed an algebraic output."""
    symbols = RestrictedParser().parse(expression, location="fixture").symbols
    initial = {
        "mode": "map",
        "expression": expression,
        "parameters": [
            {"name": name, "guess": 1} for name in ("a", "b", "c") if name in symbols
        ],
        "state_initials": {"first": "x"} if "first" in symbols else {},
        "public_initials": {
            k: v for k, v in {"second": "v", "time": "t"}.items() if k in symbols
        },
    }
    if root or shared:
        initial = {
            "mode": "value",
            "guess": 1,
            "scope": "trajectory" if root else "shared_training",
        }
    return PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "initial_family",
                "parent_candidate_id": None,
                "states": [
                    {"name": name, "kind": "latent"} for name in ("x", "z", "w")
                ],
                "state_equations": [
                    {"state": name, "rhs": "0"} for name in ("x", "z", "w")
                ],
                "processes": [{"name": "total", "expression": "w+z"}],
                "observation_mappings": [
                    {"channel": "seen", "expression": "x"},
                    {"channel": "y", "expression": "total"},
                ],
                "parameters": [],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in ("x", "z", "w")
                ],
            },
            "context": {
                "targets": ["seen", "y"],
                "external_inputs": ["u"],
                "auxiliaries": ["v"],
            },
            "graph": {
                "rules": {
                    "z": {"initial": initial},
                    "w": {
                        "initial": {
                            "mode": "map",
                            "expression": "reading",
                            "public_initials": {"reading": "u"},
                        }
                    },
                }
            },
        }
    )


def evidence(
    source, *, values=(3, 3, 4), seen=(1, 2, 1), v=(2, 2, 1), error=0, **metadata
):
    """Equal outputs under distinct seen force a=0, then the v slope is wrong."""
    rows = tuple(
        Trajectory(
            trajectory_id=f"train_{i}",
            time=np.array([0, 1.0]),
            targets={"seen": np.array([x, x]), "y": np.array([y, y])},
            external_inputs={"u": np.array([1.0, 1.0])},
            auxiliaries={"v": np.array([q, q])},
            fixed_covariates={},
            derivatives={},
        )
        for i, (y, x, q) in enumerate(zip(values, seen, v, strict=True))
    )
    contract = EvidenceContract.model_validate(
        {
            "provenance": "synthetic exact initial-observation fixture",
            "dependencies_exhaustive": True,
            "shared_parameters": True,
            "absolute_error_bounds": {"seen": 0, "y": error, "u": 0, "v": 0},
            **metadata,
        }
    )
    return build_evidence(
        DatasetSplit(SplitName.TRAIN, rows, "initial-family"), source.context, contract
    ).model_dump(mode="json")


def check(report, target="y"):
    return next(row for row in report["checks"] if row.get("target") == target)


def test_shared_affine_family_infeasibility_has_independently_verifiable_certificate():
    source = problem()
    data = evidence(source)
    before = source.model_dump(mode="json"), deepcopy(data)
    report = assess(source, data)
    result = check(report)
    assert report["status"] == "conflict"
    assert result["status"] == "conditional_conflict"
    witness = result["witnesses"][0]
    weights = list(map(Fraction, witness["weights_exact"]))
    matrix = [list(map(Fraction, row)) for row in witness["design_matrix_exact"]]
    assert sum(map(abs, weights)) == 1
    assert all(
        sum(w * row[j] for w, row in zip(weights, matrix, strict=True)) == 0
        for j in range(len(matrix[0]))
    )
    weighted_residual = sum(
        weight * (Fraction(observed) - Fraction(offset))
        for weight, observed, offset in zip(
            weights, witness["observations"], witness["offsets_exact"], strict=True
        )
    )
    assert abs(weighted_residual) == Fraction(witness["weighted_residual_abs_exact"])
    assert witness["maximum_initial_absolute_error_lower_bound"] > 0.4
    assert report == assess(source, data)
    assert before == (source.model_dump(mode="json"), data)
    assert not report["parameter_fitting_performed"]
    assert not report["numerical_rollout_performed"]
    assert (
        "ODE after time zero cannot resolve"
        in prompt_view(report)["conflicts"][0]["repair_direction"]
    )


def test_compatible_affine_family_has_no_conflict_but_no_scientific_certificate():
    source = problem()
    report = assess(source, evidence(source, values=(4, 5, 3)))  # y=1+seen+v
    assert report["status"] == "no_conflict_found"
    assert check(report)["comparisons"] == 1
    assert not report["scientific_correctness_established"]


def test_initial_checks_do_not_require_dynamical_uniqueness_or_open_loop_metadata():
    source = problem()
    assert assess(source, evidence(source, uniqueness_assumed=False, open_loop=False))[
        "conflicts"
    ]


@pytest.mark.parametrize(
    "metadata", [{"dependencies_exhaustive": None}, {"shared_parameters": False}]
)
def test_missing_shared_family_contract_defers(metadata):
    source = problem()
    assert assess(source, evidence(source, **metadata))["status"] == "deferred"


@pytest.mark.parametrize(
    "expression", ["a*b*first", "sin(a*first)", "a/(b+second)", "exp(first)"]
)
def test_unsupported_initializer_families_defer(expression):
    source = problem(expression=expression)
    result = check(assess(source, evidence(source)))
    assert result["status"] == "deferred"
    assert not result["witnesses"]


def test_relevant_trajectory_roots_defer_without_assuming_the_guess_is_shared():
    source = problem(root=True)
    data = evidence(source)
    assert check(assess(source, data))["status"] == "deferred"
    assert "trajectory-scoped" in check(assess(source, data))["unavailable"][0]
    assert check(assess(source, data), "seen")["status"] == "no_conflict_found"


def test_shared_root_is_unknown_shared_coordinate_not_a_fixed_guess():
    source = problem(shared=True)
    assert (
        check(assess(source, evidence(source, values=(7, 7, 7))))["status"]
        == "no_conflict_found"
    )
    assert check(assess(source, evidence(source)))["status"] == "conditional_conflict"


def test_nonzero_target_uncertainty_enters_rigorous_weighted_bound():
    source = problem()
    assert (
        check(assess(source, evidence(source, error=0.01)))["status"]
        == "conditional_conflict"
    )
    assert (
        check(assess(source, evidence(source, error=2)))["status"]
        == "no_conflict_found"
    )
    assert check(assess(source, evidence(source, error=None)))["status"] == "deferred"


@pytest.mark.parametrize("channel,value", [("seen", 0.01), ("v", 1e-10), ("u", None)])
def test_uncertain_conditioning_cannot_be_treated_as_exact_design(channel, value):
    source = problem()
    bounds = {"seen": 0, "y": 0, "u": 0, "v": 0, channel: value}
    result = check(assess(source, evidence(source, absolute_error_bounds=bounds)))
    assert result["status"] == "deferred"
    assert "conditioning" in result["unavailable"][0]


def test_numerical_guard_defers_small_but_exact_nonzero_residual():
    source = problem()
    result = check(assess(source, evidence(source, values=(4, 5, 3 + 1e-9))))
    assert result["status"] == "deferred"
    assert result["numerically_unresolved"]


def test_dependent_known_and_shared_initials_resolve_in_actual_graph_order():
    payload = problem().model_dump(mode="json")
    payload["graph"]["rules"]["z"]["initial"] = {
        "mode": "map",
        "expression": "a*ancestor",
        "parameters": [{"name": "a", "guess": 1}],
        "state_initials": {"ancestor": "w"},
    }
    payload["graph"]["rules"]["w"]["initial"] = {
        "mode": "known",
        "value": 2,
        "justification": "fixed preparation",
    }
    source = PrefixFitProblem.model_validate(payload)
    result = check(assess(source, evidence(source)))
    assert result["status"] == "conditional_conflict"
    assert result["exact_conditioning_channels"] == []


def test_observed_initial_identity_is_used_instead_of_base_fixed_placeholder():
    source = problem()
    result = check(assess(source, evidence(source, values=(4, 5, 3))))
    first = result["expanded_initial_readouts"]["train_1"]
    assert first["shared_coefficients"]["init_z_a"] == "2"


def test_nonlinear_output_in_shared_initial_coordinates_defers():
    payload = problem().model_dump(mode="json")
    payload["base_candidate"]["observation_mappings"][1]["expression"] = "total*total"
    source = PrefixFitProblem.model_validate(payload)
    assert check(assess(source, evidence(source)))["status"] == "deferred"


def test_known_initial_values_give_zero_column_certificate():
    payload = problem().model_dump(mode="json")
    payload["graph"]["rules"]["z"]["initial"] = {
        "mode": "known",
        "value": 2,
        "justification": "fixed preparation",
    }
    source = PrefixFitProblem.model_validate(payload)
    report = assess(source, evidence(source))
    witness = check(report)["witnesses"][0]
    assert witness["shared_coordinates"] == []
    assert witness["maximum_initial_absolute_error_lower_bound"] == 1
    compact = prompt_view(report)["conflicts"][0]["witnesses"][0]
    assert len(compact["initial_observations"]) == 1
    assert "design_matrix_exact" not in compact


def test_malformed_stale_nontraining_and_context_mismatch_evidence_rejected():
    source = problem()
    data = evidence(source)
    data["trajectories"][0]["targets"]["y"][0] = 100
    with pytest.raises(ValueError, match="digest"):
        assess(source, data)
    data = evidence(source)
    data["split"] = "test"
    with pytest.raises(ValueError):
        assess(source, data)
    data = evidence(source)
    data["context"]["time_symbol"] = "clock"
    data["sha256"] = content_hash({k: v for k, v in data.items() if k != "sha256"})
    with pytest.raises(ValueError, match="context differs"):
        assess(source, data)


def test_exact_elimination_is_bounded_and_handles_zero_column_family():
    weights = _null_relations([[], []], _Budget())
    assert weights == [[1, 0], [0, 1]]
    with pytest.raises(_Unsupported, match="operation"):
        _null_relations([[Fraction(1)]], _Budget(remaining=0))
    with pytest.raises(_Unsupported, match="bit limit"):
        _Budget().check(Fraction(1 << (LIMITS["rational_bits"] + 1)))


def test_resource_limit_defers_instead_of_partial_success(monkeypatch):
    source = problem()
    monkeypatch.setitem(LIMITS, "rows", 2)
    result = check(assess(source, evidence(source)))
    assert result["status"] == "deferred"
    assert "row limit" in result["unavailable"][0]
