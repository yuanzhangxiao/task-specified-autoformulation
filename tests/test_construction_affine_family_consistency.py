"""Finite affine response certificates preserve initializer and forcing scope."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.research.construction_affine_family_consistency import (
    SYMBOLIC_INITIAL_POLICY,
    _display,
    _null_vectors,
    _rref,
    _Unsupported,
    assess,
    prompt_view,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.topology_compatibility import EvidenceContract, build_evidence
from autoformalism.staged_topology import content_hash


def problem(*, pin_root=False, nonlinear=False, extra_root=False, map_root=False):
    states = ["x", "m"] + (["a"] if pin_root else []) + (["n"] if extra_root else [])
    equations = {"x": "-k*x + m + u", "m": "x - m/tau"}
    if nonlinear:
        equations["x"] = "-k*x + m*x + u"
    if extra_root:
        equations["x"] += " + n"
        equations["n"] = "-n"
    if pin_root:
        equations["a"] = "0"
    graph = {"m": {"initial": {"mode": "value", "scope": "trajectory", "guess": 1}}}
    if map_root:
        graph["m"] = {
            "initial": {
                "mode": "map",
                "expression": "a*p",
                "parameters": [{"name": "a", "role": "coefficient", "guess": 1}],
                "public_initials": {"p": "q"},
            }
        }
    if pin_root:
        graph["a"] = {
            "initial": {"mode": "map", "expression": "p", "public_initials": {"p": "u"}}
        }
    if extra_root:
        graph["n"] = {"initial": {"mode": "value", "scope": "trajectory", "guess": 1}}
    outputs = [{"channel": "y", "expression": "x"}]
    if pin_root:
        outputs += [{"channel": "z", "expression": "readout"}]
    return PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "affine_family_fixture",
                "parent_candidate_id": None,
                "states": [{"name": name, "kind": "latent"} for name in states],
                "state_equations": [
                    {"state": name, "rhs": rhs} for name, rhs in equations.items()
                ],
                "processes": [{"name": "readout", "expression": "a+m"}]
                if pin_root
                else [],
                "observation_mappings": outputs,
                "parameters": [
                    {"name": "k", "scope": "global", "role": "coefficient"},
                    {"name": "tau", "scope": "global", "role": "time_constant"},
                ],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in states
                ],
            },
            "context": {
                "targets": [item["channel"] for item in outputs],
                "external_inputs": ["u"],
                "fixed_covariates": ["q"] if map_root else [],
            },
            "graph": {"rules": graph},
            "parameter_guesses": {"k": 1, "tau": 1},
        }
    )


def evidence(
    source,
    y,
    *,
    z=None,
    u=None,
    error=0,
    forcing_error=0,
    mode="linear_complete_knots",
    **metadata,
):
    u = u or [[1, 1, 1]] * len(y)
    trajectories = []
    for index, values in enumerate(y):
        targets = {"y": np.array(values, dtype=float)}
        if z is not None:
            targets["z"] = np.array(z[index], dtype=float)
        trajectories.append(
            Trajectory(
                trajectory_id=f"row{index}",
                time=np.array([0, 1, 2], dtype=float),
                targets=targets,
                external_inputs={"u": np.array(u[index], dtype=float)},
                auxiliaries={},
                fixed_covariates={"q": float(index + 1)}
                if source.context.fixed_covariates
                else {},
                derivatives={},
            )
        )
    contract = EvidenceContract(
        provenance="synthetic complete training data",
        dependencies_exhaustive=True,
        shared_parameters=True,
        open_loop=True,
        reset_impulse_free=True,
        uniqueness_assumed=True,
        interpolation={"u": mode},
        absolute_error_bounds={
            **dict.fromkeys(source.context.targets, error),
            "u": forcing_error,
            "q": 0,
        },
    ).model_copy(update=metadata)
    return build_evidence(
        DatasetSplit(SplitName.TRAIN, tuple(trajectories), "fixture"),
        source.context,
        contract,
    ).model_dump(mode="json")


def conflicts(report, kind):
    return [
        row
        for row in report["conflicts"]
        if row["property"] == kind and row["target"] == "y"
    ]


def test_exact_superposition_uses_algebraic_readout_to_pin_trajectory_root():
    source = problem(pin_root=True)
    data = evidence(source, [[2, 3, 4], [1, 1, 2], [3, 4, 5]], z=[[2, 2, 2]] * 3)
    before = source.model_dump(mode="json"), deepcopy(data)
    report = assess(source, data)
    check = conflicts(report, "affine_initial_superposition")[0]
    witness = check["witnesses"][0]
    assert witness["initial_state_values"]["m"] == [1, 1, 1]
    assert sum(map(Fraction, witness["weights_exact"])) == 0
    assert (
        abs(witness["weighted_observation_defect"]) > witness["target_numerical_guard"]
    )
    assert "No bound" in witness["scope"]
    assert report == assess(source, data)
    assert before == (source.model_dump(mode="json"), data)
    assert not report["parameter_fitting_performed"]
    assert not report["numerical_rollout_performed"]
    assert prompt_view(report)["full_conflict_count"] == len(report["conflicts"])


def test_consistent_affine_family_is_not_certified_scientifically():
    source = problem(pin_root=True)
    report = assess(
        source, evidence(source, [[2, 3, 4], [1, 2, 3], [3, 4, 5]], z=[[2, 2, 2]] * 3)
    )
    assert report["status"] == "deferred"
    assert not report["conflicts"]
    assert not report["scientific_correctness_established"]


@pytest.mark.parametrize("map_root", [False, True])
def test_rank_one_conflict_allows_arbitrary_different_hidden_initials(map_root):
    source = problem(map_root=map_root)
    report = assess(source, evidence(source, [[0, 1, 1], [0, 2, 3], [0, 4, 6]]))
    check = conflicts(report, "linear_initial_difference_rank_one")[0]
    witness = check["witnesses"][0]
    assert witness["possibly_different_initial_states"] == ["m"]
    assert witness["determinant_exact"] == "-1"
    assert "arbitrary trajectory-specific" in witness["scope"]
    assert not conflicts(report, "affine_initial_superposition")


@pytest.mark.parametrize("options", [{"nonlinear": True}, {"extra_root": True}])
def test_nonlinearity_and_multiple_free_hidden_directions_defer(options):
    source = problem(**options)
    report = assess(source, evidence(source, [[0, 1, 1], [0, 2, 3], [0, 4, 6]]))
    assert not report["conflicts"]


@pytest.mark.parametrize(
    "change",
    [
        {"u": [[1, 1, 1], [1, 2, 1], [1, 3, 1]]},
        {"error": 1e-8},
        {"forcing_error": 1e-8},
        {"mode": "unknown"},
        {"uniqueness_assumed": False},
        {"dependencies_exhaustive": False},
        {"shared_parameters": False},
        {"reset_impulse_free": False},
    ],
)
def test_uncertain_conditioning_and_changed_forcing_cannot_create_proof(change):
    source = problem()
    report = assess(
        source, evidence(source, [[0, 1, 1], [0, 2, 3], [0, 4, 6]], **change)
    )
    assert not report["conflicts"]


def test_identical_sampled_forcing_requires_authenticated_interpolation():
    source = problem()
    for mode in ("linear_complete_knots", "previous_complete_knots"):
        report = assess(
            source, evidence(source, [[0, 1, 1], [0, 2, 3], [0, 4, 6]], mode=mode)
        )
        assert conflicts(report, "linear_initial_difference_rank_one")


def test_no_target_interpolation_or_numerical_derivative_is_used():
    source = problem()
    data = evidence(source, [[0, 1, 1], [0, 2, 3], [0, 4, 6]])
    assert set(data["contract"]["interpolation"]) == {"u"}
    assert conflicts(assess(source, data), "linear_initial_difference_rank_one")


def test_free_root_is_not_assumed_equal_for_superposition():
    source = problem()
    report = assess(source, evidence(source, [[2, 3, 4], [1, 1, 2], [3, 4, 5]]))
    assert not conflicts(report, "affine_initial_superposition")


def test_distinct_observed_initials_and_free_memory_are_two_directions():
    source = problem()
    report = assess(source, evidence(source, [[2, 3, 4], [1, 1, 2], [3, 4, 5]]))
    assert not conflicts(report, "linear_initial_difference_rank_one")


def test_small_target_roundoff_defers_instead_of_claiming_exact_conflict():
    source = problem()
    report = assess(source, evidence(source, [[0, 1, 1], [0, 2, 3], [0, 3, 5 + 1e-12]]))
    assert not report["conflicts"]


def test_declared_time_varying_nonlinear_coefficient_is_not_silently_constant():
    source = problem()
    payload = source.model_dump(mode="json")
    payload["base_candidate"]["state_equations"][0]["rhs"] = "t*k*x + m + u"
    source = PrefixFitProblem.model_validate(payload)
    report = assess(source, evidence(source, [[0, 1, 1], [0, 2, 3], [0, 4, 6]]))
    assert not report["conflicts"]


def test_modified_evidence_digest_is_rejected():
    source = problem()
    data = evidence(source, [[0, 1, 1], [0, 2, 3], [0, 4, 6]])
    data["trajectories"][0]["targets"]["y"][1] = 99
    with pytest.raises(ValueError, match="digest"):
        assess(source, data)


def test_bounded_exact_elimination_and_nullspace():
    matrix = [
        [Fraction(1), Fraction(1), Fraction(1)],
        [Fraction(2), Fraction(1), Fraction(3)],
    ]
    vector = _null_vectors(matrix, 3)[0]
    assert all(
        sum(a * b for a, b in zip(row, vector, strict=True)) == 0 for row in matrix
    )
    with pytest.raises(_Unsupported, match="bit budget"):
        _rref([[Fraction(1), Fraction(2**5000)]], 1)


def test_uncertain_initial_readout_does_not_pin_hidden_root():
    source = problem(pin_root=True)
    data = evidence(source, [[2, 3, 4], [1, 1, 2], [3, 4, 5]], z=[[2, 2, 2]] * 3)
    data["contract"]["absolute_error_bounds"]["z"] = 1e-10
    payload = {key: value for key, value in data.items() if key != "sha256"}
    data["sha256"] = content_hash(payload)
    report = assess(source, data)
    assert not report["conflicts"]


def test_unrenderable_exact_certificate_defers_instead_of_overflow():
    with pytest.raises(_Unsupported, match="display"):
        _display(Fraction(10**400))


def symbolic_problem(*, b_public="u", readout="a+m+c*b", extra_root=False):
    """Minimal B_glucose regression with a shared, possibly zero readout gain."""
    payload = problem(pin_root=True, extra_root=extra_root).model_dump(mode="json")
    candidate = payload["base_candidate"]
    candidate["states"].append({"name": "b", "kind": "latent"})
    candidate["state_equations"].append({"state": "b", "rhs": "-b"})
    candidate["state_equations"][0]["rhs"] += "+b"
    candidate["processes"][0]["expression"] = readout
    candidate["parameters"].append(
        {"name": "c", "scope": "global", "role": "nonnegative_coefficient"}
    )
    candidate["initial_conditions"].append(
        {"state": "b", "scope": "global", "fixed_value": 0}
    )
    payload["graph"]["rules"]["b"] = {
        "initial": {
            "mode": "map",
            "expression": "initial_b",
            "public_initials": {"initial_b": b_public},
        }
    }
    if b_public == "q":
        payload["context"]["fixed_covariates"] = ["q"]
    return PrefixFitProblem.model_validate(payload)


def symbolic_evidence(source, **options):
    """Common readout initials but an inconsistent observed affine time shape."""
    return evidence(
        source, [[2, 3, 4], [1, 1, 2], [3, 4, 5]], z=[[2, 2, 2]] * 3, **options
    )


def reseal(data):
    """Authenticate intentionally changed synthetic evidence metadata."""
    data["sha256"] = content_hash({k: v for k, v in data.items() if k != "sha256"})


def test_symbolic_shared_readout_recovers_common_root_without_fitting():
    source = symbolic_problem()
    data = symbolic_evidence(source)
    before = source.model_dump(mode="json"), deepcopy(data)
    assert not assess(source, data)["conflicts"]  # Historical policy is unchanged.
    report = assess(source, data, symbolic_initials=True)
    assert report["symbolic_initial_policy"] == SYMBOLIC_INITIAL_POLICY
    assert prompt_view(report)["symbolic_initial_policy"] == SYMBOLIC_INITIAL_POLICY
    assert source.base_candidate.parameters[-1].domain.value == "nonnegative"
    superposition = conflicts(report, "affine_initial_superposition")[0]
    witness = superposition["witnesses"][0]
    assert witness["initial_state_values"]["m"] == {
        "source": "exact initial readout with shared symbolic parameters",
        "policy": SYMBOLIC_INITIAL_POLICY,
        "constant_exact": "1",
        "shared_coefficients_exact": {"c": "-1"},
    }
    rank = conflicts(report, "linear_initial_difference_rank_one")[0]
    assert rank["witnesses"][0]["possibly_different_initial_states"] == ["x"]
    assert "m" in rank["witnesses"][0]["common_initial_readout_values"]
    assert "No bound" in witness["scope"]
    assert report == assess(source, data, symbolic_initials=True)
    assert before == (source.model_dump(mode="json"), data)
    assert not report["parameter_fitting_performed"]
    assert not report["numerical_rollout_performed"]


def test_symbolic_readout_does_not_identify_unequal_b_initials_as_equal():
    source = symbolic_problem(b_public="q")
    report = assess(source, symbolic_evidence(source), symbolic_initials=True)
    assert not report["conflicts"]


def test_symbolic_readout_consistent_identity_remains_deferred():
    source = symbolic_problem()
    data = evidence(
        source, [[2, 3, 4], [1, 2, 3], [3, 4, 5]], z=[[2, 2, 2]] * 3
    )
    report = assess(source, data, symbolic_initials=True)
    assert report["status"] == "deferred"
    assert not report["conflicts"]
    assert not report["scientific_correctness_established"]


def test_symbolic_readout_exhausted_exact_budget_cannot_create_equality(monkeypatch):
    from autoformalism.research import construction_affine_family_consistency as module

    budget_type = module._InitialBudget
    monkeypatch.setattr(module, "_InitialBudget", lambda: budget_type(remaining=1))
    source = symbolic_problem()
    report = assess(source, symbolic_evidence(source), symbolic_initials=True)
    assert not report["conflicts"]


@pytest.mark.parametrize("channel", ["z", "q"])
def test_symbolic_initial_readout_requires_exact_every_conditioning_value(channel):
    source = symbolic_problem(b_public="q")
    data = symbolic_evidence(source)
    for row in data["trajectories"]:
        row["covariates"]["q"] = 1.0
    data["contract"]["absolute_error_bounds"][channel] = 1e-10
    reseal(data)
    assert not assess(source, data, symbolic_initials=True)["conflicts"]


@pytest.mark.parametrize("role", ["coefficient", "nonnegative_coefficient", "rate"])
def test_symbolic_readout_never_divides_by_an_unknown_parameter(role):
    source = symbolic_problem(readout="a+c*m+b")
    payload = source.model_dump(mode="json")
    payload["base_candidate"]["parameters"][-1] = {
        "name": "c", "scope": "global", "role": role
    }
    source = PrefixFitProblem.model_validate(payload)
    assert not assess(
        source, symbolic_evidence(source), symbolic_initials=True
    )["conflicts"]


def test_symbolic_readout_preserves_an_unconstrained_extra_root():
    source = symbolic_problem(extra_root=True)
    report = assess(source, symbolic_evidence(source), symbolic_initials=True)
    assert not report["conflicts"]


def test_symbolic_readout_requires_shared_coefficient_contract():
    source = symbolic_problem()
    data = symbolic_evidence(source, shared_parameters=False)
    assert not assess(source, data, symbolic_initials=True)["conflicts"]
    payload = source.model_dump(mode="json")
    payload["base_candidate"]["parameters"][-1]["scope"] = "trajectory_specific"
    nonshared = PrefixFitProblem.model_validate(payload)
    with pytest.raises(ValueError, match="globally shared"):
        assess(nonshared, symbolic_evidence(nonshared), symbolic_initials=True)


def test_symbolic_readout_keeps_parameter_identities_distinct():
    source = symbolic_problem(readout="a+m+c*b")
    payload = source.model_dump(mode="json")
    payload["base_candidate"]["parameters"].append(
        {"name": "d", "scope": "global", "role": "nonnegative_coefficient"}
    )
    payload["base_candidate"]["processes"][0]["expression"] = "a+m+(c-d)*b"
    source = PrefixFitProblem.model_validate(payload)
    report = assess(source, symbolic_evidence(source), symbolic_initials=True)
    symbolic = conflicts(report, "affine_initial_superposition")[0]["witnesses"][0]
    assert symbolic["initial_state_values"]["m"]["shared_coefficients_exact"] == {
        "c": "-1", "d": "1"
    }


@pytest.mark.parametrize("coordinates", [64, 65])
def test_symbolic_readout_enforces_shared_coordinate_limit(coordinates):
    def balanced_sum(names):
        """Exercise coordinate count without exhausting expression depth first."""
        if len(names) == 1:
            return names[0]
        midpoint = len(names) // 2
        return f"({balanced_sum(names[:midpoint])}+{balanced_sum(names[midpoint:])})"

    names = ["c", *(f"coef{i}" for i in range(coordinates - 1))]
    source = symbolic_problem(readout=f"a+m+{balanced_sum(names)}*b")
    payload = source.model_dump(mode="json")
    payload["base_candidate"]["parameters"] += [
        {"name": name, "scope": "global", "role": "coefficient"}
        for name in names[1:]
    ]
    source = PrefixFitProblem.model_validate(payload)
    report = assess(source, symbolic_evidence(source), symbolic_initials=True)
    limit = report["witness_budget"]["symbolic_initial_readout"]["coordinates"]
    assert limit == 64
    if coordinates > limit:
        assert not report["conflicts"]
        assert report["status"] == "deferred"
    else:
        check = conflicts(report, "affine_initial_superposition")[0]
        value = check["witnesses"][0]["initial_state_values"]["m"]
        assert len(value["shared_coefficients_exact"]) == coordinates
