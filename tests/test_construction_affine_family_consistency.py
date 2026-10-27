"""Finite affine response certificates preserve initializer and forcing scope."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.research.construction_affine_family_consistency import (
    COMBINED_SYMBOLIC_POLICY,
    COMMON_ZERO_FORCING_POLICY,
    SYMBOLIC_INITIAL_POLICY,
    WEIGHTED_SYMBOLIC_POLICY,
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


def mapped_family_problem(*, dependent=False, nonlinear=False, free_root=False):
    """Two shared maps differ across rows but obey the same weighted identity."""
    payload = problem(extra_root=True, map_root=True).model_dump(mode="json")
    for state in ("m", "n"):
        payload["graph"]["rules"][state] = {
            "initial": {
                "mode": "map",
                "expression": "a*x0+b*q0",
                "parameters": [
                    {"name": name, "role": "coefficient", "guess": 1}
                    for name in ("a", "b")
                ],
                "state_initials": {"x0": "x"},
                "public_initials": {"q0": "q"},
            }
        }
    if dependent:
        payload["graph"]["rules"]["n"]["initial"] = {
            "mode": "map",
            "expression": "a*m0+x0" if nonlinear else "2*m0+a*x0",
            "parameters": [{"name": "a", "role": "coefficient", "guess": 1}],
            "state_initials": {"m0": "m", "x0": "x"},
        }
    if free_root:
        payload["graph"]["rules"]["n"]["initial"] = {
            "mode": "value", "scope": "trajectory", "guess": 1,
        }
    return PrefixFitProblem.model_validate(payload)


def mapped_family_evidence(source, *, equal_covariate=True, consistent=False, **kwargs):
    values = [[2, 3, 4], [1, 1, 2], [3, 4, 5]]
    if consistent:
        values = [[2, 3, 4], [1, 2, 3], [3, 4, 5]]
    data = evidence(source, values, **kwargs)
    if equal_covariate:
        for row in data["trajectories"]:
            row["covariates"]["q"] = 7.0
    reseal(data)
    return data


@pytest.mark.parametrize("dependent", [False, True])
def test_weighted_shared_maps_cancel_for_all_unknown_parameters(dependent):
    source = mapped_family_problem(dependent=dependent)
    data = mapped_family_evidence(source)
    before = source.model_dump(mode="json"), deepcopy(data)
    assert not assess(source, data, symbolic_initials=True)["conflicts"]
    report = assess(source, data, weighted_symbolic_initials=True)
    assert report["weighted_symbolic_initial_policy"] == WEIGHTED_SYMBOLIC_POLICY
    assert (
        prompt_view(report)["weighted_symbolic_initial_policy"]
        == WEIGHTED_SYMBOLIC_POLICY
    )
    witness = conflicts(report, "affine_initial_superposition")[0]["witnesses"][0]
    weights = list(map(Fraction, witness["weights_exact"]))
    assert sum(weights) == 0
    assert sum(w * Fraction(x) for w, x in zip(weights, [2, 1, 3], strict=True)) == 0
    for state in ("m", "n"):
        value = witness["initial_state_values"][state]
        assert value["policy"] == WEIGHTED_SYMBOLIC_POLICY
        columns = [
            value["constant_exact"], *value["shared_coefficients_exact"].values()
        ]
        for column in columns:
            products = (w * Fraction(x) for w, x in zip(weights, column, strict=True))
            assert sum(products) == 0
    coefficients = witness["initial_state_values"]["m"]["shared_coefficients_exact"]
    assert "init_m_a" in coefficients
    assert (
        abs(witness["weighted_observation_defect"]) > witness["target_numerical_guard"]
    )
    assert not report["parameter_fitting_performed"]
    assert not report["numerical_rollout_performed"]
    assert before == (source.model_dump(mode="json"), data)
    assert report == assess(source, data, weighted_symbolic_initials=True)


@pytest.mark.parametrize(
    "source_options,evidence_options",
    [
        ({"free_root": True}, {}),
        ({"dependent": True, "nonlinear": True}, {}),
        ({}, {"equal_covariate": False}),
        ({}, {"consistent": True}),
        ({}, {"error": 1e-8}),
        ({}, {"forcing_error": 1e-8}),
        ({}, {"u": [[1, 1, 1], [1, 2, 1], [1, 3, 1]]}),
        ({}, {"mode": "unknown"}),
        ({}, {"shared_parameters": False}),
        ({}, {"reset_impulse_free": False}),
        ({}, {"uniqueness_assumed": False}),
    ],
)
def test_weighted_shared_map_proof_preserves_negative_controls(
    source_options, evidence_options
):
    source = mapped_family_problem(**source_options)
    data = mapped_family_evidence(source, **evidence_options)
    report = assess(source, data, weighted_symbolic_initials=True)
    assert report["status"] == "deferred"
    assert not report["conflicts"]
    assert not report["scientific_correctness_established"]


def test_weighted_map_uncertain_conditioning_is_not_silently_exact():
    source = mapped_family_problem()
    data = mapped_family_evidence(source)
    data["contract"]["absolute_error_bounds"]["q"] = 1e-12
    reseal(data)
    assert not assess(source, data, weighted_symbolic_initials=True)["conflicts"]


def test_weighted_map_operation_limit_defers(monkeypatch):
    from autoformalism.research import construction_affine_family_consistency as module

    budget_type = module._InitialBudget
    monkeypatch.setattr(module, "_InitialBudget", lambda: budget_type(remaining=1))
    source = mapped_family_problem()
    data = mapped_family_evidence(source)
    assert not assess(source, data, weighted_symbolic_initials=True)["conflicts"]


def combined_problem(*, free_b=False, dependent=False, nonlinear=False):
    """A free m(0) is constrained only after expanding b(0)=beta*y(0)."""
    payload = symbolic_problem(readout="a+m+b").model_dump(mode="json")
    candidate = payload["base_candidate"]
    candidate["state_equations"][0]["rhs"] += "+c*x"
    next(eq for eq in candidate["state_equations"] if eq["state"] == "b")["rhs"] = "0"
    payload["graph"]["rules"]["b"]["initial"] = {
        "mode": "map",
        "expression": "beta*y0",
        "parameters": [{"name": "beta", "role": "coefficient", "guess": 1}],
        "public_initials": {"y0": "y"},
    }
    if free_b:
        payload["graph"]["rules"]["b"]["initial"] = {
            "mode": "value", "scope": "trajectory", "guess": 1,
        }
    if dependent:
        candidate["states"].append({"name": "n", "kind": "latent"})
        candidate["state_equations"].append({"state": "n", "rhs": "-n"})
        candidate["state_equations"][0]["rhs"] += "+n"
        candidate["initial_conditions"].append(
            {"state": "n", "scope": "global", "fixed_value": 0}
        )
        payload["graph"]["rules"]["n"] = {
            "initial": {
                "mode": "map", "expression": "2*m0+alpha*y0",
                "state_initials": {"m0": "m"},
                "public_initials": {"y0": "y"},
                "parameters": [
                    {"name": "alpha", "role": "coefficient", "guess": 1}
                ],
            }
        }
    if nonlinear:
        candidate["state_equations"][0]["rhs"] += "+x*m"
    return PrefixFitProblem.model_validate(payload)


@pytest.mark.parametrize("dependent", [False, True])
def test_combined_initial_readout_eliminates_root_for_every_shared_coefficient(
    dependent,
):
    source = combined_problem(dependent=dependent)
    data = symbolic_evidence(source)
    before = source.model_dump(mode="json"), deepcopy(data)
    historical = assess(
        source, data, symbolic_initials=True, weighted_symbolic_initials=True
    )
    assert historical["status"] == "deferred"
    assert "combined_symbolic_initial_policy" not in historical
    report = assess(source, data, combined_symbolic_initials=True)
    assert report["combined_symbolic_initial_policy"] == COMBINED_SYMBOLIC_POLICY
    assert (
        prompt_view(report)["combined_symbolic_initial_policy"]
        == COMBINED_SYMBOLIC_POLICY
    )
    witness = conflicts(report, "affine_initial_superposition")[0]["witnesses"][0]
    for state in ("m", "b", *(("n",) if dependent else ())):
        value = witness["initial_state_values"][state]
        assert value["policy"] == COMBINED_SYMBOLIC_POLICY
        assert "observed initial readouts" in value["source"]
        assert "init_m_value" not in value["shared_coefficients_exact"]
        weights = list(map(Fraction, witness["weights_exact"]))
        for column in [
            value["constant_exact"], *value["shared_coefficients_exact"].values()
        ]:
            products = (w * Fraction(x) for w, x in zip(weights, column, strict=True))
            assert sum(products) == 0
    assert witness["initial_state_values"]["m"]["shared_coefficients_exact"] == {
        "init_b_beta": ["-2", "-1", "-3"]
    }
    assert before == (source.model_dump(mode="json"), data)
    assert report == assess(source, data, combined_symbolic_initials=True)


@pytest.mark.parametrize(
    "options,evidence_options",
    [
        ({"free_b": True}, {}),
        ({"nonlinear": True}, {}),
        ({}, {"error": 1e-8}),
        ({}, {"forcing_error": 1e-8}),
        ({}, {"u": [[1, 1, 1], [1, 2, 1], [1, 3, 1]]}),
        ({}, {"uniqueness_assumed": False}),
        ({}, {"shared_parameters": False}),
        ({}, {"reset_impulse_free": False}),
    ],
)
def test_combined_initial_proof_keeps_required_assumptions(options, evidence_options):
    source = combined_problem(**options)
    data = symbolic_evidence(source, **evidence_options)
    report = assess(source, data, combined_symbolic_initials=True)
    assert report["status"] == "deferred"
    assert not report["conflicts"]


@pytest.mark.parametrize("readout", ["a+c*m+b", "a+m+b*b"])
def test_combined_initial_proof_does_not_divide_unknown_or_linearize(readout):
    source = combined_problem()
    payload = source.model_dump(mode="json")
    payload["base_candidate"]["processes"][0]["expression"] = readout
    source = PrefixFitProblem.model_validate(payload)
    assert not assess(
        source, symbolic_evidence(source), combined_symbolic_initials=True
    )["conflicts"]


def test_combined_initial_proof_cannot_equate_unrelated_free_state():
    payload = combined_problem().model_dump(mode="json")
    payload["base_candidate"]["states"].append({"name": "n", "kind": "latent"})
    payload["base_candidate"]["state_equations"].append({"state": "n", "rhs": "-n"})
    payload["base_candidate"]["state_equations"][0]["rhs"] += "+n"
    payload["base_candidate"]["initial_conditions"].append(
        {"state": "n", "scope": "global", "fixed_value": 0}
    )
    payload["graph"]["rules"]["n"] = {
        "initial": {"mode": "value", "scope": "trajectory", "guess": 1}
    }
    source = PrefixFitProblem.model_validate(payload)
    assert not assess(
        source, symbolic_evidence(source), combined_symbolic_initials=True
    )["conflicts"]


def test_combined_initial_proof_preserves_unconstrained_readout_split():
    source = combined_problem(free_b=True)
    report = assess(source, symbolic_evidence(source), combined_symbolic_initials=True)
    assert not conflicts(report, "affine_initial_superposition")
    assert not report["scientific_correctness_established"]


def test_combined_initial_proof_requires_exact_initial_readout():
    source = combined_problem()
    data = symbolic_evidence(source)
    data["contract"]["absolute_error_bounds"]["z"] = 1e-12
    reseal(data)
    assert not assess(source, data, combined_symbolic_initials=True)["conflicts"]


def test_combined_initial_proof_operation_limit_defers(monkeypatch):
    from autoformalism.research import construction_affine_family_consistency as module

    budget_type = module._InitialBudget
    monkeypatch.setattr(module, "_InitialBudget", lambda: budget_type(remaining=1))
    source = combined_problem()
    data = symbolic_evidence(source)
    assert not assess(source, data, combined_symbolic_initials=True)["conflicts"]


def test_combined_initial_proof_does_not_reject_a_consistent_weighted_family():
    source = combined_problem()
    data = evidence(
        source, [[2, 3, 4], [1, 2, 3], [3, 4, 5]], z=[[2, 2, 2]] * 3
    )
    report = assess(source, data, combined_symbolic_initials=True)
    assert report["status"] == "deferred"
    assert not report["scientific_correctness_established"]


def test_combined_initial_proof_solves_two_roots_using_two_exact_readouts():
    source = combined_problem(free_b=True)
    data = symbolic_evidence(source)
    payload = source.model_dump(mode="json")
    candidate = payload["base_candidate"]
    candidate["states"].append({"name": "n", "kind": "latent"})
    candidate["state_equations"].append({"state": "n", "rhs": "0"})
    candidate["state_equations"][0]["rhs"] += "+n"
    candidate["initial_conditions"].append(
        {"state": "n", "scope": "global", "fixed_value": 0}
    )
    candidate["processes"].append({"name": "readout_w", "expression": "m+2*b+n"})
    candidate["observation_mappings"].append(
        {"channel": "w", "expression": "readout_w"}
    )
    payload["context"]["targets"].append("w")
    payload["graph"]["rules"]["n"] = {
        "initial": {
            "mode": "map", "expression": "beta*y0",
            "parameters": [{"name": "beta", "role": "coefficient", "guess": 1}],
            "public_initials": {"y0": "y"},
        }
    }
    source = PrefixFitProblem.model_validate(payload)
    data["context"] = source.context.model_dump(mode="json")
    data["contract"]["absolute_error_bounds"]["w"] = 0.0
    for row in data["trajectories"]:
        row["targets"]["w"] = [4.0, 4.0, 4.0]
    reseal(data)
    assert not assess(
        source, data, symbolic_initials=True, weighted_symbolic_initials=True
    )["conflicts"]
    report = assess(source, data, combined_symbolic_initials=True)
    values = conflicts(report, "affine_initial_superposition")[0]["witnesses"][0][
        "initial_state_values"
    ]
    assert values["m"]["constant_exact"] == ["-2"] * 3
    assert values["b"]["constant_exact"] == ["3"] * 3
    assert values["m"]["shared_coefficients_exact"] == {
        "init_n_beta": ["2", "1", "3"]
    }
    assert values["b"]["shared_coefficients_exact"] == {
        "init_n_beta": ["-2", "-1", "-3"]
    }


def test_combined_initial_proof_coordinate_limit_defers(monkeypatch):
    from autoformalism.research import construction_affine_family_consistency as module

    monkeypatch.setitem(module.INITIAL_READOUT_LIMITS, "coordinates", 1)
    source = combined_problem()
    assert not assess(
        source, symbolic_evidence(source), combined_symbolic_initials=True
    )["conflicts"]


def saturation_problem(*, expression="V*u/(K+u)", role="rate", nonlinear_state=False):
    """The input response is nonlinear, but vanishes on certified zero inputs."""
    payload = problem(pin_root=True).model_dump(mode="json")
    candidate = payload["base_candidate"]
    candidate["parameters"] += [
        {"name": "K", "scope": "global", "role": role},
        {"name": "V", "scope": "global", "role": "coefficient"},
    ]
    candidate["processes"].append({"name": "saturation", "expression": expression})
    candidate["state_equations"][0]["rhs"] = "-k*x+m+saturation+K*x"
    if nonlinear_state:
        candidate["state_equations"][0]["rhs"] += "+x*m"
    return PrefixFitProblem.model_validate(payload)


def saturation_evidence(source, **options):
    settings = {"u": [[0, 0, 0]] * 3, **options}
    return evidence(
        source, [[2, 3, 4], [1, 1, 2], [3, 4, 5]], z=[[2, 2, 2]] * 3,
        **settings,
    )


def test_common_zero_saturation_restricts_before_affine_parsing():
    source = saturation_problem()
    data = saturation_evidence(source)
    before = source.model_dump(mode="json"), deepcopy(data)
    assert assess(source, data)["status"] == "deferred"
    assert "common_zero_forcing_policy" not in assess(source, data)
    report = assess(source, data, common_zero_forcing=True)
    assert report["common_zero_forcing_policy"] == COMMON_ZERO_FORCING_POLICY
    assert (
        prompt_view(report)["common_zero_forcing_policy"] == COMMON_ZERO_FORCING_POLICY
    )
    check = conflicts(report, "affine_initial_superposition")[0]
    assert check["complete_common_forcing"] == ["u"]
    certificate = check["common_zero_forcing"]
    assert certificate["policy"] == COMMON_ZERO_FORCING_POLICY
    assert certificate["channels"] == ["u"]
    assert certificate["interval"] == [0, 2]
    assert certificate["parameter_domains"]["K"] == "positive"
    assert not certificate["global_affinity_claim"]
    assert certificate["schedules"]["u"]["interpolation"] == "linear_complete_knots"
    for row in data["trajectories"]:
        recorded = certificate["schedules"]["u"]["trajectories"][row["trajectory_id"]]
        assert recorded["schedule_sha256"] == content_hash({
            "time": row["time"], "values": row["inputs"]["u"],
        })
    assert check["restricted_process_relations"]["saturation"] == "V * 0 / (K + 0)"
    assert "zero-input schedules" in check["repair_feedback"]
    assert not report["scientific_correctness_established"]
    assert before == (source.model_dump(mode="json"), data)
    assert report == assess(source, data, common_zero_forcing=True)


@pytest.mark.parametrize(
    "source_options,evidence_options",
    [
        ({"nonlinear_state": True}, {}),
        ({"role": "coefficient"}, {}),
        ({"role": "nonnegative_coefficient"}, {}),
        ({"expression": "V*u/u"}, {}),
        ({"expression": "V*u/(K*u)"}, {}),
        ({"expression": "0*(V/u)"}, {}),
        ({}, {"u": [[1, 1, 1]] * 3}),
        ({}, {"u": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]}),
        ({}, {"mode": "unknown"}),
        ({}, {"forcing_error": 1e-12}),
        ({}, {"error": 1e-12}),
        ({}, {"uniqueness_assumed": False}),
    ],
)
def test_zero_forcing_preserves_state_nonlinearity_and_undefined_domains(
    source_options, evidence_options
):
    source = saturation_problem(**source_options)
    report = assess(
        source,
        saturation_evidence(source, **evidence_options),
        common_zero_forcing=True,
    )
    assert report["status"] == "deferred"
    assert not report["conflicts"]


def test_zero_forcing_may_use_a_qualified_subset_of_training_rows():
    source = saturation_problem()
    data = evidence(
        source, [[2, 3, 4], [1, 1, 2], [3, 4, 5], [4, 4, 4]],
        z=[[2, 2, 2]] * 4, u=[[0, 0, 0]] * 3 + [[1, 1, 1]],
    )
    check = conflicts(
        assess(source, data, common_zero_forcing=True), "affine_initial_superposition"
    )[0]
    assert check["witnesses"][0]["trajectories"] == ["row0", "row1", "row2"]
    assert set(check["common_zero_forcing"]["schedules"]["u"]["trajectories"]) == {
        "row0", "row1", "row2"
    }


def test_zero_forcing_does_not_replace_a_modeled_auxiliary_by_measured_history():
    source = saturation_problem(expression="V*w/(K+w)")
    data = saturation_evidence(source)
    payload = source.model_dump(mode="json")
    candidate = payload["base_candidate"]
    candidate["states"].append({"name": "w", "kind": "latent"})
    candidate["state_equations"].append({"state": "w", "rhs": "x-w"})
    candidate["initial_conditions"].append({
        "state": "w", "fixed_value": 0, "scope": "global",
    })
    payload["context"]["auxiliaries"] = ["w"]
    source = PrefixFitProblem.model_validate(payload)
    data["context"] = source.context.model_dump(mode="json")
    for row in data["trajectories"]:
        row["auxiliaries"]["w"] = [0.0, 0.0, 0.0]
    data["contract"]["absolute_error_bounds"]["w"] = 0.0
    data["contract"]["interpolation"]["w"] = "linear_complete_knots"
    reseal(data)
    assert not assess(source, data, common_zero_forcing=True)["conflicts"]


def test_zero_forcing_restriction_budget_defers(monkeypatch):
    from autoformalism.research import construction_affine_family_consistency as module

    monkeypatch.setattr(module, "MAX_ZERO_RESTRICTIONS", 0)
    source = saturation_problem()
    assert not assess(
        source, saturation_evidence(source), common_zero_forcing=True
    )["conflicts"]
