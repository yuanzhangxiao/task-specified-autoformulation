"""No-fitting domain certificates reject only qualified training contradictions."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.research.construction_function_consistency import (
    _affine,
    _Unsupported,
    assess,
    prompt_view,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.topology_compatibility import EvidenceContract, build_evidence
from autoformalism.staged_topology import content_hash


def problem(
    *,
    rhs="u+k*m",
    memory="(i-m)/tau",
    root_role="nonnegative_coefficient",
    coefficient_role="nonnegative_coefficient",
    process=False,
):
    """Generic positive reservoir pair drives an observed accumulation quantity."""
    return PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "fixture",
                "parent_candidate_id": None,
                "states": [
                    {"name": name, "kind": "latent"} for name in ("x", "i", "m")
                ],
                "state_equations": [
                    {"state": "x", "rhs": "effect" if process else rhs},
                    {"state": "i", "rhs": "-i+m+u"},
                    {"state": "m", "rhs": memory},
                ],
                "processes": [{"name": "effect", "expression": rhs}] if process else [],
                "observation_mappings": [
                    {"channel": "y", "expression": "x"},
                    {"channel": "seen", "expression": "i"},
                ],
                "parameters": [
                    {"name": "k", "scope": "global", "role": coefficient_role},
                    {"name": "tau", "scope": "global", "role": "time_constant"},
                ],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in ("x", "i", "m")
                ],
            },
            "context": {"targets": ["y", "seen"], "external_inputs": ["u"]},
            "graph": {
                "rules": {
                    "m": {
                        "initial": {
                            "mode": "value",
                            "guess": 1,
                            "scope": "trajectory",
                            "role": root_role,
                        }
                    }
                }
            },
            "parameter_guesses": {"k": 1, "tau": 1},
        }
    )


def evidence(
    source,
    *,
    y=(2, 2, 2),
    seen=(1, 1, 1),
    u=(1, 1, 1),
    error=0,
    forcing_error=0,
    mode="linear_complete_knots",
    **metadata,
):
    row = Trajectory(
        trajectory_id="train_000",
        time=np.array([0, 1, 2.0]),
        targets={"y": np.array(y, float), "seen": np.array(seen, float)},
        external_inputs={"u": np.array(u, float)},
        auxiliaries={},
        fixed_covariates={},
        derivatives={},
    )
    contract = EvidenceContract(
        provenance="synthetic complete training contract",
        dependencies_exhaustive=True,
        shared_parameters=True,
        open_loop=True,
        reset_impulse_free=True,
        uniqueness_assumed=True,
        interpolation={"u": mode},
        absolute_error_bounds={"y": error, "seen": error, "u": forcing_error},
    ).model_copy(update=metadata)
    return build_evidence(
        DatasetSplit(SplitName.TRAIN, (row,), "fixture"), source.context, contract
    ).model_dump(mode="json")


def target_check(report):
    return next(row for row in report["checks"] if row.get("target") == "y")


@pytest.mark.parametrize("process", [False, True])
def test_positive_subsystem_proves_constant_output_conflict(process):
    source = problem(process=process)
    data = evidence(source)
    before = source.model_dump(mode="json"), deepcopy(data)
    report = assess(source, data)
    check = target_check(report)
    assert check["status"] == "conditional_conflict"
    assert check["derivative_interval"] == [1, "infinity"]
    assert check["proven_nonnegative_states"] == ["i", "m", "x"]
    witness = check["witnesses"][0]
    assert witness["times"] == [0, 2]
    assert witness["required_change"] == {"comparison": "at_least", "value": 2}
    assert witness["observed_change_interval"] == [0, 0]
    assert report == assess(source, data)
    assert (source.model_dump(mode="json"), data) == before
    assert not report["numerical_rollout_performed"]
    assert not report["parameter_fitting_performed"]


@pytest.mark.parametrize(
    "change",
    [
        {"root_role": "coefficient"},
        {"coefficient_role": "coefficient"},
        {"memory": "(-i-m)/tau"},
        {"rhs": "u+k*m-x"},
        {"rhs": "u+k*m*i"},
        {"rhs": "u+k*sin(m)"},
    ],
)
def test_unknown_sign_restoring_terms_and_nonlinearity_never_get_false_proof(change):
    source = problem(**change)
    report = assess(source, evidence(source))
    assert target_check(report)["status"] == "deferred"
    assert not target_check(report).get("witnesses")


@pytest.mark.parametrize(
    "options",
    [
        {"seen": (-1, 1, 1)},
        {"u": (-1, 1, 1)},
        {"mode": "unknown"},
        {"forcing_error": None},
        {"forcing_error": 2},
        {"error": 2},
    ],
)
def test_negative_initials_and_unknown_forcing_or_noise_prevent_false_conflicts(
    options,
):
    source = problem()
    report = assess(source, evidence(source, **options))
    assert not target_check(report).get("witnesses")


@pytest.mark.parametrize("mode", ["linear_complete_knots", "previous_complete_knots"])
def test_no_conflict_when_monotonic_target_meets_derivative_bound(mode):
    source = problem()
    check = target_check(assess(source, evidence(source, y=(2, 3, 4), mode=mode)))
    assert check["status"] == "no_conflict_found"
    assert not check["witnesses"]


def test_negative_slope_certificate_and_no_latent_observation_substitution():
    source = problem(rhs="-u-k*m")
    check = target_check(assess(source, evidence(source)))
    assert check["status"] == "conditional_conflict"
    assert check["derivative_interval"] == ["-infinity", -1]
    assert check["witnesses"][0]["required_change"] == {
        "comparison": "at_most",
        "value": -2,
    }
    # Later negative measured i cannot replace the proved nonnegative generated i.
    report = assess(source, evidence(source, seen=(1, -2, -4)))
    assert target_check(report)["status"] == "conditional_conflict"


def test_observation_maps_do_not_become_direct_state_readouts():
    payload = problem().model_dump(mode="json")
    payload["base_candidate"]["observation_mappings"][0]["expression"] = "2*x"
    payload["graph"]["rules"]["x"] = {
        "initial": {"mode": "known", "value": 2, "justification": "fixture"}
    }
    source = PrefixFitProblem.model_validate(payload)
    check = target_check(assess(source, evidence(source)))
    assert check["status"] == "deferred"


def test_initializer_maps_proven_by_domains_and_known_values_only():
    payload = problem().model_dump(mode="json")
    payload["graph"]["rules"]["m"]["initial"] = {
        "mode": "map",
        "expression": "a*start",
        "parameters": [{"name": "a", "role": "nonnegative_coefficient", "guess": 1}],
        "public_initials": {"start": "seen"},
    }
    source = PrefixFitProblem.model_validate(payload)
    assert (
        target_check(assess(source, evidence(source)))["status"]
        == "conditional_conflict"
    )
    payload["graph"]["rules"]["m"]["initial"]["parameters"][0]["role"] = "coefficient"
    source = PrefixFitProblem.model_validate(payload)
    assert target_check(assess(source, evidence(source)))["status"] == "deferred"


def test_state_constraint_and_positive_guess_do_not_prove_positive_trajectory():
    payload = problem(root_role="coefficient").model_dump(mode="json")
    payload["base_candidate"]["constraints"] = [
        {
            "kind": "nonnegative",
            "subject": "m",
            "source": "proposer",
            "enforcement": "hard",
        }
    ]
    source = PrefixFitProblem.model_validate(payload)
    assert target_check(assess(source, evidence(source)))["status"] == "deferred"


def test_numerical_guard_and_observation_error_prevent_roundoff_rejection():
    source = problem(rhs="0*k")
    check = target_check(assess(source, evidence(source, y=(2, 2, 2 + 1e-8))))
    assert not check["witnesses"]
    check = target_check(assess(source, evidence(source, y=(2, 2, 2.1), error=0.1)))
    assert not check["witnesses"]


def test_missing_class_contract_and_stale_evidence_are_not_accepted():
    source = problem()
    data = evidence(source, reset_impulse_free=None)
    assert assess(source, data)["status"] == "deferred"
    data["contract"]["reset_impulse_free"] = True
    with pytest.raises(ValueError, match="digest"):
        assess(source, data)
    data["sha256"] = content_hash(
        {key: value for key, value in data.items() if key != "sha256"}
    )
    assert assess(source, data)["status"] == "conflict"


def test_no_parameter_guess_or_legacy_bounds_strengthen_domain_proof():
    payload = problem(coefficient_role="coefficient").model_dump(mode="json")
    payload["base_candidate"]["parameters"][0]["bounds"] = {"lower": 1, "upper": 5}
    payload["parameter_guesses"]["k"] = 3
    source = PrefixFitProblem.model_validate(payload)
    assert target_check(assess(source, evidence(source)))["status"] == "deferred"


def test_generated_auxiliary_history_never_substitutes_for_its_state():
    source = problem(rhs="u+k*i")
    data = evidence(source)
    payload = source.model_dump(mode="json")
    payload["base_candidate"]["states"][1]["name"] = "seen"
    payload["base_candidate"]["state_equations"] = [
        {"state": "x", "rhs": "u+k*seen"},
        {"state": "seen", "rhs": "-1"},
        {"state": "m", "rhs": "(seen-m)/tau"},
    ]
    payload["base_candidate"]["initial_conditions"][1]["state"] = "seen"
    payload["base_candidate"]["observation_mappings"][1]["expression"] = "seen"
    payload["context"]["targets"] = ["y"]
    payload["context"]["auxiliaries"] = ["seen"]
    source = PrefixFitProblem.model_validate(payload)
    data["context"] = source.context.model_dump(mode="json")
    row = data["trajectories"][0]
    row["auxiliaries"]["seen"] = row["targets"].pop("seen")
    data["contract"]["interpolation"]["seen"] = "linear_complete_knots"
    data["sha256"] = content_hash(
        {key: value for key, value in data.items() if key != "sha256"}
    )
    check = target_check(assess(source, data))
    assert check["status"] == "deferred"
    assert "seen" not in check["proven_nonnegative_states"]


def test_parameter_only_denominators_require_strict_positive_domain():
    source = problem(rhs="u+k*m/(1+tau)")
    assert (
        target_check(assess(source, evidence(source)))["status"]
        == "conditional_conflict"
    )
    payload = source.model_dump(mode="json")
    payload["base_candidate"]["parameters"][1].update(
        role="nonnegative_coefficient", domain="nonnegative"
    )
    payload["base_candidate"]["state_equations"][0]["rhs"] = "u+k*m/tau"
    source = PrefixFitProblem.model_validate(payload)
    assert target_check(assess(source, evidence(source)))["status"] == "deferred"


def test_prompt_keeps_one_witness_per_target_without_duplicating_all_laws():
    source = problem()
    data = evidence(source)
    extra = deepcopy(data["trajectories"][0])
    extra["trajectory_id"] = "train_001"
    data["trajectories"].append(extra)
    data["sha256"] = content_hash(
        {key: value for key, value in data.items() if key != "sha256"}
    )
    report = assess(source, data)
    view = prompt_view(report)
    assert view["full_conflict_count"] == 2
    assert len(view["conflicts"]) == 1
    assert "proof_inputs" not in view["conflicts"][0]
    assert view["conflicts"][0]["witness"]["trajectory"] == "train_000"


def test_nested_constant_powers_abstain_at_bounded_exact_arithmetic_budget():
    expression = RestrictedParser().parse(
        "(((10**16)**16)**16)**16", location="adversarial constant"
    )
    with pytest.raises(_Unsupported, match="bit budget"):
        _affine(expression.tree, {})
