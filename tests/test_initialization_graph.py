"""Acyclic initial maps preserve dependency, scope, and public-boundary semantics."""

from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.fitting.initialization_graph import (
    InitializationGraph,
    apply_initialization_graph,
)
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import (
    simulate_trajectory,
    trajectory_initial_state,
)
from autoformalism.schemas import CandidateModel, ParameterDomain


def problem():
    """Return a model whose identity-observed state is historically labeled latent."""
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "initial_graph",
            "parent_candidate_id": None,
            "states": [
                {"name": state, "kind": "latent"} for state in ("a", "b", "m", "y")
            ],
            "processes": [{"name": "loss", "expression": "rate*a"}],
            "state_equations": [
                {"state": "a", "rhs": "-loss"},
                {"state": "b", "rhs": "0"},
                {"state": "m", "rhs": "-rate*m"},
                {"state": "y", "rhs": "m"},
            ],
            "parameters": [{"name": "rate", "scope": "global", "role": "rate"}],
            "observation_mappings": [{"channel": "signal", "expression": "y"}],
            "initial_conditions": [
                {"state": state, "scope": "global", "fixed_value": 0}
                for state in ("a", "b", "m", "y")
            ],
        }
    )
    context = ValidationContext(
        targets=("signal",),
        auxiliaries=("aux",),
        external_inputs=("u",),
        fixed_covariates=("dose",),
    )
    row = Trajectory(
        "row",
        np.array([0.0, 0.5, 1.0]),
        {"signal": np.array([4.0, 8.0, 9.0])},
        {"aux": np.array([5.0, 6.0, 7.0])},
        {"u": np.array([6.0, 7.0, 8.0])},
        {"dose": 7.0},
        {},
    )
    return compile_candidate(candidate, context), row


def graph_payload(*, trajectory=True):
    """Return a dependent map with one shared and one optionally trajectory root."""
    return {
        "rules": {
            "a": {"initial": {"mode": "value", "guess": 2.0}},
            "b": {
                "initial": {
                    "mode": "value",
                    "guess": 3.0,
                    "scope": "trajectory" if trajectory else "shared_training",
                    "role": "nonnegative_coefficient",
                }
            },
            "m": {
                "initial": {
                    "mode": "map",
                    "expression": "gain*first+second+measured+z+input+cov+time",
                    "parameters": [{"name": "gain", "guess": 2.0}],
                    "state_initials": {"first": "a", "second": "b", "measured": "y"},
                    "public_initials": {
                        "z": "aux",
                        "input": "u",
                        "cov": "dose",
                        "time": "t",
                    },
                }
            },
        }
    }


def test_dependencies_expand_with_distinct_shared_and_trajectory_coordinates():
    model, row = problem()
    before = model.validated.candidate.model_dump(mode="json")
    graph = InitializationGraph.model_validate(graph_payload())
    compiled, guesses, audit = apply_initialization_graph(model, graph)
    assert guesses == {"init_a_value": 2, "init_b_value": 3, "init_m_gain": 2}
    assert audit["shared_parameter_names"] == ["init_a_value", "init_m_gain"]
    assert audit["trajectory_parameter_names"] == ["init_b_value"]
    assert audit["trajectory_root_parameters"] == {"b": "init_b_value"}
    assert audit["legacy_shared_compatible"] is False
    assert audit["topological_order"] == ["a", "b", "m"]
    assert audit["observed_initial_channels"] == {"y": "signal"}
    assert "init_m_value" not in compiled.parameter_names
    assert audit["parameter_domains"]["init_b_value"]["domain"] == "nonnegative"
    assert (
        compiled.validated.candidate.parameters[2].domain is ParameterDomain.NONNEGATIVE
    )
    assert compiled.validated.candidate.parameters[2].bounds is None
    parameters = {"rate": 0.3, **guesses}
    np.testing.assert_allclose(
        trajectory_initial_state(compiled, row, {}, parameters=parameters),
        [2, 3, 29, 4],
    )
    parameters["init_b_value"] = 11
    np.testing.assert_allclose(
        trajectory_initial_state(compiled, row, {}, parameters=parameters),
        [2, 11, 37, 4],
    )
    simulation = simulate_trajectory(compiled, row, parameters, {}, FitConfig())
    assert simulation.success
    np.testing.assert_allclose(simulation.states[:, 0], [2, 11, 37, 4])
    assert model.validated.candidate.model_dump(mode="json") == before
    assert (
        compiled.validated.candidate.state_equations
        == model.validated.candidate.state_equations
    )
    assert (
        compiled.validated.candidate.observation_mappings
        == model.validated.candidate.observation_mappings
    )


def test_only_first_public_sample_enters_graph_and_observed_identity_is_preserved():
    model, row = problem()
    compiled, guesses, _ = apply_initialization_graph(
        model, InitializationGraph.model_validate(graph_payload())
    )
    altered = replace(
        row,
        targets={"signal": np.array([4.0, -1e9, -1e9])},
        auxiliaries={"aux": np.array([5.0, 1e9, 1e9])},
        external_inputs={"u": np.array([6.0, -1e9, -1e9])},
    )
    parameters = {"rate": 0.3, **guesses}
    original = trajectory_initial_state(compiled, row, {}, parameters=parameters)
    np.testing.assert_array_equal(
        trajectory_initial_state(compiled, altered, {}, parameters=parameters), original
    )
    altered = replace(altered, targets={"signal": np.array([8.0, -1e9, -1e9])})
    np.testing.assert_array_equal(
        trajectory_initial_state(compiled, altered, {}, parameters=parameters),
        original + np.array([0, 0, 4, 4]),
    )


def test_shared_graph_exports_replayable_legacy_candidate_with_known_dependency():
    model, row = problem()
    payload = graph_payload(trajectory=False)
    payload["rules"]["a"]["initial"] = {
        "mode": "known",
        "value": 2.0,
        "justification": "Known prepared amount",
    }
    compiled, guesses, audit = apply_initialization_graph(
        model, InitializationGraph.model_validate(payload)
    )
    assert audit["legacy_shared_compatible"]
    assert audit["trajectory_parameter_names"] == []
    assert set(guesses) == {"init_b_value", "init_m_gain"}
    restored = compile_candidate(
        CandidateModel.model_validate_json(
            compiled.validated.candidate.model_dump_json()
        ),
        ValidationContext.model_validate_json(
            compiled.validated.context.model_dump_json()
        ),
    )
    np.testing.assert_allclose(
        trajectory_initial_state(
            restored, row, {}, parameters={"rate": 0.3, **guesses}
        ),
        [2, 3, 29, 4],
    )


def test_symbolic_boundary_derivatives_include_shared_and_trajectory_dependencies():
    pytest.importorskip("casadi")
    from autoformalism.fitting.sensitivity_probe import SymbolicODE

    model, row = problem()
    compiled, guesses, _ = apply_initialization_graph(
        model, InitializationGraph.model_validate(graph_payload())
    )
    system = SymbolicODE(compiled)
    parameters = {"rate": 0.3, **guesses}
    theta = np.array([parameters[name] for name in compiled.parameter_names])
    jacobian = system.initial_sensitivity(row, theta)
    np.testing.assert_allclose(
        jacobian, [[0, 1, 0, 0], [0, 0, 1, 0], [0, 2, 1, 2], [0, 0, 0, 0]]
    )
    for column, name in enumerate(compiled.parameter_names):
        plus, minus = dict(parameters), dict(parameters)
        plus[name] += 1e-4
        minus[name] -= 1e-4
        difference = (
            trajectory_initial_state(compiled, row, {}, parameters=plus)
            - trajectory_initial_state(compiled, row, {}, parameters=minus)
        ) / 2e-4
        np.testing.assert_allclose(jacobian[:, column], difference, atol=1e-8)


def test_transitive_expansion_is_deterministic_across_mapping_order():
    model, row = problem()
    payload = graph_payload()
    payload["rules"]["a"]["initial"] = {
        "mode": "map",
        "expression": "twice*2",
        "state_initials": {"twice": "b"},
    }
    first, guesses, audit = apply_initialization_graph(
        model, InitializationGraph.model_validate(payload)
    )
    payload["rules"] = dict(reversed(list(payload["rules"].items())))
    payload["rules"]["m"]["initial"]["state_initials"] = {
        "measured": "y",
        "second": "b",
        "first": "a",
    }
    second, second_guesses, second_audit = apply_initialization_graph(
        model, InitializationGraph.model_validate(payload)
    )
    assert first.validated.candidate == second.validated.candidate
    assert guesses == second_guesses
    assert (
        audit["topological_order"]
        == second_audit["topological_order"]
        == ["b", "a", "m"]
    )
    np.testing.assert_allclose(
        trajectory_initial_state(first, row, {}, parameters={"rate": 0.3, **guesses}),
        [6, 3, 37, 4],
    )


@pytest.mark.parametrize("state", ["y", "missing"])
def test_observed_or_unknown_rule_rejected(state):
    model, _ = problem()
    payload = graph_payload()
    payload["rules"][state] = {"initial": {"mode": "value", "guess": 0}}
    with pytest.raises(ValueError, match="exactly latent states"):
        apply_initialization_graph(model, InitializationGraph.model_validate(payload))


def test_incomplete_graph_does_not_reinterpret_legacy_fixed_initials():
    model, _ = problem()
    payload = graph_payload()
    del payload["rules"]["a"]
    with pytest.raises(ValueError, match=r"missing=.*a"):
        apply_initialization_graph(model, InitializationGraph.model_validate(payload))


@pytest.mark.parametrize("dependencies", [{"self": "a"}, {"other": "m"}])
def test_self_and_indirect_cycles_rejected(dependencies):
    model, _ = problem()
    payload = graph_payload()
    payload["rules"]["a"]["initial"] = {
        "mode": "map",
        "expression": next(iter(dependencies)),
        "state_initials": dependencies,
    }
    with pytest.raises(ValueError, match="cyclic initializer"):
        apply_initialization_graph(model, InitializationGraph.model_validate(payload))


@pytest.mark.parametrize(
    "expression",
    [
        "loss",
        "rate",
        "a",
        "signal",
        "undeclared",
        "signal[1]",
        "__import__('os')",
    ],
)
def test_undeclared_or_unsafe_expression_rejected(expression):
    model, _ = problem()
    payload = graph_payload()
    payload["rules"]["m"]["initial"] = {"mode": "map", "expression": expression}
    with pytest.raises((ValueError, ModelValidationError)):
        apply_initialization_graph(model, InitializationGraph.model_validate(payload))


@pytest.mark.parametrize(
    "namespace,target",
    [
        ("state_initials", "loss"),
        ("state_initials", "undeclared"),
        ("public_initials", "loss"),
        ("public_initials", "a"),
        ("public_initials", "unavailable"),
    ],
)
def test_reference_destinations_must_be_states_or_public_channels(namespace, target):
    model, _ = problem()
    payload = graph_payload()
    payload["rules"]["m"]["initial"] = {
        "mode": "map",
        "expression": "x",
        namespace: {"x": target},
    }
    with pytest.raises(ValueError, match="unavailable initializer references"):
        apply_initialization_graph(model, InitializationGraph.model_validate(payload))


@pytest.mark.parametrize(
    "declaration",
    [
        {"parameters": [{"name": "x"}, {"name": "x"}]},
        {"parameters": [{"name": "x"}], "state_initials": {"x": "a"}},
        {"state_initials": {"x": "a"}, "public_initials": {"x": "signal"}},
        {"public_initials": {"sin": "signal"}},
    ],
)
def test_ambiguous_symbol_declarations_rejected(declaration):
    payload = graph_payload()
    payload["rules"]["m"]["initial"] = {
        "mode": "map",
        "expression": "x",
        **declaration,
    }
    with pytest.raises(ValidationError):
        InitializationGraph.model_validate(payload)


def test_unused_map_parameter_rejected():
    model, _ = problem()
    payload = graph_payload()
    payload["rules"]["m"]["initial"]["parameters"].append({"name": "unused"})
    with pytest.raises(ValueError, match="unused"):
        apply_initialization_graph(model, InitializationGraph.model_validate(payload))


def test_parameter_namespace_collision_rejected():
    model, _ = problem()
    candidate = model.validated.candidate.model_dump(mode="json")
    candidate["parameters"][0]["name"] = "init_a_value"
    candidate["processes"][0]["expression"] = "init_a_value*a"
    candidate["state_equations"][2]["rhs"] = "-init_a_value*m"
    model = compile_candidate(
        CandidateModel.model_validate(candidate), model.validated.context
    )
    with pytest.raises(ValueError, match="collision"):
        apply_initialization_graph(
            model, InitializationGraph.model_validate(graph_payload())
        )


def test_expansion_size_limit_stops_exponential_dependency_growth():
    states = [f"x{index}" for index in range(10)]
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "large_initial_graph",
            "parent_candidate_id": None,
            "states": [{"name": name, "kind": "latent"} for name in [*states, "y"]],
            "state_equations": [{"state": name, "rhs": "0"} for name in [*states, "y"]],
            "observation_mappings": [{"channel": "signal", "expression": "y"}],
            "initial_conditions": [
                {"state": name, "scope": "global", "fixed_value": 0}
                for name in [*states, "y"]
            ],
        }
    )
    model = compile_candidate(candidate, ValidationContext(targets=("signal",)))
    rules = {"x0": {"initial": {"mode": "value", "guess": 1}}}
    for index in range(1, len(states)):
        rules[states[index]] = {
            "initial": {
                "mode": "map",
                "expression": "previous+previous",
                "state_initials": {"previous": states[index - 1]},
            }
        }
    with pytest.raises(ValueError, match=r"expanded initializer.*limit"):
        apply_initialization_graph(
            model, InitializationGraph.model_validate({"rules": rules})
        )


def test_expansion_depth_limit_stops_deep_composition():
    model, _ = problem()
    payload = graph_payload()
    for state, dependency in (("b", "a"), ("m", "b")):
        payload["rules"][state]["initial"] = {
            "mode": "map",
            "expression": "sin(" * 40 + "root" + ")" * 40,
            "state_initials": {"root": dependency},
        }
    with pytest.raises(ValueError, match="AST depth limit"):
        apply_initialization_graph(model, InitializationGraph.model_validate(payload))


def test_expansion_length_limit_stops_repeated_long_names():
    model, _ = problem()
    payload = graph_payload()
    local = "p" * 110
    payload["rules"]["a"]["initial"] = {
        "mode": "map",
        "expression": local,
        "parameters": [{"name": local}],
    }
    payload["rules"]["m"]["initial"] = {
        "mode": "map",
        "expression": "+".join(["root"] * 40),
        "state_initials": {"root": "a"},
    }
    with pytest.raises(ValueError, match="expression length limit"):
        apply_initialization_graph(model, InitializationGraph.model_validate(payload))


def test_expanded_expression_retains_native_domain_validation():
    model, _ = problem()
    payload = graph_payload()
    payload["rules"]["a"]["initial"] = {
        "mode": "known",
        "value": -1,
        "justification": "Prepared deviation",
    }
    payload["rules"]["m"]["initial"] = {
        "mode": "map",
        "expression": "log(root)",
        "state_initials": {"root": "a"},
    }
    with pytest.raises(ModelValidationError, match="log"):
        apply_initialization_graph(model, InitializationGraph.model_validate(payload))


@pytest.mark.parametrize(
    "edit",
    [
        {"protocol": "unknown-version"},
        {"scope": "trajectory"},
        {"rules": {"a": {"initial": {"mode": "value", "guess": float("nan")}}}},
        {"rules": {"a": {"initial": {"mode": "value", "guess": 0, "bounds": [0, 1]}}}},
    ],
)
def test_version_and_schema_fail_closed(edit):
    with pytest.raises(ValidationError):
        InitializationGraph.model_validate({**graph_payload(), **edit})
