"""Portable scoped fitting receipts, constructed without observations or rollouts.

This adapter verifies completed function construction and exercises only the
existing M43 compilation boundary. It never optimizes, selects a prefix duration,
opens datasets, or changes the numerical adapter's supported model class.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from autoformalism.expressions import CompiledModel
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.construction_graph_formulas import (
    compile_graph_formula_result,
)
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-function-fit-handoff-1"
COMPILE_METHOD = "autoformalism.fitting.prefix_initialization._compile"


def _numerical_readiness(problem: PrefixFitProblem) -> dict[str, Any]:
    """Exercise the exact scoped compiler, without creating a fitting oracle."""
    try:
        from autoformalism.fitting.prefix_initialization import (
            NumericalPolicy,
            _compile,
        )

        system, _, audit, _, _ = _compile(problem, NumericalPolicy())
    except ImportError as exc:
        return {"status": "unavailable", "method": COMPILE_METHOD, "reason": str(exc)}
    except ValueError as exc:
        return {"status": "unsupported", "method": COMPILE_METHOD, "reason": str(exc)}
    return {
        "status": "passed",
        "method": COMPILE_METHOD,
        "reason": None,
        "state_names": list(system.model.state_names),
        "parameter_names": list(system.model.parameter_names),
        "trajectory_root_parameters": audit["trajectory_root_parameters"],
    }


def _forecast_readiness(
    model: CompiledModel, contract: GraphConstructionContract
) -> dict[str, Any]:
    """Report static information-access limits without using trajectory arrays."""
    context = model.validated.context
    forcing = set(model.validated.forcing_symbols)
    external = forcing & set(context.external_inputs)
    forbidden = forcing - set(context.external_inputs) - set(context.fixed_covariates)
    blockers = []
    if forbidden:
        blockers.append(
            {
                "code": "future_observed_forcing_unavailable",
                "channels": sorted(forbidden),
                "auxiliary_channels": sorted(forbidden & set(context.auxiliaries)),
                "reason": (
                    "The scoped prefix adapter permits only declared future "
                    "external inputs and fixed covariates as rollout forcing."
                ),
            }
        )
    policy = contract.prefix_policy
    if policy is None:
        blockers.append(
            {
                "code": "prefix_policy_unresolved",
                "reason": (
                    "Prefix duration and permitted future-input schedules remain "
                    "fitting-policy decisions; no forecast is claimed."
                ),
            }
        )
    else:
        if invalid := set(policy.future_inputs) - set(context.external_inputs):
            blockers.append(
                {"code": "nonexternal_future_input", "channels": sorted(invalid)}
            )
        if missing := external - set(policy.future_inputs):
            blockers.append(
                {"code": "future_input_not_authorized", "channels": sorted(missing)}
            )
    status = "structurally_eligible"
    if policy is None:
        status = "policy_unresolved"
    if forbidden or (policy is not None and blockers):
        status = "blocked"
    return {
        "status": status,
        "required_future_external_inputs": sorted(external),
        "future_auxiliary_channels": sorted(forbidden & set(context.auxiliaries)),
        "blockers": blockers,
        "forecast_performed": False,
        "scientific_accuracy_established": False,
    }


def _validate_numerical_receipt(
    receipt: Mapping[str, Any], model: CompiledModel, audit: dict[str, Any]
) -> dict[str, Any]:
    """Check a recorded compile outcome without reproducing its environment."""
    receipt = dict(receipt)
    if receipt.get("status") == "passed":
        expected = {
            "status": "passed",
            "method": COMPILE_METHOD,
            "reason": None,
            "state_names": list(model.state_names),
            "parameter_names": list(model.parameter_names),
            "trajectory_root_parameters": audit["trajectory_root_parameters"],
        }
        if receipt != expected:
            raise ValueError("saved numerical compilation identity differs")
    elif (
        receipt.get("status") not in {"unsupported", "unavailable"}
        or set(receipt) != {"status", "method", "reason"}
        or receipt.get("method") != COMPILE_METHOD
        or not isinstance(receipt.get("reason"), str)
        or not receipt["reason"]
    ):
        raise ValueError("invalid saved numerical compilation receipt")
    return receipt


def _handoff(
    result: Mapping[str, Any],
    provenance: Mapping[str, Any],
    numerical_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild deterministic fields, optionally retaining a recorded compile."""
    result = json.loads(json.dumps(dict(result), allow_nan=False))
    provenance = json.loads(json.dumps(dict(provenance), allow_nan=False))
    if (
        result.get("protocol") != "scientific-staged-functions-3"
        or result.get("status") != "complete"
        or result.get("complete_model") is not True
        or result.get("initialization_policy") != "initialization_graph"
        or result.get("fitting_backend") != "prefix-fit-problem-1"
        or result.get("candidate") is not None
        or result.get("parameter_fitting_performed") is not False
        or result.get("test_data_opened") is not False
        or result.get("private_reference_opened") is not False
    ):
        raise ValueError("requires a complete construction-only functions-3 result")
    initialization = result.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("missing sealed initializer formula result")
    problem = PrefixFitProblem.model_validate(result.get("prefix_fit_problem"))
    if initialization.get("prefix_fit_problem") != problem.model_dump(mode="json"):
        raise ValueError("top-level prefix fit problem differs from initialization")
    contract = GraphConstructionContract.model_validate(
        result.get("initialization_graph_contract")
    )
    handoff = GraphDependencyHandoff.model_validate(
        result.get("initialization_graph_handoff")
    )
    receipt = initialization.get("initialization_dependency_contract", {})
    compiled = compile_graph_formula_result(
        problem.base_candidate,
        problem.context,
        initialization,
        contract=contract,
        handoff=handoff,
        state_aliases=receipt.get("state_aliases"),
    )
    audit = initialization["audit"]
    numerical = _validate_numerical_receipt(
        _numerical_readiness(problem)
        if numerical_receipt is None
        else numerical_receipt,
        compiled,
        audit,
    )
    forecast = _forecast_readiness(compiled, contract)
    if numerical["status"] != "passed":
        forecast["blockers"].append(
            {
                "code": "numerical_compile_not_passed",
                "reason": numerical["reason"],
            }
        )
        forecast["status"] = "blocked"
    payload = {
        "protocol": PROTOCOL,
        "provenance": provenance,
        "source_function_result_sha256": content_hash(result),
        "source_initialization_result_sha256": initialization["result_sha256"],
        "prefix_fit_problem": problem.model_dump(mode="json"),
        "initialization_dependency_contract": receipt,
        "initialization_audit": audit,
        "runtime_observed_initials": initialization.get(
            "runtime_observed_initials", []
        ),
        "parameter_scopes": {
            "shared_dynamics": [p.name for p in problem.base_candidate.parameters],
            "shared_initialization": audit["shared_parameter_names"],
            "trajectory_roots": audit["trajectory_root_parameters"],
            "dependent_initial_states": [
                state
                for state, rule in problem.graph.rules.items()
                if rule.initial.mode == "map"
            ],
        },
        "prefix_policy": initialization["prefix_policy"],
        "window_policy_unresolved": initialization["window_policy_unresolved"],
        "legacy_shared_compatible": audit["legacy_shared_compatible"],
        "legacy_fitter_dispatch_allowed": False,
        "readiness": {
            "construction": {"status": "verified"},
            "numerical_compile": numerical,
            "forecast": forecast,
        },
        "parameter_fitting_performed": False,
        "numerical_rollout_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "scientific_correctness_established": False,
    }
    payload["artifact_sha256"] = content_hash(payload)
    return payload


def build_handoff(
    result: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a complete functions-3 result and return a sealed M43 receipt.

    ``provenance`` contains caller-owned source identities and retained scientific
    evidence. Both it and the complete problem are embedded as JSON, so the
    exported file does not depend on local absolute paths to interpret the model.
    Compilation readiness is separate from prospective forecast capability and
    from numerical reliability, neither of which is established here.
    """
    return _handoff(result, provenance)


def verify_handoff(
    result: Mapping[str, Any],
    exported: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay structural receipts without retrying environment-specific compile.

    A saved unavailable or unsupported compiler result remains recorded even when
    the reviewing host has different optional dependencies. The original seal,
    exact problem, graph, scopes, provenance and forecast limitations must match.
    """
    exported = json.loads(json.dumps(dict(exported), allow_nan=False))
    payload = {
        key: value for key, value in exported.items() if key != "artifact_sha256"
    }
    if exported.get("artifact_sha256") != content_hash(payload):
        raise ValueError("fitting handoff artifact seal differs")
    readiness = exported.get("readiness")
    numerical = (
        readiness.get("numerical_compile") if isinstance(readiness, dict) else None
    )
    if not isinstance(numerical, dict):
        raise ValueError("missing saved numerical compilation receipt")
    expected = _handoff(result, provenance, numerical)
    if exported != expected:
        raise ValueError("fitting handoff differs from deterministic source replay")
    return exported
