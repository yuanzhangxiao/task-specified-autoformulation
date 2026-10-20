"""Opt-in gain correction for saved graph construction, without calls or fits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from autoformalism.construction import finalize_functional_draft
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.fitter import _parameter_variable
from autoformalism.fitting.models import FitConfig
from autoformalism.research.interaction_fit_handoff import (
    PROTOCOL as FIT_PROTOCOL,
)
from autoformalism.research.interaction_fit_handoff import build_handoff, verify_handoff
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.construction_graph_formulas import (
    _resolve,
    _result_fields,
    compile_graph_formula_result,
)
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
)
from autoformalism.search.process_gains import lower_candidate_gains
from autoformalism.search.shared_process_contract import validate_contract
from autoformalism.search.signed_processes import conversion_value
from autoformalism.staged_topology import content_hash, lower_topology

POLICY = "scoped-declared-process-gains-1"
HANDOFF_PROTOCOL = "construction-scoped-process-gain-handoff-1"


def _plain(value: Mapping[str, Any]) -> dict:
    """Copy JSON data without retaining mutable caller-owned containers."""
    return json.loads(json.dumps(dict(value), allow_nan=False))


def _sealed(value: Mapping[str, Any]) -> dict:
    """Verify the opt-in lowering receipt before consuming its candidate."""
    value = _plain(value)
    digest = value.pop("artifact_sha256", None)
    if digest != content_hash(value) or value.get("protocol") != POLICY:
        raise ValueError("scoped gain lowering seal or policy differs")
    return {**value, "artifact_sha256": digest}


def _guesses(candidate: CandidateModel, context: ValidationContext) -> dict[str, float]:
    """Use the graph constructor's role-policy starts without any numerical work."""
    model = compile_candidate(candidate, context)
    settings = FitConfig()
    result = {}
    for parameter in candidate.parameters:
        variable = _parameter_variable(model, parameter, settings)
        result[parameter.name] = (variable.start_lower + variable.start_upper) / 2
    return result


def lower_saved_function_gains(
    result: Mapping[str, Any], source: Mapping[str, Any], *, policy: str
) -> dict:
    """Replay saved dynamics and add promised consumer gains under explicit opt-in.

    ``source`` is the campaign's prepared input (brief, case, function_source).
    Complete dynamics are required; initializer construction may have failed.
    Existing complete graphs and all dependency declarations are retained exactly.
    The caller remains responsible for verifying the source archive/file chain.
    """
    if policy != POLICY:
        raise ValueError("explicit scoped process gain policy required")
    result, source = _plain(result), _plain(source)
    frozen = source["function_source"]
    if (
        result.get("protocol") != "scientific-staged-functions-3"
        or result.get("status") not in {"complete", "failed"}
        or result.get("initialization_policy") != "initialization_graph"
        or result.get("source_topology_result_sha256") != content_hash(frozen)
        or result.get("shared_process_contract")
        != frozen.get("shared_process_contract")
        or any(
            result.get(key) is not False
            for key in (
                "parameter_fitting_performed",
                "test_data_opened",
                "private_reference_opened",
            )
        )
    ):
        raise ValueError("saved graph function result differs from frozen source")
    brief = PublicScientificBrief.model_validate(source["brief"])
    context = ValidationContext.model_validate(source["case"]["context"])
    inventory = tuple(ScientificVariable.model_validate(v) for v in frozen["inventory"])
    equations = tuple(EquationDefinition.model_validate(e) for e in frozen["equations"])
    contract = frozen.get("shared_process_contract")
    validate_contract(contract, equations, inventory)
    topology, aliases = lower_topology(brief, inventory, equations, context)
    if (
        not frozen.get("complete_topology")
        or topology.model_dump(mode="json") != frozen["topology"]
    ):
        raise ValueError("frozen topology differs from scientific declarations")
    candidate = finalize_functional_draft(
        topology, FunctionalDraft.model_validate(result["draft"]), context
    ).candidate
    graph_contract = GraphConstructionContract.model_validate(
        frozen["initialization_graph_contract"]
    )
    graph_handoff = GraphDependencyHandoff.model_validate(
        frozen["initialization_graph_handoff"]
    )
    for key in ("initialization_graph_contract", "initialization_graph_handoff"):
        if result.get(key) != frozen[key]:
            raise ValueError("saved initializer dependencies differ from frozen source")
    graph_aliases = {
        name: aliases.get(name, name)
        for name in graph_contract.state_declaration_sha256
    }
    _resolve(
        compile_candidate(candidate, context),
        context,
        graph_contract,
        graph_handoff,
        graph_aliases,
    )
    graph = None
    if result["status"] == "complete":
        initial = result["initialization"]
        compile_graph_formula_result(
            candidate,
            context,
            initial,
            contract=graph_contract,
            handoff=graph_handoff,
            state_aliases=graph_aliases,
        )
        if result.get("prefix_fit_problem") != initial["prefix_fit_problem"]:
            raise ValueError("saved prefix problem differs from initializer receipt")
        graph = initial["graph"]
    covariates = [
        row.get("fixed_covariates", {})
        for row in (source.get("training_evidence") or {}).get("trajectories", [])
    ]
    # Validate conversion syntax even when no training rows were retained. This
    # unit-valued schema probe is never exported as a physical conversion/start.
    allowed = dict.fromkeys(context.fixed_covariates, 1.0)
    for binding in (contract or {}).get("bindings", []):
        for use in binding["signed_declaration"]["uses"]:
            if use.get("conversion") is not None:
                conversion_value(use["conversion"], allowed)
    lowered, suggested, processes = lower_candidate_gains(
        candidate,
        context,
        contract,
        "declared_or_independent",
        fixed_covariates=covariates,
        state_aliases=aliases,
        preserve_signed_amplitudes=True,
    )
    payload = {
        "protocol": POLICY,
        "source_function_result_sha256": content_hash(result),
        "source_input_sha256": content_hash(source),
        "source_base_candidate": candidate.model_dump(mode="json"),
        "base_candidate": lowered.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "parameter_guesses": _guesses(lowered, context),
        "initialization_graph_contract": graph_contract.model_dump(mode="json"),
        "initialization_graph_handoff": graph_handoff.model_dump(mode="json"),
        "state_aliases": graph_aliases,
        "preserved_initialization_graph": graph,
        "gain_compilation": {
            "policy": "declared_or_independent",
            "signed_amplitude_policy": "preserve_real_domain",
            "source_contract_sha256": content_hash(contract),
            "processes": processes,
            "candidate_sha256": content_hash(lowered.model_dump(mode="json")),
            "legacy_suggested_gain_starts": suggested,
            "start_policy": "graph_constructor_role_midpoints",
            "initializer_dependencies_unchanged": True,
            "conservation_certified": False,
        },
        "parameter_fitting_performed": False,
        "numerical_rollout_performed": False,
    }
    payload["artifact_sha256"] = content_hash(payload)
    return payload


def rebind_saved_initializer(
    lowered: Mapping[str, Any], initializer_result: Mapping[str, Any]
) -> dict:
    """Migrate accepted formulas onto corrected dynamics, preserving the exact graph."""
    lowered, initial = _sealed(lowered), _plain(initializer_result)
    context = ValidationContext.model_validate(lowered["context"])
    contract = GraphConstructionContract.model_validate(
        lowered["initialization_graph_contract"]
    )
    handoff = GraphDependencyHandoff.model_validate(
        lowered["initialization_graph_handoff"]
    )
    compile_graph_formula_result(
        CandidateModel.model_validate(lowered["source_base_candidate"]),
        context,
        initial,
        contract=contract,
        handoff=handoff,
        state_aliases=lowered["state_aliases"],
    )
    candidate = CandidateModel.model_validate(lowered["base_candidate"])
    _, fields = _result_fields(
        candidate,
        context,
        compile_candidate(candidate, context),
        contract,
        handoff,
        initial["choices"],
        lowered["state_aliases"],
    )
    if (
        fields["graph"] != initial["graph"]
        or fields["graph"] != lowered["preserved_initialization_graph"]
    ):
        raise ValueError("gain migration changed the saved initializer graph")
    migration = {
        "policy": POLICY,
        "source_initializer_result_sha256": initial["result_sha256"],
        "gain_lowering_sha256": lowered["artifact_sha256"],
        "new_provider_calls": 0,
    }
    result = {
        **{key: value for key, value in initial.items() if key != "result_sha256"},
        **fields,
        "identity": content_hash(migration),
        "gain_migration": migration,
    }
    result["result_sha256"] = content_hash(result)
    compile_graph_formula_result(
        candidate,
        context,
        result,
        contract=contract,
        handoff=handoff,
        state_aliases=lowered["state_aliases"],
    )
    return result


def _function_view(lowered: dict, initial: Mapping[str, Any]) -> dict:
    """Expose the existing receipt validator without pretending the draft changed."""
    initial = _plain(initial)
    if (
        initial["base_candidate"] != lowered["base_candidate"]
        or initial["base_context"] != lowered["context"]
    ):
        raise ValueError("initializer receipt does not use gain-corrected dynamics")
    if (
        initial["prefix_fit_problem"]["parameter_guesses"]
        != lowered["parameter_guesses"]
    ):
        raise ValueError("scoped gain parameter starts differ from graph policy")
    saved_graph = lowered["preserved_initialization_graph"]
    if saved_graph is not None and initial["graph"] != saved_graph:
        raise ValueError("saved initializer graph changed")
    return {
        "protocol": "scientific-staged-functions-3",
        "status": "complete",
        "complete_model": True,
        "initialization_policy": "initialization_graph",
        "fitting_backend": "prefix-fit-problem-1",
        "candidate": None,
        "initialization": initial,
        "prefix_fit_problem": initial["prefix_fit_problem"],
        "initialization_graph_contract": lowered["initialization_graph_contract"],
        "initialization_graph_handoff": lowered["initialization_graph_handoff"],
        "gain_lowering_sha256": lowered["artifact_sha256"],
        "parameter_fitting_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


def build_scoped_gain_handoff(
    lowered: Mapping[str, Any],
    initializer_result: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict:
    """Export full corrected scopes and separate static readiness, without fitting."""
    lowered = _sealed(lowered)
    value = build_handoff(
        _function_view(lowered, initializer_result), provenance=provenance
    )
    value.update(protocol=HANDOFF_PROTOCOL, gain_lowering=lowered)
    value.pop("artifact_sha256")
    return {**value, "artifact_sha256": content_hash(value)}


def verify_scoped_gain_handoff(
    lowered: Mapping[str, Any],
    initializer_result: Mapping[str, Any],
    exported: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict:
    """Replay a portable correction receipt without rerunning numerical compilation."""
    lowered, value = _sealed(lowered), _plain(exported)
    payload = {key: val for key, val in value.items() if key != "artifact_sha256"}
    if (
        value.get("artifact_sha256") != content_hash(payload)
        or value.get("protocol") != HANDOFF_PROTOCOL
        or value.get("gain_lowering") != lowered
    ):
        raise ValueError("scoped gain handoff seal or lowering differs")
    payload.pop("gain_lowering")
    payload["protocol"] = FIT_PROTOCOL
    ordinary = {**payload, "artifact_sha256": content_hash(payload)}
    verify_handoff(
        _function_view(lowered, initializer_result), ordinary, provenance=provenance
    )
    return value
