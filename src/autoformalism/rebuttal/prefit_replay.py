"""Read-only replay of historical function slots and causal initializers.

Every slot uses its captured historical context. A newly admissible response
never advances the historical trajectory or replaces a later recorded decision.
No trajectory table, fit result, optimizer, or scientific judge is used.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from autoformalism.construction import finalize_functional_draft
from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_functions import (
    InteractionFunctionObligation,
    InteractionFunctionReply,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.causal_initialization import (
    InitializerChoice,
    _normalize_choice,
    _rule,
)
from autoformalism.search.staged_function_runner import _selected_term
from autoformalism.staged_functions import (
    bind_function_reply,
    repair_certified_outer_gain_role,
)
from autoformalism.staged_topology import content_hash, lower_topology
from autoformalism.staging import topology_commitment_sha256

PROTOCOL = "prefit-response-replay-1"


class ReplayCase(StrictSchema):
    """One independently checkable response with a frozen historical parent."""

    case_id: str
    kind: Literal["function", "initializer"]
    task_id: str
    component: str
    request_hash: str
    source_attempt: int
    historical_accepted: bool | None
    historical_acceptance_scope: str
    context: ValidationContext
    public_request: dict[str, Any]
    original_reply: Any
    source_topology: dict[str, Any]
    draft: dict[str, Any]
    selected_id: str | None = None
    selected_term: dict[str, Any] | None = None


class Diagnosis(StrictSchema):
    """Deterministic local validity, deliberately separate from scientific merit."""

    valid: bool
    scope: Literal["frozen_local_construction_context"] = (
        "frozen_local_construction_context"
    )
    diagnostics: tuple[dict[str, str], ...] = ()
    normalizations: tuple[dict[str, Any], ...] = ()
    canonical_reply: dict[str, Any] | None = None
    result_sha256: str | None = None
    protected_context_preserved: bool = False
    scientific_status: Literal["not_assessed"] = "not_assessed"


def sealed_read(path: Path) -> dict:
    """Read an immutable, content-addressed artifact."""
    value = json.loads(path.read_text())
    if value.get("artifact_sha256") != content_hash(
        {k: v for k, v in value.items() if k != "artifact_sha256"}
    ):
        raise ValueError(f"artifact digest differs: {path}")
    return value


def sealed_write(path: Path, value: dict) -> dict:
    """Write once, or verify an identical deterministic rerun."""
    result = {**value, "artifact_sha256": content_hash(value)}
    if path.exists():
        if sealed_read(path) != result:
            raise ValueError(f"frozen artifact differs: {path}")
    else:
        atomic_json(path, result)
    return result


def _topology(case: ReplayCase):
    source = case.source_topology
    brief = PublicScientificBrief.model_validate(case.public_request["public_brief"])
    topology, aliases = lower_topology(
        brief,
        tuple(ScientificVariable.model_validate(v) for v in source["inventory"]),
        tuple(EquationDefinition.model_validate(v) for v in source["equations"]),
        case.context,
    )
    if topology.model_dump(mode="json") != source["topology"]:
        raise ValueError("captured topology differs from declarations")
    return topology, aliases


def _brief_without_packet(payload: dict) -> dict:
    """Only the typed scientific brief enters topology lowering."""
    return {k: v for k, v in payload.items() if k != "training_observations"}


def diagnose(case: ReplayCase, reply: Any, *, normalize: bool = True) -> Diagnosis:
    """Apply production slot validators without fitting or advancing history."""
    # Historical parent failures are experiment errors, never proposer diagnoses.
    typed_request = {
        **case.public_request,
        "public_brief": _brief_without_packet(case.public_request["public_brief"]),
    }
    topology, aliases = _topology(
        case.model_copy(update={"public_request": typed_request})
    )
    draft = FunctionalDraft.model_validate(case.draft)
    if draft.topology_commitment_sha256 != topology_commitment_sha256(topology):
        raise ValueError("captured draft belongs to a different topology")
    if case.kind == "function":
        selected = case.selected_term
        if selected is None or not re.fullmatch(
            r"term_\d+_\d+", case.selected_id or ""
        ):
            raise ValueError("missing selected function slot")
        equation_index, term_index = map(int, case.selected_id.split("_")[1:])
        equation = EquationDefinition.model_validate(
            case.source_topology["equations"][equation_index]
        )
        obligation = InteractionFunctionObligation.model_validate(
            selected["functional_obligation"]
        )
        expected = _selected_term(
            equation,
            equation.terms[term_index],
            parameter_identity_policy=obligation.parameter_identity_policy,
        )
        if any(selected.get(k) != v for k, v in expected.items()):
            raise ValueError("captured function slot differs from source equations")
        for field, source_field in (
            ("frozen_inventory", "inventory"),
            ("frozen_equation_sketch", "equations"),
        ):
            if case.public_request[field] != case.source_topology[source_field]:
                raise ValueError("captured public function context differs")
    else:
        base = finalize_functional_draft(topology, draft, case.context).candidate
        model = compile_candidate(base, case.context)
        if case.component not in set(model.state_names) - set(
            model.direct_state_observation_channels
        ):
            raise ValueError("initializer must select a latent state")
        expected_rhs = next(
            e.rhs for e in base.state_equations if e.state == case.component
        )
        if case.public_request["selected_state"] != {
            "name": case.component,
            "rhs": expected_rhs,
        }:
            raise ValueError("captured initializer state RHS differs")
        allowed = sorted(
            set(case.context.targets)
            | set(case.context.forcing_channels)
            | {case.context.time_symbol}
        )
        if case.public_request["allowed_initial_symbols"] != allowed:
            raise ValueError("captured initial information differs")
        previous = case.public_request.get("accepted_initializers", {})
        rules = {
            name: _rule(InitializerChoice.model_validate(value))
            for name, value in previous.items()
        }
        if case.component in rules:
            raise ValueError("captured initializer was already accepted")
        apply_initialization_plan(model, LatentInitializationPlan(rules=rules))
    repairs = []
    try:
        if case.kind == "function":
            function = InteractionFunctionReply.model_validate(reply)
            if (
                selected.get("deterministic_role_repair_policy")
                == "certified_outer_gain"
            ):
                function, role_repairs = repair_certified_outer_gain_role(
                    function,
                    set(selected["sources"]),
                    outer_weight_sign=selected["outer_weight_sign"],
                )
                repairs.extend(r.model_dump(mode="json") for r in role_repairs)
            result, canonical = bind_function_reply(
                topology,
                draft,
                case.selected_id,
                function,
                case.context,
                aliases,
                InteractionFunctionObligation.model_validate(
                    selected["functional_obligation"]
                ),
            )
            protected = [
                f.model_dump(mode="json")
                for f in result.interaction_functions
                if f.interaction_id != case.selected_id
            ]
            if protected != [
                f.model_dump(mode="json") for f in draft.interaction_functions
            ]:
                raise ValueError(
                    "PROTECTED_CONTEXT_CHANGED: accepted sibling functions changed"
                )
            canonical_reply = canonical.model_dump(mode="json")
            result_value = result.model_dump(mode="json")
        else:
            choice = InitializerChoice.model_validate(reply)
            if normalize:
                reserved = (
                    set(model.state_names)
                    | set(model.parameter_names)
                    | set(case.context.targets)
                    | set(case.context.forcing_channels)
                    | set(case.context.unavailable_observed_channels)
                    | {case.context.time_symbol}
                    | {p.name for p in base.processes}
                )
                choice, repair = _normalize_choice(choice, case.component, reserved)
                if repair:
                    repair["original_expression"] = reply["initial"]["expression"]
                    repairs.append(repair)
            rules[case.component] = _rule(choice)
            compiled, _, _ = apply_initialization_plan(
                model, LatentInitializationPlan(rules=rules)
            )
            result_value = compiled.validated.candidate.model_dump(mode="json")
            if (
                result_value["state_equations"]
                != base.model_dump(mode="json")["state_equations"]
                or result_value["processes"]
                != base.model_dump(mode="json")["processes"]
            ):
                raise ValueError(
                    "PROTECTED_CONTEXT_CHANGED: initialization changed dynamics"
                )
            canonical_reply = choice.model_dump(mode="json")
        return Diagnosis(
            valid=True,
            normalizations=tuple(repairs),
            canonical_reply=canonical_reply,
            result_sha256=content_hash(result_value),
            protected_context_preserved=True,
        )
    except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
        return Diagnosis(
            valid=False,
            diagnostics=tuple(_diagnostics(exc, case.component)),
            normalizations=tuple(repairs),
        )


def _diagnostics(exc: Exception, component: str) -> list[dict[str, str]]:
    if isinstance(exc, ModelValidationError):
        return [
            {
                "code": d.code,
                "component": component,
                "location": d.location,
                "message": d.message,
            }
            for d in exc.diagnostics
        ]
    if isinstance(exc, ValidationError):
        return [
            {
                "code": "SCHEMA_ERROR",
                "component": component,
                "location": ".".join(map(str, d["loc"])),
                "message": d["msg"],
            }
            for d in exc.errors()
        ]
    message = str(exc)[:4000]
    match = re.match(r"([A-Z][A-Z0-9_]+)(?: at |:)", message)
    code = (
        match[1]
        if match
        else (
            "SOURCE_MISMATCH"
            if message.startswith("source mismatch:")
            else "INITIALIZER_SYMBOLS"
            if message.startswith("initializer has unavailable")
            else "PARAMETER_COLLISION"
            if message.startswith("parameter collides")
            else "DETERMINISTIC_VALIDATION_ERROR"
        )
    )
    return [
        {
            "code": code,
            "component": component,
            "location": component,
            "message": message,
        }
    ]


def expression_facts(case: ReplayCase, reply: Any) -> dict:
    """Expose syntax facts; do not invent a corrected scientific expression."""
    if case.kind == "initializer":
        spec = reply.get("initial", {}) if isinstance(reply, dict) else {}
        allowed = case.public_request["allowed_initial_symbols"]
        facts = {
            "requested_quantity": "initial_value",
            "allowed_initial_symbols": allowed,
            "future_observations_allowed": False,
            "other_state_dynamics_editable": False,
        }
    else:
        spec = reply if isinstance(reply, dict) else {}
        allowed = case.selected_term["sources"]
        facts = {
            "requested_quantity": "single_function_slot",
            "required_sources": allowed,
            "outer_weight_sign": case.selected_term["outer_weight_sign"],
            "functional_obligation": case.selected_term["functional_obligation"],
            "topology_editable": False,
        }
    try:
        parsed = RestrictedParser().parse(
            spec.get("expression", ""), location=case.component
        )
        parameters = {p["name"] for p in spec.get("parameters", [])}
        facts.update(
            referenced_symbols=sorted(parsed.symbols),
            declared_parameters=sorted(parameters),
            unavailable_symbols=sorted(parsed.symbols - parameters - set(allowed)),
            unused_parameters=sorted(parameters - parsed.symbols),
        )
    except (ValueError, TypeError, KeyError, ModelValidationError):
        facts["symbol_analysis_available"] = False
    return facts


def _prefix(draft: dict, selected_id: str) -> dict:
    """Retain only historically accepted slots before the selected ordered slot."""

    def order(name: str) -> tuple[int, ...]:
        return tuple(map(int, name.removeprefix("term_").split("_")))

    return {
        **draft,
        "interaction_functions": [
            v
            for v in draft["interaction_functions"]
            if order(v["interaction_id"]) < order(selected_id)
        ],
        "latent_initials": [],
    }


def replay(source: Path, output: Path) -> dict:
    """Freeze a case corpus from a terminal matched-construction campaign."""
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("replay output must be outside the historical source")
    ledger = {}

    def read(relative: str, *, sealed: bool = False) -> dict:
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("source path escapes historical root")
        data = path.read_bytes()
        ledger[relative] = hashlib.sha256(data).hexdigest()
        value = json.loads(data)
        if sealed and value.get("artifact_sha256") != content_hash(
            {k: v for k, v in value.items() if k != "artifact_sha256"}
        ):
            raise ValueError("historical artifact digest differs")
        return value

    plan = read("plan.json", sealed=True)
    if (
        plan["protocol"] != "prefit-matched-construction-1"
        or plan.get("test_data_opened") is not False
        or plan.get("private_reference_opened") is not False
    ):
        raise ValueError("expected a public-only matched-construction source")
    cases, rows, unsupported, historical = [], [], [], []
    for task in plan["tasks"]:
        task_id = task["task_id"]
        if not re.fullmatch(r"[A-Za-z0-9_]+", task_id):
            raise ValueError("invalid historical task ID")
        prefix = f"results/{task_id}"
        result = read(f"{prefix}/construction.json", sealed=True)
        namespace = content_hash([plan["artifact_sha256"], task])
        if (
            result["identity"] != namespace
            or result.get("test_data_opened") is not False
        ):
            raise ValueError("historical construction identity differs")
        context = ValidationContext.model_validate(
            plan["cells"][task["cell"]]["context"]
        )
        functions = result.get("functions")
        historical.append({"task_id": task_id, "status": result["status"]})
        events = {
            e["request_hash"]: e
            for group in [
                result["topology"].get("events", []),
                (functions or {}).get("events", []),
            ]
            for e in group
            if "request_hash" in e
        }
        init_path = source / prefix / "functions/initialization/state.json"
        if init_path.exists():
            state = read(f"{prefix}/functions/initialization/state.json")
            if state.get("state_sha256") != content_hash(
                {k: v for k, v in state.items() if k != "state_sha256"}
            ):
                raise ValueError("historical initializer digest differs")
            events.update({e["request_hash"]: e for e in state["attempts"]})
        keys = result["request_hashes"]
        if len(keys) != len(set(keys)) or set(events) - set(keys):
            raise ValueError("historical request inventory differs")
        for key in sorted(keys):
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError("invalid historical request hash")
            call = read(f"{prefix}/calls/{key}.json")
            if (
                call["request_hash"] != key
                or content_hash(call["request"]) != key
                or call["request"]["namespace"] != namespace
            ):
                raise ValueError("historical cached request identity differs")
            event = events.get(key, {})
            descriptor = {
                "task_id": task_id,
                "request_hash": key,
                "step": call["step"],
                "historical_accepted": event.get("accepted"),
                "historical_error": event.get("error"),
            }
            payload = json.loads(
                call["request"]["body"]["messages"][1]["content"].split("\n", 1)[1]
            )
            try:
                response = visible_response(call)
            except (ValueError, TypeError, KeyError) as exc:
                unsupported.append(
                    {
                        **descriptor,
                        "reason": "no_complete_visible_json",
                        "error": str(exc)[:1000],
                    }
                )
                continue
            common = {
                "task_id": task_id,
                "request_hash": key,
                "source_attempt": call["attempt"],
                "historical_accepted": event.get("accepted"),
                "context": context.model_dump(mode="json"),
                "public_request": payload,
                "source_topology": {
                    k: result["topology"][k]
                    for k in ("inventory", "equations", "topology")
                },
            }
            units = []
            if functions and call["step"].startswith("causal_initial_"):
                units.append(
                    {
                        "kind": "initializer",
                        "component": payload["selected_state"]["name"],
                        "original_reply": response,
                        "draft": functions["draft"],
                        "historical_acceptance_scope": "initializer",
                    }
                )
            elif functions and "selected_term" in payload:
                identifier = (
                    call["step"]
                    .removeprefix("atomic_repair_")
                    .removeprefix("function_")
                )
                units.append(
                    {
                        "kind": "function",
                        "component": f"{payload['selected_term']['lhs']}/{identifier}",
                        "selected_id": identifier,
                        "selected_term": payload["selected_term"],
                        "original_reply": response,
                        "draft": _prefix(functions["draft"], identifier),
                        "historical_acceptance_scope": "function_slot",
                    }
                )
            elif (
                functions
                and "selected_equation" in payload
                and isinstance(response, dict)
            ):
                terms = payload["selected_equation"]["terms"]
                supplied = response.get("functions", [])
                if not isinstance(supplied, list) or len(supplied) != len(terms):
                    unsupported.append(
                        {**descriptor, "reason": "batch_shape_requires_separate_replay"}
                    )
                    continue
                eq_index = int(call["step"].removeprefix("equation_functions_"))
                for index, (selected, value) in enumerate(
                    zip(terms, supplied, strict=True)
                ):
                    identifier = f"term_{eq_index}_{index}"
                    local_payload = {
                        k: v for k, v in payload.items() if k != "selected_equation"
                    }
                    local_payload["selected_term"] = selected
                    units.append(
                        {
                            "kind": "function",
                            "component": f"{selected['lhs']}/{identifier}",
                            "selected_id": identifier,
                            "selected_term": selected,
                            "original_reply": value,
                            "draft": _prefix(functions["draft"], identifier),
                            "historical_acceptance_scope": (
                                "batch_schema_and_count_only"
                            ),
                            "public_request": local_payload,
                        }
                    )
            else:
                unsupported.append(
                    {**descriptor, "reason": "topology_stage_not_replayed"}
                )
            for unit in units:
                data = {**common, **unit}
                case = ReplayCase(case_id=content_hash(data), **data)
                case = case.model_copy(
                    update={
                        "case_id": content_hash(
                            case.model_dump(mode="json", exclude={"case_id"})
                        )
                    }
                )
                before, after = (
                    diagnose(case, case.original_reply, normalize=False),
                    diagnose(case, case.original_reply),
                )
                cases.append(case.model_dump(mode="json"))
                rows.append(
                    {
                        "case_id": case.case_id,
                        **descriptor,
                        "component": case.component,
                        "historical_acceptance_scope": case.historical_acceptance_scope,
                        "before_normalization": before.model_dump(mode="json"),
                        "after_normalization": after.model_dump(mode="json"),
                        "normalization_recovered": not before.valid and after.valid,
                    }
                )
    for relative, digest in ledger.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("source path escapes historical root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("historical source changed during replay")
    return sealed_write(
        output,
        {
            "protocol": PROTOCOL,
            "source_root": str(source),
            "source_plan_sha256": plan["artifact_sha256"],
            "source_model_settings": plan["config"]["model_settings"],
            "source_file_sha256": ledger,
            "runtime_source_sha256": runtime_source_hash(),
            "cases": cases,
            "rows": rows,
            "unreplayed_requests": unsupported,
            "historical_constructions": historical,
            "counts": {
                "historical_tasks": len(historical),
                "historical_complete": sum(
                    t["status"] == "complete" for t in historical
                ),
                "replayed_slots": len(rows),
                "valid_before_normalization": sum(
                    r["before_normalization"]["valid"] for r in rows
                ),
                "valid_after_normalization": sum(
                    r["after_normalization"]["valid"] for r in rows
                ),
                "normalization_recovered": sum(
                    r["normalization_recovered"] for r in rows
                ),
                "unreplayed_requests": len(unsupported),
                "remaining_diagnostic_codes": dict(
                    Counter(
                        d["code"]
                        for r in rows
                        for d in r["after_normalization"]["diagnostics"]
                    )
                ),
            },
            "parameter_fitting_performed": False,
            "new_llm_calls_made": False,
            "scientific_judge_called": False,
            "trajectory_tables_opened": False,
            "test_data_opened": False,
            "private_reference_opened": False,
            "limitation": (
                "Local function/initializer admissibility only. Historical parents "
                "advance only through recorded accepted work. Topology and malformed "
                "batch requests are retained as unreplayed; scientific merit and "
                "counterfactual construction success are not inferred."
            ),
        },
    )
