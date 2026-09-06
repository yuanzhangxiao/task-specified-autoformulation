"""Fixed scientific agenda and bounded local repairs for topology construction."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    StagedTopologyClient,
    atomic_json,
    visible_response,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
    VariableReply,
    equation_reply_model,
)
from autoformalism.search.staged_topology_prompts import (
    render_equation_topology_system_prompt,
    render_equation_topology_user_prompt,
    render_variable_identification_system_prompt,
    render_variable_identification_user_prompt,
)
from autoformalism.staged_topology import (
    content_hash,
    freeze_inventory,
    lower_topology,
    merge_variable_reply,
    public_structure_checks,
    validate_equation,
)


def scientific_agenda(brief: PublicScientificBrief) -> tuple[dict[str, Any], ...]:
    """Visit shared mechanisms together, then complete each public target."""
    return tuple(
        {
            "purpose": "identify variables for this mechanism",
            "requirements": [item.id],
            "targets": list(item.targets),
        }
        for item in brief.requirements
    ) + tuple(
        {
            "purpose": "complete variables needed to generate this target",
            "requirements": [],
            "targets": [item.name],
        }
        for item in brief.public_variables
        if item.data_role == "target"
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _variables(inventory: tuple[ScientificVariable, ...]) -> str:
    return _json([item.model_dump(mode="json") for item in inventory])


def run_staged_topology(
    brief: PublicScientificBrief,
    context: ValidationContext,
    client: StagedTopologyClient,
    output: Path,
    *,
    initial_inventory: tuple[ScientificVariable, ...] | None = None,
) -> dict[str, Any]:
    """Build one topology without functions, numerical data, or a scientific judge."""
    inventory: tuple[ScientificVariable, ...] = initial_inventory or ()
    equations: tuple[EquationDefinition, ...] = ()
    events: list[dict[str, Any]] = []
    brief_json = brief.model_dump_json()
    output.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        atomic_json(
            output / "progress.json",
            {
                "inventory": [item.model_dump(mode="json") for item in inventory],
                "equations": [item.model_dump(mode="json") for item in equations],
                "events": events,
            },
        )

    def request(
        stage: str,
        system: str,
        renderer: Callable[[str | None], str],
        model: type[StrictSchema],
        validate: Callable[[Any], Any],
    ) -> Any:
        diagnostic: str | None = None
        for attempt in range(client.settings.attempts_per_step):
            rejected: object = None
            record = client.call(
                system=system,
                user=renderer(diagnostic),
                response_model=model,
                step=stage,
                attempt=attempt,
            )
            try:
                rejected = visible_response(record)
                reply = model.model_validate(rejected)
                accepted = validate(reply)
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)[:6000]
                if rejected is None:
                    choices = record.get("raw_response", {}).get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        visible = choices[0].get("message", {}).get("content")
                        if isinstance(visible, str):
                            rejected = visible
                diagnostic = _json({"rejected_response": rejected, "error": error})
                events.append(
                    {
                        "step": stage,
                        "attempt": attempt,
                        "accepted": False,
                        "request_hash": record["request_hash"],
                        "error": error,
                    }
                )
                checkpoint()
                continue
            events.append(
                {
                    "step": stage,
                    "attempt": attempt,
                    "accepted": True,
                    "request_hash": record["request_hash"],
                }
            )
            return accepted
        raise ValueError(f"bounded local repair exhausted for {stage}")

    status = "failed"
    failure: str | None = None
    revision: object = None
    topology = None
    aliases: dict[str, str] = {}
    agenda = scientific_agenda(brief) if initial_inventory is None else ()
    try:
        for index, item in enumerate(agenda):
            is_final_item = index == len(agenda) - 1

            def accept_variables(
                reply: VariableReply,
                parent=inventory,
                final_item=is_final_item,
            ) -> tuple[ScientificVariable, ...]:
                merged = merge_variable_reply(brief, parent, reply)
                return freeze_inventory(brief, merged) if final_item else merged

            inventory = request(
                f"variables_{index}",
                render_variable_identification_system_prompt(),
                lambda diagnostic,
                item=item,
                inventory=inventory: render_variable_identification_user_prompt(
                    public_brief_json=brief_json,
                    agenda_json=_json(item),
                    inventory_json=_variables(inventory),
                    diagnostics_json=diagnostic,
                ),
                VariableReply,
                accept_variables,
            )
            checkpoint()
        inventory = freeze_inventory(brief, inventory)
        allowed = tuple(item.name for item in inventory if item.definition != "unused")
        model = equation_reply_model(
            allowed, maximum_terms=brief.limits.terms_per_equation
        )
        for selected in inventory:
            if selected.definition not in {"differential", "algebraic"}:
                continue

            def accept_equation(
                reply: Any, selected=selected, equations=equations
            ) -> Any:
                if reply.inventory_revision is not None:
                    return reply
                definition = EquationDefinition(
                    name=selected.name,
                    definition=selected.definition,
                    terms=reply.terms,
                )
                validate_equation(inventory, equations, definition, brief.limits)
                return definition

            accepted = request(
                f"equation_{selected.name}",
                render_equation_topology_system_prompt(),
                lambda diagnostic,
                selected=selected,
                equations=equations: render_equation_topology_user_prompt(
                    public_brief_json=brief_json,
                    agenda_json=_json(
                        {
                            "purpose": "define the selected variable",
                            "targets": list(context.targets),
                            "requirements": [item.id for item in brief.requirements],
                        }
                    ),
                    inventory_json=_variables(inventory),
                    selected_lhs_json=_json(
                        {"name": selected.name, "definition": selected.definition}
                    ),
                    equation_sketch_json=_json(
                        [item.model_dump(mode="json") for item in equations]
                    ),
                    allowed_sources_json=_json(allowed),
                    diagnostics_json=diagnostic,
                ),
                model,
                accept_equation,
            )
            if not isinstance(accepted, EquationDefinition):
                revision = accepted.inventory_revision.model_dump(mode="json")
                status = "inventory_revision_requested"
                break
            equations += (accepted,)
            checkpoint()
        else:
            topology, aliases = lower_topology(brief, inventory, equations, context)
            status = "complete"
    except ValueError as exc:
        failure = str(exc)[:6000]
    checks = public_structure_checks(brief, equations)
    result = {
        "protocol": "scientific-staged-topology-1",
        "status": status,
        "error": failure,
        "complete_topology": topology is not None,
        "public_structure_checks_passed": topology is not None
        and all(item["passed"] for item in checks),
        "public_structure_checks": checks,
        "inventory": [item.model_dump(mode="json") for item in inventory],
        "inventory_sha256": content_hash(
            [item.model_dump(mode="json") for item in inventory]
        ),
        "equations": [item.model_dump(mode="json") for item in equations],
        "topology": topology.model_dump(mode="json") if topology else None,
        "generated_auxiliary_aliases": aliases,
        "inventory_revision": revision,
        "events": events,
        "physical_requests": len(client.records),
        "budget_charge": sum(item.get("budget_charge", 0) for item in client.records),
        "observed_total_tokens": sum(
            item.get("observed_total_tokens") or 0 for item in client.records
        ),
        "unmeasured_requests": sum(
            item.get("observed_total_tokens") is None for item in client.records
        ),
        "provider_seconds": sum(
            item.get("latency_seconds", 0) for item in client.records
        ),
        "test_data_opened": False,
        "private_reference_opened": False,
        "function_generation_performed": False,
        "diagnostic_inventory_supplied": initial_inventory is not None,
        "parameter_fitting_performed": False,
    }
    checkpoint()
    atomic_json(output / "result.json", result)
    return result
