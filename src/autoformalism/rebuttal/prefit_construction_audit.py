"""Reconstruct and describe complete pre-fitting models without numerical fitting."""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.construction import finalize_functional_draft
from autoformalism.expressions import ValidationContext
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.rebuttal import prefit_construction_campaign as campaign
from autoformalism.rebuttal.staged_function_prefit_campaign import (
    deterministic_prefit_audit,
)
from autoformalism.schemas.candidate import StateKind
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_functions import (
    InteractionFunctionObligation,
    InteractionFunctionReply,
    LatentInitialReply,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import shared_process_contract as shared
from autoformalism.search import signed_processes as signed
from autoformalism.search.causal_initialization import compile_initialization_result
from autoformalism.search.staged_function_runner import (
    _accepted_function_record,
    _scientific_review_facts,
    _selected_term,
)
from autoformalism.staged_functions import (
    apply_initial_reply,
    bind_function_reply,
    repair_certified_outer_gain_role,
)
from autoformalism.staged_topology import content_hash, lower_topology
from autoformalism.staging import topology_commitment_sha256

LIMITATION = (
    "Deterministic construction and local preservation only. Nonlinear syntax, "
    "identity mappings and parameter-role changes are review facts, not scientific "
    "verdicts. No fitting, scientific judge, validation observations, test data or "
    "private reference. This compares evidence arms, not old versus new controllers."
)


def _reply(payload: dict) -> InteractionFunctionReply:
    return InteractionFunctionReply.model_validate(
        {key: payload[key] for key in ("expression", "parameters")}
    )


def reconstruct(cell: dict, construction: dict) -> dict:
    """Rebind every slot against frozen declarations and recompile its boundary plan."""
    brief = PublicScientificBrief.model_validate(cell["brief"])
    context = ValidationContext.model_validate(cell["context"])
    source, result = construction["topology"], construction["functions"]
    inventory = tuple(ScientificVariable.model_validate(x) for x in source["inventory"])
    equations = tuple(EquationDefinition.model_validate(x) for x in source["equations"])
    bindings = shared.validate_contract(
        source.get("shared_process_contract"), equations, inventory
    )
    if source.get("shared_process_contract") != result.get("shared_process_contract"):
        raise ValueError("saved shared-process contract differs")
    topology, aliases = lower_topology(brief, inventory, equations, context)
    if topology.model_dump(mode="json") != source["topology"]:
        raise ValueError("saved topology differs from its scientific declarations")
    commitment = topology_commitment_sha256(topology)
    if result["topology_commitment_sha256"] != commitment:
        raise ValueError("function topology commitment differs")
    draft = FunctionalDraft(topology_commitment_sha256=commitment)
    slots = [
        (
            f"term_{i}_{j}",
            shared.decorate_function_term(
                {
                    **_selected_term(
                        equation, term, parameter_identity_policy="interaction_local"
                    ),
                    "deterministic_role_repair_policy": "certified_outer_gain",
                },
                bindings,
            ),
        )
        for i, equation in enumerate(equations)
        for j, term in enumerate(equation.terms)
    ]
    local = result["provider_visible_accepted_functions"]
    accepted = result["accepted_functions"]
    batches = result["batch_term_audits"]
    if not len(slots) == len(local) == len(accepted) == len(batches):
        raise ValueError("saved slot coverage differs from the complete topology")
    repairs = []
    for (identifier, selected), final, stored, batch in zip(
        slots, local, accepted, batches, strict=True
    ):
        if final["selected_term"] != selected or stored["selected_term"] != selected:
            raise ValueError("saved function obligation differs from frozen topology")
        if batch["interaction_id"] != identifier:
            raise ValueError("batch slot ordering differs")
        obligation = InteractionFunctionObligation.model_validate(
            selected["functional_obligation"]
        )
        original = _reply(batch["batch_function"])
        automatic = signed.automatic_function(selected)
        if automatic is not None and (
            original != automatic
            or _reply(final) != automatic
            or batch.get("runtime_generated") != "signed_process_identity"
            or batch["atomic_repair_attempted"]
        ):
            raise ValueError(
                "automatic process function differs from signed declaration"
            )
        original_valid = False
        prepared = None
        try:
            shared.validate_function(
                original.expression, selected.get("shared_process_use")
            )
            prepared, _ = repair_certified_outer_gain_role(
                original,
                set(selected["sources"]),
                outer_weight_sign=selected["outer_weight_sign"],
            )
            bind_function_reply(
                topology, draft, identifier, prepared, context, aliases, obligation
            )
            original_valid = True
        except (ValueError, ModelValidationError):
            pass
        final_reply = _reply(final)
        shared.validate_function(
            final_reply.expression, selected.get("shared_process_use")
        )
        if original_valid != batch["batch_accepted"]:
            raise ValueError("recorded batch validity differs from revalidation")
        if original_valid and (
            batch["atomic_repair_attempted"] or prepared != final_reply
        ):
            raise ValueError("already-valid batch slot was changed or sent for repair")
        draft, _ = bind_function_reply(
            topology, draft, identifier, final_reply, context, aliases, obligation
        )
        if _accepted_function_record(draft, identifier, selected, aliases) != stored:
            raise ValueError("saved canonical function differs from accepted reply")
        before_roles = {p.name: p.role.value for p in original.parameters}
        after_roles = {p.name: p.role.value for p in final_reply.parameters}
        repairs.append(
            {
                "interaction_id": identifier,
                "selected_term": selected,
                "original_batch": original.model_dump(mode="json"),
                "accepted_reply": final_reply.model_dump(mode="json"),
                "canonical_function": stored,
                "batch_valid": original_valid,
                "preserved_without_atomic_call": original_valid,
                "atomic_repair_attempted": batch["atomic_repair_attempted"],
                "batch_error": batch["batch_error"],
                "rhs_changed": original.expression != final_reply.expression,
                "parameter_roles_changed": before_roles != after_roles,
                "same_name_role_changes": {
                    name: {"before": before_roles[name], "after": after_roles[name]}
                    for name in before_roles.keys() & after_roles.keys()
                    if before_roles[name] != after_roles[name]
                },
                "deterministic_role_repairs": batch["deterministic_role_repairs"],
                "atomic_deterministic_role_repairs": batch[
                    "atomic_deterministic_role_repairs"
                ],
            }
        )
    for state in topology.states:
        if state.kind is StateKind.LATENT:
            draft = apply_initial_reply(
                topology,
                draft,
                state.name,
                LatentInitialReply(initial={"fixed_value": 0.0}),
                context,
                aliases,
            )
    if draft.model_dump(mode="json") != result["draft"]:
        raise ValueError("saved functional draft differs from rebound slots")
    expansion = finalize_functional_draft(topology, draft, context)
    initialization = result["initialization"]
    compiled = compile_initialization_result(
        expansion.candidate, context, initialization
    )
    if compiled.validated.candidate.model_dump(mode="json") != result["candidate"]:
        raise ValueError("saved complete model differs from reconstructed initializer")
    certificate = deterministic_prefit_audit(brief, source, result, context=context)
    certificate["checks"].update(
        topology_reconstructed=True,
        functions_rebound_from_local_replies=True,
        initializer_recompiled=True,
        valid_batch_slots_preserved=True,
    )
    certificate["passed"] = all(certificate["checks"].values())
    return {
        "certificate": certificate,
        "function_slots": repairs,
        "scientific_review_facts": _scientific_review_facts(brief, accepted),
        "handoff": {
            "schema_version": "prefit-canonical-handoff-1",
            "candidate": result["candidate"],
            "context": initialization["context"],
            "initialization": initialization,
            "parameter_values_are_optimizer_guesses": True,
            "candidate_sha256": content_hash(result["candidate"]),
        },
    }


def _inputs(root: Path, plan: dict, task: dict) -> tuple[dict | None, dict, list[dict]]:
    directory = root / "results" / task["task_id"]
    identity = campaign._identity(plan, task)
    construction = campaign._terminal(directory / "construction.json", identity)
    calls = campaign._cache_records(directory / "calls", identity)
    binding = {
        "identity": identity,
        "construction_sha256": (construction or {}).get("artifact_sha256"),
        "call_ledger_sha256": content_hash(calls),
    }
    return construction, binding, calls


def read_audit(directory: Path, binding: dict) -> dict | None:
    """Reject stale audits after any sealed construction or provider-cache change."""
    audit = campaign._terminal(directory / "audit.json", binding["identity"])
    if audit is not None and any(audit.get(k) != v for k, v in binding.items()):
        raise ValueError("audit inputs differ from construction or cached calls")
    return audit


def audit_task(root: Path, plan: dict, task: dict) -> dict | None:
    """Persist one bounded deterministic audit; resume never calls a proposer."""
    if plan["protocol"] != campaign.CONSTRUCTION_ONLY_PROTOCOL:
        raise ValueError("audit requires the construction-only protocol")
    construction, binding, _ = _inputs(root, plan, task)
    if construction is None:
        return None
    directory = root / "results" / task["task_id"]
    prior = read_audit(directory, binding)
    if prior is None:
        record = {
            **binding,
            "schema_version": "prefit-construction-audit-1",
            "task": task,
            "status": "not_constructed",
            "error": None,
            "certificate": None,
            "function_slots": [],
            "handoff": None,
            "parameter_fitting_performed": False,
            "scientific_judge_called": False,
            "validation_data_opened": False,
            "test_data_opened": False,
            "private_reference_opened": False,
            "limitation": LIMITATION,
        }
        if construction["status"] == "complete":
            try:
                record.update(reconstruct(plan["cells"][task["cell"]], construction))
                record["status"] = (
                    "passed" if record["certificate"]["passed"] else "failed"
                )
                if record["status"] != "passed":
                    record["handoff"] = None
            except (ValueError, KeyError, TypeError, ModelValidationError) as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        else:
            record["error"] = (construction.get("functions") or {}).get("error") or (
                construction.get("topology") or {}
            ).get("error")
        # Publish the readable model before making this task terminal. A killed
        # worker must not leave a terminal audit pointing to a partial report.
        _write_model(directory / "model.md", render_model(record, construction))
        campaign._sealed_write(directory / "audit.json", record)
        prior = read_audit(directory, binding)
    else:
        _write_model(directory / "model.md", render_model(prior, construction))
    return prior


def _write_model(path: Path, text: str) -> None:
    """Atomically replace a derived report without touching sealed inputs."""
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def render_model(audit: dict, construction: dict) -> str:
    """Render generated declarations as quoted data, with no provider reasoning text."""
    lines = [
        f"# {audit['task']['task_id']}",
        "",
        f"Audit: **{audit['status']}**.",
        "",
        LIMITATION,
        "",
    ]
    if audit["error"]:
        lines += [
            "Error (recorded diagnostic):",
            "",
            "    " + audit["error"].replace("\n", "\n    "),
            "",
        ]
    functions = construction.get("functions") or {}
    candidate = functions.get("candidate") or {}
    lines += ["## Assembled equations", ""]
    for equation in candidate.get("state_equations", []):
        lines += [f"    d({equation['state']})/dt = {equation['rhs']}", ""]
    for process in candidate.get("processes", []):
        lines += [f"    {process['name']} = {process['expression']}", ""]
    sections = {
        "Variables and equation types": (construction.get("topology") or {}).get(
            "inventory", []
        ),
        "Canonical derivative equations": candidate.get("state_equations", []),
        "Canonical algebraic processes": candidate.get("processes", []),
        "Observation mappings": candidate.get("observation_mappings", []),
        "Parameter roles and domains": candidate.get("parameters", []),
        "Causal initialization plan (parameters not fitted)": (
            functions.get("initialization") or {}
        ).get("plan"),
        "Term obligations and local repair changes": audit["function_slots"],
        "Scientific review facts (not certification)": audit.get(
            "scientific_review_facts"
        ),
        "Deterministic checks": audit["certificate"],
    }
    for title, payload in sections.items():
        # Quote arbitrary generated prose as data.
        lines += [
            f"## {title}",
            "",
            *[
                "    " + s
                for s in json.dumps(payload, indent=2, sort_keys=True).splitlines()
            ],
            "",
        ]
    return "\n".join(lines)


def render_index(summary: dict) -> str:
    """Link every planned task, including incomplete ones, from one review index."""
    lines = [
        "# Fresh construction review",
        "",
        LIMITATION,
        "",
        f"Campaign status: **{summary['status']}**.",
        "",
        "| Cell / seed / arm | Construction | Audit | Report |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        label = f"{row['cell']} / {row['seed']} / {row['arm']}"
        link = (
            f"[model](results/{row['task_id']}/model.md)"
            if row["model_report"]
            else "pending"
        )
        lines.append(
            f"| {label} | {row['construction_status']} | "
            f"{row['audit_status']} | {link} |"
        )
    return "\n".join(lines) + "\n"


def build_summary(root: Path, plan: dict) -> dict:
    """Retain all planned task denominators, including partial and failed work."""
    rows = []
    for task in plan["tasks"]:
        construction, binding, calls = _inputs(root, plan, task)
        directory = root / "results" / task["task_id"]
        audit = read_audit(directory, binding)
        slots = (audit or {}).get("function_slots", [])
        rows.append(
            {
                **task,
                "construction_status": (construction or {}).get("status", "pending"),
                "audit_status": (audit or {}).get("status", "pending"),
                "audit_error": (audit or {}).get("error"),
                "candidate_sha256": ((audit or {}).get("handoff") or {}).get(
                    "candidate_sha256"
                ),
                "valid_slots_preserved": sum(
                    x["preserved_without_atomic_call"] for x in slots
                ),
                "atomic_repair_slots": sum(x["atomic_repair_attempted"] for x in slots),
                "rhs_changed_slots": sum(x["rhs_changed"] for x in slots),
                "parameter_role_changed_slots": sum(
                    x["parameter_roles_changed"] for x in slots
                ),
                "model_report": str(directory / "model.md") if audit else None,
                "cost": campaign._cost(calls),
            }
        )
    arms = {}
    for arm in campaign.ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        arms[arm] = {
            "expected": len(selected),
            "constructed": sum(
                r["construction_status"] == "complete" for r in selected
            ),
            "audit_passed": sum(r["audit_status"] == "passed" for r in selected),
            "audit_failed": sum(r["audit_status"] == "failed" for r in selected),
            "not_constructed": sum(
                r["audit_status"] == "not_constructed" for r in selected
            ),
            "pending_audit": sum(r["audit_status"] == "pending" for r in selected),
            **{
                key: sum(r[key] for r in selected)
                for key in (
                    "valid_slots_preserved",
                    "atomic_repair_slots",
                    "rhs_changed_slots",
                    "parameter_role_changed_slots",
                )
            },
            **{
                key: sum(r["cost"][key] for r in selected)
                for key in (
                    "physical_requests",
                    "observed_tokens",
                    "budget_charge",
                    "requests_with_unknown_usage",
                    "provider_seconds",
                )
            },
        }
    return {
        "protocol": plan["protocol"],
        "plan_sha256": plan["artifact_sha256"],
        "status": "complete"
        if all(r["audit_status"] != "pending" for r in rows)
        else "partial",
        "construction_complete": all(
            r["construction_status"] != "pending" for r in rows
        ),
        "arms": arms,
        "rows": rows,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
        "limitation": LIMITATION,
    }
