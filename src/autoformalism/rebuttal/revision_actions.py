"""Versioned, checkpointed component actions for the multiround pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from autoformalism.expressions import RestrictedParser, compile_candidate
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.revision_decision import (
    expressions,
    interaction_contract,
    nonlinear_target_paths,
    parameter_aliases,
    translate_names,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.staged_topology import content_hash

ACTION_PROMPT = """Revise the displayed continuous-time model for the stated objective.
Return revisions for components you change and keep_components for companions you
intentionally leave unchanged. Cover every pending component exactly once across
these two lists. Do not resend provisionally retained components.
For function actions, preserve the MULTISET of outer signs and nonparameter source
sets of the displayed additive interactions, not just the whole-equation source
union. Reordering is allowed. A new offset adds an interaction; coupling another
variable into an existing term changes an interaction. Both require topology scope.
Topology actions may change those interactions within the existing inventory, but
must preserve the public required pathway. Do not change unselected equations,
states, mappings or initializers. Parent parameter aliases are already declared and
keep their roles. Declare only new parameters. Use role null for a direct gain or
offset, and a specific qualitative role for an internal shape parameter.
Address the numerical evidence without treating a failed fit as proof that the
model is wrong. Finite trajectories alone are not evidence of a useful fit. Do not
remove public nonlinear requirements just to obtain a finite solution. Uncertified
state denominators are potential domain risks, not proven reached singularities.
Do not numerically tune parameters, add ranges, or emit prose or a complete model.
"""


def request_revision_action(
    *,
    client,
    route,
    candidate,
    selected,
    context,
    scientific_context,
    numerical_feedback,
    round_index,
    failure_checkpoint: Path,
    transaction_path: Path,
    nonlinear_targets: tuple[str, ...] = (),
):
    """Resume retained components without spending calls on them again.

    Native fitting is deliberately outside this transaction. No draft becomes a
    committed candidate until every selected component and the aggregate pass.
    """
    from autoformalism.rebuttal import staged_multiround_feedback_campaign as campaign

    class ActionReply(StrictSchema):
        revisions: tuple[campaign.ComponentRevision, ...] = Field(
            default=(), max_length=4
        )
        keep_components: tuple[Identifier, ...] = Field(default=(), max_length=4)

        @model_validator(mode="after")
        def disjoint_actions(self):
            names = [item.component for item in self.revisions] + list(
                self.keep_components
            )
            if not names or len(names) != len(set(names)):
                raise ValueError("one change or keep action per component is required")
            return self

    parent_sha = campaign._candidate_hash(candidate)
    contract_sha = content_hash(
        [
            parent_sha,
            route,
            selected,
            scientific_context,
            nonlinear_targets,
            sorted(context.forcing_channels),
        ]
    )
    transaction = {
        "schema_version": "component-action-transaction-1",
        "status": "pending",
        "contract_sha256": contract_sha,
        "parent_sha256": parent_sha,
        "route": route,
        "selected": list(selected),
        "pending": list(selected),
        "kept": [],
        "accepted": {},
        "attempts": [],
        "diagnostic": None,
        "provisional_candidate": candidate.model_dump(mode="json"),
        "aliases": parameter_aliases(candidate),
    }
    if transaction_path.exists():
        saved = json.loads(transaction_path.read_text())
        if saved["status"] == "pending" or (
            saved.get("completed_round_index") == round_index
            and saved["contract_sha256"] == contract_sha
        ):
            if saved["contract_sha256"] != contract_sha:
                raise ValueError("pending revision transaction contract differs")
            transaction = saved
    provisional = CandidateModel.model_validate(transaction["provisional_candidate"])
    pending = transaction["pending"]
    accepted = transaction["accepted"]
    kept = transaction["kept"]
    attempts = transaction["attempts"]
    aliases = transaction["aliases"]

    def persist() -> None:
        transaction["provisional_candidate"] = provisional.model_dump(mode="json")
        atomic_json(transaction_path, transaction)
        atomic_json(
            failure_checkpoint,
            {
                "schema_version": "component-revision-failures-3",
                "parent_sha256": parent_sha,
                "route": route,
                "selected_components": list(selected),
                "pending_components": pending,
                "provisionally_retained_components": accepted,
                "kept_components": kept,
                "attempts": [
                    item for item in attempts if item["round_index"] == round_index
                ],
            },
        )

    def finish():
        """Complete a recovered transaction without another provider call."""
        nonlocal provisional
        compile_candidate(provisional, context)
        payload = provisional.model_dump(mode="json")
        payload["parent_candidate_id"] = candidate.candidate_id
        if accepted:
            provisional = CandidateModel.model_validate(payload)
        transaction["status"] = "committed" if accepted else "no_change"
        transaction["completed_round_index"] = round_index
        persist()
        return provisional, {
            "schema_version": "component-revision-record-4",
            "status": transaction["status"],
            "route": route,
            "parent_sha256": parent_sha,
            "kept_components": kept,
            "accepted_component_revisions": accepted,
            "audit": campaign._combined_revision_audit(
                candidate, provisional, accepted, context
            ),
            "attempts": [
                item for item in attempts if item["round_index"] == round_index
            ],
            "transaction_attempt_count": len(attempts),
        }

    if not pending:
        return finish()

    start_attempt = sum(item["round_index"] == round_index for item in attempts)
    for attempt in range(start_attempt, client.settings.attempts_per_step):
        # Preserve aliases across retries/rounds; new identities get new aliases.
        occupied = set(aliases.values()) | set(expressions(provisional))
        for parameter in provisional.parameters:
            if parameter.name not in aliases:
                index = len(aliases) + 1
                while f"par_{index:03d}" in occupied:
                    index += 1
                aliases[parameter.name] = f"par_{index:03d}"
                occupied.add(aliases[parameter.name])
        registry = {p.name: p for p in (*candidate.parameters, *provisional.parameters)}
        equation_kinds = {
            **{item.state: "ode_rhs" for item in provisional.state_equations},
            **{item.name: "algebraic" for item in provisional.processes},
        }
        request = {
            "schema_version": "component-revision-request-4",
            "route": route,
            "scientific_context": scientific_context,
            "objective": (
                "improve fit while preserving required mechanisms"
                if route == "function_refinement"
                else "address the recorded failure within the authorized scope"
            ),
            "selected_pending_components": [
                {
                    "component": name,
                    "equation_kind": equation_kinds[name],
                    "current_expression": expressions(provisional)[name],
                    "immutable_signed_interactions": interaction_contract(
                        provisional, name
                    )
                    if route != "topology_revision"
                    else None,
                }
                for name in pending
            ],
            "provisionally_retained_components": sorted(accepted),
            "intentionally_kept_components": kept,
            "all_current_equations": expressions(provisional),
            "equation_kinds": equation_kinds,
            "available_parent_parameters": {
                name: p.role.value for name, p in registry.items()
            },
            "available_nonparameter_symbols": sorted(
                set(expressions(provisional))
                | set(context.forcing_channels)
                | {context.time_symbol}
            ),
            "public_nonlinearity_targets": list(nonlinear_targets),
            "numerical_feedback": numerical_feedback,
            "runtime_diagnostic": transaction["diagnostic"],
        }
        record = client.call(
            system=ACTION_PROMPT,
            user=json.dumps(translate_names(request, aliases), sort_keys=True),
            response_model=ActionReply,
            step=f"round_{round_index}_{route}",
            attempt=attempt,
        )
        raw: Any = None
        results = []
        try:
            raw = visible_response(record)
            # Short aliases are presentation only; canonical identities stay frozen.
            reply = ActionReply.model_validate(
                translate_names(raw, {v: k for k, v in aliases.items()})
            )
            returned = set(reply.keep_components) | {
                item.component for item in reply.revisions
            }
            if returned - set(pending):
                raise ValueError(
                    f"unrequested components: {sorted(returned - set(pending))}"
                )
            for component in list(pending):
                item = next(
                    (
                        value
                        for value in reply.revisions
                        if value.component == component
                    ),
                    None,
                )
                try:
                    if component not in returned:
                        raise campaign.RevisionContractError(
                            "MISSING_COMPONENT_ACTION",
                            f"pending component {component} was omitted",
                            component=component,
                            allowed_actions=["revise", "keep"],
                        )
                    unchanged = component in reply.keep_components
                    if item is not None:
                        # Preserve parent identities pruned by earlier partial edits.
                        used = set(
                            RestrictedParser()
                            .parse(item.expression, location=component)
                            .symbols
                        )
                        payload = provisional.model_dump(mode="json")
                        existing = {p.name for p in provisional.parameters}
                        payload["parameters"].extend(
                            p.model_dump(mode="json")
                            for name, p in registry.items()
                            if name in used and name not in existing
                        )
                        available = CandidateModel.model_validate(payload)
                        try:
                            revised, audit = campaign.apply_component_revision(
                                available,
                                campaign.ComponentRevisionReply(revisions=(item,)),
                                context,
                                selected=(component,),
                                route=route,
                                detailed_contract=True,
                            )
                        except ValueError as exc:
                            if (
                                str(exc)
                                != "revision did not change the executable functions"
                            ):
                                raise
                            unchanged = True
                    if unchanged:
                        all_findings = campaign.state_dependent_denominator_findings(
                            provisional
                        )
                        findings = [
                            finding
                            for finding in all_findings
                            if finding["component"] == component
                        ]
                        if findings:
                            raise campaign.RevisionContractError(
                                "KEEP_UNSAFE_COMPONENT",
                                "kept component has unresolved domain findings",
                                component=component,
                                findings=findings,
                            )
                        kept.append(component)
                        results.append(
                            {"component": component, "accepted": True, "action": "keep"}
                        )
                    else:
                        coverage = nonlinear_target_paths(revised, nonlinear_targets)
                        if not all(coverage.values()):
                            raise campaign.RevisionContractError(
                                "PUBLIC_NONLINEAR_OBLIGATION_LOST",
                                "revision removed all non-affine generated-variable "
                                "syntax from a required target path",
                                component=component,
                                target_coverage=coverage,
                                interpretation=(
                                    "necessary syntax condition only, "
                                    "not a scientific score"
                                ),
                            )
                        provisional = revised
                        accepted[component] = {
                            "revision": item.model_dump(mode="json"),
                            "audit": audit,
                        }
                        results.append(
                            {
                                "component": component,
                                "accepted": True,
                                "action": "revise",
                                "audit": audit,
                            }
                        )
                    pending.remove(component)
                except (ValueError, TypeError, KeyError) as exc:
                    results.append(
                        {
                            "component": component,
                            "accepted": False,
                            "diagnostic": {
                                **campaign._named_revision_diagnostic(exc),
                                "component": component,
                            },
                        }
                    )
        except (ValueError, TypeError, KeyError) as exc:
            results.append(
                {
                    "component": None,
                    "accepted": False,
                    "diagnostic": campaign._named_revision_diagnostic(exc),
                }
            )
        attempts.append(
            {
                "round_index": round_index,
                "attempt": attempt,
                "request_hash": record["request_hash"],
                "accepted": not pending,
                "rejected_response": raw,
                "parameter_aliases": dict(aliases),
                "component_results": results,
                "pending_components_after": list(pending),
            }
        )
        transaction["diagnostic"] = {
            "pending_components": list(pending),
            "provisionally_retained_components": sorted(accepted),
            "component_failures": [
                item["diagnostic"] for item in results if not item["accepted"]
            ],
        }
        persist()
        if not pending:
            return finish()
    raise campaign.RevisionContractError(
        "COMPONENT_REVISION_ATTEMPTS_EXHAUSTED",
        f"bounded component revision exhausted for {route}",
        pending_components=list(pending),
        provisionally_retained_components=sorted(accepted),
        transaction_path=str(transaction_path),
        final_diagnostic=transaction["diagnostic"],
    )
