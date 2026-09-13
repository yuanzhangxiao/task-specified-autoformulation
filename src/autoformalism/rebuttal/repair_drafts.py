"""Versioned typed repair drafts; omission retains, withdrawal cancels explicitly."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting.initialization import LatentInitializationPlan
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.repair_evidence import model_hash
from autoformalism.rebuttal.repair_feedback import PROTOCOL as FEEDBACK_PROTOCOL
from autoformalism.rebuttal.repair_transactions import (
    SYSTEM_PROMPT as LEGACY_PROMPT,
)
from autoformalism.rebuttal.repair_transactions import (
    EquationEdit,
    InitializerEdit,
    MappingEdit,
    RepairAction,
    _local_edit,
    commit_action,
    shared_declarations,
)
from autoformalism.rebuttal.revision_decision import (
    expressions,
    parameter_aliases,
    translate_names,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.staged_topology import content_hash

PROTOCOL = "repair-action-2"
KINDS = {
    "equation": ("equations", "component"),
    "mapping": ("mappings", "channel"),
    "initializer": ("initializers", "state"),
    "remove": ("remove", None),
}


class PendingAction(StrictSchema):
    """A draft operation, not a request to remove a committed model variable."""

    kind: Literal["equation", "mapping", "initializer", "remove"]
    target: Identifier


class RepairActionV2(StrictSchema):
    """Unchanged model components are implicit; only draft withdrawal is explicit."""

    scope: Literal["function", "model", "no_change"]
    hypothesis: str = Field(default="", max_length=800)
    equations: tuple[EquationEdit, ...] = Field(default=(), max_length=6)
    remove: tuple[Identifier, ...] = Field(default=(), max_length=2)
    mappings: tuple[MappingEdit, ...] = Field(default=(), max_length=3)
    initializers: tuple[InitializerEdit, ...] = Field(default=(), max_length=4)
    withdraw: tuple[PendingAction, ...] = Field(default=(), max_length=15)

    @model_validator(mode="after")
    def scoped(self):
        raw = self.model_dump(mode="json", exclude={"withdraw"})
        RepairAction.model_validate(raw)
        withdrawn = {(w.kind, w.target) for w in self.withdraw}
        if len(withdrawn) != len(self.withdraw):
            raise ValueError("duplicate withdrawal")
        touched = {
            (kind, entry[key] if key else entry)
            for kind, (field, key) in KINDS.items()
            for entry in raw[field]
        }
        if withdrawn & touched:
            raise ValueError("cannot replace and withdraw the same draft action")
        if self.scope == "no_change" and self.withdraw:
            raise ValueError("no_change abandons the whole draft; omit withdrawals")
        return self


SYSTEM_PROMPT = (
    LEGACY_PROMPT[: LEGACY_PROMPT.index("Keep lists for known")]
    + """\
Unmentioned model components stay unchanged: there is no keep field.
Omitted draft edits remain provisional, never silently disappear. On retry replace
only invalid entries, or cancel one with:
"withdraw": [{"kind": "initializer", "target": "z"}]
Kinds are equation, mapping, initializer, remove. Withdrawal
only cancels a draft operation; remove deletes a model variable. Valid equations
remain provisional while other operations are repaired. Pending actions identify
the exact operation requiring correction, not necessarily an equation.
Initializer edits apply only to latent dynamic states after the proposed mapping
and type edits. Directly measured states use their public initial measurement;
algebraic variables and inputs have no fitted state initializer. causal_map=null
requests a training-fitted shared latent value, NOT withdrawal. Use explicit
withdrawal to cancel an initializer. scope=no_change abandons the entire draft.
Read initialization_facts: latent-boundary policies and collocation optimizer
initialization are different. A collocation timeout does not mean a latent
initial-state policy is missing. mode=value is already training-fitted; guess=0
is only an optimizer start. Do not numerically tune initial values yourself.
Read recent_actions.actual_effects before proposing a repair: repeating the same
equation, mapping or initializer is an executable no-op, not a recovered model.
Only the runtime before/after diff establishes what changed; hypothesis is a
proposed explanation, never evidence. Do not invent public sign requirements.
Scientific evidence may include named term references with certified outer signs;
these are syntax facts, not a claim that a state or the term value is positive.
An indeterminate scientific review is unavailable advice, not scientific approval.
No unseen data, arbitrary code, full model, or prose outside hypothesis.
"""
)


def empty_draft() -> dict:
    return {
        "schema_version": PROTOCOL,
        "edits": {k: {} for k in KINDS},
        "accepted": {},
        "extras": {},
        "pending_actions": [],
        "action_states": [],
        "pending": [],
        "scope": "function",
        "equation_scopes": {},
        "attempts": [],
        "status": "pending",
    }


def diagnostic(exc, kind=None, target=None) -> dict:
    details = getattr(exc, "details", {})
    kind, target = details.get("action_kind", kind), details.get("target", target)
    return {
        "component": target,
        "action_kind": kind or "transaction",
        "code": getattr(
            exc, "code", "ACTION_SCHEMA" if kind is None else "ACTION_CONTRACT"
        ),
        "message": str(exc),
        "details": details,
    }


def advance(
    parent, plan, context, draft, action, nonlinear_targets=(), memory_targets=()
):
    """Revalidate the entire draft, preserving invalid entries as explicitly pending.

    No rejected operation is silently applied or discarded. Related operations may
    become valid after a later mapping/type edit; only a complete closure commits.
    """
    revised, initial = parent, plan
    diagnostics = []
    if action.scope == "no_change":
        return (
            parent,
            plan,
            {
                **empty_draft(),
                "status": "no_change",
                "audit": {
                    "scope": "no_change",
                    "status": "no_change",
                    "no_change_reason": "explicit_decline",
                    "actual_effects": [],
                },
            },
            [],
        )
    edits = copy.deepcopy(draft["edits"])
    for withdrawal in action.withdraw:
        if withdrawal.target not in edits[withdrawal.kind]:
            return (
                parent,
                plan,
                draft,
                [
                    {
                        "component": withdrawal.target,
                        "action_kind": withdrawal.kind,
                        "code": "WITHDRAWAL_NOT_PENDING",
                        "message": "withdrawal must name an existing draft operation",
                        "details": {},
                    }
                ],
            )
        del edits[withdrawal.kind][withdrawal.target]
    raw = action.model_dump(mode="json")
    for kind, (field, key) in KINDS.items():
        for entry in raw[field]:
            edits[kind][entry[key] if key else entry] = entry
    result = {
        **draft,
        "edits": edits,
        "scope": "model" if "model" in (draft["scope"], action.scope) else "function",
    }
    result["equation_scopes"] = {
        k: v for k, v in draft["equation_scopes"].items() if k in edits["equation"]
    }
    result["equation_scopes"].update(
        {e.component: action.scope for e in action.equations}
    )
    combined = {field: list(edits[kind].values()) for kind, (field, _) in KINDS.items()}
    combined.update(scope=result["scope"], hypothesis=action.hypothesis)
    accepted = {}
    try:
        internal = RepairAction.model_validate(combined)
        inventory = (set(expressions(parent)) | set(edits["equation"])) - set(
            internal.remove
        )
        shared = shared_declarations(internal.equations)
        for edit in internal.equations:
            try:
                _local_edit(
                    parent,
                    edit,
                    result["equation_scopes"][edit.component],
                    context,
                    inventory,
                    shared,
                )
                accepted[edit.component] = edit.model_dump(mode="json")
            except (ValueError, ModelValidationError) as exc:
                diagnostics.append(diagnostic(exc, "equation", edit.component))
        if not diagnostics:
            revised, initial, audit = commit_action(
                parent, plan, internal, context, nonlinear_targets, memory_targets
            )
            result.update(status=audit["status"], audit=audit)
    except (ValueError, ModelValidationError) as exc:
        diagnostics.append(diagnostic(exc))
    result.update(
        accepted=accepted,
        extras={
            field: combined[field] for field in ("remove", "mappings", "initializers")
        },
        pending_actions=[
            {"kind": d["action_kind"], "target": d["component"], "code": d["code"]}
            for d in diagnostics
        ],
        pending=[d["component"] for d in diagnostics if d["action_kind"] == "equation"],
    )
    invalid = {(d["action_kind"], d["component"]) for d in diagnostics}
    result["action_states"] = [
        {
            "kind": kind,
            "target": target,
            "status": "committed"
            if result["status"] == "committed"
            else "unchanged"
            if result["status"] == "no_change"
            else "pending_invalid"
            if (kind, target) in invalid
            else "locally_valid_provisional"
            if kind == "equation" and target in accepted
            else "provisional_requires_global_check",
        }
        for kind, entries in edits.items()
        for target in entries
    ]
    return revised, initial, result, diagnostics


def request_repair(
    parent,
    plan,
    context,
    report,
    public_prompt,
    client,
    directory: Path,
    round_index,
    nonlinear_targets=(),
    memory_targets=(),
):
    """Persist typed pending work and replies without resetting consumed budgets."""
    identity = content_hash(
        [
            PROTOCOL,
            FEEDBACK_PROTOCOL,
            model_hash(parent),
            plan.model_dump(mode="json"),
            report,
            public_prompt,
            nonlinear_targets,
            memory_targets,
        ]
    )
    path = directory / "transaction.json"
    draft = (
        json.loads(path.read_text())
        if path.exists()
        else {**empty_draft(), "identity": identity}
    )
    if draft.get("schema_version") != PROTOCOL or draft["identity"] != identity:
        raise ValueError("transaction provenance differs; use a fresh output root")
    if draft["status"] == "exhausted":
        return parent, plan, draft
    if draft["status"] in {"committed", "no_change"}:
        return (
            CandidateModel.model_validate(draft["candidate"]),
            LatentInitializationPlan.model_validate(draft["initialization"]),
            draft,
        )
    aliases = parameter_aliases(parent)
    for attempt in range(len(draft["attempts"]), client.settings.attempts_per_step):
        request = {
            "action_protocol": PROTOCOL,
            "feedback_protocol": FEEDBACK_PROTOCOL,
            "public_task": public_prompt,
            "report": report,
            "equations": expressions(parent),
            "kinds": {s.name: "dynamic" for s in parent.states}
            | {p.name: "algebraic" for p in parent.processes},
            "parent_parameters": [
                {"name": p.name, "role": p.role.value} for p in parent.parameters
            ],
            "available_forcing": sorted(context.forcing_channels),
            "target_mappings": [
                m.model_dump(mode="json") for m in parent.observation_mappings
            ],
            "initialization_plan": plan.model_dump(mode="json"),
            "eligible_parent_initializer_states": sorted(plan.rules),
            "provisional_edits": draft["accepted"],
            "provisional_related_edits": draft["extras"],
            "draft_actions": draft["edits"],
            "pending_components": draft["pending"],
            "pending_actions": draft["pending_actions"],
            "action_states": draft["action_states"],
            "retry_findings": draft["attempts"][-1:],
        }
        record = client.call(
            system=SYSTEM_PROMPT,
            user=json.dumps(translate_names(request, aliases), sort_keys=True),
            response_model=RepairActionV2,
            step=f"repair_round_{round_index}",
            attempt=attempt,
        )
        raw, diagnostics = None, []
        revised, initial = parent, plan
        prior = copy.deepcopy(draft)
        try:
            raw = translate_names(
                visible_response(record), {v: k for k, v in aliases.items()}
            )
            action = RepairActionV2.model_validate(raw)
            revised, initial, draft, diagnostics = advance(
                parent, plan, context, draft, action, nonlinear_targets, memory_targets
            )
            draft["hypothesis"] = action.hypothesis
        except (ValueError, ModelValidationError) as exc:
            diagnostics = [diagnostic(exc)]
        draft.update(identity=identity, attempts=prior["attempts"])
        draft["attempts"].append(
            {
                "attempt": attempt,
                "request_hash": record["request_hash"],
                "response": raw,
                "diagnostics": diagnostics,
                "pending_actions": draft["pending_actions"],
            }
        )
        if draft["status"] in {"committed", "no_change"}:
            draft.update(
                candidate=revised.model_dump(mode="json"),
                initialization=initial.model_dump(mode="json"),
            )
        atomic_json(path, draft)
        if draft["status"] != "pending":
            return revised, initial, draft
    draft["status"] = "exhausted"
    atomic_json(path, draft)
    return parent, plan, draft
