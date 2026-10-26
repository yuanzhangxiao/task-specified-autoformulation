"""Bounded construction search policies with explicit non-adopting bookkeeping.

Each trial is one complete patch against the immutable parent. Alternatives are
never merged and no fitting, rollout, held-out access or model promotion occurs.
A caller supplies one durable budgeted client for the whole three-call episode.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from pydantic import Field, ValidationError

from autoformalism.llm.staged_topology import visible_response
from autoformalism.research import construction_structural_feedback as feedback
from autoformalism.research.construction_repair_identity import (
    diagnostic_identity,
    family_identity,
)
from autoformalism.research.construction_revision_edits import (
    RevisionValidationError,
    model_view,
)
from autoformalism.research.construction_structural_revision import (
    StructuralRevisionReply,
    apply_structural_revision,
)
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.staged_topology import content_hash
from scripts import structural_revision_campaign as base

PROTOCOL = "construction-search-policy-2"
POLICIES = ("sequential_restart", "independent", "options_shallow")
SYSTEM = (
    base.SYSTEM
    + "\n"
    + """
Search bookkeeping is separate from scientific declarations. Return only changes
to current_model, not a complete model: untouched definitions are kept automatically.
For each changed equation give its complete new RHS; for a new state give its RHS
and initializer. Reuse existing parameters without declaring them again. Declare
initializer-local parameters only inside that initializer. The runtime derives
dependencies and installs all consistent edits together; it does not invent laws.
Example: to change x' = A+B into A+C, return only the x edit with RHS A+C, plus
any genuinely new declarations needed for C. Do not re-declare unchanged A.
Every patch must contain all changes still intended relative to the parent, even
on retry. An intended trial
shown in feedback was rejected, not installed. Retain every earlier correction
you still intend; omitted edits revert to the parent. Never merge alternatives.
Renaming private states or freely fitted parameters does not create a different
repair when equations, initial rules, parameter sharing and domains are unchanged.
Changing a time constant's name cannot clear the same model-data contradiction.
A repeated unchanged failure requires a different scientific premise, not another
explanation of the same model. A formatting correction can be useful even when its
equations are unchanged; the runtime validates it before judging repetition.
""".strip()
)


class RepairOption(StrictSchema):
    """One independently applicable alternative with an explicit premise change."""

    option_id: Identifier
    affected_target: str = Field(min_length=1, max_length=500)
    premise_changed: str = Field(min_length=1, max_length=1500)
    patch: StructuralRevisionReply


class OptionsReply(StrictSchema):
    """Exactly three alternatives share one completion's output allowance."""

    options: tuple[RepairOption, ...] = Field(min_length=3, max_length=3)


class _OptionMetadata(StrictSchema):
    """Validate wrapper metadata without rejecting another option's valid patch."""

    option_id: Identifier
    affected_target: str = Field(min_length=1, max_length=500)
    premise_changed: str = Field(min_length=1, max_length=1500)


def _diagnostic(exc: ValueError) -> dict:
    if isinstance(exc, RevisionValidationError):
        return exc.as_dict()
    return {"code": "RESPONSE_SCHEMA_OR_AVAILABILITY", "message": str(exc)[:4000]}


def _model_display(problem: dict, aliases: dict | None) -> dict:
    """Show full equations/initial rules without repeating the public context."""
    view = model_view(
        PrefixFitProblem.model_validate(problem), scientific_aliases=aliases
    )
    return {
        key: view[key]
        for key in (
            "equations",
            "parameters",
            "observation_mappings",
            "initialization_graph",
            "initialization_scopes",
        )
    }


def assess_candidate(
    parent: dict,
    raw: Any,
    *,
    candidate_id: str = "saved",
    call_index: int = 0,
    option_index: int = 0,
    branch_id: str | None = None,
    premise: str = "",
    affected_target: str = "",
) -> dict:
    """Normalize and validate one detached candidate before judging repetition.

    The static result is kept even when qualified data checks reject it, allowing
    complete-model identity comparison. It is never an adopted model or fit export.
    """
    result = {
        "id": candidate_id,
        "call_index": call_index,
        "option_index": option_index,
        "branch_id": branch_id or candidate_id,
        "raw_reply": deepcopy(raw),
        "premise_changed": premise,
        "affected_target": affected_target,
        "status": "rejected",
        "problem": None,
        "audit": {},
        "diagnostic": None,
        "family_identity": {"status": "deferred", "sha256": None},
        "normalized_reply": None,
        "patch_sha256": feedback.patch_signature(raw),
        "diagnostic_sha256": None,
        "repeated_unchanged_failure": False,
        "repeats_candidate_ids": [],
        "omitted_previous_actions": [],
    }
    try:
        if not isinstance(raw, dict):
            raise ValueError("one complete repair must be a JSON object")
        normalized, _ = base._normalize_displayed_citations(
            parent, raw, weighted_symbolic_initials=True
        )
        static = apply_structural_revision(
            PrefixFitProblem.model_validate(parent["problem"]),
            normalized,
            evidence_ids=["public_task", "training_consistency", "parameter_audit"],
            brief=PublicScientificBrief.model_validate(parent["public_brief"]),
            scientific_aliases=parent.get("scientific_aliases"),
            normalize_parent_declarations=True,
        )
        result["audit"] = static["audit"]
        result["normalized_reply"] = (
            static["audit"].get("normalization", {}).get("normalized_reply")
        )
        if static["status"] == "requires_wider_revision":
            result["status"] = "requires_wider_revision"
            return result
        problem = static["problem"].model_dump(mode="json")
        result["problem"] = problem
        result["family_identity"] = family_identity(parent["problem"], problem)
        # Keep acceptance exactly at the existing qualified, candidate-specific
        # gate boundary rather than maintaining a second acceptance algorithm.
        applied = base._apply(
            parent,
            raw,
            normalize_parent_declarations=True,
            weighted_symbolic_initials=True,
        )
        result.update(status="eligible", audit=applied["audit"])
    except (ValueError, TypeError, KeyError) as exc:
        diagnostic = (
            _diagnostic(exc)
            if isinstance(exc, ValueError)
            else {"code": "RESPONSE_SCHEMA_OR_AVAILABILITY", "message": str(exc)[:4000]}
        )
        result["diagnostic"] = diagnostic
        result["diagnostic_sha256"] = diagnostic_identity(
            diagnostic,
            private_symbol_mapping=result["family_identity"].get(
                "private_symbol_mapping"
            ),
        )
    return result


def _family_key(candidate: dict) -> str | None:
    identity = candidate.get("family_identity", {})
    if identity.get("sha256"):
        return "model:" + identity["sha256"]
    if candidate.get("patch_sha256"):
        return "patch:" + candidate["patch_sha256"]
    return None


def _omitted_actions(candidate: dict, previous: dict) -> list[dict]:
    """Expose omitted prior intentions without inferring usefulness or merging."""
    current = candidate.get("normalized_reply") or candidate.get("raw_reply")
    prior = previous.get("normalized_reply") or previous.get("raw_reply")
    if not isinstance(current, dict) or not isinstance(prior, dict):
        return []
    omitted = []
    for field, key in (
        ("edits", "component"),
        ("new_states", "name"),
        ("new_processes", "name"),
        ("new_parameters", "name"),
        ("initialization_edits", "state"),
        ("remove_states", None),
        ("remove_processes", None),
    ):
        old_rows, new_rows = prior.get(field, []), current.get(field, [])
        if not isinstance(old_rows, list) or not isinstance(new_rows, list):
            continue
        new_keys = set()
        for row in new_rows:
            name = row.get(key) if key and isinstance(row, dict) else row
            if isinstance(name, str):
                new_keys.add(name)
        for row in old_rows:
            name = row.get(key) if key and isinstance(row, dict) else row
            if isinstance(name, str) and name not in new_keys:
                omitted.append({"field": field, "previous_action": deepcopy(row)})
    return omitted


def annotate_repetition(candidate: dict, history: list[dict]) -> dict:
    """Compare model family AND remaining failure after current validation."""
    result = deepcopy(candidate)
    same_branch = [row for row in history if row["branch_id"] == result["branch_id"]]
    if same_branch:
        result["omitted_previous_actions"] = _omitted_actions(result, same_branch[-1])
    key = _family_key(candidate)
    if result["status"] == "rejected" and key:
        result["repeats_candidate_ids"] = [
            row["id"]
            for row in history
            if row["status"] == "rejected"
            and _family_key(row) == key
            and row.get("diagnostic_sha256") == candidate.get("diagnostic_sha256")
        ]
        result["repeated_unchanged_failure"] = bool(result["repeats_candidate_ids"])
    return result


def _targets(value: Any) -> list[str]:
    found, pending = set(), [value]
    for _ in range(4096):
        if not pending:
            break
        item = pending.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"target", "channel", "property", "state"} and isinstance(
                    child, str
                ):
                    found.add(child[:160])
                elif isinstance(child, (dict, list)):
                    pending.append(child)
        elif isinstance(item, list):
            pending.extend(item)
    return sorted(found)[:12]


def ledger_view(history: list[dict]) -> list[dict]:
    """Keep all bounded attempts visible without copying replies or witnesses."""
    rows = []
    for candidate in history:
        reply = candidate.get("normalized_reply") or candidate.get("raw_reply")
        reply = reply if isinstance(reply, dict) else {}
        edits = reply.get("edits", [])
        states = reply.get("new_states", [])
        edits = edits if isinstance(edits, list) else []
        states = states if isinstance(states, list) else []
        premise = candidate.get("premise_changed") or reply.get("hypothesis", "")
        rows.append(
            {
                "candidate": candidate["id"],
                "branch": candidate["branch_id"],
                "family": _family_key(candidate),
                "premise_changed": premise[:500] if isinstance(premise, str) else "",
                "status": candidate["status"],
                "remaining_error": (candidate.get("diagnostic") or {}).get("code"),
                "affected_targets_or_properties": _targets(candidate.get("diagnostic")),
                "edited_components": [
                    item.get("component") for item in edits if isinstance(item, dict)
                ][:32],
                "new_states": [
                    item.get("name") for item in states if isinstance(item, dict)
                ][:8],
                "repeated_unchanged_failure": candidate["repeated_unchanged_failure"],
                "not_installed": True,
            }
        )
    return rows


def request(
    parent: dict,
    policy: str,
    *,
    history: list[dict] | None = None,
    intended: dict | None = None,
    fresh: bool = False,
    batch: bool = False,
) -> str:
    """Build one exact request; history never silently becomes an adopted model."""
    if policy not in POLICIES:
        raise ValueError("unknown construction search policy")
    history = history or []
    repair = None
    if intended is not None and not fresh:
        diagnostic = intended.get("diagnostic") or {}
        guidance = feedback.feedback_guidance(
            diagnostic.get("code", ""),
            diagnostic.get("details", {}),
            parent,
            intended.get("raw_reply"),
        )
        guidance.pop("details", None)
        repair = {
            "status": "rejected_in_full_parent_unchanged",
            "diagnostic": diagnostic,
            "guidance": guidance,
            "last_reply": intended.get("normalized_reply") or intended.get("raw_reply"),
            "complete_intended_patch": intended.get("normalized_reply")
            or intended.get("raw_reply"),
            "instruction": (
                "This complete intended trial was rejected and is not installed. "
                "Return a complete corrected patch against current_model. Preserve "
                "every earlier correction you still intend; omitting it restores "
                "the parent's definition. Correct the stated failure, not only "
                "the explanation."
            ),
        }
        # There is one copy of the full patch, under a descriptive field. Base
        # request's trial-reference text is adjusted below, not left dangling.
        repair.pop("last_reply")
        if intended.get("omitted_previous_actions"):
            repair["previous_trial_actions_now_omitted"] = {
                "actions": intended["omitted_previous_actions"],
                "instruction": "These intentions were present in the previous trial "
                "but absent in the latest complete patch. Their scientific usefulness "
                "is not assumed. Preserve those still intended, or deliberately "
                "replace/revert them. Nothing has been automatically merged.",
            }
        if intended.get("problem"):
            repair["intended_trial_model"] = _model_display(
                intended["problem"], intended.get("audit", {}).get("scientific_aliases")
            )
    packet = json.loads(base.request(parent, repair, weighted_symbolic_initials=True))
    packet["editing_scope"]["patch_semantics"] = (
        "List only changes to current_model; all untouched definitions are kept. "
        "Each changed equation is a complete RHS replacement, not an added term. "
        "Include all changes still intended from rejected trials, because none "
        "was installed. Reuse existing parameter/process names without redeclaring "
        "them. Initializer-local parameters belong only in initial.parameters."
    )
    training = packet.get("training_consistency", {})
    if isinstance(training, dict) and "model_reference" in training:
        training["model_reference"] = (
            "Trial evidence refers to repair_feedback.complete_intended_patch; "
            "current_model is the unchanged parent."
        )
    packet["search_instruction"] = {
        "policy": policy,
        "mode": "alternative_batch"
        if batch
        else "fresh_alternative"
        if fresh
        else "repair",
        "instruction": (
            "Return exactly three self-contained repair options. This is mandatory; "
            "a reason for returning fewer is not accepted. Each option names the "
            "affected target and the premise of its conflict that changes. A "
            "private-symbol rename or another freely fitted coefficient name is "
            "not diversity. Change a scientific dependence, function family or "
            "allowed initialization premise in each alternative. Each option is "
            "a separate complete set of changes to the same parent, not a full "
            "model or an incremental edit to another option. All three share "
            "this call's output allowance; keep each concise and complete."
            if batch
            else "Start a fresh scientific alternative from current_model. Avoid the "
            "failed families in attempted_repairs; do not copy the last rejected "
            "patch. Return one complete patch."
            if fresh and history
            else "Return one complete scientific patch against current_model."
        ),
        "maximum_episode_calls": 3,
        "no_fitting_or_promotion": True,
    }
    if history:
        packet["attempted_repairs"] = ledger_view(history)
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))


def _options(raw: Any) -> list[tuple[Any, dict]]:
    """Bound the envelope; assess malformed sibling options independently."""
    if not isinstance(raw, dict) or set(raw) != {"options"}:
        raise ValueError("option reply must contain only the options array")
    options = raw.get("options")
    if not isinstance(options, list) or len(options) != 3:
        raise ValueError("return exactly three complete options; fewer is not allowed")
    result, identifiers = [], set()
    for option in options:
        try:
            if not isinstance(option, dict) or set(option) != {
                "option_id",
                "affected_target",
                "premise_changed",
                "patch",
            }:
                raise ValueError("each option requires metadata and one complete patch")
            metadata = _OptionMetadata.model_validate(
                {key: value for key, value in option.items() if key != "patch"}
            ).model_dump(mode="json")
            if isinstance(option["patch"], dict) and option["patch"].get(
                "decision"
            ) in {"retain", "requires_wider_revision"}:
                raise ValueError("each repair option must propose decision=revise")
            if metadata["option_id"] in identifiers:
                raise ValueError("option_id must be unique within this reply")
            identifiers.add(metadata["option_id"])
            result.append((option["patch"], metadata))
        except (ValueError, ValidationError) as exc:
            result.append((None, {"wrapper_error": str(exc)[:2000]}))
    return result


def _failed_branches(candidates: list[dict], seed: int) -> list[dict]:
    """Choose two distinct failed families by frozen seeded order, never by fit."""
    chosen, used = [], set()
    for candidate in sorted(
        candidates, key=lambda row: content_hash([seed, row["id"]])
    ):
        if candidate["status"] != "rejected":
            continue
        key = _family_key(candidate) or "invalid:" + content_hash(
            candidate["raw_reply"]
        )
        if key not in used:
            chosen.append(candidate)
            used.add(key)
        if len(chosen) == 2:
            break
    return chosen


def run_episode(parent, client, settings, policy: str, *, seed: int = 0) -> dict:
    """Run at most three cached calls with policy-specific, deterministic routing."""
    if policy not in POLICIES:
        raise ValueError("unknown construction search policy")
    if settings.maximum_requests != 3:
        raise ValueError("the search experiment requires a three-call episode ceiling")
    candidates, attempts, selected_branches = [], [], []
    intended, fresh = None, False
    for call_index in range(3):
        batch = policy == "options_shallow" and call_index == 0
        if policy == "options_shallow" and call_index:
            if call_index > len(selected_branches):
                break
            intended = selected_branches[call_index - 1]
        visible_history = [] if policy == "independent" else candidates
        record = client.call(
            system=SYSTEM,
            user=request(
                parent,
                policy,
                history=visible_history,
                intended=intended,
                fresh=fresh or policy == "independent",
                batch=batch,
            ),
            response_model=OptionsReply if batch else StructuralRevisionReply,
            step="search_policy",
            attempt=call_index,
        )
        raw, parse_error = None, None
        try:
            raw = visible_response(record)
        except ValueError as exc:
            parse_error = _diagnostic(exc)
        if parse_error is None:
            try:
                options = _options(raw) if batch else [(raw, {})]
            except ValueError as exc:
                parse_error = {"code": "OPTION_OR_RESPONSE_FORMAT", "message": str(exc)}
        if parse_error is not None:
            options = [(None, {"wrapper_error": parse_error["message"]})]
        new_candidates = []
        for option_index, (payload, metadata) in enumerate(options):
            candidate_id = f"call{call_index}-option{option_index}"
            candidate = assess_candidate(
                parent,
                payload,
                candidate_id=candidate_id,
                call_index=call_index,
                option_index=option_index,
                branch_id=intended["branch_id"] if intended else candidate_id,
                premise=metadata.get("premise_changed", ""),
                affected_target=metadata.get("affected_target", ""),
            )
            if metadata.get("wrapper_error"):
                candidate["diagnostic"] = {
                    "code": parse_error["code"]
                    if parse_error
                    else "OPTION_OR_RESPONSE_FORMAT",
                    "message": metadata["wrapper_error"],
                }
                if (
                    parse_error
                    and parse_error["code"] == "RESPONSE_SCHEMA_OR_AVAILABILITY"
                ):
                    candidate["status"] = "unavailable"
                candidate["diagnostic_sha256"] = diagnostic_identity(
                    candidate["diagnostic"]
                )
                candidate["option_wrapper_valid"] = False
            else:
                candidate["option_wrapper_valid"] = True
            candidate = annotate_repetition(candidate, candidates)
            candidates.append(candidate)
            new_candidates.append(candidate)
        attempts.append(
            {
                "attempt": call_index,
                "request_hash": record["request_hash"],
                "mode": "options"
                if batch
                else "fresh"
                if fresh or policy == "independent"
                else "repair",
                "branch_id": intended["branch_id"] if intended else None,
                "candidate_ids": [row["id"] for row in new_candidates],
                "parse_error": parse_error,
            }
        )
        if batch:
            selected_branches = _failed_branches(
                [row for row in new_candidates if row["option_wrapper_valid"]], seed
            )
        elif policy == "sequential_restart":
            latest = new_candidates[0]
            if latest["status"] in {"eligible", "requires_wider_revision"}:
                break
            intended = latest
            fresh = latest["repeated_unchanged_failure"]
            if fresh:
                intended = None
        # Independent breadth deliberately collects all three proposals even if
        # an earlier one is eligible, to measure useful diversity under its cap.
    eligible = [row["id"] for row in candidates if row["status"] == "eligible"]
    return {
        "protocol": PROTOCOL,
        "policy": policy,
        "status": "eligible_found" if eligible else "no_eligible_candidate",
        "candidates": candidates,
        "attempts": attempts,
        "eligible_candidate_ids": eligible,
        "attempted_repairs": ledger_view(candidates),
        "generation_calls": len(attempts),
        "candidate_count": len(candidates),
        "parent_unchanged": True,
        "model_promoted": False,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "interpretation": "Eligible means no currently certified conflict under finite "
        "checks; deferred evidence is not scientific or numerical adequacy.",
    }
