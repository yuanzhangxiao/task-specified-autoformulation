"""Opt-in evidence feedback and explicit public boundary-effect bookkeeping.

This policy adds declaration obligations, not a scientific proof or a fitted
accuracy gate. Historical realization schemas and prompts are left unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import Field, create_model, field_validator, model_validator

from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_deferred as deferred
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_realizations as prompts
from autoformalism.search import mechanism_realizations as core
from autoformalism.search.public_graph_obligations import _path
from autoformalism.staged_topology import content_hash

POLICY = "minimal-evidence-feedback-1"
SYSTEM = prompts.SYSTEM
REPAIR_REQUESTS = prompts.REPAIR_REQUESTS
REPAIR_TOKENS = prompts.REPAIR_TOKENS
REPEATED_REPAIRS = prompts.REPEATED_REPAIRS
repair_budget = prompts.repair_budget
repair_signature = prompts.repair_signature


class BoundaryObligation(StrictSchema):
    """A reviewed public requirement for a distinct effect leaving the system."""

    id: Identifier
    target: Identifier
    public_quote: str = Field(min_length=1, max_length=1500)
    interpretation: str = Field(min_length=1, max_length=1500)
    kind: Literal["external_loss"] = "external_loss"


class BoundaryContract(StrictSchema):
    """Public-only interpretation, frozen independently of benchmark files."""

    policy: Literal["reviewed-public-boundaries-1"] = "reviewed-public-boundaries-1"
    source_brief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    obligations: tuple[BoundaryObligation, ...] = ()

    @model_validator(mode="after")
    def unique_ids(self):
        """Do not permit conflicting identities for the same obligation."""
        ids = [rule.id for rule in self.obligations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate public boundary obligation ID")
        return self

    def validate_public(self, brief: PublicScientificBrief) -> None:
        """Check exact public provenance, without inferring science from text."""
        if self.source_brief_sha256 != content_hash(brief.model_dump(mode="json")):
            raise ValueError("public boundary contract brief hash differs")
        targets = {v.name for v in brief.public_variables if v.data_role == "target"}
        text = " ".join(brief.scientific_context.split())
        for rule in self.obligations:
            if rule.target not in targets:
                raise ValueError("public boundary target is not a public target")
            if " ".join(rule.public_quote.split()) not in text:
                raise ValueError("public boundary quote differs from public task")


class ExternalExchangeChoice(StrictSchema):
    """Proposer-selected existing contribution; no new process is implied."""

    obligation_id: Identifier
    receiver: Identifier
    local_term_index: int | None = Field(default=None, ge=0, le=31)
    shared_process: Identifier | None = None
    explanation: str = Field(min_length=1, max_length=700)

    @model_validator(mode="after")
    def one_reference(self):
        """A role must point to exactly one ordinary term or shared use."""
        if (self.local_term_index is None) == (self.shared_process is None):
            raise ValueError("select exactly one local_term_index or shared_process")
        if not self.explanation.strip():
            raise ValueError("external exchange explanation must be nonempty")
        return self


class ExternalExchange(ExternalExchangeChoice):
    """Runtime-bound reference; later edits cannot silently retarget a role."""

    reference_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class CompatibilityDecision(StrictSchema):
    """A response to conditional evidence, separate from scientific declarations."""

    question_id: str = Field(min_length=1, max_length=160)
    disposition: Literal["revise", "retain_approximation", "dispute_assumption"]
    explanation: str = Field(min_length=1, max_length=1000)

    @field_validator("question_id", "explanation")
    @classmethod
    def nonempty(cls, value: str) -> str:
        """Whitespace alone is not a decision or a justification."""
        if not value.strip():
            raise ValueError("compatibility decision fields must be nonempty")
        return value


class Draft(core.Draft):
    """Retain explicit boundary roles with the scientific construction ledger."""

    external_exchanges: tuple[ExternalExchange, ...] = ()

    @model_validator(mode="after")
    def unique_external_ids(self):
        """Each public role has one current reference."""
        ids = [row.obligation_id for row in self.external_exchanges]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate external_exchanges obligation_id")
        return self


class DraftPatch(core.DraftPatch):
    """Keep advisory answers out of the scientific draft and its type system."""

    external_exchanges: tuple[ExternalExchangeChoice, ...] = ()
    remove_external_exchanges: tuple[Identifier, ...] = ()
    compatibility_decisions: tuple[CompatibilityDecision, ...] = Field(
        default=(), max_length=16
    )

    @model_validator(mode="after")
    def unique_new_keys(self):
        """Reject ambiguous replacements and duplicate advisory answers."""
        for rows, key in (
            (self.external_exchanges, "obligation_id"),
            (self.compatibility_decisions, "question_id"),
        ):
            ids = [getattr(row, key) for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {key}")
        return self


def base(draft: Draft) -> core.Draft:
    """Build an exact disposable v1 view without mutating saved declarations."""
    return core.Draft(
        **{name: getattr(draft, name) for name in core.Draft.model_fields}
    )


def project(draft: Draft) -> ledger.Draft:
    """Reuse historical compilation while retaining new records separately."""
    return core.project(base(draft))


def load_draft(raw: dict):
    """Read the declared schema without silently upgrading historical artifacts."""
    if "external_exchanges" in raw:
        return Draft.model_validate(raw)
    return core.load_draft(raw)


def from_legacy(draft) -> Draft:
    """Explicit preview projection; no boundary effect is inferred."""
    previous = draft if isinstance(draft, core.Draft) else core.from_legacy(draft)
    return Draft(**previous.model_dump(mode="json"))


def _base_patch(patch: DraftPatch) -> core.DraftPatch:
    return core.DraftPatch(
        **{name: getattr(patch, name) for name in core.DraftPatch.model_fields},
    )


def _reference(draft: Draft, choice: ExternalExchangeChoice) -> dict | None:
    """Resolve a selected contribution in the exact current declaration view."""
    variable = next((v for v in draft.variables if v.name == choice.receiver), None)
    if variable is None:
        return None
    if choice.local_term_index is not None:
        equation = next((e for e in draft.equations if e.name == choice.receiver), None)
        if equation is None or choice.local_term_index >= len(equation.terms):
            return None
        return {
            "receiver": choice.receiver,
            "definition": variable.definition,
            "local_term": equation.terms[choice.local_term_index].model_dump(
                mode="json"
            ),
        }
    process = next(
        (p for p in draft.processes if p.name == choice.shared_process), None
    )
    if process is None or not any(u.target == choice.receiver for u in process.uses):
        return None
    return {
        "receiver": choice.receiver,
        "definition": variable.definition,
        "process": process.model_dump(mode="json"),
    }


def _bind(draft: Draft, choice: ExternalExchangeChoice) -> ExternalExchange:
    reference = _reference(draft, choice)
    return ExternalExchange(
        **choice.model_dump(mode="json"),
        reference_sha256=content_hash(reference) if reference is not None else None,
    )


def apply_patch(brief, draft: Draft, patch: DraftPatch) -> Draft:
    """Atomically install proposer edits; record controls never become variables."""
    controls = {
        "external_exchanges",
        "remove_external_exchanges",
        "compatibility_decisions",
    }
    if any(
        row.name in controls
        for field in ("variables", "equations", "processes")
        for row in getattr(patch, field)
    ):
        raise ValueError(
            "reply-control fields are not scientific declarations; no edit committed"
        )
    records = {r.obligation_id: r for r in draft.external_exchanges}
    replacements = {r.obligation_id: r for r in patch.external_exchanges}
    removed = patch.remove_external_exchanges
    if len(removed) != len(set(removed)) or set(removed) & replacements.keys():
        raise ValueError(
            "ambiguous external_exchanges replacement/removal; no edit committed"
        )
    unknown = sorted(set(removed) - records.keys())
    if unknown:
        raise ValueError(
            f"cannot remove unknown external_exchanges: {unknown}; use obligation_id, "
            f"available keys={sorted(records)}. No edit committed."
        )
    # A boundary-only edit or advisory response is invisible to the v1 view.
    previous = core.apply_patch(
        brief,
        base(draft),
        _base_patch(patch).model_copy(update={"stage_complete": True}),
    )
    for name in removed:
        del records[name]
    candidate = Draft(
        **previous.model_dump(mode="json"), external_exchanges=tuple(records.values())
    )
    records.update(
        (name, _bind(candidate, choice)) for name, choice in replacements.items()
    )
    candidate = candidate.model_copy(
        update={"external_exchanges": tuple(records.values())}
    )
    if (
        candidate == draft
        and not patch.stage_complete
        and not patch.compatibility_decisions
    ):
        raise ValueError("no draft change; make an explicit edit or finish this stage")
    return candidate


def normalize_reply(brief, raw, *, draft=None, **kwargs):
    """Normalize existing delivery fields while preserving new structured choices."""
    value = deepcopy(raw)
    fields = {}
    if isinstance(value, dict):
        fields = {
            name: value.pop(name)
            for name in (
                "external_exchanges",
                "remove_external_exchanges",
                "compatibility_decisions",
            )
            if name in value
        }
    value, changes = core.normalize_reply(
        brief, value, draft=base(draft) if draft is not None else None, **kwargs
    )
    if isinstance(value, dict):
        value.update(fields)
    return value, changes


def normalize_consumers(draft: Draft, patch: DraftPatch, previous: Draft):
    """Bind explicitly selected roles after duplicate-use normalization."""
    updated, changes = core.normalize_consumers(
        base(draft), _base_patch(patch), base(previous)
    )
    candidate = Draft(
        **updated.model_dump(mode="json"), external_exchanges=draft.external_exchanges
    )
    replacements = {r.obligation_id: r for r in patch.external_exchanges}
    candidate = candidate.model_copy(
        update={
            "external_exchanges": tuple(
                _bind(candidate, replacements[row.obligation_id])
                if row.obligation_id in replacements
                else row
                for row in candidate.external_exchanges
            )
        }
    )
    return candidate, changes


def response_model(stage: str):
    """Present only available stage edits and non-scientific advisory controls."""
    fields = set(core.response_model(stage).model_fields)
    if stage in {"shared_laws", "equations", "repair"}:
        fields |= {"external_exchanges", "remove_external_exchanges"}
    if stage == "repair":
        fields.add("compatibility_decisions")
    return create_model(
        "Evidence" + stage.title().replace("_", "") + "Reply",
        __base__=StrictSchema,
        **{
            name: (field.annotation, deepcopy(field))
            for name, field in DraftPatch.model_fields.items()
            if name in fields
        },
    )


def edit_effects(before: Draft, after: Draft) -> dict:
    """Include role edits in the durable transaction receipt."""
    result = core.edit_effects(base(before), base(after))
    old = {
        r.obligation_id: r.model_dump(mode="json") for r in before.external_exchanges
    }
    new = {r.obligation_id: r.model_dump(mode="json") for r in after.external_exchanges}
    result["external_exchanges"] = [
        {"key": name, "before": old.get(name), "after": new.get(name)}
        for name in sorted(old.keys() | new.keys())
        if old.get(name) != new.get(name)
    ]
    return result


def pending(brief, draft: Draft) -> dict:
    """Preserve historical pending work; final role coverage is checked separately."""
    return core.pending(brief, base(draft))


def variable_checklist(brief, draft: Draft, target_definitions, graph_contract=None):
    """Allow provisional source-only records before topology exists."""
    return {
        **core.variable_checklist(
            brief, base(draft), target_definitions, graph_contract
        ),
        "policy": POLICY,
    }


def _source_only_issues(result: dict, draft: Draft) -> list[dict]:
    generated = {v.name for v in draft.variables} | {p.name for p in draft.processes}
    issues = []
    for row in result["mechanism_realization_checks"]:
        if row["declaration_status"] != "satisfied" or generated.intersection(
            row["components"]
        ):
            continue
        paths = row.get("path_checks", [])
        path_status = (
            "passed"
            if paths and all(p["passed"] for p in paths)
            else ("failed" if paths else "not_assessed")
        )
        row["implementation_record_status"] = "source_only"
        issues.append(
            {
                "code": "mechanism_realization_source_only",
                "requirement": row["requirement"],
                "components": row["components"],
                "public_path_status": path_status,
                "path_checks": paths,
                "field_to_edit": "mechanism_realizations",
                "action": (
                    "The displayed public path exists. This is an incomplete "
                    "implementation "
                    "record, not a missing-path claim. "
                    if path_status == "passed"
                    else "This record names only supplied sources; inspect the "
                    "separate pathway verdict. "
                )
                + "Name an existing generated variable whose local equation "
                "implements the effect, or an existing shared process, and preserve "
                "intended source references. Do not invent another state or process "
                "merely to complete this record. If no generated component implements "
                "it, propose the scientifically intended topology and then identify "
                "that component.",
            }
        )
    return issues


def boundary_checks(
    brief, draft: Draft, contract: BoundaryContract | None
) -> list[dict]:
    """Check claimed roles/references; loss sign and conservation remain deferred."""
    if contract is not None:
        contract.validate_public(brief)
    obligations = {r.id: r for r in contract.obligations} if contract else {}
    records = {r.obligation_id: r for r in draft.external_exchanges}
    definitions = {v.name: v.definition for v in draft.variables}
    # Readout reachability permits equivalent storage coordinates, without
    # letting an unrelated dynamic state merely feeding that storage stand in.
    readouts: dict[str, set[str]] = {}
    for equation in draft.equations:
        if definitions.get(equation.name) == "algebraic":
            for term in equation.terms:
                for source in term.sources:
                    readouts.setdefault(source, set()).add(equation.name)
    for process in draft.processes:
        for source in process.depends_on:
            readouts.setdefault(source, set()).add(process.name)
        for use in process.uses:
            if definitions.get(use.target) == "algebraic":
                readouts.setdefault(process.name, set()).add(use.target)
    rows = []
    for name in sorted(obligations.keys() | records.keys()):
        obligation, record = obligations.get(name), records.get(name)
        problems = []
        reference = _reference(draft, record) if record else None
        if obligation is None:
            problems.append("unknown_public_boundary_obligation")
        if record is None:
            problems.append("missing_external_exchange_assignment")
        else:
            if definitions.get(record.receiver) != "differential":
                problems.append("receiver_is_not_generated_balance_state")
            if reference is None:
                problems.append("contribution_reference_unresolved")
            elif record.reference_sha256 != content_hash(reference):
                problems.append("contribution_reference_changed")
            if (reference or {}).get("process", {}).get("kind") == "transfer":
                problems.append("internal_transfer_is_not_distinct_external_loss")
            if (
                obligation
                and record.receiver != obligation.target
                and not _path(readouts, record.receiver, obligation.target)
            ):
                problems.append("balance_coordinate_has_no_algebraic_target_readout")
        rows.append(
            {
                "id": name,
                "target": obligation.target if obligation else None,
                "public_quote": obligation.public_quote if obligation else None,
                "status": "incomplete" if problems else "reference_covered",
                "issues": problems,
                "assignment": record.model_dump(mode="json") if record else None,
                "referenced_contribution": reference,
                "behavior_status": "unverified",
                "action": "Identify the intended external loss using "
                "external_exchanges: obligation_id, its generated differential "
                "receiver, and either the displayed zero-based local_term_index or "
                "shared_process. An ordinary local process is sufficient; do not "
                "invent a second receiver or a named shared process. An internal "
                "transfer between modeled balances is not this external loss. "
                "Equivalent storage coordinates with an algebraic readout are "
                "allowed. If a selected term changed, repeat the assignment only "
                "after checking its new meaning. Do not relabel a genuine internal "
                "transfer to satisfy this check. Numerical sign, threshold behavior "
                "and conservation require interaction functions.",
            }
        )
    return rows


def assess(
    brief,
    context,
    target_definitions,
    draft: Draft,
    *,
    graph_contract=None,
    clarify_overlaps=False,
    boundary_contract: BoundaryContract | None = None,
):
    """Add final bookkeeping coverage without turning data diagnostics into gates."""
    result = core.assess(
        brief,
        context,
        target_definitions,
        base(draft),
        graph_contract=graph_contract,
        clarify_overlaps=clarify_overlaps,
    )
    result["errors"].extend(_source_only_issues(result, draft))
    rows = boundary_checks(brief, draft, boundary_contract)
    result["boundary_checks"] = rows
    result["errors"].extend(
        {"code": "external_exchange_coverage", **row}
        for row in rows
        if row["status"] != "reference_covered"
    )
    result["eligible"] = result["eligible"] and not result["errors"]
    return result


def snapshot(brief, draft: Draft):
    """Expose the new saved declarations without omitting boundary choices."""
    return {
        **core.snapshot(brief, base(draft)),
        "declarations": draft.model_dump(mode="json"),
    }


def variable_snapshot(draft: Draft):
    """The variable stage still has no premature topology obligation."""
    return {
        **core.variable_snapshot(base(draft)),
        "declarations": draft.model_dump(mode="json"),
    }


def final_handoff(draft: Draft, assessment: dict):
    """Carry explicit roles into later interaction review without verifying them."""
    return {
        **core.final_handoff(base(draft), assessment),
        "policy": POLICY,
        "external_exchanges": [
            r.model_dump(mode="json") for r in draft.external_exchanges
        ],
        "boundary_checks": assessment.get("boundary_checks", []),
    }


def interaction_handoff(brief, draft: Draft) -> dict:
    """Save concrete unresolved law questions without executing interaction work."""
    result = deferred.interaction_handoff(brief, project(draft))
    result["internal_transfer_candidates"] = [
        {
            "process": process.model_dump(mode="json"),
            "status": "deferred_to_interaction",
            "question_for_interaction": (
                "Using the proposed function and complete assembled contributions, "
                "does this transfer remove the same physical amount from its "
                "source balance that it adds to its destination? Identify the "
                "balance coordinates and include every gain and conversion. "
                "Opposite topology signs alone do not establish cancellation; "
                "an absent conversion is unknown, not one. Clarify the intended "
                "direction and positive-rate versus signed-flux interpretation."
            ),
        }
        for process in draft.processes
        if process.kind == "transfer"
    ]
    result["external_role_candidates"] = []
    for assignment in draft.external_exchanges:
        reference = _reference(draft, assignment)
        result["external_role_candidates"].append(
            {
                "assignment": assignment.model_dump(mode="json"),
                "current_reference": reference,
                "reference_matches_assignment": reference is not None
                and assignment.reference_sha256 == content_hash(reference),
                "status": "deferred_to_interaction",
                "question_for_interaction": (
                    "Does this assembled contribution implement the claimed "
                    "external loss, with consistent function and use signs in "
                    "the stated physical coordinates? Determine whether the "
                    "function denotes a nonnegative outward rate or signed flux; "
                    "inspect its full law, parameters, conversions and all uses. "
                    "Do not infer its direction from the topology sign alone. "
                    "Check the relevant public boundary behavior and ensure an "
                    "internal transfer is not counted as an additional loss."
                ),
            }
        )
    result["additional_scope"] = (
        "Boundary assignments and transfer labels identify intended review "
        "candidates only. Equivalent inlined processes still require assembled-law "
        "inspection; an absent named candidate is not a conservation verdict."
    )
    result["function_checks_executed"] = 0
    result["scientific_clarification_calls"] = 0
    return result


def model_view(draft: Draft):
    """Display ordinary-term indices only where precise role references need them."""
    view = prompts.model_view(base(draft))
    for balance in view["balances"]:
        meanings = balance.pop("local_process_meanings_in_order")
        balance["local_process_meanings_by_index"] = {
            str(index): meaning for index, meaning in enumerate(meanings)
        }
    view["external_exchanges"] = [
        r.model_dump(mode="json", exclude={"reference_sha256"})
        for r in draft.external_exchanges
    ]
    view["notation"] += (
        " Local-process indices start at 0 in the displayed ordinary f terms, "
        "with meanings shown by index. Shared uses are separate and are not "
        "counted in these indices. Indices are references, not new variables."
    )
    return view


def stage_text(stage: str):
    """Keep the model/data instruction short and stage appropriate."""
    text = prompts.stage_text(stage)
    if stage == "variables":
        text += (
            "\nRead the training highlights alongside the public requirements; "
            "use them to consider which generated quantities may be needed. "
            "A source-only realization record is provisional until topology "
            "is complete."
        )
    else:
        text += (
            "\nCompare the proposed dependencies with the training highlights. "
            "By topology completion, every realization must identify a generated "
            "implementing component, not supplied sources alone."
        )
    if stage == "repair":
        text += (
            "\nAnswer any compatibility_questions using compatibility_decisions. "
            "These are conditional exact-reproduction conflicts, not fitted-error "
            "tests. Choose a scientifically justified revision, retain an "
            "approximation "
            "with an explanation, or identify an unsupported assumption. A revise "
            "answer alone does not edit the model."
        )
    return text


def payload(
    stage,
    brief,
    context,
    target_definitions,
    enriched_brief,
    draft: Draft,
    contract,
    last=None,
    continuation=None,
    *,
    boundary_contract=None,
    training_highlights=None,
    compatibility_questions=(),
):
    """Use one readable model plus explicit public coverage and advisory evidence."""
    packet = prompts.payload(
        stage,
        brief,
        context,
        target_definitions,
        enriched_brief,
        base(draft),
        contract,
        last,
        continuation,
    )
    packet["stage_instructions"] = stage_text(stage)
    packet["model_view"] = model_view(draft)
    packet["response_template"] = response_model(stage)(
        stage_complete=False
    ).model_dump(mode="json")
    if training_highlights is not None:
        packet["training_highlights"] = training_highlights
    if boundary_contract is not None:
        boundary_contract.validate_public(brief)
        if boundary_contract.obligations and stage != "variables":
            packet["required_external_effects"] = [
                r.model_dump(mode="json") for r in boundary_contract.obligations
            ]
            packet["external_effect_rules"] = (
                "Identify the existing contribution realizing each listed external "
                "loss. Use external_exchanges with obligation_id, receiver and "
                "exactly one of "
                "local_term_index or shared_process, plus its physical explanation. "
                "Indices refer to the displayed ordinary terms, starting at 0; shared "
                "uses are listed separately. Keep an ordinary local process ordinary. "
                "An assignment is tied to that exact term: after changing it, repeat "
                "the assignment to confirm its meaning. remove_external_exchanges uses "
                "obligation_id. This role claim does not verify loss signs "
                "or conservation."
            )
    if stage in {"equations", "repair"}:
        check = assess(
            brief,
            context,
            target_definitions,
            draft,
            graph_contract=contract,
            boundary_contract=boundary_contract,
        )
        added = [
            row
            for row in check["errors"]
            if row["code"]
            in {"mechanism_realization_source_only", "external_exchange_coverage"}
        ]
        packet["feedback"]["issues"].extend(added)
        for row in packet["requirements_checklist"]:
            if row["id"] == "structural_validity":
                row["status"] = "passed" if check["eligible"] else "failed"
            if any(
                issue.get("requirement") == row.get("requirement")
                and issue["code"] == "mechanism_realization_source_only"
                for issue in added
            ):
                row["implementation_record_status"] = "source_only"
                row["status"] = "incomplete"
        packet["requirements_checklist"].extend(
            {**row, "id": f"external_exchange:{row['id']}", "checked_at": "topology"}
            for row in check["boundary_checks"]
        )
    if compatibility_questions:
        packet["compatibility_questions"] = list(compatibility_questions)
    if stage == "repair":
        packet["editing_rules"] += (
            "\ncompatibility_decisions is a reply-control list, never a scientific "
            "quantity. Each answer has question_id, disposition (revise, "
            "retain_approximation, or dispute_assumption) and explanation. "
            "Boundary assignments are separate from general mechanism_realizations "
            "and do not introduce equations."
        )
    feedback = packet["feedback"]
    if "repair_agenda" in feedback:
        feedback["repair_agenda"]["representation_issues"] = list(
            dict.fromkeys(row["code"] for row in feedback["issues"])
        )
        feedback["repair_agenda"]["failed_requirements"] = [
            row["id"]
            for row in packet["requirements_checklist"]
            if row["status"] in {"failed", "missing", "inconsistent", "incomplete"}
        ]
    return packet
