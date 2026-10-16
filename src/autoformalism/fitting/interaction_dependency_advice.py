"""Optional questions about additive restrictions, never missing-edge diagnoses.

This read-only helper consumes M37/M38 evidence. It does not change their
policies, fit a model, choose a source, or authorize a constructor edit.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.expressions import ModelValidationError, RestrictedParser
from autoformalism.fitting import interaction_advice as advice
from autoformalism.fitting import interaction_feedback as feedback
from autoformalism.fitting import interaction_mismatches as mismatches
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import Sha256


class Policy(StrictSchema):
    """An explicit bounded investigation allowance, not an arity or fit quota."""

    protocol: Literal["interaction-dependency-advice-policy-1"] = (
        "interaction-dependency-advice-policy-1"
    )
    enabled: bool = False
    maximum_questions: int = Field(default=1, ge=1, le=3)
    extraction: mismatches.Policy = Field(default_factory=mismatches.Policy)


class LawInspection(StrictSchema):
    """Conservative expression structure with ODE values treated as leaves."""

    location_id: str
    status: Literal["additive_unary", "joint_or_unresolved", "single_dynamic_source"]
    additive_term_arguments: tuple[tuple[str, ...], ...]
    dynamic_arguments: tuple[str, ...]
    affected_targets: tuple[str, ...]
    reason: str
    mathematical_separability_proved: Literal[False] = False


class Question(StrictSchema):
    """A request for discriminating evidence, not a requested complexity increase."""

    target: str
    law_refs: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    affected_targets: tuple[str, ...]
    question: str
    alternatives: tuple[str, ...]
    required_comparison: str
    disconfirmation: str
    missing_dependency_diagnosed: Literal[False] = False
    edit_authorized: Literal[False] = False


class Review(StrictSchema):
    """Deterministic training-only projection; no constructor transition occurs."""

    protocol: Literal["interaction-dependency-advice-1"] = (
        "interaction-dependency-advice-1"
    )
    policy: Policy
    source_packet_sha256: Sha256
    source_verification: Literal["caller_required", "verified_snapshot"]
    status: Literal["disabled", "withheld", "inapplicable", "investigation_available"]
    reason: str
    readiness: advice.Readiness
    inspections: tuple[LawInspection, ...]
    initialization_dependencies: dict[str, tuple[str, ...]]
    questions: tuple[Question, ...]
    omitted_questions: int = Field(ge=0)
    limitations: tuple[str, ...]
    optimization_calls: Literal[0] = 0
    rollout_calls: Literal[0] = 0
    llm_calls: Literal[0] = 0
    model_changed: Literal[False] = False
    validation_used: Literal[False] = False
    test_data_opened: Literal[False] = False


class _Arguments:
    """Track source sets through aliases without evaluation or algebraic expansion."""

    def __init__(self, packet: feedback.Packet) -> None:
        view = packet.model_view
        self.dynamic = (
            {s.name for s in view.states}
            | set(view.public_channel_roles.auxiliaries)
            | set(view.public_channel_roles.external_inputs)
        )
        self.fixed = {p.name for p in view.parameters} | set(
            view.public_channel_roles.fixed_covariates
        )
        parser = RestrictedParser()
        self.processes = {
            p.name: parser.parse(p.expression, location=f"process:{p.name}").tree.body
            for p in view.processes
        }
        self.visits = 0

    def terms(
        self, node: ast.AST, visiting: frozenset[str] = frozenset()
    ) -> tuple[frozenset[str], ...]:
        """Split sums and fixed scaling only; nonlinear compositions stay whole."""
        self.visits += 1
        if self.visits > 4096:
            raise ValueError("dependency inspection node budget exceeded")
        if isinstance(node, ast.Constant):
            return (frozenset(),)
        if isinstance(node, ast.Name):
            if node.id in self.dynamic:
                return (frozenset({node.id}),)
            if node.id in self.fixed:
                return (frozenset(),)
            if node.id in self.processes and node.id not in visiting:
                return self.terms(self.processes[node.id], visiting | {node.id})
            raise ValueError("unresolved or cyclic process dependency")
        if isinstance(node, ast.UnaryOp):
            return self.terms(node.operand, visiting)
        if isinstance(node, ast.BinOp):
            left, right = (
                self.terms(node.left, visiting),
                self.terms(node.right, visiting),
            )
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return left + right
            left_sources, right_sources = (
                frozenset().union(*left),
                frozenset().union(*right),
            )
            if isinstance(node.op, ast.Mult):
                if not left_sources:
                    return right
                if not right_sources:
                    return left
            if isinstance(node.op, ast.Div) and not right_sources:
                return left
            return (left_sources | right_sources,)
        if isinstance(node, ast.Call):
            arguments = [self.terms(arg, visiting) for arg in node.args]
            return (frozenset().union(*(s for group in arguments for s in group)),)
        raise ValueError("unsupported dependency expression")


def _inspect(packet: feedback.Packet) -> tuple[LawInspection, ...]:
    rows = []
    for law in packet.laws:
        if law.kind == "observation_mapping":
            continue
        try:
            parsed = RestrictedParser().parse(law.expression, location=law.location_id)
            terms = _Arguments(packet).terms(parsed.tree.body)
            arguments = tuple(sorted(frozenset().union(*terms)))
            status = (
                "joint_or_unresolved"
                if any(len(term) > 1 for term in terms)
                else "additive_unary"
                if len(arguments) > 1
                else "single_dynamic_source"
            )
            reason = (
                "Only sums, signs and fixed scaling separate the effective "
                "dynamic arguments. This describes the current expression, "
                "not the allowed constructor function family."
                if status == "additive_unary"
                else "No multi-source additive-unary restriction is certified."
            )
        except (ValueError, ModelValidationError):
            status, terms, arguments = "joint_or_unresolved", (), ()
            reason = "Expression or process expansion is unresolved; no arity advice."
        rows.append(
            LawInspection(
                location_id=law.location_id,
                status=status,
                additive_term_arguments=tuple(tuple(sorted(t)) for t in terms),
                dynamic_arguments=arguments,
                affected_targets=law.affected_targets,
                reason=reason,
            )
        )
    return tuple(rows)


def _initial_dependencies(packet: feedback.Packet) -> dict[str, tuple[str, ...]]:
    """Report initial-time information separately; never add it to RHS arguments."""
    view, parser = packet.model_view, RestrictedParser()
    parameters = {p.name for p in view.parameters}
    result = {}
    for initial in view.initial_conditions:
        result[initial.state] = (
            tuple(
                sorted(
                    parser.parse(
                        initial.expression, location=f"initial:{initial.state}"
                    ).symbols
                    - parameters
                )
            )
            if initial.expression is not None
            else ()
        )
    for state, rule in view.initialization_plan.rules.items():
        initial = rule.initial
        result[state] = (
            tuple(
                sorted(
                    parser.parse(
                        initial.expression, location=f"initial:{state}"
                    ).symbols
                    - {p.name for p in initial.parameters}
                )
            )
            if initial.mode == "map"
            else ()
        )
    return result


def draft(packet: feedback.Packet, policy: Policy | None = None) -> Review:
    """Pure projection; callers must verify M37 provenance before using advice."""
    policy = policy or Policy()
    route = advice.readiness(packet)
    inspections = _inspect(packet) if policy.enabled else ()
    questions = []
    if not policy.enabled:
        status, reason = "disabled", "Optional dependency investigation is disabled."
    elif route.hypothesis_status != "permitted":
        status, reason = (
            "withheld",
            (
                "Verified poor output and a matching retained-point stationarity or "
                "restricted-optimum witness are required. " + route.rationale
            ),
        )
    else:
        extracted = mismatches.extract(packet, policy.extraction)
        for target in extracted.targets:
            findings = [
                f
                for f in extracted.findings
                if f.target == target and f.availability == "measured"
            ]
            laws = [
                law
                for law in inspections
                if law.status == "additive_unary" and target in law.affected_targets
            ]
            evidence = tuple(sorted({ref for f in findings for ref in f.evidence_ids}))
            if (
                not findings
                or not laws
                or not set(evidence) <= packet.evidence_catalog.keys()
            ):
                continue
            questions.append(
                Question(
                    target=target,
                    law_refs=tuple(law.location_id for law in laws),
                    evidence_ids=evidence,
                    affected_targets=tuple(
                        sorted({t for law in laws for t in law.affected_targets})
                    ),
                    question=(
                        "The displayed current laws separate their dynamic "
                        "arguments additively, and the cited training discrepancies "
                        "remain after the scoped fitting check. Is there controlled "
                        "training evidence that the effect of one quantity varies "
                        "with another after accounting for coefficients, causal "
                        "initialization and existing feedback? If not, retain this "
                        "representation and identify the missing evidence. No new "
                        "argument or increase in complexity is requested."
                    ),
                    alternatives=(
                        route.alternative_action,
                        "Allowed initialization rules or their fitted values "
                        "may explain the mismatch.",
                        "Limited excitation, measurement variation or an "
                        "inappropriate comparison may explain apparent "
                        "conditional effects; do not change the data contract.",
                        "Existing nonlinear forms or indirect state feedback "
                        "may suffice without new dependencies.",
                    ),
                    required_comparison=(
                        "First inspect matched training contrasts and the permitted "
                        "initial information; selected extreme examples are not "
                        "controlled contrasts. Only with supporting evidence and "
                        "the authoritative constructor contract, compare a bounded "
                        "hypothesis/refit with continued fitting of the incumbent "
                        "under predetermined equal budgets. State a predicted "
                        "change in the cited discrepancy and preserve all targets "
                        "and shared consumers. New sources require topology review."
                    ),
                    disconfirmation=(
                        "Absent conditional evidence, or failure of the predicted "
                        "discrepancy change after a verified refit, leaves the "
                        "hypothesis unsupported. Lower aggregate error alone does "
                        "not identify a missing dependency or its mechanism."
                    ),
                )
            )
        status = "investigation_available" if questions else "inapplicable"
        reason = (
            "A bounded investigative option is available; "
            "joint dependence is not diagnosed."
            if questions
            else "No measured, cited discrepancy and additive-unary "
            "multi-source law coincide."
        )
    return Review(
        policy=policy,
        source_packet_sha256=public.content_sha256(packet.model_dump(mode="json")),
        source_verification="caller_required",
        status=status,
        reason=reason,
        readiness=route,
        inspections=inspections,
        initialization_dependencies=_initial_dependencies(packet)
        if policy.enabled
        else {},
        questions=tuple(questions[: policy.maximum_questions]),
        omitted_questions=max(0, len(questions) - policy.maximum_questions),
        limitations=(
            "A restricted fitted optimum concerns this candidate and parameter "
            "domain, not global model adequacy.",
            "The packet has no controlled joint-effect test; residuals do not "
            "locate missing arguments.",
            "Initial-time dependencies can change future trajectories but are "
            "not continuous RHS dependencies.",
            "ODE states are leaves; indirect dynamics and nonlinear algebraic "
            "composition can express joint behavior.",
            "Syntactic inspection is conservative, not a general separability "
            "proof or an arity quota.",
            "The packet omits the time-symbol binding; unclassified symbols "
            "remain unresolved instead of being guessed as time.",
            "No function family, source, sign, observation or initialization "
            "edit is authorized; preserve the incumbent on unsupported revisions.",
        ),
    )


def review(
    diagnostic_folder: Path, packet_folder: Path, policy: Policy | None = None
) -> Review:
    """Verify the immutable training packet before drafting optional questions."""
    packet = feedback.completed(diagnostic_folder, packet_folder)
    return draft(packet, policy).model_copy(
        update={"source_verification": "verified_snapshot"}
    )
