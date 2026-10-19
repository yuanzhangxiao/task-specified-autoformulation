"""Opt-in conditional evidence for declared, non-executable initializer graphs.

Shared graph maps can imply equality without knowing their formulas. A trajectory
root cannot: neither matching first readings nor matching prefixes establish an
estimator equality contract. Historical evidence and policies remain untouched.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    dependency_summary,
    validate_contract,
    validate_handoff,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    InitialRule,
    TrainingCompatibilityEvidence,
    _closure,
    _graph,
    _pair,
    diagnose,
    validate_evidence,
)
from autoformalism.search.topology_initialization import (
    INITIAL_TIME_POLICY,
    _initial_scope,
    _saved_pair,
    initial_time_questions,
    recheck_closed_witnesses,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-initialization-graph-evidence-1"
PREFIX_REASON = "prefix-inferred root; t0 equality insufficient"


def _derived_evidence(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    contract: GraphConstructionContract,
    handoff: GraphDependencyHandoff,
    summary: dict,
) -> TrainingCompatibilityEvidence:
    """Seal conservative equality rules without evaluating or selecting formulas."""
    states = {
        variable.name
        for variable in draft.variables
        if variable.definition == "differential"
    }
    for state in sorted(states):
        old = evidence.contract.initial_rules.get(state)
        if (
            old is not None
            and old.mode == "observed_initial"
            and contract.observed_initial_channels.get(state) != state
        ):
            raise ValueError(f"fixed observed initialization cannot change for {state}")
    rules = {}
    for state in sorted(states):
        dependencies = summary[state]
        arguments = tuple(dependencies["public_initials"])
        if dependencies["trajectory_roots"]:
            rule = InitialRule(mode="unknown")
        elif arguments:
            rule = InitialRule(mode="common_map", arguments=arguments)
        else:
            rule = InitialRule(mode="shared_parameter")
        rules[state] = rule.model_dump(mode="json")
    raw = evidence.model_dump(mode="json", exclude={"sha256"})
    raw["contract"]["initial_rules"] = rules
    raw["contract"]["provenance"] += (
        f" {PROTOCOL}; original evidence={evidence.sha256}; "
        f"graph contract={content_hash(contract.model_dump(mode='json'))}; "
        f"graph declarations={content_hash(handoff.model_dump(mode='json'))}. "
        "Only transitive public t0 dependencies and shared-root equality were "
        "lowered; trajectory-root equality remains unavailable. No initializer "
        "formula, fitted value, or prefix estimator was evaluated."
    )
    derived = TrainingCompatibilityEvidence.model_validate(
        {**raw, "sha256": content_hash(raw)}
    )
    validate_evidence(derived)
    return derived


def _trajectory_roots(scope: set[str], summary: dict) -> list[str]:
    """Return only roots relevant to the compared states and their initial DAGs."""
    return sorted(
        {
            root
            for state in scope
            for root in summary.get(state, {}).get("trajectory_roots", ())
        }
    )


def _annotate_report(report: dict, summary: dict, prefix_reasons: list[str]) -> dict:
    """Explain only affected subsystem deferrals; retain independent diagnostics."""
    result = deepcopy(report)
    for diagnostic in result["diagnostics"]:
        if diagnostic["property"] != "closed_subsystem":
            continue
        roots = _trajectory_roots(set(diagnostic["scope"]), summary)
        if roots:
            # The derived unknown rule prevents any equality comparison; fail
            # closed if a future change accidentally introduces such a witness.
            if diagnostic["status"] == "conditional_conflict":
                raise ValueError("trajectory-root equality was unexpectedly asserted")
            diagnostic["status"] = "deferred"
            diagnostic["unavailable"] = sorted(
                {*diagnostic["unavailable"], *prefix_reasons}
            )
            diagnostic["prefix_inferred_roots"] = roots
    return result


def _recheck_questions(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    questions: list[dict],
    summary: dict,
    prefix_reasons: list[str],
) -> dict:
    """Recompute saved pairs and retain all original records with their provenance."""
    replay = recheck_closed_witnesses(
        draft, evidence, questions, initial_time_policy=INITIAL_TIME_POLICY
    )
    result = replay.model_dump(mode="json")
    result["original_questions"] = deepcopy(questions)
    graph, kinds, _ = _graph(draft, evidence.context)
    checks = []
    for index, question in enumerate(questions):
        if (
            not isinstance(question, dict)
            or question.get("property") != "closed_subsystem"
        ):
            checks.append(
                {
                    "source_question_index": index,
                    "status": "retained_unresolved",
                    "reason": "Saved question is outside matched-subsystem replay; "
                    "independent current topology diagnostics remain available.",
                    "retained_unresolved": True,
                    "scientific_repair_established": False,
                }
            )
            continue
        witnesses = question.get("witnesses")
        items = (
            witnesses if isinstance(witnesses, (tuple, list)) and witnesses else [None]
        )
        for witness in items:
            selected = deepcopy(question)
            if witness is not None:
                selected["witnesses"] = [deepcopy(witness)]
            checked = recheck_closed_witnesses(
                draft, evidence, [selected], initial_time_policy=INITIAL_TIME_POLICY
            )
            for original in checked.checks:
                check: dict[str, Any] = deepcopy(original)
                check.update(
                    source_question_index=index,
                    original_witness=deepcopy(witness),
                    recheck_status=original["status"],
                    scientific_repair_established=False,
                )
                target = question.get("target")
                scope = (
                    (
                        _initial_scope(target, graph, kinds)
                        if check.get("conditioning_policy") == INITIAL_TIME_POLICY
                        else _closure(target, graph)
                    )
                    if isinstance(target, str)
                    else set()
                )
                roots = _trajectory_roots(scope, summary)
                if check["status"] != "invalid_saved_witness":
                    target, saved, left, right = _saved_pair(
                        question, witness, evidence
                    )
                    check["recomputed_witness"] = _pair(
                        left,
                        saved.sample_indices[0],
                        right,
                        saved.sample_indices[1],
                        target,
                        evidence,
                    ).model_dump(mode="json")
                if roots and check["status"] != "invalid_saved_witness":
                    check.update(
                        status="deferred",
                        reason=PREFIX_REASON,
                        reasons=sorted({*check.get("reasons", ()), *prefix_reasons}),
                        prefix_inferred_roots=roots,
                    )
                check["retained_unresolved"] = check["status"] in {
                    "deferred",
                    "unavailable",
                    "invalid_saved_witness",
                    "uncertainty_limited",
                }
                checks.append(check)
    result["checks"] = checks
    result["scientific_pass"] = False
    return result


def inspect_graph_initialization(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    contract: GraphConstructionContract,
    handoff: GraphDependencyHandoff,
    previous_questions: list[dict] | tuple[dict, ...] = (),
) -> dict:
    """Inspect one explicitly selected graph policy using sealed training evidence.

    This is structural qualification with conditional equality diagnostics. It
    does not select analytic functions, infer roots, fit parameters, prove state
    identifiability, or establish that a changed declaration repairs the science.
    """
    validate_evidence(evidence)
    contract = validate_contract(contract, draft=draft, context=evidence.context)
    handoff = validate_handoff(handoff, contract, draft=draft, context=evidence.context)
    summary = dependency_summary(handoff, contract)
    derived = _derived_evidence(draft, evidence, contract, handoff, summary)
    prefix_reasons = [PREFIX_REASON]
    if contract.prefix_policy is None:
        prefix_reasons.append("observation-window policy unresolved")
    declared = _annotate_report(
        diagnose(draft, derived).model_dump(mode="json"), summary, prefix_reasons
    )
    initial = _recheck_questions(
        draft, derived, initial_time_questions(draft, evidence), summary, prefix_reasons
    )
    retained = _recheck_questions(
        draft, derived, list(previous_questions), summary, prefix_reasons
    )
    return {
        "protocol": PROTOCOL,
        "original_evidence_sha256": evidence.sha256,
        "derived_evidence_sha256": derived.sha256,
        "derived_evidence_provenance": derived.contract.provenance,
        "initialization_contract": contract.model_dump(mode="json"),
        "handoff": handoff.model_dump(mode="json"),
        "initialization_dependency_summary": summary,
        "prefix_policy": (
            contract.prefix_policy.model_dump(mode="json")
            if contract.prefix_policy is not None
            else None
        ),
        "declared_report": declared,
        "initial_time_report": initial,
        "retained_witness_report": retained,
        "qualification": "structural_only",
        "qualification_reason": "Initializer dependencies are declared; analytic "
        "formulas and a numerical prefix estimator have not been supplied here. "
        "Equal prefixes do not establish root equality or unique hidden states. "
        "Deferred checks are unresolved, not scientific passes or repairs.",
        "executable_initializers_selected": False,
        "scientific_pass": False,
        "acceptance_gate": False,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }
