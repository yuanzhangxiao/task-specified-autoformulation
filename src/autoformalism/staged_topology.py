"""Scientific inventory validation and lowering into the existing graph compiler."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

from autoformalism.expressions import ValidationContext
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.schemas.staged import (
    InteractionPolarity,
    InteractionTargetKind,
    ProposedTopologyCandidate,
    ProposedTopologyProcess,
    ProposedTopologyState,
    TopologyCandidate,
    TopologyInteraction,
    TopologyTargetMapping,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    EquationPolarityPolicy,
    EquationTerm,
    ExactSourcePolarityRule,
    InventoryRevision,
    LegacyEquationTerm,
    ModelingLimits,
    OuterWeightSign,
    PublicScientificBrief,
    PublicVariable,
    ScientificRequirement,
    ScientificVariable,
    TargetDependency,
    VariableReply,
)
from autoformalism.staging import enrich_topology_proposal
from autoformalism.targets import PublicTargetContract


def content_hash(value: object) -> str:
    """Hash canonical JSON without relying on filesystem locations."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_scientific_brief(
    public_prompt: str,
    context: ValidationContext,
    target_contract: PublicTargetContract,
    mechanism_spec: MechanismEvaluationSpec,
    *,
    limits: ModelingLimits | None = None,
) -> PublicScientificBrief:
    """Keep public Sections A-E and positive requirements; never load trajectories.

    This separately versioned protocol deliberately does not import the old
    mass/rate keyword inference as a mandatory representation rule.
    """
    digest = hashlib.sha256(public_prompt.encode()).hexdigest()
    if digest != target_contract.public_prompt_sha256:
        raise ValueError("public prompt differs from target contract")
    if digest != mechanism_spec.public_prompt_sha256:
        raise ValueError("public prompt differs from mechanism specification")
    sections = re.split(r"(?m)^F\.\s+Required response\s*$", public_prompt, maxsplit=1)
    if len(sections) != 2:
        raise ValueError("missing reviewed Required response section boundary")
    if {item.target_channel for item in target_contract.targets} != set(
        context.targets
    ):
        raise ValueError("public target sets differ")
    variables = tuple(
        PublicVariable(name=name, data_role=role)
        for role, names in (
            ("target", context.targets),
            ("auxiliary", context.auxiliaries),
            ("external_input", context.external_inputs),
            ("covariate", context.fixed_covariates),
            ("time", (context.time_symbol,)),
        )
        for name in names
    )
    return PublicScientificBrief(
        scientific_context=sections[0].rstrip(),
        public_variables=variables,
        requirements=tuple(
            ScientificRequirement(
                id=item.id,
                public_requirement=item.public_requirement,
                targets=item.required_targets,
                drivers=item.required_drivers,
                positive_requirements=(
                    ("Represent dynamic memory in the required pathway.",)
                    if item.requires_dynamic_memory
                    else ()
                ),
                public_pathway_sign=item.required_sign,
                requires_dynamic_memory=item.requires_dynamic_memory,
            )
            for item in mechanism_spec.required_mechanisms
        ),
        target_dependencies=tuple(
            TargetDependency(
                target=target.target_channel,
                acceptable_sources=dependency.acceptable_symbols,
                public_requirement=dependency.public_requirement,
            )
            for target in target_contract.targets
            for dependency in target.required_dependencies
        ),
        limits=limits or ModelingLimits(),
    )


def compile_equation_polarity_policy(
    brief: PublicScientificBrief,
    selected_lhs: str,
    *,
    proposer_owns_unfixed_sources: bool = False,
) -> EquationPolarityPolicy:
    """Compile only exact fixed signs certified by reviewed public evidence.

    A public mechanism sign applies here only when its complete driver set is a
    direct term of one of its public targets. Indirect nonlinear path signs are
    not inferred by multiplying edge labels. Every other term remains signed
    and is therefore assigned the explicit ``unrestricted`` default.
    """
    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for requirement in brief.requirements:
        if (
            selected_lhs not in requirement.targets
            or requirement.public_pathway_sign not in {"positive", "negative"}
            or not requirement.drivers
        ):
            continue
        key = tuple(sorted(requirement.drivers))
        existing = grouped.get(key)
        if existing is not None and existing["outer_weight_sign"] != (
            requirement.public_pathway_sign
        ):
            raise ValueError(
                "conflicting public polarity evidence for exact source set "
                f"{list(key)} on {selected_lhs}"
            )
        if existing is None:
            grouped[key] = {
                "sources": key,
                "outer_weight_sign": requirement.public_pathway_sign,
                "requirement_ids": [requirement.id],
            }
        else:
            identifiers = existing["requirement_ids"]
            assert isinstance(identifiers, list)
            identifiers.append(requirement.id)
    return EquationPolarityPolicy(
        selected_lhs=selected_lhs,
        unfixed_source_ownership=(
            "proposer"
            if proposer_owns_unfixed_sources
            else "runtime_requires_unrestricted"
        ),
        fixed_exact_source_sets=tuple(
            ExactSourcePolarityRule.model_validate(item)
            for _, item in sorted(grouped.items())
        ),
    )


def audit_equation_polarity_policy(
    equation: EquationDefinition, policy: EquationPolarityPolicy
) -> dict[str, object]:
    """Compare every accepted term with its public-evidence polarity policy."""
    if equation.name != policy.selected_lhs:
        raise ValueError("equation and polarity-policy left-hand sides differ")
    fixed = {
        tuple(sorted(item.sources)): item.outer_weight_sign
        for item in policy.fixed_exact_source_sets
    }
    rows: list[dict[str, object]] = []
    for index, term in enumerate(equation.terms):
        key = tuple(sorted(term.sources))
        expected = fixed.get(key, policy.default_outer_weight_sign)
        if isinstance(term, LegacyEquationTerm):
            observed = "positive" if term.outer_sign == "add" else "negative"
        else:
            observed = term.outer_weight_sign.value
        rows.append(
            {
                "term_index": index,
                "sources": list(key),
                "expected_outer_weight_sign": expected,
                "observed_outer_weight_sign": observed,
                "public_fixed_evidence": key in fixed,
                "correct": observed == expected,
            }
        )
    return {
        "selected_lhs": equation.name,
        "term_count": len(rows),
        "fixed_evidence_term_count": sum(
            bool(item["public_fixed_evidence"]) for item in rows
        ),
        "unrestricted_term_count": sum(
            item["expected_outer_weight_sign"] == "unrestricted" for item in rows
        ),
        "correct_term_count": sum(bool(item["correct"]) for item in rows),
        "passed": all(bool(item["correct"]) for item in rows),
        "rows": rows,
    }


def audit_explicit_equation_polarity(
    equation: EquationDefinition, policy: EquationPolarityPolicy
) -> dict[str, object]:
    """Check explicit public signs while leaving all other signs proposer-owned."""
    if equation.name != policy.selected_lhs:
        raise ValueError("equation and polarity-policy left-hand sides differ")
    fixed = {
        tuple(sorted(item.sources)): item.outer_weight_sign
        for item in policy.fixed_exact_source_sets
    }
    rows: list[dict[str, object]] = []
    for index, term in enumerate(equation.terms):
        key = tuple(sorted(term.sources))
        expected = fixed.get(key)
        if isinstance(term, LegacyEquationTerm):
            observed = "positive" if term.outer_sign == "add" else "negative"
        else:
            observed = term.outer_weight_sign.value
        rows.append(
            {
                "term_index": index,
                "sources": list(key),
                "expected_outer_weight_sign": expected,
                "observed_outer_weight_sign": observed,
                "public_fixed_evidence": expected is not None,
                "proposer_owned": expected is None,
                "correct": expected is None or observed == expected,
            }
        )
    return {
        "selected_lhs": equation.name,
        "term_count": len(rows),
        "fixed_evidence_term_count": sum(
            bool(item["public_fixed_evidence"]) for item in rows
        ),
        "proposer_owned_term_count": sum(bool(item["proposer_owned"]) for item in rows),
        "correct_fixed_evidence_term_count": sum(
            bool(item["public_fixed_evidence"]) and bool(item["correct"])
            for item in rows
        ),
        "passed": all(bool(item["correct"]) for item in rows),
        "rows": rows,
    }


def merge_variable_reply(
    brief: PublicScientificBrief,
    inventory: tuple[ScientificVariable, ...],
    reply: VariableReply,
) -> tuple[ScientificVariable, ...]:
    """Validate an atomic addition/reuse step without semantic deduplication."""
    public = {item.name: item.data_role for item in brief.public_variables}
    merged = {item.name: item for item in inventory}
    for item in reply.variables:
        if item.name in APPROVED_FUNCTION_ARITY or item.name.startswith("af_internal_"):
            raise ValueError(f"reserved variable name: {item.name}")
        role = public.get(item.name, "internal")
        generated = item.definition in {"differential", "algebraic"}
        if role in {"target", "internal"} and not generated:
            raise ValueError(f"{role} {item.name} must be differential or algebraic")
        if role in {"external_input", "covariate", "time"} and generated:
            raise ValueError(f"{role} {item.name} must be supplied or unused")
        existing = merged.get(item.name)
        if (
            existing
            and existing.definition == "unused"
            and item.definition != "unused"
            and role != "internal"
        ):
            merged[item.name] = item
            continue
        if existing and existing.definition != item.definition:
            raise ValueError(
                f"changing definition of {item.name} requires inventory revision"
            )
        # Rephrasing a reused role does not change the accepted hypothesis.
        merged.setdefault(item.name, item)
    if (
        sum(
            item.definition in {"differential", "algebraic"} for item in merged.values()
        )
        > brief.limits.generated_variables
    ):
        raise ValueError("generated variable limit exceeded")
    return tuple(merged.values())


def merge_variable_reply_partially(
    brief: PublicScientificBrief,
    inventory: tuple[ScientificVariable, ...],
    reply: VariableReply,
) -> tuple[tuple[ScientificVariable, ...], tuple[dict[str, object], ...]]:
    """Retain each independently valid variable and explain every rejection.

    Provider-schema failures still reject the response before this function is
    called.  This boundary handles cross-item runtime rules: public-role
    compatibility, immutable accepted definitions, reserved names, and the
    generated-variable limit.
    """
    merged = inventory
    decisions: list[dict[str, object]] = []
    for item in reply.variables:
        before = {variable.name: variable for variable in merged}
        try:
            candidate = merge_variable_reply(
                brief, merged, VariableReply(variables=(item,))
            )
        except ValueError as exc:
            decisions.append(
                {
                    "name": item.name,
                    "definition": item.definition,
                    "accepted": False,
                    "changed_inventory": False,
                    "error": str(exc),
                }
            )
            continue
        after = {variable.name: variable for variable in candidate}
        changed = before != after
        merged = candidate
        decisions.append(
            {
                "name": item.name,
                "definition": item.definition,
                "accepted": True,
                "changed_inventory": changed,
                "error": None,
            }
        )
    return merged, tuple(decisions)


def freeze_inventory(
    brief: PublicScientificBrief, inventory: tuple[ScientificVariable, ...]
) -> tuple[ScientificVariable, ...]:
    """Check target declarations; unselected public channels remain unused."""
    inventory = merge_variable_reply(brief, (), VariableReply(variables=inventory))
    names = {item.name for item in inventory}
    missing = [
        item.name
        for item in brief.public_variables
        if item.data_role == "target" and item.name not in names
    ]
    if missing:
        raise ValueError(f"missing generated targets: {missing}")
    active = {item.name for item in inventory if item.definition != "unused"}
    missing_drivers = {
        driver for item in brief.requirements for driver in item.drivers
    } - active
    if missing_drivers:
        raise ValueError(
            f"select required mechanism drivers: {sorted(missing_drivers)}"
        )
    for dependency in brief.target_dependencies:
        if not active.intersection(dependency.acceptable_sources):
            raise ValueError(
                f"select a required source for {dependency.target}: "
                f"{list(dependency.acceptable_sources)}"
            )
    return inventory


def validate_inventory_revision(
    brief: PublicScientificBrief,
    inventory: tuple[ScientificVariable, ...],
    revision: InventoryRevision,
) -> None:
    """Check a proposed inventory change without applying it or routing backward."""
    proposed = revision.variable
    existing = next((item for item in inventory if item.name == proposed.name), None)
    if existing is not None and existing.definition == proposed.definition:
        raise ValueError(
            f"inventory revision leaves {proposed.name} and its definition unchanged; "
            "return equation terms using the current inventory, or request an "
            "actual variable addition or definition change. A differential "
            "variable may be its own source; fitted parameters belong to the "
            "later functional-form stage, not the variable inventory"
        )
    candidate = tuple(
        proposed if item.name == proposed.name else item for item in inventory
    )
    if existing is None:
        candidate = (*candidate, proposed)
    # Revalidate public roles, required drivers and construction limits on the
    # hypothetical inventory. Accepted equations and inventory remain untouched.
    freeze_inventory(brief, candidate)


def validate_equation(
    inventory: tuple[ScientificVariable, ...],
    equations: tuple[EquationDefinition, ...],
    equation: EquationDefinition,
    limits: ModelingLimits,
) -> None:
    """Check local closure and algebraic cycles before accepting an equation."""
    selected = next((item for item in inventory if item.name == equation.name), None)
    if selected is None or selected.definition != equation.definition:
        raise ValueError("equation LHS does not match selected generated variable")
    if any(item.name == equation.name for item in equations):
        raise ValueError("equation is already defined")
    allowed = {item.name for item in inventory if item.definition != "unused"}
    if len(equation.terms) > limits.terms_per_equation:
        raise ValueError("terms per equation limit exceeded")
    if sum(len(item.terms) for item in (*equations, equation)) > limits.total_terms:
        raise ValueError("total term limit exceeded")
    unknown = {name for term in equation.terms for name in term.sources} - allowed
    if unknown:
        raise ValueError(f"undeclared or unused sources: {sorted(unknown)}")
    algebraics = {item.name for item in inventory if item.definition == "algebraic"}
    dependencies = {
        item.name: {source for term in item.terms for source in term.sources}
        & algebraics
        for item in (*equations, equation)
        if item.name in algebraics
    }
    for name in dependencies:
        if name in _ancestors(name, dependencies):
            raise ValueError(
                f"algebraic cycle through {name}; revise the current equation"
            )


def public_structure_checks(
    brief: PublicScientificBrief, equations: tuple[EquationDefinition, ...]
) -> tuple[dict[str, object], ...]:
    """Report necessary public paths separately from scientific correctness."""
    dependencies = {
        item.name: {source for term in item.terms for source in term.sources}
        for item in equations
    }
    differential = {
        item.name for item in equations if item.definition == "differential"
    }
    checks: list[dict[str, object]] = []
    for item in brief.target_dependencies:
        checks.append(
            {
                "requirement": item.public_requirement,
                "kind": "composition_path",
                "passed": bool(
                    set(item.acceptable_sources) & _ancestors(item.target, dependencies)
                ),
            }
        )
    for requirement in brief.requirements:
        for target in requirement.targets:
            ancestors = _ancestors(target, dependencies)
            for driver in requirement.drivers:
                checks.append(
                    {
                        "requirement": requirement.id,
                        "kind": "driver_path",
                        "target": target,
                        "driver": driver,
                        "passed": driver in ancestors,
                    }
                )
                if requirement.positive_requirements:
                    mediators = differential - {target, driver}
                    memory = any(
                        node in ancestors and driver in _ancestors(node, dependencies)
                        for node in mediators
                    )
                    checks.append(
                        {
                            "requirement": requirement.id,
                            "kind": "memory_path",
                            "target": target,
                            "driver": driver,
                            "passed": memory,
                        }
                    )
    return tuple(checks)


def _ancestors(name: str, dependencies: Mapping[str, set[str]]) -> set[str]:
    pending = list(dependencies.get(name, set()))
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node not in visited:
            visited.add(node)
            pending.extend(dependencies.get(node, set()) - visited)
    return visited


def lower_topology(
    brief: PublicScientificBrief,
    inventory: tuple[ScientificVariable, ...],
    equations: tuple[EquationDefinition, ...],
    context: ValidationContext,
) -> tuple[TopologyCandidate, dict[str, str]]:
    """Compile a closed scientific topology, preserving generated auxiliary meaning.

    Generated auxiliary names receive compiler-only aliases so the legacy
    expression evaluator cannot substitute their supplied future trajectories.
    Auxiliary supervision is disabled in this topology-only milestone.
    """
    inventory = freeze_inventory(brief, inventory)
    generated = {
        item.name
        for item in inventory
        if item.definition in {"differential", "algebraic"}
    }
    if generated != {item.name for item in equations} or len(equations) != len(
        generated
    ):
        raise ValueError("every generated variable must have exactly one equation")
    accepted: tuple[EquationDefinition, ...] = ()
    for equation in equations:
        validate_equation(inventory, accepted, equation, brief.limits)
        accepted += (equation,)
    aliases = {
        item.name: f"af_internal_aux_{index}"
        for index, item in enumerate(inventory)
        if item.definition in {"algebraic", "differential"}
        and item.name in context.auxiliaries
    }

    def name_of(name: str) -> str:
        return aliases.get(name, name)

    proposal = ProposedTopologyCandidate(
        candidate_id="staged_"
        + content_hash([item.model_dump(mode="json") for item in equations])[:16],
        states=tuple(
            ProposedTopologyState(name=name_of(item.name))
            for item in inventory
            if item.definition == "differential"
        ),
        processes=tuple(
            ProposedTopologyProcess(name=name_of(item.name))
            for item in inventory
            if item.definition == "algebraic"
        ),
        target_mappings=tuple(
            TopologyTargetMapping(channel=name, source=name) for name in context.targets
        ),
        interactions=tuple(
            TopologyInteraction(
                interaction_id=f"term_{equation_index}_{term_index}",
                target=name_of(equation.name),
                target_kind=(
                    InteractionTargetKind.STATE_DERIVATIVE
                    if equation.definition == "differential"
                    else InteractionTargetKind.ALGEBRAIC_PROCESS
                ),
                sources=tuple(name_of(name) for name in term.sources),
                polarity=_interaction_polarity(term),
                description=term.scientific_role,
            )
            for equation_index, equation in enumerate(equations)
            for term_index, term in enumerate(equation.terms)
        ),
    )
    topology = enrich_topology_proposal(proposal, context)
    roles = {name_of(item.name): item.scientific_role for item in inventory}
    topology = topology.model_copy(
        update={
            "states": tuple(
                item.model_copy(update={"description": roles[item.name]})
                for item in topology.states
            ),
            "processes": tuple(
                item.model_copy(update={"description": roles[item.name]})
                for item in topology.processes
            ),
        }
    )
    return topology, aliases


def _interaction_polarity(
    term: EquationTerm | LegacyEquationTerm,
) -> InteractionPolarity:
    """Lower new sign decisions while preserving legacy replay semantics."""
    if isinstance(term, LegacyEquationTerm):
        return (
            InteractionPolarity.ADDITIVE
            if term.outer_sign == "add"
            else InteractionPolarity.SUBTRACTIVE
        )
    return {
        OuterWeightSign.POSITIVE: InteractionPolarity.POSITIVE,
        OuterWeightSign.NEGATIVE: InteractionPolarity.NEGATIVE,
        OuterWeightSign.UNRESTRICTED: InteractionPolarity.UNRESTRICTED,
    }[term.outer_weight_sign]
