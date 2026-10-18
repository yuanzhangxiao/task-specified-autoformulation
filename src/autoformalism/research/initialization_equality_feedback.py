"""Concise, opt-in explanations of already qualified topology/data equalities.

This is a presentation adapter, not another diagnostic or a scientific repair.
It checks that the saved witness conditioning covers the current graph and the
currently declared initializers before restating its conditional implication.
Underlying questions, observations and assumptions remain available unchanged.
"""

from __future__ import annotations

import math
from copy import deepcopy

from autoformalism.expressions import ValidationContext
from autoformalism.search.construction_initialization_focus import (
    expand_compatibility_details,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import _closure, _graph
from autoformalism.search.topology_initialization import INITIAL_TIME_POLICY

POLICY = "model-specific-initialization-equality-1"
_MAX_SUMMARIES = 2


def _context(packet: dict) -> ValidationContext:
    """Recover declared public roles, without treating unknown symbols as inputs."""
    rows = packet["public_brief"]["public_variables"]
    roles = {
        role: tuple(row["name"] for row in rows if row["data_role"] == role)
        for role in ("target", "auxiliary", "external_input", "covariate", "time")
    }
    if len(roles["time"]) > 1:
        raise ValueError("ambiguous public time symbol")
    return ValidationContext(
        targets=roles["target"],
        auxiliaries=roles["auxiliary"],
        external_inputs=roles["external_input"],
        fixed_covariates=roles["covariate"],
        time_symbol=next(iter(roles["time"]), "t"),
    )


def _initial_leaves(target: str, graph: dict, kinds: dict) -> set[str]:
    """Expand only algebraic definitions; an ODE RHS cannot set a t0 value."""
    leaves, seen, pending = set(), set(), [target]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if kinds.get(name) == "algebraic":
            pending.extend(graph[name])
        else:
            leaves.add(name)
    return leaves


def _number(value: object) -> bool:
    """Accept finite recorded numbers only, never booleans or numeric strings."""
    return type(value) in (int, float) and math.isfinite(value)


def _pair(witness: dict) -> bool:
    """Require a paired, uncertainty-qualified witness at a common instant."""
    if not isinstance(witness, dict):
        return False
    for key in ("trajectories", "times", "observations", "sample_indices"):
        values = witness.get(key)
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            return False
    names, times, values, indices = (
        witness[key]
        for key in ("trajectories", "times", "observations", "sample_indices")
    )
    return (
        all(isinstance(name, str) and name for name in names)
        and names[0] != names[1]
        and all(_number(value) for value in [*times, *values])
        and times[0] == times[1]
        and values[0] != values[1]
        and all(type(index) is int and index >= 0 for index in indices)
        and witness.get("error_intervals_disjoint") is True
    )


def _rules(initialization: dict, *, consistent_runtime_rules: bool = False) -> dict:
    """Combine rules, optionally accepting an exact repeat of a fixed runtime rule.

    Historical projections reject every duplicate. The opt-in display path merges
    only identical runtime/retained rules; conflicting or proposer-only duplicate
    declarations remain unsupported. The original packet is never normalized.
    """
    fixed = initialization.get("fixed_runtime_rules", {})
    rules = deepcopy(fixed)
    seen_retained = set()
    for record in initialization.get("retained_handoff", {}).get("records", []):
        state = record["state"]
        if consistent_runtime_rules and state in seen_retained:
            raise ValueError("duplicate current initialization rule")
        seen_retained.add(state)
        if state in rules:
            if (
                consistent_runtime_rules
                and state in fixed
                and rules[state] == record["rule"]
            ):
                continue
            raise ValueError("duplicate current initialization rule")
        rules[state] = record["rule"]
    return rules


def _initial_equations(
    states: list[str], rules: dict, conditioning: dict, witness: dict
) -> list[str] | None:
    """Check saved equal arguments against each current declared initializer."""
    matched = conditioning.get("matched_initial_information", {})
    pair = [matched.get(name) for name in witness["trajectories"]]
    if not all(isinstance(row, dict) for row in pair):
        return None
    equations = []
    for state in states:
        rule = rules.get(state, {})
        mode = rule.get("mode")
        arguments = rule.get("arguments", [])
        if mode not in {"observed_initial", "shared_parameter", "common_map"}:
            return None
        if mode == "observed_initial":
            arguments = [state]
        elif (mode == "shared_parameter" and arguments) or (
            mode == "common_map" and not arguments
        ):
            return None
        recorded = []
        for row in pair:
            entry = row.get(state, {})
            values = entry.get("recorded_arguments", {})
            if (
                entry.get("rule") != mode
                or set(values) != set(arguments)
                or not all(_number(value) for value in values.values())
            ):
                return None
            recorded.append(values)
        if recorded[0] != recorded[1]:
            return None
        if mode == "shared_parameter":
            expression = "one shared constant"
        elif mode == "observed_initial":
            expression = f"recorded {state}(t0)={recorded[0][state]:.8g}"
        else:
            args = ", ".join(f"{name}(t0)" for name in arguments)
            expression = f"h_{state}({args}), with equal recorded arguments"
        equations.append(f"{state}(t0) = {expression}")
    return equations


def _qualified(question: dict, active: list[str]) -> bool:
    """Use only active conflicts with an explicit complete shared t0 contract."""
    qualification = question.get("initialization_qualification", {})
    contract = qualification.get("contract", {})
    assumptions = question.get("assumptions")
    return (
        question.get("question_id") in active
        and question.get("property") == "closed_subsystem"
        and question.get("repair_status") in {"actionable_conflict", "repair_required"}
        and question.get("status", "conditional_conflict") == "conditional_conflict"
        and not question.get("unavailable")
        and isinstance(assumptions, list)
        and bool(assumptions)
        and all(isinstance(value, str) and value for value in assumptions)
        and qualification.get("family_available") is True
        and contract.get("complete") is True
        and contract.get("shared_deterministic") is True
        and contract.get("scope") == "recorded_t0"
    )


def _summary(
    question: dict,
    witness: dict,
    graph: dict,
    kinds: dict,
    rules: dict,
) -> dict | None:
    """Restate current dependency equality, distinguishing t0 from propagation."""
    target = question.get("target")
    if target not in graph:
        return None
    conditioning = question.get("conditioning", {})
    initial = conditioning.get("initial_time_policy") == INITIAL_TIME_POLICY
    if initial:
        if kinds[target] != "algebraic" or list(witness["sample_indices"]) != [0, 0]:
            return None
        scope = _initial_leaves(target, graph, kinds)
        channel_key = "matched_instantaneous_channels"
    else:
        if 0 in witness["sample_indices"]:
            return None
        scope = _closure(target, graph)
        channel_key = "matched_complete_forcing_channels"
    states = sorted(name for name in scope if kinds.get(name) == "differential")
    supplied = sorted(scope - set(graph))
    channels = conditioning.get(channel_key)
    if not isinstance(channels, list) or set(supplied) - set(channels):
        return None
    initials = _initial_equations(states, rules, conditioning, witness)
    if initials is None or (not initial and not states):
        return None
    time = witness["times"][0]
    if initial:
        args = ", ".join(f"{name}(t0)" for name in sorted(scope))
        relation = f"{target}(t0) = H_{target}({args})"
        equal = (
            f"Equal initial values of {', '.join(states) or 'all readout arguments'}"
            + (f" and matched instantaneous {', '.join(supplied)}" if supplied else "")
            + f" imply equal {target}(t0). Changing ODE RHS arguments alone cannot "
            "change these already fixed initial values."
        )
    else:
        rhs = ", ".join(sorted(set(states) | set(supplied)))
        relation = f"d({', '.join(states)})/dt = F({rhs})"
        equal = (
            f"Equal initial values of {', '.join(states)}"
            + (
                f" and matched supplied histories/covariates {', '.join(supplied)}"
                if supplied
                else ""
            )
            + f" imply the same subsystem solution and {target}({time:g}) under "
            "the stated uniqueness assumptions. Terms using only that same "
            "information cannot distinguish these trajectories."
        )
    observed = "; ".join(
        f"{name}: {target}({when:g})={value:.8g}"
        for name, when, value in zip(
            witness["trajectories"],
            witness["times"],
            witness["observations"],
            strict=True,
        )
    )
    return {
        "question_id": question["question_id"],
        "target": target,
        "comparison": "initial_readout" if initial else "later_subsystem",
        "model_relation": relation,
        "initial_equalities": initials,
        "equality_argument": equal,
        "observed_difference": observed,
        "assumptions": deepcopy(question["assumptions"]),
        "qualification": "Conditional on these assumptions and the saved disjoint "
        "observation-error intervals; not a fit-quality conclusion. F, H and h "
        "denote dependency shorthand, not selected formulas. Identify a justified "
        "missing distinction or support a specific assumption challenge; no "
        "scientific edit is inferred.",
    }


def project(
    packet: dict, draft: Draft, *, consistent_runtime_rules: bool = False
) -> dict:
    """Prepend up to two concise arguments after the focused request projection.

    Unsupported or incomplete evidence leaves the packet untouched. This adapter
    does not validate raw data again: evidence references identify the existing
    qualified diagnostics. It checks coverage and current initializer agreement
    to avoid attaching a stale equality argument to a changed model.
    The optional duplicate-rule display correction leaves historical defaults
    unchanged and never resolves conflicting initializer declarations.
    """
    result = deepcopy(packet)
    if "model_data_conflicts" in result:
        raise ValueError("model/data equality projection is already applied")
    if result.get("stage") != "repair":
        return result
    expanded = expand_compatibility_details(packet)
    active = (
        expanded.get("feedback", {})
        .get("repair_work", {})
        .get("actionable_conflicts", [])
    )
    if not active:
        return result
    try:
        context = _context(expanded)
        graph, kinds, errors = _graph(draft, context)
        rules = _rules(
            expanded.get("initialization", {}),
            consistent_runtime_rules=consistent_runtime_rules,
        )
    except (KeyError, TypeError, ValueError):
        return result
    if errors:
        return result
    summaries = []
    for index, question in enumerate(expanded.get("compatibility_questions", [])):
        if not isinstance(question, dict) or not _qualified(question, active):
            continue
        for witness_index, witness in enumerate(question.get("witnesses", [])):
            if not _pair(witness):
                continue
            summary = _summary(question, witness, graph, kinds, rules)
            if summary is not None:
                summary["evidence_reference"] = (
                    f"compatibility_questions[{index}].witnesses[{witness_index}]"
                )
                summaries.append(summary)
                break
        if len(summaries) == _MAX_SUMMARIES:
            break
    if not summaries:
        return result
    leading = {"stage": result.pop("stage"), "model_data_conflicts": summaries}
    return leading | result
