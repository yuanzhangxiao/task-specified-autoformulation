"""Bounded dependency explanations for already validated, rejected trials.

This is a presentation aid, not an influence proof, scientific law selector, or
acceptance gate. Expression references can disappear by cancellation or fitted
zero coefficients. A path only records where a component could enter a target.
"""

from __future__ import annotations

from collections import deque

from pydantic import ValidationError

from autoformalism.expressions import ModelValidationError
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.staged_topology import content_hash

POLICY = "validated-repair-dependency-view-1"
MAX_COMPONENTS = 128
MAX_EDGES = 4096
MAX_DISPLAY_COMPONENTS = 8
MAX_DISPLAY_TARGETS = 4
MAX_PATH_LENGTH = 16


def _bounded_problem(raw: dict) -> PrefixFitProblem:
    """Bound parsing work before accepting the typed, data-free model boundary."""
    if not isinstance(raw, dict) or not isinstance(raw.get("base_candidate"), dict):
        raise ValueError("validated model is unavailable")
    base = raw["base_candidate"]
    for field in (
        "states",
        "state_equations",
        "processes",
        "observation_mappings",
        "initial_conditions",
        "parameters",
        "equations",
    ):
        rows = base.get(field, [])
        if not isinstance(rows, list) or len(rows) > MAX_COMPONENTS:
            raise ValueError("dependency model exceeds bounded record allowance")
    if len(base.get("states", [])) + len(base.get("processes", [])) > MAX_COMPONENTS:
        raise ValueError("dependency model exceeds component allowance")
    return PrefixFitProblem.model_validate(raw)


def _symbols(expression: str, location: str) -> set[str]:
    return set(RestrictedParser().parse(expression, location=location).symbols)


def _graph(problem: PrefixFitProblem) -> tuple[dict, dict, set[str]]:
    """Build forward references without executing or expanding expression text."""
    model = problem.base_candidate
    states = {row.name for row in model.states}
    processes = {row.name for row in model.processes}
    parameters = {row.name for row in model.parameters}
    definitions = {row.state: row.rhs for row in model.state_equations}
    if (
        len(definitions) != len(model.state_equations)
        or set(definitions) != states
        or len(states) != len(model.states)
        or len(parameters) != len(model.parameters)
    ):
        raise ValueError("state equation identities do not resolve")
    if states & processes or (states | processes) & parameters:
        raise ValueError("component identities overlap")
    if len(processes) != len(model.processes):
        raise ValueError("process identities do not resolve")
    definitions.update({row.name: row.expression for row in model.processes})
    mappings = {row.channel: row.expression for row in model.observation_mappings}
    if len(mappings) != len(model.observation_mappings):
        raise ValueError("readout identities do not resolve")
    if set(mappings) != set(problem.context.targets):
        raise ValueError("readouts do not cover the public targets")
    public = set(problem.context.forcing_channels) | {problem.context.time_symbol}
    allowed = states | processes | parameters | public
    dynamic: dict[str, set[str]] = {}
    initial: dict[str, set[str]] = {}
    edges = 0

    def add(graph: dict, source: str, receiver: str) -> None:
        nonlocal edges
        graph.setdefault(source, set()).add(receiver)
        edges += 1
        if edges > MAX_EDGES:
            raise ValueError("dependency graph exceeds edge allowance")

    algebraic_dependencies = {}
    for name, expression in definitions.items():
        symbols = _symbols(expression, name)
        if symbols - allowed:
            raise ValueError("an equation contains unresolved symbols")
        for source in sorted(symbols & (states | processes)):
            add(dynamic, source, name)
        if name in processes:
            algebraic_dependencies[name] = symbols & processes
    _acyclic(algebraic_dependencies)
    for target, expression in mappings.items():
        symbols = _symbols(expression, f"output:{target}")
        if symbols - allowed:
            raise ValueError("a readout contains unresolved symbols")
        for source in sorted(symbols & (states | processes)):
            add(dynamic, source, f"output:{target}")

    initial = {key: set(values) for key, values in dynamic.items()}
    for state in sorted(states):
        add(initial, f"initial:{state}", state)
    initializer_dependencies = {}
    for state, rule in problem.graph.rules.items():
        if state not in states:
            raise ValueError("an initializer names an unavailable state")
        value = rule.initial
        if value.mode != "map":
            continue
        symbols = _symbols(value.expression, f"initial:{state}")
        declared = (
            set(value.state_initials)
            | set(value.public_initials)
            | {row.name for row in value.parameters}
        )
        if symbols != declared:
            raise ValueError("initializer-local symbols do not resolve")
        if set(value.state_initials.values()) - states:
            raise ValueError("initializer state references do not resolve")
        if set(value.public_initials.values()) - (public | set(mappings)):
            raise ValueError("initializer public references do not resolve")
        for source in sorted(value.state_initials.values()):
            add(initial, f"initial:{source}", f"initial:{state}")
        initializer_dependencies[state] = set(value.state_initials.values())
    _acyclic(initializer_dependencies)
    return dynamic, initial, set(mappings)


def _acyclic(dependencies: dict[str, set[str]]) -> None:
    """Reject algebraic/initialization cycles, leaving ODE feedback unrestricted."""
    remaining = {
        name: set(values) & dependencies.keys() for name, values in dependencies.items()
    }
    while remaining:
        ready = {name for name, values in remaining.items() if not values}
        if not ready:
            raise ValueError("algebraic or initializer dependencies are cyclic")
        remaining = {
            name: values - ready
            for name, values in remaining.items()
            if name not in ready
        }


def _path(graph: dict, source: str, destination: str) -> list[str] | None:
    """Find one shortest path with a visited set; ODE feedback cycles are allowed."""
    queue = deque([(source, [source])])
    visited = {source}
    while queue:
        node, path = queue.popleft()
        if node == destination:
            return path
        for child in sorted(graph.get(node, ())):
            if child not in visited:
                visited.add(child)
                queue.append((child, [*path, child]))
    return None


def _targets(intended: dict, valid: set[str]) -> list[str]:
    """Use targets explicitly named by the qualified certificate, not prose."""
    assessment = (intended.get("diagnostic", {}).get("details") or {}).get(
        "assessment", {}
    )
    if not isinstance(assessment, dict):
        raise ValueError("qualified certificate targets are unavailable")
    groups = [assessment]
    for key in (
        "affine_family_consistency",
        "initial_family_consistency",
        "function_consistency",
    ):
        group = assessment.get(key, {})
        if isinstance(group, dict):
            groups.append(group)
    targets = set()
    for group in groups:
        conflicts = group.get("conflicts", [])
        if not isinstance(conflicts, list) or len(conflicts) > MAX_COMPONENTS:
            raise ValueError("qualified conflict list is unavailable or too large")
        for conflict in conflicts:
            target = conflict.get("target") if isinstance(conflict, dict) else None
            if target is None:
                continue
            if not isinstance(target, str) or target not in valid:
                raise ValueError("a qualified target has no resolved readout")
            targets.add(target)
    if not targets:
        raise ValueError("no qualified failing target was identified")
    return sorted(targets)


def view(parent: dict, intended: dict) -> dict:
    """Explain new components' references only for a sealed static trial result."""
    result = {
        "policy": POLICY,
        "status": "unknown",
        "components": [],
        "acceptance_changed": False,
        "interpretation": "Expression-reference paths are possible connections, "
        "not proof of nonzero influence, scientific necessity, or fitting quality. "
        "Initialization-only paths refer to initial values, not later state values. "
        "No new hard check or scientific edit is introduced.",
    }
    if (intended.get("diagnostic") or {}).get(
        "code"
    ) != "QUALIFIED_MODEL_DATA_CONFLICT":
        result["reason"] = "The pending trial has no qualified mathematical conflict."
        return result
    try:
        raw = intended.get("problem")
        base = _bounded_problem(parent.get("problem"))
        trial = _bounded_problem(raw)
        audit = intended.get("audit") or {}
        if not (
            audit.get("static_eligibility") == "eligible"
            and audit.get("child_problem_sha256") == content_hash(raw)
            and audit.get("parent_problem_sha256") == content_hash(parent["problem"])
        ):
            raise ValueError("static trial provenance does not match the two models")
        # Resolve parent identities; never infer additions from malformed input.
        _graph(base)
        dynamic, initial, valid_targets = _graph(trial)
        targets = _targets(intended, valid_targets)
        old = {row.name for row in base.base_candidate.states} | {
            row.name for row in base.base_candidate.processes
        }
        added = [
            (row.name, "differential_state")
            for row in trial.base_candidate.states
            if row.name not in old
        ] + [
            (row.name, "algebraic_process")
            for row in trial.base_candidate.processes
            if row.name not in old
        ]
        for name, kind in sorted(added)[:MAX_DISPLAY_COMPONENTS]:
            connections = []
            disconnected = []
            for target in targets[:MAX_DISPLAY_TARGETS]:
                path = _path(dynamic, name, f"output:{target}")
                status = "equation_path" if path else "disconnected"
                if path is None and kind == "differential_state":
                    path = _path(initial, f"initial:{name}", f"output:{target}")
                    if path:
                        status = "initialization_only_path"
                item = {"target": target, "status": status}
                if path:
                    item["path"] = path[:MAX_PATH_LENGTH]
                    item["path_truncated"] = len(path) > MAX_PATH_LENGTH
                else:
                    disconnected.append(target)
                connections.append(item)
            component = {"name": name, "kind": kind, "target_connections": connections}
            if disconnected:
                component["question"] = (
                    f"How is {name} intended to enter {', '.join(disconnected)}? "
                    "No equation or initializer-reference path was found. If this "
                    "addition is meant to repair those targets, supply the intended "
                    "connection; otherwise explain its role."
                )
            result["components"].append(component)
        result.update(
            status="reported",
            omitted_components=max(0, len(added) - MAX_DISPLAY_COMPONENTS),
            omitted_targets=max(0, len(targets) - MAX_DISPLAY_TARGETS),
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        ValidationError,
        ModelValidationError,
        RecursionError,
    ) as exc:
        result["components"] = []
        result["reason"] = (
            f"Dependency explanation deferred: {type(exc).__name__}: {str(exc)[:180]}"
        )
    return result
