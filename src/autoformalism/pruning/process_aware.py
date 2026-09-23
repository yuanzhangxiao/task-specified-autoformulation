"""Deletion-only pruning units and training open-rollout contribution ranking."""

from __future__ import annotations

import ast
from time import monotonic

import numpy as np

from autoformalism.expressions import RestrictedParser
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory, trajectory_forcing
from autoformalism.pruning.pruner import _additive_terms
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from autoformalism.staged_topology import content_hash


def names(expression: str) -> set[str]:
    """Read identifiers without executing proposer text."""
    return {
        n.id
        for n in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(n, ast.Name)
    }


def definitions(candidate) -> dict[str, str]:
    """Return the single canonical definition for every generated symbol."""
    return {
        **{p.name: p.expression for p in candidate.processes},
        **{e.state: e.rhs for e in candidate.state_equations},
    }


def terms(candidate) -> dict[str, dict]:
    """Reuse the existing pruner's signed additive decomposition."""
    return {
        f"{target}:{i}": {"target": target, "index": i, "expression": ast.unparse(term)}
        for target, expression in definitions(candidate).items()
        for i, term in enumerate(
            _additive_terms(ast.parse(expression, mode="eval").body)
        )
    }


def complexity(request: PublicFitRequest) -> dict:
    """Count the executable model, including fitted causal initial parameters."""
    model, _, _ = public._lower(request)
    candidate = model.validated.candidate
    expressions = (
        list(definitions(candidate).values())
        + [o.expression for o in candidate.observation_mappings]
        + [i.expression for i in candidate.initial_conditions if i.expression]
    )
    return {
        "states": len(candidate.states),
        "processes": len(candidate.processes),
        "parameters": len(candidate.parameters),
        "terms": len(terms(candidate)),
        "expression_nodes": sum(
            sum(1 for _ in ast.walk(ast.parse(e, mode="eval"))) for e in expressions
        ),
    }


def smaller(before: dict, after: dict) -> bool:
    """Require a Pareto reduction in executable counts, with no count increasing."""
    return all(after[k] <= before[k] for k in before) and any(
        after[k] < before[k] for k in before
    )


def _linear_use(expression: str, process: str) -> bool:
    """Certify one multiplicative occurrence; reject nonlinear/denominator uses."""
    node = ast.parse(expression, mode="eval").body

    def visit(n):
        if not any(isinstance(x, ast.Name) and x.id == process for x in ast.walk(n)):
            return 0
        if isinstance(n, ast.Name):
            return 1
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            return visit(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            a, b = visit(n.left), visit(n.right)
            return 1 if sorted((a, b)) == [0, 1] else -1
        if (
            isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Div)
            and visit(n.right) == 0
        ):
            return visit(n.left)
        return -1

    return visit(node) == 1


def removal_units(request: PublicFitRequest) -> tuple[list[dict], list[dict]]:
    """Shared occurrences are atomic; law subterms remain single shared definitions."""
    candidate, units, blocked = request.base_candidate, [], []
    slots = terms(candidate)
    uses = {
        p.name: [k for k, t in slots.items() if p.name in names(t["expression"])]
        for p in candidate.processes
    }
    shared = {
        p for p, ids in uses.items() if len({slots[k]["target"] for k in ids}) > 1
    }
    for key, term in slots.items():
        referenced = names(term["expression"])
        if referenced & shared:
            continue  # Whole shared-process unit below owns all consumers.
        if (
            term["target"] in uses
            and sum(t["target"] == term["target"] for t in slots.values()) == 1
        ):
            continue  # Whole law removal is represented by removing its consumers.
        units.append({"id": f"term:{key}", "kind": "term", "slots": [key]})
    for process, ids in uses.items():
        reason = None
        if any(process in names(o.expression) for o in candidate.observation_mappings):
            reason = "process participates directly in an observation mapping"
        elif not ids or any(
            not _linear_use(slots[k]["expression"], process) for k in ids
        ):
            reason = "no unambiguous additive linear consumer group"
        # Do not delete one consumer of another shared law as collateral damage.
        elif any(
            set(ids) & set(others) and not set(others) <= set(ids)
            for p, others in uses.items()
            if p in shared and p != process
        ):
            reason = "would partially delete another shared process"
        if reason:
            blocked.append({"id": f"process:{process}", "reason": reason})
        else:
            units.append({"id": f"process:{process}", "kind": "process", "slots": ids})
    return units, blocked


def remove(request: PublicFitRequest, unit: dict) -> tuple[PublicFitRequest, dict]:
    """Delete a unit and unreachable definitions, including orphan latent initials."""
    if unit not in removal_units(request)[0]:
        raise ValueError("unit is not a permitted atomic removal")
    raw = request.model_dump(mode="json")
    candidate, slots = raw["base_candidate"], terms(request.base_candidate)
    if not unit["slots"] or set(unit["slots"]) - set(slots):
        raise ValueError("invalid pruning slots")
    equations = definitions(request.base_candidate)
    changes = {}
    for target in {slots[k]["target"] for k in unit["slots"]}:
        remaining = [
            t["expression"]
            for k, t in slots.items()
            if t["target"] == target and k not in unit["slots"]
        ]
        after = " + ".join(f"({e})" for e in remaining) or "0"
        changes[target] = {"before": equations[target], "after": after}
        equations[target] = after
    needed = set().union(
        *(names(o.expression) for o in request.base_candidate.observation_mappings)
    )
    while True:
        expanded = needed | set().union(
            *(names(equations[n]) for n in needed & equations.keys())
        )
        if expanded == needed:
            break
        needed = expanded
    removed = {
        "states": [s["name"] for s in candidate["states"] if s["name"] not in needed],
        "processes": [
            p["name"] for p in candidate["processes"] if p["name"] not in needed
        ],
    }
    candidate["states"] = [s for s in candidate["states"] if s["name"] in needed]
    candidate["processes"] = [
        {**p, "expression": equations[p["name"]]}
        for p in candidate["processes"]
        if p["name"] in needed
    ]
    candidate["state_equations"] = [
        {**e, "rhs": equations[e["state"]]}
        for e in candidate["state_equations"]
        if e["state"] in needed
    ]
    candidate["initial_conditions"] = [
        i for i in candidate["initial_conditions"] if i["state"] in needed
    ]
    rules = raw["initialization_plan"]["rules"]
    removed["initializers"] = sorted(set(rules) - needed)
    raw["initialization_plan"]["rules"] = {
        k: v for k, v in rules.items() if k in needed
    }
    for initial in candidate["initial_conditions"]:
        if initial.get("expression"):
            needed |= names(initial["expression"])
    removed["parameters"] = [
        p["name"] for p in candidate["parameters"] if p["name"] not in needed
    ]
    candidate["parameters"] = [
        p for p in candidate["parameters"] if p["name"] in needed
    ]
    deleted = set(removed["states"] + removed["processes"] + removed["parameters"])
    removed["constraints"] = [
        c for c in candidate["constraints"] if c["subject"] in deleted
    ]
    candidate["constraints"] = [
        c for c in candidate["constraints"] if c["subject"] not in deleted
    ]
    # There is no scientific boundary edit: only removed states lose their rules.
    raw["parameter_guesses"] = {
        k: v for k, v in raw["parameter_guesses"].items() if k in needed
    }
    raw["source"] = {
        "stage": "controller_revision",
        "task_id": request.source.task_id + "/pruning",
        "artifact_sha256": content_hash([request.model_dump(mode="json"), unit]),
    }
    child = PublicFitRequest.model_validate(raw)
    old_definitions = definitions(request.base_candidate)
    new_definitions = definitions(child.base_candidate)
    for process in request.base_candidate.processes:
        consumers = {
            target
            for target, rhs in old_definitions.items()
            if process.name in names(rhs)
        }
        if len(consumers) > 1 and process.name in new_definitions:
            remaining = {
                target
                for target, rhs in new_definitions.items()
                if process.name in names(rhs)
            }
            if remaining != consumers:
                raise ValueError("cleanup would partially remove a shared process")
    before, after = complexity(request), complexity(child)
    if not smaller(before, after):
        raise ValueError("pruning did not reduce executable complexity")
    return child, {
        "unit": unit,
        "changes": changes,
        "removed": removed,
        "before": before,
        "after": after,
    }


def contributions(
    request: PublicFitRequest,
    parameters: dict,
    training: PublicSplit,
    *,
    seconds: float = 300,
) -> dict[str, float]:
    """RMS term/full-law ratios on training open rollouts at observation times only."""
    if training.name != "train":
        raise ValueError("pruning ranks require training data")
    model, _, _ = public._lower(request)
    if set(parameters) != set(model.parameter_names):
        raise ValueError("complete fitted parameter vector required")
    slots, parser = terms(request.base_candidate), RestrictedParser()
    parsed = {k: parser.parse(t["expression"], location=k) for k, t in slots.items()}
    total = {
        k: parser.parse(e, location=k)
        for k, e in definitions(request.base_candidate).items()
    }
    squares, denominators = dict.fromkeys(slots, 0.0), dict.fromkeys(total, 0.0)
    if seconds <= 0:
        raise TimeoutError("training contribution budget exhausted")
    deadline = monotonic() + seconds
    settings = FitConfig(
        integration_method="Radau",
        relative_tolerance=1e-7,
        absolute_tolerance=1e-9,
        allow_derivative_regression=False,
    )
    count = 0
    for row in public.unpack_split(training).trajectories:
        sim = simulate_trajectory(
            model,
            row,
            parameters,
            {},
            settings,
            deadline=deadline,
            reset_observed_states=False,
        )
        if not sim.success:
            raise ValueError(f"training contribution replay failed: {sim.message}")
        forcing = trajectory_forcing(model, row)
        for i, t in enumerate(row.time):
            if monotonic() >= deadline:
                raise TimeoutError("training contribution budget exhausted")
            state = sim.states[:, i]
            for pool, expressions in ((squares, parsed), (denominators, total)):
                for k, expression in expressions.items():
                    value = model.evaluate_expression(
                        expression, float(t), state, parameters, forcing
                    )
                    pool[k] += value * value
            count += 1
    values = {
        k: float(
            np.sqrt(v / count)
            / max(np.sqrt(denominators[slots[k]["target"]] / count), 1e-12)
        )
        for k, v in squares.items()
    }
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("nonfinite training contributions")
    return values
