"""Propose one graph first, then enumerate bounded additive edits on that graph."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from typing import Literal

from pydantic import ValidationError

from autoformalism.research import preproposal_topology as topology
from autoformalism.research.preproposal_cases import screen_case
from autoformalism.research.preproposal_selection import SelectionCase

MAX_PATCH_CHECKS = 1024
PASS = "topology_screen_passed"


@dataclass(frozen=True, order=True)
class Edit:
    """Add one RHS dependency or one time-zero argument to a latent initializer."""

    kind: Literal["add_source", "add_public_initial"]
    receiver: str
    source: str


def graph_first_messages(case: SelectionCase, frozen: dict) -> tuple[str, str]:
    """Give the same public task and observed discrepancy without a source menu."""
    system, user = topology.topology_messages(case, None, frozen)
    packet = json.loads(user)
    # This v1 summary also narrates the menu and names its surviving sources.
    # The graph-first proposer receives the underlying training observation.
    packet.pop("training_evidence")
    screen = screen_case(frozen)
    required = set(screen.required_options)
    witness = next(
        (
            w
            for w in screen.witnesses
            if required.isdisjoint(w.different) and required.isdisjoint(w.unknown)
        ),
        None,
    )
    packet["training_observation"] = {
        "target": case.target,
        "minimum_target_gap": screen.minimum_target_gap,
        "fixed_information": screen.required_options,
        "trajectory_pair_count": screen.pair_count,
        "example_with_equal_fixed_information": (
            {
                "trajectories": witness.trajectories,
                "time": witness.time,
                "observations": witness.observations,
            }
            if witness
            else None
        ),
    }
    packet["permitted_initial_values"] = packet.pop("permitted_baseline_readings")
    return system, json.dumps(packet, ensure_ascii=False)


def _catalogue(case: SelectionCase, frozen: dict, graph: dict) -> list[Edit]:
    """Enumerate additions inside the fixed inventory and positive permissions."""
    context = frozen["evidence"]["context"]
    names = {v.name for v in case.inventory}
    initial_names = set(frozen["initials"]) | {case.target}
    fixed = set(context.get("fixed_covariates", ())) & initial_names
    ongoing = names | set(frozen["histories"]) | fixed
    edits = [
        Edit("add_source", target, source)
        for target, deps in graph["sources"].items()
        for source in sorted(ongoing - set(deps) - {target})
    ]
    if not frozen.get("shared_hidden_initials", False):
        for initial in graph["latent_initializers"]:
            state, rule = initial["state"], initial["rule"]
            if state not in names:
                continue
            for source in sorted(initial_names - set(rule.get("public_initials", ()))):
                edits.append(Edit("add_public_initial", state, source))
    return sorted(set(edits))


def _apply(graph: dict, patch: tuple[Edit, ...]) -> dict:
    child = deepcopy(graph)
    records = {r["state"]: r for r in child["latent_initializers"]}
    for edit in patch:
        if edit.kind == "add_source":
            child["sources"][edit.receiver].append(edit.source)
            continue
        record = records[edit.receiver]
        rule = record["rule"]
        record["rule"] = {
            "mode": "map",
            "state_initials": rule.get("state_initials", []),
            "public_initials": [*rule.get("public_initials", []), edit.source],
        }
        record["justification"] = "Algorithm-proposed time-zero dependency."
    return child


def assess_and_repair(
    case: SelectionCase,
    value: dict,
    frozen: dict,
    *,
    max_checks: int = MAX_PATCH_CHECKS,
) -> dict:
    """Retain every minimal passing one- or two-edit patch within the check budget.

    All edits are relative to this exact parent. A child with a passing proper subset
    cannot be a minimal patch, regardless of whether the public checks are monotone.
    """
    if max_checks < 1:
        raise ValueError("max_checks must be positive")
    evidence, _ = topology._inputs(case, frozen)

    @lru_cache(maxsize=512)
    def roots(target: str, actual: tuple[str, ...], gap: float) -> dict:
        return topology._actual_assessment(target, set(actual), evidence, gap)

    def cached(target, actual, packet, gap):
        return roots(target, tuple(sorted(actual)), gap)

    def check(reply):
        return topology.assess_topology(case, None, reply, frozen, root_assessor=cached)

    parent = check(value)
    result = {
        **parent,
        "parent": value,
        "repair_status": "already_passed",
        "repairs": [],
        "checked": 0,
        "max_edits": 2,
        "max_checks": max_checks,
        "budget_exhausted": False,
        "child_status_counts": {},
    }
    if parent["status"] == PASS:
        return result
    try:
        schema = topology.topology_schema(
            case, shared_hidden_initials=frozen.get("shared_hidden_initials", False)
        )
        graph = schema.model_validate(value).model_dump(mode="json", by_alias=True)
    except ValidationError:
        result["repair_status"] = "invalid_parent_schema"
        return result
    edits = _catalogue(case, frozen, graph)
    result["atomic_edits"] = len(edits)
    successful_singles = set()
    seen = set()
    counts = Counter()
    for size in (1, 2):
        for patch in combinations(edits, size):
            if successful_singles.intersection(patch):
                continue
            if result["checked"] == max_checks:
                result["budget_exhausted"] = True
                break
            child = _apply(graph, patch)
            outcome = check(child)
            result["checked"] += 1
            counts[outcome["status"]] += 1
            if outcome["status"] != PASS:
                continue
            if size == 1:
                successful_singles.add(patch[0])
            digest = outcome["structure_sha256"]
            if digest not in seen:
                seen.add(digest)
                result["repairs"].append(
                    {
                        "edits": [asdict(e) for e in patch],
                        "graph": child,
                        "structure_sha256": digest,
                        "status": PASS,
                    }
                )
        if result["budget_exhausted"]:
            break
    result["child_status_counts"] = dict(counts)
    if result["repairs"]:
        result["repair_status"] = "repaired"
    elif result["budget_exhausted"]:
        result["repair_status"] = "check_budget_exhausted"
    else:
        result["repair_status"] = "no_passing_patch_in_grammar"
    return result
