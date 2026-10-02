"""Inspect declared topology; do not confuse graph witnesses with fitted science."""

from __future__ import annotations

import html
import json
import statistics
from collections import Counter, deque
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_baseline as baseline
from autoformalism.research import construction_trace


def path_witness(source: str, target: str, dependencies: dict) -> list[str] | None:
    """Return a deterministic source-to-target path from declared hyperedges."""
    pending = deque([(target, [target])])
    visited = set()
    while pending:
        node, path = pending.popleft()
        if node == source:
            return list(reversed(path))
        if node in visited:
            continue
        visited.add(node)
        pending.extend(
            (n, [*path, n])
            for n in sorted(dependencies.get(node, ()))
            if n not in visited
        )
    return None


def structural_evidence(brief: dict, topology: dict) -> dict:
    """Report declared RHS usage and selected-memory paths, retaining unknowns."""
    equations = topology.get("equations", [])
    dependencies = {
        e["name"]: {s for t in e["terms"] for s in t["sources"]} for e in equations
    }
    complete = topology.get("complete_topology", False)
    bindings = topology.get("memory_candidates", {})
    checks = []
    for requirement in brief["requirements"]:
        if not requirement["drivers"] or not requirement["targets"]:
            checks.append(
                {
                    "requirement": requirement["id"],
                    "status": "unresolved",
                    "kind": "graph_scope",
                    "reason": "no explicit driver/target graph predicate",
                }
            )
        for target in requirement["targets"]:
            for driver in requirement["drivers"]:
                direct = path_witness(driver, target, dependencies)
                memories = [
                    {"memory": m, "driver_to_memory": left, "memory_to_target": right}
                    for m in bindings.get(requirement["id"], [])
                    if (left := path_witness(driver, m, dependencies))
                    and (right := path_witness(m, target, dependencies))
                ]
                passed = bool(
                    memories if requirement["requires_dynamic_memory"] else direct
                )
                checks.append(
                    {
                        "requirement": requirement["id"],
                        "driver": driver,
                        "target": target,
                        "kind": "bound_memory_path"
                        if requirement["requires_dynamic_memory"]
                        else "driver_path",
                        "status": "unresolved"
                        if not complete
                        else "pass"
                        if passed
                        else "fail",
                        "driver_to_target": direct,
                        "memory_witnesses": memories,
                    }
                )
    for dependency in brief.get("target_dependencies", []):
        paths = [
            p
            for source in dependency["acceptable_sources"]
            if (p := path_witness(source, dependency["target"], dependencies))
        ]
        checks.append(
            {
                "kind": "composition_path",
                "target": dependency["target"],
                "status": "unresolved" if not complete else "pass" if paths else "fail",
                "paths": paths,
            }
        )
    usage = [
        {
            **v,
            "generated_lhs": v["name"] in dependencies,
            "declared_rhs_in": sorted(
                n for n, sources in dependencies.items() if v["name"] in sources
            ),
            "scope": "declared topology, functions not yet generated",
        }
        for v in topology.get("inventory", [])
    ]
    return {
        "channel_usage": usage,
        "path_checks": checks,
        "interpretation": (
            "Declared dependencies can vanish in later functions; "
            "path presence does not certify scientific correctness."
        ),
    }


def skeletons(topology: dict) -> list[str]:
    """Render unknown interaction laws without suggesting known fitted equations."""
    processes = {
        b["proposal"]["name"]: b
        for b in topology.get("shared_process_contract", {}).get("bindings", [])
    }
    result = []
    for i, equation in enumerate(topology.get("equations", [])):
        lhs = (
            f"d({equation['name']})/dt"
            if equation["definition"] == "differential"
            else equation["name"]
        )
        parts = []
        for j, term in enumerate(equation["terms"]):
            sources = term["sources"]
            shared = processes.get(sources[0]) if len(sources) == 1 else None
            if shared and any(u["target"] == equation["name"] for u in shared["uses"]):
                law = sources[0]
                use = next(
                    u
                    for u in shared["signed_declaration"]["uses"]
                    if u["target"] == equation["name"]
                )
                if use["conversion"] not in (None, "1"):
                    law = f"({use['conversion']}) * {law}"
            else:
                law = f"phi_{i}_{j}({', '.join(sources)})"
            sign = "-" if term["outer_weight_sign"] == "negative" else "+"
            weight = "b" if term["outer_weight_sign"] == "unrestricted" else "a"
            parts.append(f"{sign} {weight}_{i}_{j} * {law}")
        result.append(f"{lhs} = {' '.join(parts)}")
    return result


def report(root: Path, plan: dict) -> dict:
    """Preserve failed routes and exact prompts, including unfinished checkpoints."""
    rows = []
    for task in plan["tasks"]:
        directory = baseline.location(root, task)
        identity = baseline.namespace(plan, task)
        records = _cache_records(directory / "calls", identity)
        outcome = baseline.read_outcome(directory / "proposal.json", plan, task) or {}
        if directory.exists():
            construction_trace.render(directory, identity)
        routes = []
        for route in ("process", "ordinary_fallback"):
            stage = directory / "construction" / route / "topology_stage.json"
            progress = directory / "construction" / route / "topology/progress.json"
            if stage.exists():
                topology = sealed_read(stage)["result"]
            elif progress.exists():
                topology = {**json.loads(progress.read_text()), "status": "in_progress"}
            else:
                continue
            routes.append(
                {
                    "route": route,
                    "status": topology["status"],
                    "error": topology.get("error"),
                    "skeletons": skeletons(topology),
                    "structural_evidence": structural_evidence(
                        plan["cells"][task["benchmark_id"]]["brief"], topology
                    ),
                    "topology": topology,
                }
            )
        saved = plan["inputs"][task["task_id"]]
        events = [
            e
            for r in routes
            for e in r["topology"].get("events", [])
            if e.get("step", "").startswith("equation_")
        ]
        rows.append(
            {
                "task": task["task_id"],
                "benchmark_id": task["benchmark_id"],
                "arm": task["arm"],
                "seed": task["seed"],
                "status": outcome.get(
                    "status", "in_progress" if records else "pending"
                ),
                "selected_route": outcome.get("selected_route"),
                "fallback_used": outcome.get("fallback_used"),
                "origin": saved["origin"],
                "historical_variable_cost": saved["historical_variable_cost"],
                "new_topology_cost": _cost(records),
                "equation_first_replies": sum(e["attempt"] == 0 for e in events),
                "equation_first_accepted": sum(
                    e["attempt"] == 0 and e["accepted"] for e in events
                ),
                "equation_repairs": sum(e["attempt"] > 0 for e in events),
                "routes": routes,
                "scientific_adequacy": "not_assessed",
            }
        )
    terminal = all(r["status"] not in {"pending", "in_progress"} for r in rows)
    tokens = [r["new_topology_cost"]["observed_tokens"] for r in rows]
    median = statistics.median(tokens) if terminal else None
    summary = {
        "protocol": plan["protocol"],
        "identity": plan["artifact_sha256"],
        "planned_constructions": len(rows),
        "all_tasks_terminal": terminal,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "inventory_origins": dict(Counter(r["origin"]["source"] for r in rows)),
        "fallback_constructions": sum(r["fallback_used"] is True for r in rows),
        "new_physical_calls": sum(
            r["new_topology_cost"]["physical_requests"] for r in rows
        ),
        "new_observed_tokens": sum(tokens),
        "new_tokens_per_construction_median": median,
        "new_tokens_per_construction_MAD": statistics.median(
            abs(t - median) for t in tokens
        )
        if terminal
        else None,
        "unknown_usage_calls": sum(
            r["new_topology_cost"]["requests_with_unknown_usage"] for r in rows
        ),
        "equation_first_replies": sum(r["equation_first_replies"] for r in rows),
        "equation_first_accepted": sum(r["equation_first_accepted"] for r in rows),
        "equation_repairs": sum(r["equation_repairs"] for r in rows),
        "scientific_adequacy": (
            "Inspect skeletons manually; independent fitted "
            "mechanism compliance unavailable until functions "
            "and parameters exist."
        ),
        "function_generation_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "rows": rows,
    }
    atomic_json(root / "summary.json", summary)
    text = [
        "# Phase C topology confirmation",
        "",
        "Saved inventories; optional processes and topology. No functions or fits.",
        "Declared paths and assembly signs are not scientific certification.",
        "Historical variable costs and new topology costs are separate.",
        "",
        f"Status counts: {summary['status_counts']}",
        "",
        "| Task | Status | Route | Calls | Tokens |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    panels = [
        "<!doctype html><meta charset='utf-8'><title>Topology inspection</title>",
        "<style>body{max-width:1200px;margin:2em auto;font:16px/1.5 system-ui;"
        "padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f4f6f8;padding:1em}summary{cursor:pointer;margin:1em 0}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;"
        "padding:8px}</style>",
        "<h1>Topology inspection</h1><p>phi denotes an unknown interaction law; "
        "a is a nonnegative outer magnitude; b is unrestricted. Signs refer to "
        "outer weights, not global monotonicity of phi. Shared process names "
        "refer to one later law reused at its consumers. No parameters have "
        "been fitted. Initialization remains a later stage.</p>",
    ]
    for r in rows:
        c = r["new_topology_cost"]
        text.append(
            f"| {r['task']} | {r['status']} | {r['selected_route']} | "
            f"{c['physical_requests']} | {c['observed_tokens']} |"
        )
        panels.append(
            f"<h2>{html.escape(r['task'])} — {html.escape(r['status'])}</h2>"
            f"<p>Inventory origin: {html.escape(r['origin']['source'])}. "
            f"<a href='results/{html.escape(r['task'], quote=True)}/TRACE.html'>"
            "Exact prompts, replies and repair feedback</a></p>"
        )
        for route in r["routes"]:
            panels.append(
                f"<h3>{html.escape(route['route'])}: "
                f"{html.escape(route['status'])}</h3>"
                f"<pre>{html.escape(chr(10).join(route['skeletons']))}</pre>"
                "<details><summary>Terms, roles, channel usage and path witnesses"
                "</summary><pre>"
                f"{html.escape(json.dumps(route, indent=2))}</pre></details>"
            )
    (root / "SUMMARY.md").write_text("\n".join(text) + "\n")
    (root / "TOPOLOGY.html").write_text("\n".join(panels))
    return summary
