"""Replay nominations after replacing every scientific identifier with an opaque ID."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.shape_equation_nomination import build_shape_shortlist
from autoformalism.staged_topology import content_hash
from scripts.matched_equation_experiment import DEFAULT


def opaque_packet(packet: dict) -> tuple[dict, dict]:
    graph = packet["graph"]
    roles = ("targets", "inputs", "auxiliaries", "covariates")
    names = set(graph["variable_types"])
    for role in roles:
        names.update(packet["train_rows"][0][role])
    forward = {old: f"v{i:03d}" for i, old in enumerate(sorted(names, reverse=True))}

    def remap(table):
        return {forward[k]: v for k, v in table.items()}

    initials = {}
    for k, rule in graph["initializers"].items():
        renamed = rule.copy()
        if "channel" in rule:
            renamed["channel"] = forward[rule["channel"]]
        for field in ("state_initials", "public_initials"):
            if field in rule:
                renamed[field] = [forward[n] for n in rule[field]]
        initials[forward[k]] = renamed
    return {
        "graph": {
            "variable_types": remap(graph["variable_types"]),
            "sources": {
                forward[k]: [forward[s] for s in v] for k, v in graph["sources"].items()
            },
            "initializers": initials,
        },
        "training_rows": [
            {**row, **{k: remap(row[k]) for k in roles}} for row in packet["train_rows"]
        ],
        "interpolation": remap(packet["interpolation"]),
    }, forward


def formula_signature(formula: dict, original_to_current: dict) -> str:
    """Compare syntax, parameter sharing and starts after undoing renaming."""
    inverse = {v: k for k, v in original_to_current.items()}
    inverse.update(
        {f"state0_{v}": f"state0_{k}" for k, v in original_to_current.items()}
    )
    positions, starts, expressions = {}, [], {}
    for field in ("equations", "initializers"):
        for name in sorted(formula[field], key=lambda v: inverse.get(v, v)):
            tree = ast.parse(formula[field][name], mode="eval")
            for node in ast.walk(tree):
                if not isinstance(node, ast.Name):
                    continue
                source = node.id
                if source in formula["parameters"]:
                    if source not in positions:
                        positions[source] = len(starts)
                        starts.append(formula["parameters"][source])
                    node.id = f"coef_{positions[source]}"
                else:
                    node.id = inverse.get(source, source)
            expressions[f"{field}/{inverse.get(name, name)}"] = ast.dump(tree)
    return content_hash([expressions, starts])


def compare(packet: dict) -> dict:
    opaque, mapping = opaque_packet(packet)
    original = packet["shape"]["candidates"]
    signatures = [formula_signature(model, {}) for model in original]
    checks = {}
    for order, inputs in (
        ("preserve_graph_order", opaque),
        ("resort_opaque_keys", json.loads(json.dumps(opaque, sort_keys=True))),
    ):
        renamed = build_shape_shortlist(**inputs)["candidates"]
        recovered = [formula_signature(model, mapping) for model in renamed]
        checks[order] = [r == s for r, s in zip(recovered, signatures, strict=True)]
    return {"panel_id": packet["panel_id"], **checks}


def run(root: Path) -> dict:
    plan = sealed_read(root / "plan.json")
    rows = []
    for panel in plan["panels"]:
        packet = sealed_read(root / "packets" / f"{panel['panel_id']}.json")
        if packet["artifact_sha256"] != panel["packet"]:
            raise ValueError("Packet differs from the frozen experiment")
        rows.append(compare(packet))
    return sealed_write(
        root / "opaque-name-audit.json", {"plan": plan["artifact_sha256"], "rows": rows}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    print(json.dumps(run(parser.parse_args().root), indent=2))
