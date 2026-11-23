"""Export actual graph selections and admitted graph-first revisions for functions."""

from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import preproposal_topology as topology
from autoformalism.research.preproposal_selection import SelectionCase
from autoformalism.staged_topology import content_hash


def _canonical(case, frozen, schema, graph, digest):
    """Recover all generated types and initial rules; verify the admission identity."""
    reply = schema.model_validate(graph).model_dump(mode="json", by_alias=True)
    kinds = {row.name: row.kind for row in case.inventory}
    kinds.update({r["name"]: r["kind"] for r in reply.get("new_latents", ())})
    states = sorted(name for name, kind in kinds.items() if kind == "differential")
    sources = {k: sorted(v) for k, v in topology.graph_sources(reply).items()}
    initials = {}
    for record in reply["latent_initializers"]:
        rule = record["rule"].copy()
        if rule["mode"] == "map":
            for key in ("public_initials", "state_initials"):
                rule[key] = sorted(rule[key])
        initials[record["state"]] = rule
    if (
        content_hash({"states": states, "sources": sources, "initializers": initials})
        != digest
    ):
        raise ValueError("Exported graph differs from the admitted structural identity")
    permitted = set(frozen["initials"]) | {case.target}
    contract = frozen["evidence"]["contract"]["initial_rules"]
    for state in set(states) - initials.keys():
        if (
            state not in permitted
            or contract.get(state, {}).get("mode") != "observed_initial"
        ):
            raise ValueError(f"Missing admitted initial rule for {state}")
        initials[state] = {"mode": "observed_initial", "channel": state}
    return {"variable_types": kinds, "sources": sources, "initializers": initials}


def _selected(root, key, group):
    """Yield each admitted selection or successful revision with its provenance."""
    pool = group["complete_graphs"]
    candidates = {row["candidate_id"]: row for row in pool["candidates"]}
    for draw in pool["draws"]:
        ids_by_arm = {"uniform_graph_selection": draw["uniform_graph_ids"]}
        if draw["scientific"]["status"] == "selected":
            ids_by_arm["scientific_graph_selection"] = draw["scientific"]["graph_ids"]
        for arm, ids in ids_by_arm.items():
            for rank, candidate_id in enumerate(ids, 1):
                row = candidates[candidate_id]
                yield (
                    row["structure_sha256"],
                    row["graph"],
                    {
                        "arm": arm,
                        "seed": draw["seed"],
                        "rank": rank,
                        "candidate_id": candidate_id,
                    },
                )

    for draw in group["draws"]:
        seed, first = draw["seed"], draw["graph_first"]
        first_slot = f"{key}--seed-{seed}--graph-first"
        origin = {"arm": "graph_first", "seed": seed, "slot": first_slot}
        if first["status"] == "topology_screen_passed":
            yield (
                first["structure_sha256"],
                first["parent"],
                origin | {"origin": "parent"},
            )
        for child in first.get("repairs", ()):
            if child["status"] == "topology_screen_passed":
                yield (
                    child["structure_sha256"],
                    child["graph"],
                    origin | {"origin": "repair", "edits": child["edits"]},
                )
        for arm in ("scientific", "uniform"):
            for rank, branch in enumerate(draw[arm], 1):
                assessed = branch["topology"]
                if assessed["status"] != "topology_screen_passed":
                    continue
                slot = branch["slot"]
                reply = sealed_read(root / "steps" / slot / "reply.json")
                yield (
                    assessed["structure_sha256"],
                    reply["value"],
                    {
                        "arm": arm + "_source_selection",
                        "seed": seed,
                        "candidate_id": branch["candidate_id"],
                        "slot": slot,
                        "rank": rank,
                    },
                )


def admitted_graphs(root: Path, summary: dict, intake: dict) -> dict:
    """Deduplicate independently within each inventory, retaining every selection."""
    rows = {}
    cases = {
        c["case_id"]: SelectionCase.model_validate(c)
        for c in intake["selection"]["cases"]
    }
    case_info = {}
    for group in summary["cases"]:
        key = group["case_id"]
        case, frozen = cases[key], intake["frozen_cases"][key]
        schema = topology.reply_schema(case, frozen)
        pool = group["complete_graphs"]
        case_info[key] = {
            "target": case.target,
            "minimum_target_gap": frozen.get("minimum_target_gap", 0),
            "evidence_sha256": case.evidence_sha256,
            "pool_sha256": pool["pool_sha256"],
        }

        for digest, graph, origin in _selected(root, key, group):
            identity = (key, digest)
            if identity not in rows:
                rows[identity] = {
                    "case_id": key,
                    "structure_sha256": digest,
                    "graph": _canonical(case, frozen, schema, graph, digest),
                    "origins": [],
                }
            rows[identity]["origins"].append(origin)
    graphs = [rows[k] for k in sorted(rows)]
    return {
        "protocol": "admitted-complete-graphs-1",
        "plan_sha256": summary["plan_sha256"],
        "run_status": summary["status"],
        "stage": "topology",
        "pending": ["interaction functions and signs", "parameter fitting"],
        "scientific_correctness_assessed": False,
        "distinct_case_graphs": len(graphs),
        "cases": case_info,
        "graphs": graphs,
    }
