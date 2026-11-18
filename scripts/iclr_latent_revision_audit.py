"""Audit retained latent introductions and required-pathway membership.

Mechanism membership is an explicit graph operationalization, not a claim of
physical identification. Feedback citation is recorded provenance, not a causal
ablation of the proposer's use of numerical information.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts.collect_iclr_latent_histories import digest
from scripts.iclr_active_history import graph_for, reachable


def pathway_nodes(graph: dict[str, set[str]], target: str, driver: str) -> set[str]:
    """Union nodes on simple driver-to-output paths; do not admit side cycles.

    Dependencies point backward. Stopping at the specified driver prevents a
    feedback cycle from attributing upstream secretion memory to insulin action.
    The same state may belong to multiple independently specified pathways.
    """
    found: set[str] = set()
    root = "output:" + target
    visits = 0

    def visit(node: str, path: tuple[str, ...]) -> None:
        nonlocal visits
        visits += 1
        if visits > 200_000:
            raise ValueError("path enumeration exceeded audit bound")
        if node == driver:
            found.update((*path, node))
            return
        if node in path:
            return
        for dependency in sorted(graph.get(node, ())):
            visit(dependency, (*path, node))

    visit(root, ())
    return found


def anatomy(model: dict) -> dict:
    """Count fitted, target-connected differential coordinates per requirement."""
    candidate, parameters = model["candidate"], model["parameters"]
    context = model["context"]
    active = reachable(
        {
            "candidate": candidate,
            "parameters": parameters,
            "lowered_context": context,
        }
    )
    graph = graph_for(candidate, parameters)
    requirements = (model.get("brief") or {}).get("requirements")
    if requirements is None:
        raise ValueError("actual public requirements unavailable")
    latents = set(active["active_latents"])
    auxiliary_aliases = latents & set(model.get("modeled_observed_auxiliaries", []))
    latents -= auxiliary_aliases
    active["active_latents"] = sorted(latents)
    active["modeled_observed_auxiliary_states"] = sorted(auxiliary_aliases)
    mechanism_rows = []
    for requirement in requirements:
        drivers = requirement.get("drivers", [])
        targets = requirement.get("targets", [])
        nodes: set[str] = set()
        present = False
        if drivers:
            paths = [pathway_nodes(graph, t, d) for t in targets for d in drivers]
            present = any(paths)
            for path in paths:
                nodes |= path
        else:
            # Some briefs specify the whole reactor balance without driver bindings.
            for target in targets:
                nodes.update(active["target_upstream_states"].get(target, []))
            # With no declared driver binding, the broad balance's presence
            # cannot be established by this driver-to-target path audit.
            present = None
        mechanism_rows.append(
            {
                "id": requirement["id"],
                "requirement": requirement,
                "latent_states": sorted(latents & nodes),
                "path_present": present,
                "mapping_rule": "driver_to_output_simple_paths"
                if drivers
                else "broad_target_balance_all_upstream_states",
            }
        )
    assigned = set().union(*(set(m["latent_states"]) for m in mechanism_rows))
    return {
        **active,
        "requirements": mechanism_rows,
        "outside_required_paths": sorted(latents - assigned),
        "required_path_latents": sorted(assigned),
    }


def feedback_attribution(visit: dict) -> dict:
    """Separate observable numerical citations, other advice, and missing traces."""
    if "guided_feedback" in visit:
        return visit["guided_feedback"]
    accepted = [a for a in visit.get("attempts", []) if a.get("accepted")]
    if not accepted:
        return {"status": "accepted_reply_missing", "numerical_evidence_cited": False}
    attempt = accepted[-1]
    raw = attempt.get("raw") or {}
    matching = [
        r
        for r in visit.get("requests", [])
        if r.get("request_hash") == attempt.get("request_hash")
    ]
    request = matching[-1] if matching else {}
    payload = request.get("payload") or {}
    training = payload.get("training_evidence") or {}
    references = set(raw.get("evidence_refs") or [])
    catalog = {r["ref"] for r in payload.get("evidence_catalog", [])}
    known = sorted(references & catalog)
    numerical = bool(known and training.get("split") == "train")
    advice = payload.get("scientific_advice") or {}
    return {
        "status": request.get("status", "request_missing"),
        "numerical_evidence_cited": numerical,
        "cited_training_refs": known,
        "reply_evidence_refs": sorted(references),
        "hypothesis": raw.get("hypothesis"),
        "request_hash": attempt.get("request_hash"),
        "scientific_advice_present": bool(advice.get("findings")),
        "public_requirement_findings": payload.get("public_requirement_findings"),
        "retry_feedback": payload.get("retry_feedback"),
        "cited_records": [
            r for r in payload.get("evidence_catalog", []) if r["ref"] in known
        ],
    }


def state_births(run: dict, models: dict) -> dict:
    """Track retained lineages, treating removal/reintroduction and gaps explicitly."""
    births = {}
    prior, prior_round = None, None
    first_fit = None

    @cache
    def inspect(uid: str) -> dict:
        return anatomy(models[uid])

    prior_uid = None
    for visit in sorted(run["visits"], key=lambda x: x["round"]):
        uid = visit.get("selected_model")
        if uid is None:
            prior, prior_round = None, visit["round"]
            continue
        current = models[uid]
        current_states = {s["name"] for s in current["candidate"]["states"]}
        previous_states = (
            set()
            if prior is None
            else {s["name"] for s in prior["candidate"]["states"]}
        )
        for removed in previous_states - current_states:
            births.pop(removed, None)
        added = current_states - previous_states
        first_fit = visit["round"] if first_fit is None else first_fit
        previous_anatomy = inspect(prior_uid) if prior is not None and added else None
        for name in added:
            before_latents = (
                set()
                if prior is None
                else {
                    s["name"]
                    for s in prior["candidate"]["states"]
                    if s["kind"] == "latent"
                }
            )
            current_latents = {
                s["name"]
                for s in current["candidate"]["states"]
                if s["kind"] == "latent"
            }
            consecutive = prior_round is not None and visit["round"] == prior_round + 1
            initial = visit["round"] == first_fit
            births[name] = {
                "round": visit["round"],
                "initial": initial,
                "history_consecutive": consecutive,
                "post_fit_introduction": prior is not None
                and consecutive
                and not initial,
                "pure_augmentation": bool(
                    prior is not None
                    and consecutive
                    and not (previous_states - current_states)
                    and len(current_latents) > len(before_latents)
                ),
                "simultaneous_removals": sorted(previous_states - current_states),
                "prior_model": None if prior is None else digest(prior),
                "introduced_model": uid,
                "prior_mechanisms": []
                if previous_anatomy is None
                else previous_anatomy["requirements"],
                "lineage_mechanisms": [],
                "feedback": feedback_attribution(visit)
                if not initial
                else {
                    "status": "initial_construction_or_first_fit",
                    "numerical_evidence_cited": False,
                },
            }
        current_anatomy = inspect(uid)
        for name in current_states:
            if name in births:
                births[name]["lineage_mechanisms"] = sorted(
                    set(births[name]["lineage_mechanisms"])
                    | {
                        m["id"]
                        for m in current_anatomy["requirements"]
                        if name in m["latent_states"]
                    }
                )
        prior, prior_round = current, visit["round"]
        prior_uid = uid
    return births


def arm_name(task: dict) -> str:
    """Distinguish the fully enabled method from specification/component controls."""
    if "treatment" in task:
        return task["treatment"]
    arm = task["arm"]
    switches = (
        task.get("critic", True),
        task.get("scientific_verifier", True),
        task.get("shared_processes", True),
    )
    if arm == "full" and all(switches):
        return "full"
    return arm + f":c{int(switches[0])}v{int(switches[1])}s{int(switches[2])}"


def audit_campaign(name: str, campaign: dict) -> tuple[list, list, list]:
    """Produce final and separately labelled end-search snapshots for every run."""
    models = campaign["models"]

    @cache
    def inspect(uid: str) -> dict:
        return anatomy(models[uid])

    runs, states, mechanisms = [], [], []
    for run in campaign["runs"]:
        task = run["task"]
        base = {
            "run_id": campaign["plan_sha256"] + ":" + task["task_id"],
            "cohort": name,
            "proposer": campaign["model_settings"]["model"],
            "run_class": "guided_alternative"
            if name.startswith("guided")
            else "search",
            "benchmark": task["cell"],
            "seed": task["seed"],
            "arm": arm_name(task),
            "planned_visits": campaign["rounds"],
            "recorded_visits": len(run["visits"]),
            "search_complete": run["search_complete"],
            "final_status": run["final_status"],
        }
        births = state_births(run, models)
        for snapshot, uid in [
            ("final", run.get("final_model")),
            ("end_search", run.get("latest_model")),
            ("latest_available", run.get("final_model") or run.get("latest_model")),
        ]:
            row = {
                **base,
                "snapshot": snapshot,
                "model_id": uid,
                "available": uid is not None,
                "endpoint_stage": "after_endpoint_selection"
                if uid and uid == run.get("final_model") and snapshot != "end_search"
                else "end_search_before_pruning",
            }
            if not uid:
                runs.append(row)
                continue
            a = inspect(uid)
            row.update(
                {
                    "active_latent_count": len(a["active_latents"]),
                    "outside_required_count": len(a["outside_required_paths"]),
                    "required_latent_count": len(a["required_path_latents"]),
                    "inactive_state_count": len(a["inactive_states"]),
                }
            )
            local_states = []
            for state in a["active_latents"]:
                birth = births.get(
                    state,
                    {
                        "initial": None,
                        "post_fit_introduction": False,
                        "pure_augmentation": False,
                        "feedback": {
                            "status": "lineage_unavailable",
                            "numerical_evidence_cited": False,
                        },
                    },
                )
                roles = [
                    m["id"] for m in a["requirements"] if state in m["latent_states"]
                ]
                prior = {m["id"]: m for m in birth.get("prior_mechanisms", [])}
                existing = [
                    rid for rid in roles if prior.get(rid, {}).get("latent_states")
                ]
                s = {
                    **base,
                    "snapshot": snapshot,
                    "state": state,
                    "model_id": uid,
                    "required_mechanisms": roles,
                    "outside_required_paths": not roles,
                    "historically_required_path": bool(birth.get("lineage_mechanisms")),
                    "state_declaration": next(
                        d
                        for d in models[uid]["candidate"]["states"]
                        if d["name"] == state
                    ),
                    "equation": next(
                        e
                        for e in models[uid]["candidate"]["state_equations"]
                        if e["state"] == state
                    ),
                    "prior_latent_mechanisms": existing,
                    "birth": birth,
                    "retained_post_fit_addition": birth["post_fit_introduction"],
                    "retained_pure_augmentation": birth["pure_augmentation"],
                    "numerically_cited_addition": bool(
                        birth["pure_augmentation"]
                        and birth["feedback"]["numerical_evidence_cited"]
                    ),
                    "supplements_existing_mechanism_latent": bool(
                        existing and birth["pure_augmentation"]
                    ),
                }
                local_states.append(s)
                states.append(s)
            for flag in [
                "retained_post_fit_addition",
                "retained_pure_augmentation",
                "numerically_cited_addition",
                "supplements_existing_mechanism_latent",
            ]:
                row[flag + "_count"] = sum(s[flag] for s in local_states)
            for m in a["requirements"]:
                mechanisms.append(
                    {
                        **base,
                        "snapshot": snapshot,
                        "model_id": uid,
                        **m,
                        "latent_count": len(m["latent_states"]),
                        "added_after_existing_latent_count": sum(
                            m["id"] in s["prior_latent_mechanisms"]
                            and s["retained_pure_augmentation"]
                            for s in local_states
                        ),
                        "numerically_cited_addition_count": sum(
                            m["id"] in s["required_mechanisms"]
                            and s["numerically_cited_addition"]
                            for s in local_states
                        ),
                    }
                )
            runs.append(row)
    return runs, states, mechanisms


def summarize(rows: list[dict]) -> dict:
    """Conditional state distributions never convert missing models to zero."""
    ready = [r for r in rows if r["available"]]
    fields = [
        "active_latent_count",
        "outside_required_count",
        "required_latent_count",
        "retained_post_fit_addition_count",
        "retained_pure_augmentation_count",
        "numerically_cited_addition_count",
        "supplements_existing_mechanism_latent_count",
    ]
    out = {
        "planned": len(rows),
        "available": len(ready),
        "final_statuses": dict(Counter(r["final_status"] for r in rows)),
    }
    for field in fields:
        hist = Counter(r[field] for r in ready)
        nonzero = sum(n for k, n in hist.items() if k)
        out[field] = {
            "histogram": dict(sorted(hist.items())),
            "models_with_any": nonzero,
            "percentage_of_available": 100 * nonzero / len(ready) if ready else None,
            "percentage_of_all_assignments": 100 * nonzero / len(rows)
            if rows
            else None,
            "total_states": sum(k * n for k, n in hist.items()),
        }
    return out


def summarize_mechanisms(rows: list[dict]) -> dict:
    """Keep missing pathways distinct from represented zero-latent pathways."""
    histogram = dict(sorted(Counter(r["latent_count"] for r in rows).items()))
    zero_present = sum(
        r["latent_count"] == 0 and r["path_present"] is True for r in rows
    )
    absent = sum(r["path_present"] is False for r in rows)
    return {
        "requirement_model_pairs": len(rows),
        "latent_count_histogram": histogram,
        "represented_without_latents": zero_present,
        "path_absent": absent,
        "broad_balance_presence_unassessed": sum(
            r["path_present"] is None for r in rows
        ),
        "zero_latent_broad_balance_models": sum(
            r["path_present"] is None and r["latent_count"] == 0 for r in rows
        ),
        "represented_without_latents_percent": 100 * zero_present / len(rows)
        if rows
        else None,
        "with_supplemental_latents": sum(
            r["added_after_existing_latent_count"] > 0 for r in rows
        ),
        "supplemental_state_count_histogram": dict(
            sorted(
                Counter(r["added_after_existing_latent_count"] for r in rows).items()
            )
        ),
    }


def write_table(path: Path, rows: list[dict]) -> None:
    """Write portable tabular audit records with nested cells as JSON."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Separate immutable analysis directory"
    )
    args = parser.parse_args()
    destination = args.output or args.root
    destination.mkdir(parents=True, exist_ok=True)
    campaigns, runs, states, mechanisms = [], [], [], []
    seen = set()
    for path in sorted((args.root / "sources").glob("*.json.gz")):
        d = json.loads(gzip.decompress(path.read_bytes()))
        if d["artifact_sha256"] != digest(
            {k: v for k, v in d.items() if k != "artifact_sha256"}
        ):
            raise ValueError("source export hash mismatch")
        if d["plan_sha256"] in seen:
            raise ValueError("duplicate campaign plan")
        seen.add(d["plan_sha256"])
        name = path.name.removesuffix(".json.gz")
        r, s, m = audit_campaign(name, d)
        runs.extend(r)
        states.extend(s)
        mechanisms.extend(m)
        campaigns.append(
            {
                "cohort": name,
                "plan": d["plan_sha256"],
                "root": d["root"],
                "errors": d["errors"],
                "runs": len(d["runs"]),
            }
        )
        print(name, "audited", len(d["runs"]), flush=True)
    summary = {}
    for keys in [
        ("snapshot", "run_class", "proposer"),
        ("snapshot", "run_class", "proposer", "arm"),
        ("snapshot", "cohort", "arm"),
        ("snapshot", "cohort", "benchmark", "arm"),
    ]:
        groups = defaultdict(list)
        for r in runs:
            groups[tuple(r[k] for k in keys)].append(r)
        summary["/".join(keys)] = [
            {**dict(zip(keys, key, strict=True)), **summarize(rows)}
            for key, rows in sorted(groups.items())
        ]
    mechanism_summary = {}
    for keys in [
        ("snapshot", "run_class", "proposer", "arm"),
        ("snapshot", "cohort", "benchmark", "arm", "id"),
    ]:
        groups = defaultdict(list)
        for row in mechanisms:
            groups[tuple(row[k] for k in keys)].append(row)
        mechanism_summary["/".join(keys)] = [
            {**dict(zip(keys, key, strict=True)), **summarize_mechanisms(rows)}
            for key, rows in sorted(groups.items())
        ]
    # JSON normalization makes integer histogram keys stable across resumes.
    sealed_write(
        destination / "audit.json",
        json.loads(
            json.dumps(
                {
                    "campaigns": campaigns,
                    "runs": runs,
                    "states": states,
                    "mechanisms": mechanisms,
                }
            )
        ),
    )
    sealed_write(destination / "summary.json", json.loads(json.dumps(summary)))
    sealed_write(
        destination / "mechanism_summary.json",
        json.loads(json.dumps(mechanism_summary)),
    )
    write_table(destination / "runs.csv", runs)
    write_table(destination / "states.csv", states)
    write_table(destination / "mechanisms.csv", mechanisms)


if __name__ == "__main__":
    main()
