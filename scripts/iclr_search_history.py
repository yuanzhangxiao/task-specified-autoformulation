"""Reconstruct paired search histories, equation stability and marginal envelopes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from statistics import median

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from autoformalism.search.identity import candidate_identity
from autoformalism.staged_topology import content_hash
from scripts.iclr_submission_evidence import (
    ARCHIVE,
    CUTOFF,
    PLAN,
    REPO,
    digest,
    load_archive,
    table,
)

INVENTORY = REPO / "artifacts/dalla-all-models-review-2026-09-25/source/inventory.json"
SUCCESS = "cell08_seed0_full_c1v1s1"
FAILED = "cell07_seed0_full_c1v1s1"


def learning(history: list[dict], assessments: dict) -> list[dict]:
    """Pair measurements on each retained model and label separate marginal extrema."""
    records = []
    ordered = sorted(history, key=lambda r: (r["task_id"], r["round"]))
    for task, entries in groupby(ordered, key=lambda r: r["task_id"]):
        lowest, best_graph, best_functional = None, None, None
        previous = -1
        for r in entries:
            step = r["round"]
            if step != previous + 1:
                raise ValueError(f"missing or duplicate planned round at {task}/{step}")
            previous = step
            value = r["retained_validation_nmse"]
            graph = r["all_graph_requirements_certified"]
            functional = assessments.get((task, step, r["result_sha256"]))
            for candidate in (value, r["trial_validation_nmse"]):
                if candidate is not None:
                    lowest = candidate if lowest is None else min(lowest, candidate)
            if graph is not None:
                best_graph = max(float(graph), best_graph or 0.0)
            if functional is not None:
                best_functional = max(functional, best_functional or 0.0)
            records.append(
                {
                    "cell": r["cell"],
                    "task": task,
                    "configuration": task.rsplit("_", 1)[1],
                    "seed": r["seed"],
                    "prompt": r["arm"],
                    "round": step,
                    "status": r["status"],
                    "selected_origin": r["selected_origin_round"],
                    "result_sha256": r["result_sha256"],
                    "selected_validation_nmse": value,
                    "selected_train_nmse": r["retained_train_nmse"],
                    "selected_graph_certified": graph,
                    "selected_functional_fraction": functional,
                    "trial_validation_nmse": r["trial_validation_nmse"],
                    "best_validation_seen": lowest,
                    "best_retained_graph_seen": best_graph,
                    "best_retained_functional_seen": best_functional,
                    "cumulative_proposer_tokens": r["cumulative_tokens"],
                    "cumulative_proposer_calls": r["cumulative_requests"],
                    "cumulative_fit_attempts": r["cumulative_fit_attempts"],
                    "provider_unknown_usage": r["provider_calls_with_unknown_usage"],
                }
            )
    return records


def assessed_visits(folder: Path, history: list[dict]) -> dict:
    """Accept only sealed fitted-model assessments for this exact retained record."""
    results = {}
    archived = {(r["task_id"], r["round"]): r["result_sha256"] for r in history}
    if (folder / "plan.json").is_file():
        plan = sealed_read(folder / "plan.json")
        if plan.get("protocol") != "iclr-retained-functional-1" or plan["plan"] != PLAN:
            raise ValueError("incompatible local assessment export")
        for row in plan["rows"]:
            key = (row["task"], row["round"], row["result_sha256"])
            if archived.get(key[:2]) != key[2]:
                raise ValueError("local assessment was made on another visit")
            if row["job"] is None:
                continue
            path = folder / "jobs" / row["job"] / "score.json"
            if path.is_file():
                value = sealed_read(path)
                if (
                    value["job"] != row["job"]
                    or value["plan_sha256"] != plan["artifact_sha256"]
                ):
                    raise ValueError("local assessment was made on another model")
                results[key] = value["confirmed"]
        return results
    for planfile in sorted(folder.glob("visit_*/functional/plan.json")):
        plan = sealed_read(planfile)
        if plan["protocol"] != "fitted-public-mechanism-tests-1":
            raise ValueError("unexpected operational assessment protocol")
        for row in plan["rows"]:
            task = row["task"]["task_id"]
            step = row["source_round"]
            key = (task, step, row["source_result_sha256"])
            path = planfile.parent / "results" / f"{row['index']:04d}" / "result.json"
            if row["status"] != "ready" or not path.exists():
                continue
            if (task, step) in archived and archived[task, step] is None:
                continue  # A later worker receipt cannot fill the historical cutoff.
            if (
                row["source_plan_sha256"] != PLAN
                or archived.get((task, step)) != key[2]
            ):
                raise ValueError("functional records and search snapshot differ")
            r = sealed_read(path)
            if (
                r["plan_sha256"] != plan["artifact_sha256"]
                or r["index"] != row["index"]
                or key in results
            ):
                raise ValueError(
                    "functional result index, identity or uniqueness differs"
                )
            statuses = [m["status"] for m in r["mechanisms"]]
            if not statuses or set(statuses) - {"pass", "fail", "unresolved"}:
                raise ValueError("invalid functional outcomes")
            results[key] = statuses.count("pass") / len(statuses)
    return results


@lru_cache(maxsize=512)
def describe(payload: str) -> dict:
    """Preserve sign and public channels when comparing canonicalized expressions."""
    model = CandidateModel.model_validate_json(payload)
    identity = candidate_identity(model)
    return {
        "topology_sha256": identity.topology_sha256,
        "functional_sha256": identity.functional_sha256,
        "states": len(model.states),
        "latent_states": sum(s.kind.value == "latent" for s in model.states),
        "parameters": len(model.parameters),
        "processes": len(model.processes),
        "declared_roles": "|".join(
            sorted({tag for s in model.states for tag in s.mechanisms})
        ),
        "equations": len(model.state_equations),
    }


def matched_models(inv: dict, snapshots: dict, complete: Path | None) -> dict:
    """Index selected models at the fixed cutoff from verified campaign records."""
    models = {}
    for item in inv["occurrences"]:
        if item["round"] != CUTOFF or item["role"] != "selected":
            continue
        task = item["task"]["task_id"]
        if task in models or item["source_sha256"] != snapshots[task]["result_sha256"]:
            raise ValueError("duplicate or mismatched selected model")
        models[task] = inv["models"][item["model_id"]]["candidate"]
    if complete:
        bundle = sealed_read(complete)
        if (
            bundle["source_plan_sha256"] != PLAN
            or bundle["snapshot_round"] != CUTOFF
            or bundle["test_data_opened"] is not False
        ):
            raise ValueError("expected exact six-visit development export")
        for r in bundle["rows"]:
            if r["status"] != "ready":
                continue
            task = r["task"]["task_id"]
            if r["source_result_sha256"] != snapshots[task]["result_sha256"]:
                raise ValueError("complete export and archived result differ")
            if task in models and models[task] != r["candidate"]:
                raise ValueError(
                    "candidate disagreement across matching source records"
                )
            models[task] = r["candidate"]
    return models


def stability(history: list[dict], models: dict) -> tuple[list[dict], list[dict]]:
    """Keep every expected seed pair, including models without an equation export."""
    groups = defaultdict(dict)
    rows = []
    for r in history:
        if r["round"] != CUTOFF:
            continue
        task = r["task"]
        present = task in models
        status = (
            "available"
            if present
            else (
                "no_retained_fit"
                if r["selected_validation_nmse"] is None
                else "model_export_missing"
            )
        )
        row = {
            k: r[k]
            for k in (
                "cell",
                "prompt",
                "configuration",
                "seed",
                "task",
                "selected_validation_nmse",
                "selected_graph_certified",
                "selected_functional_fraction",
            )
        }
        row.update(status=status)
        if present:
            row.update(describe(json.dumps(models[task], sort_keys=True)))
        rows.append(row)
        block = groups[r["cell"], r["prompt"], r["configuration"]]
        if r["seed"] in block:
            raise ValueError("duplicate stability seed")
        block[r["seed"]] = row
    pairs = []
    for (cell, prompt, arm), block in sorted(groups.items()):
        if set(block) != {0, 1}:
            raise ValueError("expected both planned submission seeds")
        a, b = block[0], block[1]
        compared = a["status"] == b["status"] == "available"
        pairs.append(
            {
                "cell": cell,
                "prompt": prompt,
                "configuration": arm,
                "compared": compared,
                "seed0_status": a["status"],
                "seed1_status": b["status"],
                "same_topology": a["topology_sha256"] == b["topology_sha256"]
                if compared
                else None,
                "same_functions": a["functional_sha256"] == b["functional_sha256"]
                if compared
                else None,
                "same_dimension": a["states"] == b["states"] if compared else None,
                "seed0_nmse": a["selected_validation_nmse"],
                "seed1_nmse": b["selected_validation_nmse"],
            }
        )
    return rows, pairs


def trace(inv: dict, rows: list[dict], task: str) -> tuple[list[str], dict]:
    """Render fitted trials, selection decisions and observed diagnostics."""
    visits = {(v["task"]["task_id"], v["round"]): v for v in inv["visits"]}
    retained = {}
    for o in inv["occurrences"]:
        if (
            o["role"] in ("trial", "selected")
            and o["task"]["task_id"] == task
            and o["round"] <= CUTOFF
        ):
            retained[o["round"], o["role"]] = o
    lines = [f"## {task}", ""]
    payload = {}
    for r in (x for x in rows if x["task"] == task and x["round"] <= CUTOFF):
        i = r["round"]
        visit = visits[task, i]
        lines.extend(
            [
                f"### Visit {i + 1}",
                "",
                f"Recorded status `{r['status']}`. "
                f"Retained validation NMSE: {r['selected_validation_nmse']}.",
                "",
            ]
        )
        for field in ("proposal_error", "error"):
            if visit[field]:
                lines += [f"Recorded diagnostic: {visit[field]}", ""]
        for role in ("trial", "selected"):
            o = retained.get((i, role))
            if not o:
                continue
            if o["source_sha256"] != r["result_sha256"]:
                raise ValueError("lineage identity differs")
            uid = o["model_id"]
            m = inv["models"][uid]
            lines += [
                f"{role.title()} model `{uid[:12]}`. Validation NMSE "
                f"{o['fit']['validation'].get('normalized_mse')}.",
                "",
            ]
            if uid not in payload:
                candidate = m["candidate"]
                lines += [
                    "~~~text",
                    *[
                        f"d{x['state']}/dt = {x['rhs']}"
                        for x in candidate["state_equations"]
                    ],
                    *[
                        f"{x['name']} = {x['expression']}"
                        for x in candidate["processes"]
                    ],
                    "~~~",
                    "",
                ]
                payload[uid] = {
                    "candidate": candidate,
                    "fit_parameters": m["parameters"],
                    "initialization": m["initialization_plan"],
                }
    return lines, payload


def plot(rows: list[dict], root: Path) -> None:
    """Show the fixed planned denominator alongside observed validation medians."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 8), layout="constrained")
    top, mechanism, bottom = fig.subplots(3, 1, sharex=True)
    for prompt in ("full", "brief_only"):
        selected = [
            r for r in rows if r["prompt"] == prompt and r["configuration"] == "c1v1s1"
        ]
        available = []
        values = []
        steps = []
        for i in range(15):
            xs = [
                r["selected_validation_nmse"]
                for r in selected
                if r["round"] == i and r["selected_validation_nmse"] is not None
            ]
            steps.append(i + 1)
            available.append(len(xs))
            values.append(median(xs) if xs else float("nan"))
        top.plot(steps, values, "o-", label=prompt.replace("_", " ").title())
        bottom.plot(steps, available, "o-")
        operational = []
        counts = []
        for i in range(15):
            fs = [
                r["selected_functional_fraction"]
                for r in selected
                if r["round"] == i and r["selected_functional_fraction"] is not None
            ]
            operational.append(sum(fs) / len(fs) if fs else float("nan"))
            counts.append(len(fs))
        mechanism.plot(steps, operational, "o-")
        bottom.plot(steps, counts, ":")
    top.set(
        ylabel="Available-run median validation NMSE",
        title="Retained models with critic and verifier",
    )
    top.legend()
    mechanism.set(ylabel="Mean confirmed fraction\n(assessed runs)", ylim=(-0.05, 1.05))
    bottom.set(
        xlabel="Search visit",
        ylabel="Coverage of 18\nsolid fit · dotted assessment",
        ylim=(-1, 19),
    )
    fig.savefig(root / "learning_curves.png", dpi=155)


def run(args) -> dict:
    search, campaign = load_archive(args.archive)
    inv = sealed_read(args.inventory)
    if inv["plan_sha256"] != PLAN or inv["test_data_opened"] is not False:
        raise ValueError("incompatible equation inventory")
    snapshots = {r["task_id"]: r for r in search["rows"] if r["round"] == CUTOFF}
    assessments = (
        assessed_visits(args.functional_root, search["rows"])
        if args.functional_root
        else {}
    )
    models = matched_models(inv, snapshots, args.complete_bundle)
    rows = learning(search["rows"], assessments)
    seeds, pairs = stability(rows, models)
    summary = sealed_write(
        args.output / "snapshot.json",
        {
            "protocol": "iclr-search-history-1",
            "plan": PLAN,
            "archive": digest(args.archive),
            "inventory": inv["artifact_sha256"],
            "code": digest(Path(__file__)),
            "roster": len(snapshots),
            "planned_visits": len(rows),
            "seed_pairs": len(pairs),
            "complete_structure_pairs": sum(r["compared"] for r in pairs),
            "same_structure_count": sum(r["same_topology"] is True for r in pairs),
            "same_function_count": sum(r["same_functions"] is True for r in pairs),
            "same_state_count": sum(r["same_dimension"] is True for r in pairs),
            "functional_measurements": len(assessments),
            "assessed_scores_sha256": content_hash(
                sorted([*key, value] for key, value in assessments.items())
            ),
            "complete_bundle": digest(args.complete_bundle)
            if args.complete_bundle
            else None,
            "graph_evidence": "Boolean graph certification of the incumbent",
            "full_critic_usage_at_snapshot": campaign["judge_cost"],
        },
    )
    table(args.output / "learning_curves.csv", rows)
    table(args.output / "seed_structures.csv", seeds)
    table(args.output / "seed_stability.csv", pairs)
    lines = [
        "# Search histories",
        "",
        "Both examples use named Full seed 0 with critic and verifier. "
        "T2-hard has six observed visits. T2-easy records an interruption. "
        "The archive contains fit and selection receipts. Critic text is unavailable.",
        "",
    ]
    for task in (SUCCESS, FAILED):
        out, ms = trace(inv, rows, task)
        lines.extend(out)
        sealed_write(
            args.output / f"{task}.json",
            {"task": task, "models": ms, "source": inv["artifact_sha256"]},
        )
    (args.output / "candidate_lineages.md").write_text("\n".join(lines) + "\n")
    plot(rows, args.output)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--inventory", type=Path, default=INVENTORY)
    p.add_argument("--complete-bundle", type=Path)
    p.add_argument("--functional-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(p.parse_args()), indent=2))
