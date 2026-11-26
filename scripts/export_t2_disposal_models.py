"""Choose complete balanced T2 models using the same four events across topologies."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import validate_formula
from autoformalism.search import bounded_event_selection as selection
from scripts.construction_functions import file_hash
from scripts.matched_equation_experiment import _registered
from scripts.run_t2_balanced_disposal import DEFAULT, load_endpoint


def protocol(root: Path) -> dict:
    """Register the topology and equation decision before inspecting held outcomes."""
    return sealed_write(
        root / "model-choice/plan.json",
        {
            "protocol": "balanced-disposal-graph-and-equation-selection-1",
            "candidates": (
                "All dependency-eligible balanced laws on both fitted parent graphs."
            ),
            "decision": (
                "Require a fit and qualification on every selection event. Minimize "
                "the sum of four event NMSEs, break exact ties by parameter count then "
                "graph/family ID."
            ),
            "order": (
                "Candidates by training NMSE then ID. Events by ID. "
                "Prune on strict lower bound."
            ),
            "sources": {
                p.name: file_hash(p) for p in (Path(__file__), Path(selection.__file__))
            },
        },
    )


def shared_events(panels: list[dict], rotation: int) -> tuple[list[str], list[str]]:
    """Check both graphs used identical disjoint fit and selection events."""
    if len(panels) != 2 or len({p["graph_name"] for p in panels}) != 2:
        raise ValueError("Expected two distinct parent graphs")
    designs = {
        (
            tuple(sorted(p["fit_ids"])),
            tuple(sorted(p["choice_rotations"][rotation]["selection_ids"])),
        )
        for p in panels
    }
    if len(designs) != 1:
        raise ValueError("Cross-graph event assignments differ")
    fit, choose = designs.pop()
    if (
        len(fit) != 8
        or len(choose) != 4
        or len(set(fit + choose)) != 12
        or any(not e.startswith("train_") for e in fit + choose)
    ):
        raise ValueError("Expected eight plus four distinct training events")
    return list(fit), list(choose)


def export(root: Path) -> dict:
    """Apply the registered decision to selection receipts and export fitted ODEs."""
    destination = root / "model-choice"
    policy = protocol(root)
    plan = sealed_read(root / "plan.json")
    manifest = sealed_read(root / "choices.json")
    if manifest["plan"] != plan["artifact_sha256"]:
        raise ValueError("Choices belong to another fitting experiment")
    summaries = []
    for fold in (0, 1):
        panels = [
            p
            for p in plan["panels"]
            if p["panel_id"] == f"{p['graph_name']}-fold{fold}"
        ]
        for rotation in (0, 1):
            fit_ids, ids = shared_events(panels, rotation)
            models, endpoints, receipt_map, exclusions = [], {}, {}, {}
            for p in panels:
                pid = p["panel_id"]
                decision = sealed_read(
                    root / "choices" / pid / f"rotation{rotation}.json"
                )
                if decision["artifact_sha256"] not in manifest["receipts"] or (
                    decision["policy"],
                    decision["panel"],
                    decision["rotation"],
                    sorted(decision["ids"]),
                ) != (manifest["policy"], pid, rotation, ids):
                    raise ValueError("Selection receipt identity differs")
                for family, c in p["candidates"].items():
                    if c["arm"] != "balanced":
                        continue
                    key = pid + "/" + family
                    graph, start, fit, digest = load_endpoint(root, plan, p, family)
                    if manifest["endpoints"][pid][family] != digest:
                        raise ValueError("Different fit endpoint")
                    if not c["dependency_eligible"] or fit is None:
                        exclusions[key] = (
                            "source contradiction"
                            if not c["dependency_eligible"]
                            else "no fit"
                        )
                        continue
                    models.append(
                        selection.FittedCandidate(
                            key, len(fit["parameters"]), fit["training_nmse"]
                        )
                    )
                    endpoints[key] = (graph, start, fit, digest)
                    receipt_map[key] = {
                        eid: decision["curves"][eid][family] for eid in ids
                    }

            def loss(key, event, receipt_map=receipt_map):
                record = _registered(
                    root / "curves" / key / f"{event}.json", receipt_map[key][event]
                )
                return record["nmse"] if record["status"] == "scored" else None

            fast = selection.select_on_all_events(models, ids, loss)
            full = selection.select_on_all_events(models, ids, loss, early_stop=False)
            if (fast["winner"], fast["selection_nmse"]) != (
                full["winner"],
                full["selection_nmse"],
            ):
                raise AssertionError("Bounded and exhaustive decisions differ")
            model = None
            if fast["winner"] is not None:
                graph, start, fit, digest = endpoints[fast["winner"]]
                formula = validate_formula(
                    graph, {**start, "parameters": fit["parameters"]}
                )
                model = {
                    "graph": graph,
                    "fitted_formula": formula,
                    "fit_endpoint": digest,
                    "training_nmse": fit["training_nmse"],
                }
            key = f"fold{fold}-rotation{rotation}"
            data = sealed_write(
                destination / f"{key}.json",
                {
                    "design": key,
                    "policy": policy["artifact_sha256"],
                    "choices": manifest["artifact_sha256"],
                    "fit_ids": fit_ids,
                    "selection_ids": ids,
                    "fit_exclusions": exclusions,
                    "decision": fast,
                    "without_loss_bound": full["event_evaluations"],
                    "model": model,
                },
            )
            summaries.append(
                {
                    k: fast[k]
                    for k in (
                        "winner",
                        "selection_nmse",
                        "event_evaluations",
                        "full_grid_size",
                    )
                }
                | {
                    "design": key,
                    "without_loss_bound": full["event_evaluations"],
                    "export": data["artifact_sha256"],
                }
            )
    return sealed_write(
        destination / "summary.json",
        {"rows": summaries, "choices": manifest["artifact_sha256"]},
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=["plan", "export"])
    cli.add_argument("--root", type=Path, default=DEFAULT)
    args = cli.parse_args()
    print((protocol if args.stage == "plan" else export)(args.root)["artifact_sha256"])
