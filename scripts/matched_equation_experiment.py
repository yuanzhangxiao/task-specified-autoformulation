"""Compare three equation constructors on frozen graphs, holding out whole events."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import shape_equation_nomination as shape
from autoformalism.research import trajectory_function_families as legacy
from autoformalism.research.preproposal_cases import T2
from autoformalism.staged_topology import content_hash
from scripts import matched_topology_forecast as topology
from scripts import trajectory_functions as original
from scripts.construction_functions import file_hash

DEFAULT = topology.ROOT / "output/matched-equation-construction-20261125"
TRAIN_FILE = topology.ROOT / "output/trajectory-functions-20261123/inputs/train.json"
BOUND_PLAN = topology.ROOT / "output/matched-nonnegative-forecast-20261124/plan.json"
GRAPH_PLAN_ID = "62e9bd0fda7c92f0981671278006c81d6ab9d3917a01220881acd3fae83939d5"
BOUND_PLAN_ID = "e6eb96e11049fbfaa2ee62c15a2b706b8165f1d6b2a300acc71c98af2a412c15"
ARMS = ("original", "shape", "hybrid")


def _registered(path: Path, digest: str) -> dict:
    record = sealed_read(path)
    if record["artifact_sha256"] != digest:
        raise ValueError(f"Unexpected registered source at {path}")
    return record


def select_graphs(cohort: dict, dataset: str) -> list[dict]:
    """First graph per original, hash-ordered inventory/origin stratum."""
    chosen = {}
    for row in cohort["rows"]:
        if row["dataset"] == dataset:
            chosen.setdefault((row["inventory"], row["arm"]), row)
    if not chosen:
        raise ValueError("No admitted graphs for the specified study")
    return [chosen[key] for key in sorted(chosen)]


def split_events(rows: list[dict], fold: int) -> tuple[list, list]:
    """Two complementary folds assigned by sorted training-event ID alone."""
    if fold not in (0, 1):
        raise ValueError("Expected fold 0 or 1")
    ordered = sorted(rows, key=lambda r: r["trajectory_id"])
    ids = [r["trajectory_id"] for r in ordered]
    if (
        len(ids) < 4
        or len(set(ids)) != len(ids)
        or any(not name.startswith("train_") for name in ids)
    ):
        raise ValueError("Need distinct training events for fitting and scoring")
    return ordered[1 - fold :: 2], ordered[fold::2]


def original_menu(graph: dict, case: dict, training: list[dict]) -> dict:
    """Use only this fold in the historical constructor's generic branch."""
    source = {**case["evidence"], "trajectories": training}
    source.pop("sha256", None)
    source["sha256"] = content_hash(source)
    return legacy.build_shortlist(graph, {"evidence": source, "case_id": "anonymous"})


def prepare(root: Path) -> dict:
    roster = _registered(topology.DEFAULT / "plan.json", GRAPH_PLAN_ID)["cohort"]
    bounds = _registered(BOUND_PLAN, BOUND_PLAN_ID)["constraint"]["targets"]
    all_rows = _registered(TRAIN_FILE, original.TRAIN)["train"][T2]
    selected = select_graphs(roster, T2)
    panels = []
    with _lock(root / "prepare-lock"):
        for index, row in enumerate(selected):
            case = roster["cases"][row["case"]]
            evidence = case["evidence"]
            for fold in (0, 1):
                train, held = split_events(all_rows, fold)
                panel = f"graph{index:02}-fold{fold}"
                graph = row["graph"]
                interpolation = evidence["contract"]["interpolation"]
                packet = sealed_write(
                    root / "packets" / f"{panel}.json",
                    {
                        "panel_id": panel,
                        "graph": graph,
                        "public_task": case["public_brief"]["scientific_context"],
                        "train_rows": train,
                        "interpolation": interpolation,
                        "shape": shape.build_shape_shortlist(
                            graph,
                            train,
                            time_symbol=evidence["context"]["time_symbol"],
                            interpolation=interpolation,
                        ),
                    },
                )
                baseline = sealed_write(
                    root / "original" / f"{panel}.json",
                    {
                        "packet": packet["artifact_sha256"],
                        "menu": original_menu(graph, case, train),
                    },
                )
                panels.append(
                    {
                        "panel_id": panel,
                        "inventory": row["inventory"],
                        "origin": row["arm"],
                        "graph_id": row["graph_id"],
                        "fold": fold,
                        "packet": packet["artifact_sha256"],
                        "original": baseline["artifact_sha256"],
                        "held_ids": [r["trajectory_id"] for r in held],
                        "held_sha256": content_hash(held),
                    }
                )
        return sealed_write(
            root / "plan.json",
            {
                "protocol": "matched-equation-constructors-1",
                "dataset": T2,
                "source_graph_plan": GRAPH_PLAN_ID,
                "registered_training": original.TRAIN,
                "fit_constraint_plan": BOUND_PLAN_ID,
                "nonnegative_targets": bounds,
                "arms": list(ARMS),
                "slots_per_panel_per_arm": 2,
                "fit_budget": original.BUDGET,
                "seeds": list(original.SEEDS),
                "panels": panels,
                "constructor_sources": {
                    Path(m.__file__).name: file_hash(Path(m.__file__))
                    for m in (legacy, shape)
                },
                "graph_admission_scope": (
                    "Previously admitted on all sixteen training events."
                ),
            },
        )


def _fit_one(args):
    from scripts.matched_equation_forecasts import fit_arm

    root, panel, arm, targets = args
    name = panel["panel_id"]
    packet = _registered(root / "packets" / f"{name}.json", panel["packet"])
    if arm == "original":
        menus = _registered(root / "original" / f"{name}.json", panel["original"])
        formulas = menus["menu"]["candidates"]
    elif arm == "shape":
        formulas = packet["shape"]["candidates"]
    else:
        from scripts.matched_equation_hybrid import result_for

        outcome = result_for(packet, root)
        if outcome["status"] in {"pending", "reserved"}:
            raise RuntimeError(
                f"20B result pending for {name}; resume after collection"
            )
        formulas = [packet["shape"]["candidates"][0], outcome.get("formula")]
    if len(formulas) > 2:
        raise ValueError("Constructor exceeded two formula slots")
    fit_arm(
        root,
        panel_id=name,
        arm=arm,
        graph=packet["graph"],
        train_rows=packet["train_rows"],
        formulas=[*formulas, *([None] * (2 - len(formulas)))],
        nonnegative_targets=targets,
    )
    return name, arm


def fit(root: Path, arms: list[str], workers: int) -> None:
    plan = sealed_read(root / "plan.json")
    if workers < 1 or any(a not in plan["arms"] for a in arms):
        raise ValueError("Expected registered arms and a positive worker count")
    jobs = [
        (root, p, a, plan["nonnegative_targets"]) for p in plan["panels"] for a in arms
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for panel, arm in pool.map(_fit_one, jobs):
            print(f"Fit complete {panel} {arm}", flush=True)


def score(root: Path, arms: list[str]) -> None:
    from scripts.matched_equation_forecasts import score_arm

    plan = sealed_read(root / "plan.json")
    all_rows = _registered(TRAIN_FILE, original.TRAIN)["train"][T2]
    for p in plan["panels"]:
        name = p["panel_id"]
        packet = _registered(root / "packets" / f"{name}.json", p["packet"])
        _, held = split_events(all_rows, p["fold"])
        if content_hash(held) != p["held_sha256"]:
            raise ValueError("Held event set differs from plan")
        for arm in arms:
            score_arm(
                root, panel_id=name, arm=arm, graph=packet["graph"], held_rows=held
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "fit", "score"])
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        plan = prepare(args.root)
        print(f"Sealed {len(plan['panels'])} graph/fold panels at {args.root}")
    elif args.stage == "fit":
        fit(args.root, args.arms, args.workers)
    else:
        score(args.root, args.arms)


if __name__ == "__main__":
    main()
