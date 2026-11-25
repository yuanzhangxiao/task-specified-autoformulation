"""Run unchanged generic equation constructors on the two archived basin graphs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_cases import BASIN
from autoformalism.research.shape_equation_nomination import build_shape_shortlist
from autoformalism.staged_topology import content_hash
from scripts import matched_equation_experiment as experiment
from scripts import matched_equation_forecasts as evaluation
from scripts import trajectory_function_intake as intake
from scripts import trajectory_functions as original
from scripts.construction_functions import file_hash

DEFAULT = intake.REPO / "output/shape-portability-20261125"


def registered_basin_graphs(source: dict, replay: Path) -> list[dict]:
    """Verify both complete graphs against the original training-only selection."""
    if source["artifact_sha256"] != original.TRAIN:
        raise ValueError("Unexpected training intake")
    if (
        source["source_replay"] != intake.REPLAY_FILE
        or file_hash(replay) != intake.REPLAY_FILE
    ):
        raise ValueError("Unexpected topology replay")
    rows = [r for r in source["graphs"] if r["dataset"] == BASIN]
    selected = [
        r
        for r in intake.select_graphs(json.loads(replay.read_text()))
        if r["dataset"] == BASIN
    ]
    if rows != selected or len({r["structure_sha256"] for r in rows}) != 2:
        raise ValueError(
            "Archived basin graphs differ from original selected topologies"
        )
    return rows


def prepare(root: Path) -> dict:
    """Freeze two arms and disjoint training folds on every archived basin graph."""
    source = sealed_read(experiment.TRAIN_FILE)
    graphs = registered_basin_graphs(source, intake.REPLAY)
    panels = []
    for i, row in enumerate(graphs):
        graph = row["graph"]
        case = source["cases"][row["case"]]
        ev = case["evidence"]
        for fold in (0, 1):
            training, held = experiment.split_events(source["train"][BASIN], fold)
            panel = f"graph{i:02}-fold{fold}"
            args = {
                "time_symbol": ev["context"]["time_symbol"],
                "interpolation": ev["contract"]["interpolation"],
            }
            menus = {
                "original": experiment.original_menu(graph, case, training),
                "shape": build_shape_shortlist(graph, training, **args),
            }
            if "isolated_outlet" in menus["original"]["evidence"]:
                raise ValueError(
                    "Specialized outlet shortcut entered a generic constructor"
                )
            packet = sealed_write(
                root / "packets" / f"{panel}.json",
                {
                    "panel": panel,
                    "graph": graph,
                    "structure_sha256": row["structure_sha256"],
                    "graph_sha256": content_hash(graph),
                    "fitting_rows": training,
                    "held_event_ids": [r["trajectory_id"] for r in held],
                    "menus": menus,
                },
            )
            panels.append(
                {"panel": panel, "packet": packet["artifact_sha256"], "fold": fold}
            )
    return sealed_write(
        root / "plan.json",
        {
            "protocol": "generic-equation-portability-1",
            "source": source["artifact_sha256"],
            "selection_replay": source["source_replay"],
            "dataset": BASIN,
            "panels": panels,
            "budget": evaluation.BUDGET,
            "seeds": list(evaluation.SEEDS),
            "nonnegative_targets": [],
            "code": {
                path.name: file_hash(path)
                for path in (
                    Path(__file__),
                    Path(experiment.shape.__file__),
                    Path(experiment.legacy.__file__),
                    Path(evaluation.__file__),
                    Path(experiment.__file__),
                )
            },
            "test_data_opened": False,
            "validation_data_opened": False,
        },
    )


def run_arm(item: tuple[Path, dict, str, list[dict]]) -> dict:
    root, row, arm, held = item
    panel = row["panel"]
    packet = sealed_read(root / "packets" / f"{panel}.json")
    if (
        row["packet"] != packet["artifact_sha256"]
        or [r["trajectory_id"] for r in held] != packet["held_event_ids"]
    ):
        raise ValueError("Panel or held events differ from the frozen design")
    menu = packet["menus"][arm]
    formulas = [*menu["candidates"], None, None][:2]
    selection = evaluation.fit_arm(
        root,
        panel_id=panel,
        arm=arm,
        graph=packet["graph"],
        train_rows=packet["fitting_rows"],
        formulas=formulas,
        nonnegative_targets=[],
    )
    scored = evaluation.score_arm(
        root, panel_id=panel, arm=arm, graph=packet["graph"], held_rows=held
    )
    return {
        "panel": panel,
        "graph": packet["structure_sha256"],
        "fold": row["fold"],
        "arm": arm,
        "selected_slot": selection["selected_slot"],
        "training_nmse": selection["training_nmse"],
        "fit_statuses": selection["fit_statuses"],
        "held_status": scored["status"],
        "held_nmse": scored["pooled_nmse"],
        "held_targets": scored.get("targets"),
        "selection_sha256": selection["artifact_sha256"],
        "score_sha256": scored["artifact_sha256"],
    }


def run(root: Path, workers: int) -> dict:
    """Re-use verified fit and scoring receipts on deterministic resume."""
    if workers < 1:
        raise ValueError("Workers must be positive")
    with _lock(root / "prepare-lock"):
        plan = prepare(root)
    rows = sealed_read(experiment.TRAIN_FILE)["train"][BASIN]
    items = [
        (root, row, arm, experiment.split_events(rows, row["fold"])[1])
        for row in plan["panels"]
        for arm in ("original", "shape")
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_arm, items))
    return sealed_write(
        root / "summary.json", {"plan": plan["artifact_sha256"], "results": results}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = run(args.root, args.workers)
    for item in result["results"]:
        print(
            item["panel"],
            item["arm"],
            item["fit_statuses"],
            item["held_status"],
            item["held_nmse"],
        )


if __name__ == "__main__":
    main()
