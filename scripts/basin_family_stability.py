"""Leave out each training event to test selection among the frozen basin powers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import _check_grid, _rows
from autoformalism.staged_topology import content_hash
from scripts import basin_function_family as family
from scripts.construction_functions import file_hash

STARTS = {"original": [None], "geometric": [None, 1.0, 10.0, 100.0]}


def _at_start(formula: dict, start: float | None) -> dict:
    """Set both rate initial guesses in the formula's public units."""
    if start is None:
        return formula
    return formula | {"parameters": dict.fromkeys(formula["parameters"], start)}


def _freeze_fold(
    folder: Path, study: dict, model_plan: dict, rows: list[dict], held: str
) -> tuple[dict, list[dict]]:
    """Fit each form without the held event and freeze its five-event rank."""
    training = [row for row in rows if row["trajectory_id"] != held]
    models = model_plan["models"]
    starts = study["starts"]
    attempts = [
        model | {"formula": _at_start(model["formula"], start)}
        for start in starts
        for model in models
    ]
    plan = sealed_write(
        folder / "plan.json",
        {
            "study": study["artifact_sha256"],
            "training": [r["trajectory_id"] for r in training],
            "held": held,
            "graph": model_plan["graph"],
            "models": attempts,
            "budget": model_plan["budget"],
        },
    )
    fits = [family._fit(folder, plan, i, training) for i in range(len(attempts))]
    selected = []
    for slot in range(len(models)):
        eligible = [
            i
            for i in range(slot, len(attempts), len(models))
            if family._admissible(fits[i]["result"])
        ]
        selected.append(
            min(eligible, key=lambda i: (fits[i]["result"]["training_nmse"], i))
            if eligible
            else None
        )
    order = sorted(
        (slot for slot, i in enumerate(selected) if i is not None),
        key=lambda slot: (fits[selected[slot]]["result"]["training_nmse"], slot),
    )
    return sealed_write(
        folder / "ranking.json",
        {
            "plan": plan["artifact_sha256"],
            "held": held,
            "fit_receipts": [fit["artifact_sha256"] for fit in fits],
            "training_order": order,
            "selected_fit_indices": selected,
            "evaluations": sum(fit["result"].get("evaluations", 0) for fit in fits),
        },
    ), fits


def _held_score(graph: dict, formula: dict, fit: dict, event: dict) -> dict:
    """Score the original train_* event at fixed parameters and five-event scales."""
    if (fit["graph_sha256"], fit["formula_sha256"]) != (
        content_hash(graph),
        content_hash(formula),
    ):
        raise ValueError("Fold fit differs from the graph or formula")
    rows = _rows([event], "train")
    names = sorted(rows[0].targets)
    scales = np.array([fit["scales"][n] for n in names])
    return _check_grid(graph, formula, rows, fit["parameters"], names, scales)


def run(campaign: Path, root: Path, start_policy: str = "original") -> dict:
    """Freeze every fold selection before scoring any of the six held-out events."""
    if start_policy not in STARTS:
        raise ValueError("Unrecognized coefficient starting policy")
    starts = STARTS[start_policy]
    if root.resolve().is_relative_to(campaign.resolve()):
        raise ValueError("Use a separate study directory outside the returned campaign")
    with _lock(root):
        model_plan, rows = family.prepare(campaign, root / "family")
        ids = [row["trajectory_id"] for row in rows]
        if (
            len(ids) != 6
            or len(set(ids)) != 6
            or not all(name.startswith("train_") for name in ids)
        ):
            raise ValueError("Expected six unique training events")
        study = sealed_write(
            root / "plan.json",
            {
                "protocol": "basin-leave-one-training-event-out-2",
                "start_policy": start_policy,
                "starts": starts,
                "code": file_hash(Path(__file__)),
                "model_plan": model_plan["artifact_sha256"],
                "events": ids,
                "maximum_fits": len(rows) * len(model_plan["models"]) * len(starts),
                "selection": model_plan["selection"],
                "benchmark_validation_opened": False,
                "test_data_opened": False,
                "llm_calls": 0,
            },
        )
        folds = [
            _freeze_fold(root / f"fold-{i}", study, model_plan, rows, held)
            for i, held in enumerate(ids)
        ]
        freeze = sealed_write(
            root / "rankings.json",
            {
                "study": study["artifact_sha256"],
                "rankings": [ranking["artifact_sha256"] for ranking, _ in folds],
            },
        )
        summaries = []
        for i, (ranking, fits) in enumerate(folds):
            order = ranking["training_order"]
            winner = order[0] if order else None
            scores = []
            for slot, model in enumerate(model_plan["models"]):
                selected = ranking["selected_fit_indices"][slot]
                index = slot if selected is None else selected
                fit = fits[index]
                start = starts[index // len(model_plan["models"])]
                formula = _at_start(model["formula"], start)
                result = fit["result"]
                identity = {
                    "rankings": freeze["artifact_sha256"],
                    "fit": fit["artifact_sha256"],
                    "event": ids[i],
                }
                path = root / f"fold-{i}" / "scores" / f"{slot:02}.json"
                if path.exists():
                    receipt = sealed_read(path)
                    if any(receipt[k] != v for k, v in identity.items()):
                        raise ValueError("Held-event score differs from the frozen fit")
                else:
                    score = (
                        _held_score(model_plan["graph"], formula, result, rows[i])
                        if slot in order
                        else {"status": "no_admissible_training_fit"}
                    )
                    receipt = sealed_write(path, {**identity, "score": score})
                scores.append(
                    {
                        "slot": slot,
                        "up": model["up"],
                        "down": model["down"],
                        "selected_fit_index": selected,
                        "start": start,
                        "training_status": result["status"],
                        "training_nmse": result.get("training_nmse"),
                        "held_status": receipt["score"]["status"],
                        "held_nmse": receipt["score"].get("pooled_nmse"),
                        "receipt": receipt["artifact_sha256"],
                    }
                )
            summaries.append({"held": ids[i], "winner": winner, "scores": scores})
        return sealed_write(
            root / "summary.json",
            {
                "rankings": freeze["artifact_sha256"],
                "folds": summaries,
                "start_policy": start_policy,
                "fit_receipts": sum(len(fits) for _, fits in folds),
                "evaluations": sum(ranking["evaluations"] for ranking, _ in folds),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-policy", choices=STARTS, default="original")
    args = parser.parse_args()
    result = run(args.campaign, args.root, args.start_policy)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
