"""Check the saved T2 disposal vectors and their equation families at time zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.initial_algebraic_order import check_initial_order
from autoformalism.research.trajectory_function_fitting import initial_target_events
from autoformalism.staged_topology import content_hash
from scripts import run_t2_disposal_coupled as coupled
from scripts.t2_initial_disposal_family import assess_family

FREEZE = "bbbbd572f688479dd08b6f0daa3a8d477b101e0805a16632f35e7dd29e2796b6"
REPORT = "2aabfd39da68591b6a22aacd09fa5440047ccbfa18f8d183177faae452c2baf8"
OUT = Path("output/t2-initial-order-20261123")


def verify_initials(events: list[dict], saved: dict, model_id: str) -> None:
    """Cross-check every initial sample against the original training rollout."""
    if saved["model_sha256"] != model_id:
        raise ValueError("Saved rollout comes from a different parameter vector")
    for row, old in zip(events, saved["events"], strict=True):
        if (
            row["event"] != old["event"]
            or old["status"] != "scored"
            or old["time"][0] != 0
            or any(row[k]["U"][0] != old[k]["U"][0] for k in ("observed", "predicted"))
        ):
            raise ValueError(
                f"Initial sample differs from training replay: {row['event']}"
            )


def run(output: Path) -> dict:
    """Replay six vectors and bound the initial errors for the two refitted laws."""
    freeze = sealed_read(coupled.OUT / "training-freeze.json")
    report = sealed_read(coupled.OUT / "train-report.json")
    if (
        freeze["artifact_sha256"] != FREEZE
        or freeze["validation_opened_at_freeze"]
        or report["artifact_sha256"] != REPORT
        or report["freeze"] != FREEZE
        or report["report"]["split"] != "train"
    ):
        raise ValueError("Expected Orion's matched training freeze and report")
    rows = coupled.data.load_training(coupled.INPUTS / "train.json")
    checks, predictions, families = {}, {}, {}
    for name, model in sorted(freeze["models"].items()):
        model = coupled.insulin.with_parameters(model, model["formula"]["parameters"])
        events = initial_target_events(model["graph"], model["formula"], rows, ["U"])
        verify_initials(events, report["report"]["models"][name], content_hash(model))
        checks[name] = check_initial_order(events, "U", tolerance=0.0)
        predictions[name] = {e["event"]: e["predicted"]["U"][0] for e in events}
        if name in {"immediate_refit", "delayed_refit"}:
            families[name] = assess_family(model, rows)
    with _lock(output / "audit-lock"):
        return sealed_write(
            output / "audit.json",
            {
                "protocol": "t2-algebraic-initial-order-replay-1",
                "training_freeze": FREEZE,
                "training_report": REPORT,
                "vectors": checks,
                "observed": {r["trajectory_id"]: r["targets"]["U"][0] for r in rows},
                "initial_predictions": predictions,
                "families": families,
                "ordering_observation_allowance": 0.0,
                "absolute_fit_requirement": None,
                "optimizer_calls": 0,
                "llm_calls": 0,
                "validation_opened": False,
                "test_opened": False,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(args.root)
    print(
        json.dumps(
            {
                name: {
                    key: row[key]
                    for key in ("comparable_pairs", "reversed_pairs", "max_abs_error")
                }
                for name, row in result["vectors"].items()
            },
            indent=2,
        )
    )
