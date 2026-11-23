"""Compare nine conserved basin models on a frozen public development cohort."""

from __future__ import annotations

import argparse
import json
from itertools import product
from math import isfinite
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research.preproposal_cases import BASIN
from autoformalism.research.trajectory_function_fitting import (
    fit_formula,
    score_formula,
)
from autoformalism.research.trajectory_function_revision import assess_revision
from autoformalism.staged_topology import content_hash
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash

INTAKE = "4c28072b2deb510a7712e62d08a7afdc4b959f0d1464d32105006f313849411d"
SOURCE_PLAN = "f3d2e4e0a4585e7014d92495a51e21374132bf8628d13e49c3e5a98c76dac3fa"
PARENT = "a6b51e36260aa5714827d66a0055143a8746a7888a2ba90eb3b3aefc6d0a391c"
REVISED = "4d7aeece8426da1e43840b8d9a3ff11314be68bbeabcf3758e470741bbe340d9"
POWERS = (1.0, 1.5, 2.0)
REASON = "Use surveyed depths and areas in the public two-basin volume balance."
BUDGET = {"max_evaluations": 1000, "wall_seconds": 180.0, "starts": 2, "seed": 100}


def formula(up: float, down: float) -> dict:
    """Return the fixed balance with independently chosen threshold responses."""

    def head(basin: str, power: float) -> str:
        term = f"max(h_{basin}-crest_{basin},0)"
        if power == 1.0:
            return term
        if power == 1.5:
            return f"({term}*sqrt({term}))"
        if power == 2.0:
            return f"({term}**2)"
        raise ValueError("Power outside frozen menu")

    transfer = f"k_overflow*{head('up', up)}"
    discharge = f"k_outflow*{head('down', down)}"
    return {
        "equations": {
            "h_down": f"(inflow_down + {transfer} - {discharge})/area_down",
            "h_up": f"(inflow_up - {transfer})/area_up",
        },
        "initializers": {},
        "parameters": {"k_outflow": 0.05, "k_overflow": 0.1},
    }


def _code() -> dict:
    return {
        "script": file_hash(Path(__file__)),
        "runtime": runtime_source_hash(),
        "campaign_entrypoint": file_hash(Path(trial.__file__)),
    }


def prepare(campaign: Path, root: Path) -> tuple[dict, list]:
    """Seal the nine training designs and their common graph before fitting."""
    if root.resolve().is_relative_to(campaign.resolve()):
        raise ValueError("Use a separate experiment root outside the source campaign")
    intake = sealed_read(campaign / "intake.json")
    source_plan = sealed_read(campaign / "plan.json")
    if (
        source_plan["artifact_sha256"] != SOURCE_PLAN
        or source_plan["intake_sha256"] != INTAKE
        or intake["artifact_sha256"] != INTAKE
        or intake["source"] != trial.TRAIN
    ):
        raise ValueError("Unexpected public training cohort")
    index = next(
        i for i, r in enumerate(intake["graphs"]) if r["structure_sha256"] == PARENT
    )
    row = intake["graphs"][index]
    frozen = trial.training_case(intake, row)
    isolated = intake["menus"][index]["evidence"]["isolated_outlet"]
    if row["dataset"] != BASIN or {s["power"] for s in isolated["scores"]} != set(
        POWERS
    ):
        raise ValueError(
            "Threshold response menu differs from public training evidence"
        )
    models = []
    for up, down in product(POWERS, repeat=2):
        model = formula(up, down)
        result = assess_revision(row["graph"], frozen, model, REASON)
        if (
            result["status"] != "eligible"
            or result["child_graph_sha256"] != REVISED
            or result["balance_check"]["status"] != "pass"
        ):
            raise ValueError(f"Basin balance or graph admission failed: {result}")
        models.append({"up": up, "down": down, "formula": result["formula"]})
    return sealed_write(
        root / "plan.json",
        {
            "protocol": "basin-function-family-1",
            "training_cohort": INTAKE,
            "source_plan": SOURCE_PLAN,
            "validation_cohort": trial.VALIDATION,
            "code": _code(),
            "parent": PARENT,
            "graph": result["graph"],
            "models": models,
            "budget": BUDGET,
            "isolated_outlet": isolated,
            "selection": (
                "Require a verified training score, positive fitted rates and "
                "nonnegative downstream predictions. Rank by training NMSE, "
                "breaking ties by slot."
            ),
            "validation": "Report all nine fixed designs, including the linear control",
            "test_data_opened": False,
        },
    ), intake["train"][BASIN]


def _fit(root: Path, plan: dict, i: int, rows: list) -> dict:
    formula_ = plan["models"][i]["formula"]
    identity = {
        "plan": plan["artifact_sha256"],
        "slot": i,
        "formula": content_hash(formula_),
    }
    folder = root / "fits" / f"{i:02}"
    result_file = folder / "result.json"
    reservation_file = folder / "reservation.json"
    if result_file.exists():
        saved = sealed_read(result_file)
        if any(saved[k] != value for k, value in identity.items()):
            raise ValueError("Fit receipt belongs to another model")
        return saved
    if reservation_file.exists():
        sealed_write(reservation_file, identity)
        result = {"status": "interrupted", "training_nmse": None}
    else:
        sealed_write(reservation_file, identity)
        result = fit_formula(plan["graph"], formula_, rows, **plan["budget"])
    return sealed_write(result_file, {**identity, "result": result})


def _admissible(result: dict) -> bool:
    return (
        result["status"] == "fitted"
        and isfinite(result["training_nmse"])
        and all(isfinite(v) and v > 0 for v in result["parameters"].values())
        and result["negative_samples"]["h_down"] == 0
    )


def train(campaign: Path, root: Path) -> dict:
    """Complete bounded training fits, then freeze the ranking and every fit receipt."""
    with _lock(root):
        plan, rows = prepare(campaign, root)
        fits = [_fit(root, plan, i, rows) for i in range(len(plan["models"]))]
        eligible = [r for r in fits if _admissible(r["result"])]
        eligible.sort(key=lambda r: (r["result"]["training_nmse"], r["slot"]))
        return sealed_write(
            root / "ranking.json",
            {
                "plan": plan["artifact_sha256"],
                "training_order": [r["slot"] for r in eligible],
                "fit_receipts": [r["artifact_sha256"] for r in fits],
                "linear_control_slot": 0,
                "validation_opened": False,
            },
        )


def validate(root: Path, validation: Path) -> dict:
    """Score the prespecified cohort with frozen parameters after training selection."""
    with _lock(root):
        plan = sealed_read(root / "plan.json")
        ranking = sealed_read(root / "ranking.json")
        if (
            ranking["plan"] != plan["artifact_sha256"]
            or plan["code"] != _code()
            or len(ranking["fit_receipts"]) != len(plan["models"])
        ):
            raise ValueError("Training freeze differs from the current plan")
        fits = []
        for i, digest in enumerate(ranking["fit_receipts"]):
            fit = sealed_read(root / "fits" / f"{i:02}" / "result.json")
            if (
                fit["artifact_sha256"] != digest
                or fit["plan"] != plan["artifact_sha256"]
                or fit["slot"] != i
                or fit["formula"] != content_hash(plan["models"][i]["formula"])
            ):
                raise ValueError("Fit differs from the training selection")
            fits.append(fit["result"])
        if validation.name != "validation.json":
            raise ValueError("Expected registered validation.json")
        saved = sealed_read(validation)
        if (
            saved["artifact_sha256"] != trial.VALIDATION
            or saved["training_intake_sha256"] != trial.TRAIN
        ):
            raise ValueError("Unexpected validation data")
        summaries = []
        for i, (model, result) in enumerate(zip(plan["models"], fits, strict=True)):
            identity = {"ranking": ranking["artifact_sha256"], "slot": i}
            path = root / "validation" / f"{i:02}.json"
            if path.exists():
                receipt = sealed_read(path)
                if any(receipt[k] != v for k, v in identity.items()):
                    raise ValueError(
                        "Validation receipt differs from training selection"
                    )
            else:
                score = (
                    score_formula(
                        plan["graph"], model["formula"], result, saved["rows"][BASIN]
                    )
                    if i in ranking["training_order"]
                    else {"status": "no_admissible_training_fit"}
                )
                receipt = sealed_write(path, {**identity, "score": score})
            score = receipt["score"]
            summaries.append(
                {
                    "slot": i,
                    "up": model["up"],
                    "down": model["down"],
                    "training_status": result["status"],
                    "training_nmse": result["training_nmse"],
                    "evaluations": result.get("evaluations"),
                    "parameters": result.get("parameters"),
                    "validation_status": score["status"],
                    "validation_nmse": score.get("pooled_nmse"),
                    "per_target_nmse": score.get("per_target_nmse"),
                    "minimum_prediction": score.get("minimum_prediction"),
                    "negative_samples": score.get("negative_samples"),
                    "validation_receipt": receipt["artifact_sha256"],
                }
            )
        return sealed_write(
            root / "summary.json",
            {
                "ranking": ranking["artifact_sha256"],
                "training_order": ranking["training_order"],
                "linear_control_slot": ranking["linear_control_slot"],
                "rows": summaries,
                "test_data_opened": False,
                "llm_calls": 0,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage, name in (("train", "campaign"), ("validate", "validation")):
        command = sub.add_parser(stage)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = (
        train(args.campaign, args.root)
        if args.stage == "train"
        else validate(args.root, args.validation)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
