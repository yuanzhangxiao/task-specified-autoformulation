"""Fit the three frozen 20B basin proposals and score the train-selected model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import (
    fit_formula,
    score_formula,
)
from scripts import basin_process_proposals as proposals
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash


def fitted_checks(candidate: dict, result: dict) -> list[str]:
    """Check the fitted law at the crest and retain the required dependencies."""
    if result["status"] != "fitted" or not math.isfinite(result["training_nmse"]):
        return [result["status"]]
    failures = []
    for basin in ("up", "down"):
        label = "upstream_factor" if basin == "up" else "downstream_factor"
        try:
            proposals._factor(
                candidate["authored_factors"][label], basin, result["parameters"]
            )
        except (ValueError, ArithmeticError, ModelValidationError) as exc:
            failures.append(f"{label} at the fitted parameters: {exc}")
    for name, sources in candidate["revised_graph"]["sources"].items():
        missing = set(sources) - set(result["effective_sources"][name])
        if missing:
            failures.append(f"{name} lost fitted dependencies {sorted(missing)}")
    if result["negative_samples"]["h_down"]:
        failures.append("The training rollout predicted negative downstream depth")
    return failures


def train(campaign: Path, root: Path) -> dict:
    """Fit at most once per frozen candidate, then seal the training order."""
    plan, data = proposals.load(campaign)
    if proposals.status(campaign)["status"] != "complete":
        raise ValueError("Generate all three proposals before fitting")
    frozen = trial._frozen_candidates(campaign, plan)
    with _lock(root):
        receipts, order = [], []
        fit_plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "basin-process-proposal-fitting-1",
                "code": file_hash(Path(__file__)),
                "generation": frozen["artifact_sha256"],
                "proposal_plan": plan["artifact_sha256"],
                "budget": {**trial.BUDGET, "seed": 100},
                "validation": trial.VALIDATION,
                "selection": "Rank eligible verified fits by training NMSE, then slot.",
                "test_data_opened": False,
            },
        )
        for slot, digest in enumerate(frozen["candidates"]):
            candidate = trial._read_candidate(campaign, plan, slot)
            if candidate["artifact_sha256"] != digest:
                raise ValueError("Candidate differs from the generation freeze")
            identity = {"fit_plan": fit_plan["artifact_sha256"], "candidate": digest}
            folder = root / "fits" / f"{slot:03d}"
            path = folder / "result.json"
            if path.exists():
                receipt = sealed_read(path)
                if any(receipt[k] != v for k, v in identity.items()):
                    raise ValueError("Candidate fit checkpoint differs")
            else:
                if candidate["status"] != "eligible":
                    result = {"status": candidate["status"]}
                elif (folder / "reservation.json").exists():
                    sealed_write(folder / "reservation.json", identity)
                    result = {"status": "interrupted"}
                else:
                    sealed_write(folder / "reservation.json", identity)
                    result = fit_formula(
                        candidate["revised_graph"],
                        candidate["formula"],
                        data["train"][proposals.BASIN],
                        **fit_plan["budget"],
                    )
                failures = fitted_checks(candidate, result)
                receipt = sealed_write(
                    path, {**identity, "result": result, "failures": failures}
                )
            receipts.append(receipt["artifact_sha256"])
            if not receipt["failures"]:
                order.append((receipt["result"]["training_nmse"], slot))
        return sealed_write(
            root / "ranking.json",
            {
                "plan": fit_plan["artifact_sha256"],
                "fits": receipts,
                "training_order": [slot for _, slot in sorted(order)],
            },
        )


def validate(campaign: Path, root: Path, validation: Path) -> dict:
    """Score the three prespecified slots after the independent training freeze."""
    plan, data = proposals.load(campaign)
    with _lock(root):
        frozen = trial._frozen_candidates(campaign, plan)
        ranking = sealed_read(root / "ranking.json")
        fit_plan = sealed_read(root / "plan.json")
        if (
            ranking["plan"] != fit_plan["artifact_sha256"]
            or fit_plan["code"] != file_hash(Path(__file__))
            or fit_plan["generation"] != frozen["artifact_sha256"]
            or fit_plan["proposal_plan"] != plan["artifact_sha256"]
            or len(ranking["fits"]) != len(frozen["candidates"])
        ):
            raise ValueError("Training ranking or generation has changed")
        holdout = sealed_read(validation)
        if (
            holdout["artifact_sha256"] != trial.VALIDATION
            or holdout["training_intake_sha256"] != data["source"]
        ):
            raise ValueError("Unexpected public validation release")
        rows = []
        for slot, digest in enumerate(ranking["fits"]):
            fit = sealed_read(root / "fits" / f"{slot:03d}" / "result.json")
            cand = trial._read_candidate(campaign, plan, slot)
            if (
                fit["artifact_sha256"] != digest
                or fit["candidate"] != cand["artifact_sha256"]
                or fit["candidate"] != frozen["candidates"][slot]
            ):
                raise ValueError("Fit differs from the frozen candidate")
            path = root / "validation" / f"{slot:03d}.json"
            identity = {
                "ranking": ranking["artifact_sha256"],
                "fit": digest,
                "validation": trial.VALIDATION,
            }
            if path.exists():
                receipt = sealed_read(path)
                if any(receipt[k] != v for k, v in identity.items()):
                    raise ValueError("Validation receipt differs from the frozen fit")
            else:
                score = {"status": "no_qualified_fit"}
                if slot in ranking["training_order"]:
                    score = score_formula(
                        cand["revised_graph"],
                        cand["formula"],
                        fit["result"],
                        holdout["rows"][proposals.BASIN],
                    )
                receipt = sealed_write(path, {**identity, "score": score})
            score = receipt["score"]
            rows.append(
                {
                    "slot": slot,
                    "seed": cand["seed"],
                    "candidate_status": cand["status"],
                    "fit_status": fit["result"]["status"],
                    "failures": fit["failures"],
                    "training_nmse": fit["result"].get("training_nmse"),
                    "validation_status": score["status"],
                    "validation_nmse": score.get("pooled_nmse"),
                    "minimum_prediction": score.get("minimum_prediction"),
                    "receipt": receipt["artifact_sha256"],
                }
            )
        return sealed_write(
            root / "summary.json",
            {
                "rows": rows,
                "training_order": ranking["training_order"],
                "ranking": ranking["artifact_sha256"],
                "test_data_opened": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validate"))
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    if (args.validation is not None) != (args.stage == "validate"):
        parser.error("Pass --validation only for the validation stage")
    if args.root.resolve().is_relative_to(args.campaign.resolve()):
        parser.error("Choose a new result directory outside the source campaign")
    report = (
        train(args.campaign, args.root)
        if args.stage == "train"
        else validate(args.campaign, args.root, args.validation)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
