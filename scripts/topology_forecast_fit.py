"""Apply the same train-only function constructor and fitter to both graph origins."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.trajectory_function_families import build_shortlist
from autoformalism.research.trajectory_function_fitting import (
    fit_formula,
    score_formula,
)
from autoformalism.research.trajectory_function_revision import assess_fixed
from autoformalism.staged_topology import content_hash
from scripts import pointwise_repair_trial as pointwise
from scripts import trajectory_function_intake as intake
from scripts.preproposal_graph_portfolio import with_reviewed_initial_bindings
from scripts.trajectory_functions import BUDGET as FIT_BUDGET
from scripts.trajectory_functions import (
    FIXED,
    SEEDS,
    TRAIN,
    VALIDATION,
    pointwise_admission,
)

INPUTS = intake.REPO / "output/trajectory-functions-20261123/inputs"


@lru_cache(maxsize=3)
def _registered(path: Path, expected: str) -> dict:
    """Read a sealed artifact and verify its identity against the experiment."""
    value = sealed_read(path)
    if value["artifact_sha256"] != expected:
        raise ValueError(f"Unexpected registered artifact at {path}")
    return value


@lru_cache(maxsize=8)
def _training(case: str, dataset: str) -> dict:
    """Bind the inventory's frozen training evidence to the fitter's exact events."""
    topology = _registered(intake.INSPECTION, intake.ORIGINAL_INTAKE)
    frozen = with_reviewed_initial_bindings(topology["frozen_cases"][case])
    evidence = frozen["evidence"]
    rows = _registered(INPUTS / "train.json", TRAIN)["train"][dataset]
    if evidence["split"] != "train" or evidence["trajectories"] != rows:
        raise ValueError(f"Training evidence and release disagree for {case}/{dataset}")
    return frozen


def construct(row: dict) -> list[dict]:
    """Build the ranked fixed-graph formulas from public training events alone."""
    frozen = _training(row["case"], row["dataset"])
    return build_shortlist(row["graph"], frozen)["candidates"]


def prefit(row: dict, formula: dict) -> dict:
    """Apply registered fixed-graph and pointwise checks without invoking the solver."""
    case, dataset, graph = row["case"], row["dataset"], row["graph"]
    frozen = _training(case, dataset)
    checked = assess_fixed(graph, frozen, formula)
    digest = content_hash(graph)
    # The existing certificate resolves a slot's parent through its graph hash.
    payload = {
        "cases": {case: frozen},
        "train": {dataset: frozen["evidence"]["trajectories"]},
        "profile": FIXED,
        "pointwise_tolerances": {
            pointwise.DATASET: {pointwise.TARGET: pointwise.TOLERANCE}
        },
    }
    result = pointwise_admission(
        payload, {**row, "structure_sha256": digest}, {**checked, "graph": digest}
    )
    return {k: v for k, v in result.items() if k not in {"graph", "formula"}}


def fit(row: dict, formula: dict, rank: int) -> dict:
    """Run the registered fitter with identical starts and budget for each origin."""
    if not 0 <= rank < len(SEEDS):
        raise ValueError("Invalid formula rank")
    frozen = _training(row["case"], row["dataset"])
    return fit_formula(
        row["graph"],
        formula,
        frozen["evidence"]["trajectories"],
        seed=SEEDS[rank],
        **FIT_BUDGET,
    )


def validate(row: dict, formula: dict, fit_result: dict) -> dict:
    """Replay a frozen training winner on the separate public validation split."""
    saved = _registered(INPUTS / "validation.json", VALIDATION)
    if saved["training_intake_sha256"] != TRAIN or saved["split"] != "val":
        raise ValueError("Validation release is not paired with the training intake")
    score = score_formula(
        row["graph"], formula, fit_result, saved["rows"][row["dataset"]]
    )
    if score["status"] == "scored":
        per_event = {}
        for curve in score["curves"]:
            per_target = {}
            for name, scale in fit_result["scales"].items():
                residual = np.subtract(
                    curve["predicted"][name], curve["observed"][name]
                )
                per_target[name] = float(np.mean(np.square(residual / scale)))
            per_event[curve["trajectory_id"]] = {
                "per_target_nmse": per_target,
                "pooled_nmse": float(np.mean(list(per_target.values()))),
            }
        score["per_event"] = per_event
    return score
