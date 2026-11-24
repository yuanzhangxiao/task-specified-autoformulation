"""One-round power-law revisions of the coupled basin's linear discharge parent."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import isfinite
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.trajectory_basin_balance import check_transfer_balance
from autoformalism.research.trajectory_function_fitting import (
    _rows,
    fit_formula,
    score_formula,
    validate_formula,
)
from autoformalism.staged_topology import content_hash
from scripts import basin_balance_inference as balance
from scripts import basin_family_stability as folds
from scripts import basin_function_family as family
from scripts import trajectory_functions as trial

ROOT = Path(__file__).resolve().parents[1] / "output"
INPUTS = ROOT / "trajectory-functions-20261123/inputs"
PLAN = ROOT / "basin-family-autopilot-20261123/verified/plan.json"
POWERS = (1.0, 1.25, 1.5, 1.75, 2.0)
PARENT = {"up": 1.0, "down": 1.0}


@lru_cache(maxsize=1)
def _training() -> tuple[dict, list[dict]]:
    """Read the registered training events and the saved linear parent graph."""
    intake = sealed_read(INPUTS / "train.json")
    plan = sealed_read(PLAN)
    graph = plan["graph"]
    if (
        intake["artifact_sha256"] != trial.TRAIN
        or content_hash(graph) != family.REVISED
    ):
        raise ValueError("Unexpected basin training data or parent graph")
    rows = intake["train"][family.BASIN]
    if [r["trajectory_id"] for r in rows] != [f"train_{i:03}" for i in range(6)]:
        raise ValueError("Expected the six registered training events in order")
    return graph, rows


def normalize(edit: dict) -> dict:
    """Enforce the numeric edit schema and canonicalize equal powers."""
    if not isinstance(edit, dict) or set(edit) != {"up", "down"}:
        raise ValueError('Return exactly {"up": number, "down": number}')
    if any(type(p) not in (int, float) or p not in POWERS for p in edit.values()):
        raise ValueError(f"Choose each power from {list(POWERS)}")
    return {side: float(edit[side]) for side in ("up", "down")}


def _formula(edit: dict) -> dict:
    """Assemble the declared powers and apply matched two-gain starts."""
    edit = normalize(edit)
    if all(p in family.POWERS for p in edit.values()):
        return folds._at_start(family.formula(**edit), 10.0)
    form = family.formula(1.0, 1.0)
    for side, p in edit.items():
        head = f"max(h_{side}-crest_{side},0)"
        sqrt = f"sqrt({head})"
        quarter = f"sqrt({sqrt})"
        factors = {
            1: head,
            1.25: f"{head}*{quarter}",
            1.5: f"{head}*{sqrt}",
            1.75: f"{head}*{sqrt}*{quarter}",
            2: f"{head}**2",
        }
        for name, expression in form["equations"].items():
            form["equations"][name] = expression.replace(head, f"({factors[p]})")
    return folds._at_start(form, 10.0)


def packet() -> dict:
    """Rank the power grid using normalized errors on measured training intervals."""
    _, rows = _training()
    profiles = balance.estimate(_rows(rows, "train"), stride=1)
    errors = {}
    for side in ("upstream", "downstream"):
        fit = profiles[side]
        if fit["status"] != "estimated" or fit["zero_law_rmse_m3_per_min"] <= 0:
            raise ValueError(f"Training interval data do not identify the {side} law")
        errors[side] = {
            item["power"]: item["flow_rmse_m3_per_min"]
            / fit["zero_law_rmse_m3_per_min"]
            for item in fit["profile"]
            if item["power"] in POWERS
        }
    pairs = [p for p in product(POWERS, repeat=2) if p != (1.0, 1.0)]
    pairs.sort(
        key=lambda p: errors["upstream"][p[0]] ** 2 + errors["downstream"][p[1]] ** 2
    )
    return {
        "parent": dict(PARENT),
        "algorithm_edits": [{"up": u, "down": d} for u, d in pairs[:3]],
        "edit_schema": {
            "type": "object",
            "properties": {
                side: {"type": "number", "enum": list(POWERS)} for side in PARENT
            },
            "required": ["up", "down"],
            "additionalProperties": False,
        },
        "edit_contract": (
            "Choose numeric powers up and down from 1, 1.25, 1.5, 1.75, 2. "
            "Q=k_up*positive_upstream_head^up, D=k_down*positive_downstream_head^down. "
            "Fit both gains. Area_up*dh_up/dt=inflow_up-Q and "
            "area_down*dh_down/dt=inflow_down+Q-D. Each head is depth above its "
            "surveyed crest, clipped at zero. Initial upstream depth is measured. "
            "Only downstream depth is observed later."
        ),
        "evidence": {
            "training_intake": trial.TRAIN,
            "training_events": [r["trajectory_id"] for r in rows],
            "basis": (
                "Estimate outlet discharge before upstream water reaches its crest. "
                "Infer transfer and upstream depth from both balances conditional "
                "on this outlet law. For each power, fit a gain and measure flow "
                "RMSE relative to a zero law. No hidden flow readings are available."
            ),
            "interval_relative_rmse": {
                side: [{"power": p, "rmse": value} for p, value in profile.items()]
                for side, profile in errors.items()
            },
        },
    }


def score(edit: dict) -> dict:
    """Fit both gains on identical budgets and rank by held-training-event NMSE."""
    try:
        form = _formula(edit)
    except ValueError as exc:
        return {"eligible": False, "reason": str(exc)}
    graph, rows = _training()
    form = validate_formula(graph, form)
    balance_check = check_transfer_balance(
        {"provenance": {"case": family.BASIN}}, graph, form
    )
    if balance_check["status"] != "pass":
        return {
            "eligible": False,
            "reason": "Internal transfer cancellation unverified",
        }
    fit = fit_formula(graph, form, rows, **family.BUDGET)
    evaluations = fit["evaluations"]
    if not family._admissible(fit):
        return {
            "eligible": False,
            "reason": "Training fit failed admissibility",
            "evaluations": evaluations,
        }
    held = {}
    for i, event in enumerate(rows):
        fold_fit = fit_formula(graph, form, rows[:i] + rows[i + 1 :], **family.BUDGET)
        evaluations += fold_fit["evaluations"]
        if not family._admissible(fold_fit):
            return {
                "eligible": False,
                "reason": "Five-event training fit failed admissibility",
                "evaluations": evaluations,
            }
        replay = folds._held_score(graph, form, fold_fit, event)
        if replay["status"] != "scored" or replay["negative_samples"]["h_down"]:
            return {
                "eligible": False,
                "reason": "Held training event failed rollout verification",
                "evaluations": evaluations,
            }
        held[event["trajectory_id"]] = replay["pooled_nmse"]
    return {
        "eligible": True,
        "train_nmse": fit["training_nmse"],
        "selection_score": sum(held.values()) / len(held),
        "selection_metric": "Mean omitted-event depth NMSE, using fitting-fold scales",
        "held_event_nmse": held,
        "evaluations": evaluations,
        "fit": {
            k: fit[k]
            for k in (
                "parameters",
                "graph_sha256",
                "formula_sha256",
                "scales",
                "status",
            )
        },
        "training_physical_checks": {
            "transfer_balance": balance_check["status"],
            "minimum_downstream_m": fit["minimum_prediction"]["h_down"],
            "max_storage_gain_above_inflow_m3": fit[
                "maximum_storage_gain_above_inflow_m3"
            ],
        },
    }


def validate(edit: dict, score_result: dict) -> dict:
    """Evaluate the frozen edit and fitted gains on the registered development split."""
    graph, _ = _training()
    form = validate_formula(graph, _formula(edit))
    fit = score_result.get("fit", {})
    if not score_result.get("eligible") or fit.get("formula_sha256") != content_hash(
        form
    ):
        raise ValueError("Frozen fit and revision differ")
    data = sealed_read(INPUTS / "validation.json")
    if (
        data["artifact_sha256"] != trial.VALIDATION
        or data["training_intake_sha256"] != trial.TRAIN
    ):
        raise ValueError("Unexpected validation data")
    result = score_formula(graph, form, fit, data["rows"][family.BASIN])
    if result["status"] != "scored" or not isfinite(result["pooled_nmse"]):
        return {"status": result["status"], "nmse": None}
    return {
        "status": "scored",
        "nmse": result["pooled_nmse"],
        "per_target_nmse": result["per_target_nmse"],
        "negative_samples": result["negative_samples"],
        "minimum_prediction": result["minimum_prediction"],
        "max_storage_gain_above_inflow_m3": result[
            "maximum_storage_gain_above_inflow_m3"
        ],
    }
