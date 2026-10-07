"""Training-only prediction then polishing, sharing one time and call allowance."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from time import monotonic

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import profiled_campaign as profiled
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.bounded_screening import TrainingOnlySplit


def certificate(backend: dict) -> dict | None:
    """Return a complete independent check of exactly the selected vector."""
    selected = backend.get("selected")
    if not selected:
        return None
    for record in backend.get("certificates", []):
        value, process = record.get("value") or {}, record.get("process") or {}
        if (
            record.get("parameters_sha256")
            == public.content_sha256(selected["parameters"])
            and value.get("parameters") == selected["parameters"]
            and process.get("status") == "complete"
            and process.get("termination_confirmed")
            and value.get("complete")
        ):
            return value
    return None


def passes(value: dict | None, policy: dict) -> bool:
    """Recheck recorded training evidence against a prospective tolerance."""
    if not value or not value.get("complete"):
        return False
    for metric, threshold in (
        ("training_nmse", "training_nmse"),
        ("maximum_trajectory_nmse", "trajectory_nmse"),
        ("maximum_solver_difference", "solver_agreement"),
    ):
        number = value.get(metric)
        if (
            number is None
            or not math.isfinite(number)
            or not 0 <= number <= policy[threshold]
        ):
            return False
    return True


def call_count(backend: dict) -> int | None:
    """Unknown worker usage blocks continuation; it never becomes zero cost."""
    rollout = backend.get("rollout") or {}
    value = rollout.get("value") or {}
    count = value.get("actual_residual_calls")
    return count if type(count) is int and count >= 0 else None


def fit(base: dict, arm: str, policy: dict, folder: Path) -> dict:
    """Preserve a certified prediction before one parameter warm-start polish.

    Optimizer state is restarted once, but time and evaluation budgets are not.
    Interrupted coordinators are terminal; durable first checkpoints survive.
    """
    if arm not in profiled.ARMS or set(base) != {
        "request",
        "training",
        "coordinates",
        "nodes",
        "start",
    }:
        raise ValueError("polishing requires an eligible arm and training-only base")
    TrainingOnlySplit.model_validate(base["training"])
    identity = public.content_sha256({"base": base, "arm": arm, "policy": policy})
    started, progress, finished = (
        folder / name
        for name in ("polishing-started.json", "progress.json", "finished.json")
    )
    first_path = folder / "first-prediction.json"
    if started.exists():
        if read_seal(started) != {"identity": identity}:
            raise ValueError("polishing resume identity differs")
        if finished.exists():
            saved = read_seal(finished)
            if saved["identity"] != identity:
                raise ValueError("polishing terminal identity differs")
            return saved["result"]
        saved = public._read(progress)
        if first_path.exists():
            first = read_seal(first_path)
            if first["identity"] != identity:
                raise ValueError("first prediction identity differs")
            saved["first_prediction"] = first["value"]
            if saved["selected"] is None:
                saved["selected"] = first["value"]["selected"]
        return saved | {
            "stop_reason": "interrupted_no_fit_restart",
            "fit_seconds": None,
            "budget_restarted": False,
            "accounting_complete": False,
        }

    begun = monotonic()
    deadline = begun + policy["seconds"]
    initial = {k: policy[k] for k in common.RecoveryPolicy.model_fields}
    strict = initial | {
        "training_nmse": policy["polish_training_nmse"],
        "trajectory_nmse": policy["polish_trajectory_nmse"],
    }
    result = {
        "selected": None,
        "levels": [],
        "certificates": [],
        "stages": [],
        "first_prediction": None,
        "polishing_attempted": False,
        "strict_prediction_certified": False,
        "retained_stage": None,
        "budget_restarted": False,
        "reference_values_used": False,
        "validation_used_for_fitting": False,
        "accounting_complete": True,
        "fitting_allowance_seconds": policy["seconds"],
    }
    # A crash after the start marker must still have a readable empty checkpoint.
    folder.mkdir(parents=True, exist_ok=True)
    public._write(progress, result)
    seal(started, {"identity": identity})

    def save():
        public._write(progress, result)

    def phase(name, point, settings):
        stage_base = deepcopy(base)
        stage_base["start"] = point
        settings = settings | {"seconds": max(0, deadline - monotonic())}
        value = profiled.fit(stage_base, arm, settings, folder / name)
        stage = {"name": name, "policy": settings, "backend": value}
        seal(folder / f"{name}.json", {"identity": identity, "stage": stage})
        result["stages"].append(stage)
        result["certificates"].extend(value.get("certificates", []))
        save()
        return value

    first = phase("prediction", base["start"], initial)
    result["selected"] = first.get("selected")
    result["retained_stage"] = "prediction" if result["selected"] else None
    checked = certificate(first)
    used = call_count(first)
    reason = "prediction_not_certified"
    if first["stop_reason"] == "cleanup_unconfirmed":
        reason = "cleanup_unconfirmed"
    elif passes(checked, initial):
        result["first_prediction"] = {
            "selected": deepcopy(result["selected"]),
            "certificate": checked,
            "fit_seconds": monotonic() - begun,
            "residual_calls": used,
        }
        seal(first_path, {"identity": identity, "value": result["first_prediction"]})
        save()
        if passes(checked, strict):
            reason = "strict_target_already_reached"
        elif used is None:
            reason = "unknown_call_usage_no_polish"
        elif policy["maximum_rollout_calls"] - used < 5:
            reason = "call_budget_no_polish"
        elif deadline - monotonic() < policy["certificate_seconds"] + 5:
            reason = "time_budget_no_polish"
        else:
            result["polishing_attempted"] = True
            save()
            polished = phase(
                "polish",
                result["selected"]["parameters"],
                strict
                | {"maximum_rollout_calls": policy["maximum_rollout_calls"] - used},
            )
            candidate_check = certificate(polished)
            reason = "polish_stopped_prediction_preserved"
            if (
                passes(candidate_check, initial)
                and candidate_check["training_nmse"] < checked["training_nmse"]
            ):
                result["selected"] = polished["selected"]
                result["retained_stage"] = "polish"
                checked = candidate_check
            if polished["stop_reason"] == "cleanup_unconfirmed":
                reason = "cleanup_unconfirmed"
    counts = [call_count(s["backend"]) for s in result["stages"]]
    strict_passed = reason != "cleanup_unconfirmed" and passes(checked, strict)
    result.update(
        strict_prediction_certified=strict_passed,
        actual_residual_calls=sum(counts)
        if all(c is not None for c in counts)
        else None,
        accounting_complete=all(c is not None for c in counts),
        stop_reason="training_prediction_certified" if strict_passed else reason,
        fit_seconds=monotonic() - begun,
        polishing_seconds=(
            monotonic() - begun - result["first_prediction"]["fit_seconds"]
        )
        if result["polishing_attempted"]
        else 0.0,
    )
    save()
    seal(finished, {"identity": identity, "result": result})
    return result
