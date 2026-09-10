"""Training-only, budgeted comparison of ordinary and collocation starts."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.sensitivity_probe import SymbolicODE, SymbolicOracle
from autoformalism.fitting.stagnation import instrumented_fit
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    read_json,
    write_json,
)
from autoformalism.rebuttal.fitter_stagnation import checkpoint
from autoformalism.staged_topology import content_hash


class LimitedOracle(SymbolicOracle):
    """Count every actual residual call, including implicit Jacobian refreshes."""

    call_limit: int
    call_limit_reached: bool = False

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if self.calls >= self.call_limit:
            self.call_limit_reached = True
            raise TimeoutError("portfolio residual-call budget reached")
        return super().__call__(values)


def sensitivity_stage(
    system: SymbolicODE,
    training: DatasetSplit,
    scale: float,
    settings: FitConfig,
    start: dict,
    root: Path,
    identity: str,
    seconds: float,
    calls: int,
    outer_deadline: float,
) -> dict:
    """Checkpoint a fresh rollout refinement and its best valid training evaluation."""
    if training.name is not SplitName.TRAIN:
        raise ValueError("portfolio requires training split")
    key = content_hash([identity, start, seconds, calls])
    saved = checkpoint(root / "stage.json", key)
    if saved is not None:
        return saved
    started = monotonic()
    attempt = 0
    while (root / f"attempt-{attempt}").exists():
        attempt += 1
    directory = root / f"attempt-{attempt}"
    oracle = LimitedOracle(
        system,
        training,
        scale,
        settings,
        directory,
        min(outer_deadline, started + seconds),
        sensitivities=True,
    )
    oracle.call_limit = calls
    report = {}
    try:

        def optimizer(fun, x, **kwargs):
            kwargs["jac"] = oracle.jacobian
            return least_squares(fun, x, **kwargs)

        report = instrumented_fit(
            oracle,
            start,
            diff_step=None,
            max_nfev=calls,
            settings=settings,
            optimizer=optimizer,
        )
        if oracle.call_limit_reached:
            report.update(
                message="portfolio residual-call budget reached", optimizer_status=-3
            )
    except (ValueError, RuntimeError, ArithmeticError) as error:
        report = {"optimizer_success": False, "message": str(error)[-1200:]}
    initial = directory / "000001.json"
    first = read_json(initial) if initial.exists() else {}
    result = _finite_payload(
        {
            "identity": key,
            "start": start,
            "best": oracle.best,
            "report": report,
            "seconds": monotonic() - started,
            "calls": oracle.calls,
            "residual_seconds": oracle.seconds,
            "integration_failures": oracle.failures.count,
            "initial_rollout_valid": first.get("status") == "evaluated",
            "initial_cost": first.get("cost"),
            "call_limit": calls,
            "seconds_limit": seconds,
            "training_only": True,
            "latent_nodes_used": False,
        }
    )
    write_json(root / "stage.json", result)
    return result


def choose_stage(stages: dict[str, dict]) -> str | None:
    """Choose refined training cost; retain insertion order on numerical ties."""
    winner = None
    for name, stage in stages.items():
        best = stage.get("best")
        if best is None or not np.isfinite(best["cost"]):
            continue
        if winner is None:
            winner = name
        else:
            previous = stages[winner]["best"]["cost"]
            tolerance = 1e-12 * max(1, abs(previous), abs(best["cost"]))
            if best["cost"] < previous - tolerance:
                winner = name
    return winner


def portfolio_fit(
    system: SymbolicODE,
    training: DatasetSplit,
    scale: float,
    settings: FitConfig,
    ordinary: dict,
    initializer: dict,
    root: Path,
    identity: str,
    *,
    total_seconds: float,
    pilot_seconds: float,
    pilot_calls: int,
) -> dict:
    """Charge initialization and both pilots before continuing the better training fit.

    Completed stages are reused with their original time/call charges. A killed
    optimizer restarts only its unfinished stage at the same frozen stage start.
    No truth, validation data, or collocation node objective enters selection.
    """
    if training.name is not SplitName.TRAIN:
        raise ValueError("portfolio requires training split")
    started = monotonic()
    deadline = started + max(0, total_seconds - initializer["seconds"])
    total_calls = settings.maximum_function_evaluations
    stages: dict[str, dict] = {}
    candidate = initializer.get("parameters") if initializer.get("success") else None
    source = "converged" if candidate is not None else "unavailable"
    saved_iterate = initializer.get("last_finite_iterate") or {}
    if candidate is None and saved_iterate.get("finite_in_domain"):
        candidate = saved_iterate.get("parameters")
        source = "unfinished_iterate"
    if candidate == ordinary:
        candidate, source = None, "duplicate_ordinary"

    def remaining():
        return (
            max(
                0,
                total_seconds
                - initializer["seconds"]
                - sum(s["seconds"] for s in stages.values()),
            ),
            max(0, total_calls - sum(s["calls"] for s in stages.values())),
        )

    def run(name: str, start: dict, is_pilot: bool) -> None:
        seconds, calls = remaining()
        if calls < 1 or seconds <= 0:
            return
        if is_pilot:
            seconds, calls = min(seconds, pilot_seconds), min(calls, pilot_calls)
        stages[name] = sensitivity_stage(
            system,
            training,
            scale,
            settings,
            start,
            root / name,
            content_hash([identity, name]),
            seconds,
            calls,
            deadline,
        )

    if candidate is None:
        run("ordinary_only", ordinary, False)
        pilot_winner = "ordinary_only" if choose_stage(stages) else None
    else:
        run("ordinary_pilot", ordinary, True)
        run("collocation_pilot", candidate, True)
        pilot_winner = choose_stage(stages)
        if pilot_winner is not None:
            winner = stages[pilot_winner]
            # A converged winning endpoint needs no artificial optimizer restart.
            converged = winner["report"].get("optimizer_success", False) and (
                winner["report"].get("parameters") == winner["best"]["parameters"]
            )
            if not converged:
                run("continuation", winner["best"]["parameters"], False)
    selected = choose_stage(stages)
    picked = stages[selected] if selected is not None else {}
    best = picked.get("best")
    fit_seconds = sum(s["seconds"] for s in stages.values())
    fit = {
        "parameters": best["parameters"] if best else None,
        "cost": best["cost"] if best else None,
        "selection": "best_valid_training_evaluation_across_portfolio_stages",
        "selected_stage": selected,
        "best_evaluated": best,
        "actual_residual_calls": sum(s["calls"] for s in stages.values()),
        "residual_seconds": sum(s["residual_seconds"] for s in stages.values()),
        "fit_seconds": fit_seconds,
        "integration_failures": sum(s["integration_failures"] for s in stages.values()),
        "optimizer_success": bool(picked.get("report", {}).get("optimizer_success")),
        "message": picked.get("report", {}).get(
            "message", "no valid training evaluation"
        ),
        "nfev": sum(s["report"].get("nfev") or 0 for s in stages.values())
        if all(s["report"].get("nfev") is not None for s in stages.values())
        else None,
    }
    collocation = stages.get("collocation_pilot", {})
    return _finite_payload(
        {
            "identity": identity,
            "fit": fit,
            "initializer": initializer,
            "ordinary_start": ordinary,
            "refinement_start": ordinary,
            "training_fingerprint": training.fingerprint,
            "fit_budget_seconds": total_seconds,
            "total_fit_seconds": initializer["seconds"] + fit_seconds,
            "initializer_fallback": not initializer.get("success", False),
            "symbolic_solver_counts": None,
            "portfolio": {
                "stages": stages,
                "pilot_winner": pilot_winner,
                "collocation_source": source,
                "collocation_start": candidate,
                "unfinished_iterate_usable": source == "unfinished_iterate"
                and collocation.get("initial_rollout_valid", False),
                "selection_uses": "training residual cost after pilot refinement",
                "validation_used": False,
                "reference_used": False,
                "remaining_seconds": remaining()[0],
                "remaining_calls": remaining()[1],
                "observed_elapsed_seconds": monotonic() - started,
                "training_rows": sum(t.number_of_rows for t in training.trajectories),
                "pilot_seconds": pilot_seconds,
                "pilot_calls": pilot_calls,
            },
        }
    )
