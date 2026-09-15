"""Checkpointed warm-start windows for an explicitly authorized diagnostic.

The original fitter and one-window pilot remain unchanged. This measures a
sequence of fresh optimizer invocations, not restored trust-region state.
"""

from __future__ import annotations

import math
from pathlib import Path
from time import time

from autoformalism.fitting import fit_continuation as pilot
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.fitter import _training_variables
from autoformalism.fitting.models import FitConfig
from autoformalism.schemas.fit_continuation import (
    ContinuationResult,
    ContinuationSelection,
)
from autoformalism.schemas.fit_convergence import ConvergenceSelection, RetainedFit
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def _seal(value: dict) -> dict:
    return {"result": value, "sha256": public.content_sha256(value)}


def _read_sealed(path: Path) -> dict:
    saved = public._read(path)
    if public.content_sha256(saved["result"]) != saved["sha256"]:
        raise ValueError(f"result digest differs: {path.name}")
    return saved


def _read_freeze(path: Path) -> dict:
    value = public._read(path)
    if (
        public.content_sha256({k: v for k, v in value.items() if k != "identity"})
        != value["identity"]
    ):
        raise ValueError("freeze digest differs")
    return value


def _validate_parameters(parent: dict, parameters: dict) -> None:
    request = PublicFitRequest.model_validate(parent["freeze"]["request"])
    model, _, _ = public._lower(request)
    training = public.unpack_split(
        PublicSplit.model_validate(parent["freeze"]["training"])
    )
    variables = _training_variables(model, training, FitConfig())
    if {v.name.removeprefix("parameter:") for v in variables} != set(parameters):
        raise ValueError(
            "handoff must contain every equation and initializer parameter"
        )
    if any(
        not math.isfinite(parameters[v.name.removeprefix("parameter:")])
        or not v.lower <= parameters[v.name.removeprefix("parameter:")] <= v.upper
        for v in variables
    ):
        raise ValueError("handoff parameter is nonfinite or outside its domain")


def _historical_seed(
    parent_path: Path, continuation_path: Path, selection: ConvergenceSelection
) -> dict:
    """Verify old artifacts without substituting their historical source hash."""
    parent = pilot._verified_parent(parent_path)
    frozen = _read_freeze(continuation_path / "freeze.json")
    if (
        frozen["protocol"] != "public-fit-continuation-freeze-1"
        or frozen["parent"] != parent
    ):
        raise ValueError("continuation parent differs")
    old_selection = ContinuationSelection.model_validate(frozen["selection"])
    parent_result = parent["result"]["result"]
    if (
        old_selection.parent_lowered_candidate_sha256
        != parent_result["lowered_candidate_sha256"]
        or old_selection.parent_backend_result_sha256
        != parent_result["backend_result_sha256"]
    ):
        raise ValueError("historical pilot selection differs")
    if (
        pilot.progress_gate(parent, old_selection) != frozen["gate"]
        or not frozen["gate"]["eligible"]
    ):
        raise ValueError("historical pilot gate differs")
    pilot._check_ledger(frozen, continuation_path)
    envelope = _read_sealed(continuation_path / "result.json")
    result = ContinuationResult.model_validate(envelope["result"])
    if (
        result.identity != frozen["identity"]
        or result.identity != selection.continuation_identity
        or result.parent_identity != parent_result["identity"]
        or result.parent_result_sha256 != parent["result"]["sha256"]
    ):
        raise ValueError("continuation result lineage differs")
    raw = public._read(continuation_path / "backend_result.json")
    digest = public.content_sha256(raw)
    if (
        digest != result.backend_result_sha256
        or digest != selection.continuation_backend_sha256
    ):
        raise ValueError("continuation backend digest differs")
    for key, value in raw["result_fields"].items():
        if public._jsonable(getattr(result, key)) != value:
            raise ValueError(f"continuation public/backend evidence differs: {key}")
    if result.status != "complete" or not result.training.available:
        raise ValueError("diagnostic requires a completed finite training continuation")
    if (
        raw["initial_parameters"] != parent_result["parameters"]
        or not raw["start_check"]["agrees"]
    ):
        raise ValueError("historical continuation start differs")
    cost = frozen["gate"]["parent_training_cost"]
    if result.selected == "extension":
        if raw["best_evaluated"]["parameters"] != dict(result.parameters):
            raise ValueError(
                "selected continuation vector differs from its best evaluation"
            )
        cost = raw["best_evaluated"]["cost"]
        if cost >= frozen["gate"]["parent_training_cost"]:
            raise ValueError("selected continuation cost did not improve")
    elif dict(result.parameters) != parent_result["parameters"]:
        raise ValueError("retained parent parameters differ")
    _validate_parameters(parent, dict(result.parameters))
    state = RetainedFit(
        parameters=result.parameters,
        training=result.training,
        validation=result.validation,
        training_cost=cost,
        actual_residual_calls=result.cumulative_residual_calls,
    )
    return {
        "parent": parent,
        "continuation": {"freeze": frozen, "result": envelope, "backend": raw},
        "state": state.model_dump(mode="json"),
    }


def _reservation(frozen: dict, directory: Path) -> dict:
    return {
        "identity": frozen["identity"],
        "output": str(directory.resolve()),
        "continuation_identity": frozen["selection"]["continuation_identity"],
    }


def prepare_convergence(
    parent_path: Path,
    continuation_path: Path,
    selection: ConvergenceSelection,
    directory: Path,
) -> dict:
    """Reserve one diagnostic; changing paths cannot renew its budget."""
    parent_path, continuation_path, directory = (
        p.resolve() for p in (parent_path, continuation_path, directory)
    )
    for source in (parent_path.parent, continuation_path.parent):
        if directory.is_relative_to(source) or source.is_relative_to(directory):
            raise ValueError(
                "diagnostic output must be separate from both historical experiments"
            )
    seed = _historical_seed(parent_path, continuation_path, selection)
    ledger = (
        continuation_path.parent.parent
        / ".public-fit-convergence"
        / selection.continuation_identity
    )
    payload = {
        "protocol": "public-fit-convergence-freeze-1",
        "selection": selection.model_dump(mode="json"),
        "seed": seed,
        "parent_path": str(parent_path),
        "continuation_path": str(continuation_path),
        "ledger_path": str(ledger),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
    }
    frozen = {**payload, "identity": public.content_sha256(payload)}
    with public._lock(ledger), public._lock(directory):
        reservation = ledger / "reservation.json"
        expected = _reservation(frozen, directory)
        if reservation.exists() and public._read(reservation) != expected:
            raise ValueError("diagnostic already reserved; no fresh budget")
        path = directory / "freeze.json"
        if path.exists():
            if public._read(path) != frozen:
                raise ValueError("existing diagnostic freeze differs")
            if not reservation.exists():
                raise ValueError("missing diagnostic reservation")
        else:
            if any(p.name != ".lock" for p in directory.iterdir()):
                raise ValueError("diagnostic output is not empty")
            public._write(reservation, expected)
            public._write(path, frozen)
    return report_convergence(directory)


def _load(directory: Path) -> dict:
    frozen = _read_freeze(directory / "freeze.json")
    if frozen["source_sha256"] != public._source_identity():
        raise ValueError("diagnostic source changed")
    if frozen["runtime"] != public._runtime():
        raise ValueError("diagnostic runtime changed")
    selection = ConvergenceSelection.model_validate(frozen["selection"])
    if (
        _historical_seed(
            Path(frozen["parent_path"]), Path(frozen["continuation_path"]), selection
        )
        != frozen["seed"]
    ):
        raise ValueError("historical artifacts changed")
    if public._read(Path(frozen["ledger_path"]) / "reservation.json") != _reservation(
        frozen, directory
    ):
        raise ValueError("diagnostic reservation differs")
    return frozen


def _numerical_input(frozen: dict, state: dict) -> dict:
    """Adapt the retained point to the unchanged numerical window implementation.

    This is an in-memory backend argument, not a rewritten public-fit artifact.
    """
    return {
        "parent": {
            "freeze": frozen["seed"]["parent"]["freeze"],
            "result": {"result": state},
        },
        "gate": {"parent_training_cost": state["training_cost"]},
        "selection": {
            "additional_seconds": frozen["selection"]["window_seconds"],
            "additional_residual_calls": frozen["selection"]["window_residual_calls"],
        },
    }


def _window_request(frozen: dict, index: int, state: dict, previous: str) -> dict:
    return {
        "identity": frozen["identity"],
        "index": index,
        "input_sha256": public.content_sha256(state),
        "previous_result_sha256": previous,
    }


def _record(
    frozen: dict,
    index: int,
    state: dict,
    previous: str,
    raw: dict | None,
    error: str | None = None,
) -> dict:
    """Derive a reproducible record; validation never controls any routing decision."""
    selected, reason, next_state = "parent", None, dict(state)
    report, calls, seconds, setup, scoring = {}, None, None, None, None
    native, stationary, integration_failures = None, False, None
    budget_exhausted = None
    if raw is None:
        reason = "interrupted" if error is None else "numerical_failure"
        next_state["actual_residual_calls"] = None
    else:
        fields, report = raw["result_fields"], raw.get("optimizer") or {}
        if raw["initial_parameters"] != state["parameters"]:
            raise ValueError("window initial parameters differ from retained incumbent")
        selected = fields["selected"]
        calls, seconds = raw["actual_residual_calls"], raw["numerical_seconds"]
        setup, scoring = raw["setup_seconds"], raw["scoring_seconds"]
        integration_failures, budget_exhausted = (
            raw["integration_failures"],
            raw["budget_exhausted"],
        )
        if selected == "extension":
            best = raw["best_evaluated"]
            if (
                not raw["start_check"]["agrees"]
                or best["parameters"] != fields["parameters"]
                or not best["cost"] < state["training_cost"]
                or not fields["training"]["available"]
                or not fields["training"]["normalized_mse"]
                < state["training"]["normalized_mse"]
            ):
                raise ValueError("window violates training-only incumbent selection")
            _validate_parameters(frozen["seed"]["parent"], fields["parameters"])
            next_state.update(
                parameters=fields["parameters"],
                training=fields["training"],
                validation=fields["validation"],
                training_cost=best["cost"],
            )
        elif selected != "parent" or fields["parameters"] != state["parameters"]:
            raise ValueError("window retained-parent selection differs")
        next_state["actual_residual_calls"] = (
            None
            if state["actual_residual_calls"] is None
            else state["actual_residual_calls"] + calls
        )
        optimum = report.get("optimality")
        matches = report.get("native_optimizer_parameters") == next_state["parameters"]
        native = report.get("optimizer_native_success") if matches else None
        stationary = (
            native is True
            and report.get("selected_training_rollout_verified") is True
            and report.get("optimizer_status") in (1, 2, 3, 4)
            and isinstance(optimum, (float, int))
            and math.isfinite(optimum)
            and 0 <= optimum <= frozen["selection"]["stationarity_tolerance"]
        )
        failure = raw.get("failure")
        # Validation scoring is descriptive; failure there cannot allocate or stop
        # fitting. A training/start/integration failure remains a separate stop.
        train_scoring_failure = str(raw.get("scoring_error") or "").startswith(
            "training:"
        )
        if (
            failure and not (failure.get("type") == "TimeoutError" and budget_exhausted)
        ) or train_scoring_failure:
            reason = "numerical_failure"
        elif stationary:
            reason = "local_stationarity_reached"
    next_state = RetainedFit.model_validate(next_state).model_dump(mode="json")
    drop = (state["training_cost"] - next_state["training_cost"]) / max(
        state["training_cost"], 1e-30
    )
    if reason is None and index == frozen["selection"]["maximum_windows"]:
        reason = "diagnostic_cap_reached"
    return {
        **_window_request(frozen, index, state, previous),
        "backend_sha256": None if raw is None else public.content_sha256(raw),
        "state": next_state,
        "selected": selected,
        "stop_reason": reason,
        "error": error,
        "failure": None if raw is None else raw.get("failure"),
        "scoring_error": None if raw is None else raw.get("scoring_error"),
        "native_optimizer_converged": native,
        "stationarity_criterion_met": stationary,
        "optimizer_status": report.get("optimizer_status"),
        "optimizer_message": report.get("message"),
        "optimality": report.get("optimality"),
        "start_cost": state["training_cost"],
        "retained_cost": next_state["training_cost"],
        "relative_training_cost_drop": drop,
        "slow_progress": drop < 0.01,
        "recent_iterations": [
            {k: row[k] for k in ("iteration", "nfev", "cost")}
            for row in report.get("iterations", [])[-6:]
        ],
        "residual_calls": calls,
        "numerical_seconds": seconds,
        "setup_seconds": setup,
        "scoring_seconds": scoring,
        "integration_failures": integration_failures,
        "budget_exhausted": budget_exhausted,
    }


def _scan(frozen: dict, directory: Path) -> tuple[list[dict], dict, str]:
    """Read consecutive sealed windows and verify the complete handoff chain."""
    rows, state = [], frozen["seed"]["state"]
    previous = frozen["seed"]["continuation"]["result"]["sha256"]
    for index in range(1, frozen["selection"]["maximum_windows"] + 1):
        window = directory / "windows" / f"{index:03d}"
        if not (window / "result.json").exists():
            break
        envelope = _read_sealed(window / "result.json")
        saved = envelope["result"]
        marker = public._read(window / "started.json")
        if marker["request"] != _window_request(frozen, index, state, previous):
            raise ValueError("window started lineage differs")
        raw = (
            public._read(window / "backend_result.json")
            if saved["backend_sha256"]
            else None
        )
        if saved != _record(frozen, index, state, previous, raw, saved["error"]):
            raise ValueError("window evidence or lineage differs")
        rows.append(saved)
        state, previous = saved["state"], envelope["sha256"]
        if saved["stop_reason"]:
            break
    return rows, state, previous


def _run_window(frozen: dict, state: dict, directory: Path) -> dict:
    from autoformalism.fitting.continuation_numerics import run_extension

    return run_extension(_numerical_input(frozen, state), directory)


def execute_convergence(directory: Path) -> dict:
    """Resume completed windows; an interrupted started window is never rerun."""
    with public._lock(directory):
        frozen = _load(directory)
        rows, state, previous = _scan(frozen, directory)
        while not (rows and rows[-1]["stop_reason"]):
            index = len(rows) + 1
            window = directory / "windows" / f"{index:03d}"
            window.mkdir(parents=True, exist_ok=True)
            marker = window / "started.json"
            request = _window_request(frozen, index, state, previous)
            raw, error = None, None
            if marker.exists():
                if public._read(marker)["request"] != request:
                    raise ValueError("window started lineage differs")
                # Safe publication recovery only: no optimizer invocation here.
                if (window / "backend_result.json").exists():
                    raw = public._read(window / "backend_result.json")
            else:
                if any(window.iterdir()):
                    raise ValueError("unstarted window contains unexpected artifacts")
                public._write(marker, {"request": request, "utc_seconds": time()})
                try:
                    raw = _run_window(frozen, state, window)
                    public._write(window / "backend_result.json", raw)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            row = _record(frozen, index, state, previous, raw, error)
            envelope = _seal(row)
            public._write(window / "result.json", envelope)
            rows.append(row)
            state, previous = row["state"], envelope["sha256"]
            _publish_report(frozen, directory, rows, state)
        return _publish_report(frozen, directory, rows, state)


def _publish_report(
    frozen: dict, directory: Path, rows: list[dict], state: dict
) -> dict:
    reason = rows[-1]["stop_reason"] if rows else None
    numerical = sum(row["numerical_seconds"] or 0 for row in rows)
    status = reason or "pending_or_running"
    seed = frozen["seed"]["state"]
    result = {
        "protocol": "public-fit-convergence-report-1",
        "identity": frozen["identity"],
        "status": status,
        "recorded_windows": len(rows),
        "completed_windows": sum(row["backend_sha256"] is not None for row in rows),
        "maximum_windows": frozen["selection"]["maximum_windows"],
        "maximum_additional_numerical_seconds": frozen["selection"]["maximum_windows"]
        * 180,
        "consumed_window_budget_seconds": len(rows) * 180,
        "additional_numerical_seconds": numerical,
        "timing_incomplete": any(row["numerical_seconds"] is None for row in rows),
        "additional_residual_calls": sum(row["residual_calls"] or 0 for row in rows),
        "residual_accounting_incomplete": any(
            row["residual_calls"] is None for row in rows
        ),
        "prior_allocated_refinement_seconds": 360,
        "prior_allocated_collocation_seconds": 120,
        "initial_state": seed,
        "retained_state": state,
        "windows": rows,
        "training_only_selection_and_stopping": True,
        "validation_initials_fitted": False,
        "independent_replay": "not_performed",
        "structural_failure_established": False,
        "final_protocol_limit_decided": False,
        "optimizer_state_restored": False,
        "collocation_reruns": 0,
        "automatic_followup_after_diagnostic": False,
    }
    text = [
        "# Public-fit convergence diagnostic",
        "",
        f"Status: {status}. Recorded windows: {len(rows)}/{result['maximum_windows']}.",
        "",
        "NMSE scores v01 only. Selection and stopping use training only. "
        "Validation is descriptive.",
        "Each window restarts optimizer state from the complete retained "
        "parameter vector.",
        "Local stationarity is not global convergence or scientific correctness. "
        "A cap does not prove nonconvergence.",
        "Final pipeline continuation limit remains undecided. "
        "No independent solver replay or automatic follow-up.",
        "",
        "| Window | Numerical s | Cumulative additional s | Calls | Train NMSE | "
        "Validation NMSE | Cost drop | Native success | Optimality | Stop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]

    def fmt(value):
        return "—" if value is None else f"{value:.6g}"

    cumulative = 0.0
    for row in rows:
        cumulative += row["numerical_seconds"] or 0
        text.append(
            f"| {row['index']} | {fmt(row['numerical_seconds'])} | {fmt(cumulative)} | "
            f"{fmt(row['residual_calls'])} | "
            f"{fmt(row['state']['training']['normalized_mse'])} | "
            f"{fmt(row['state']['validation']['normalized_mse'])} | "
            f"{row['relative_training_cost_drop']:.2%} | "
            f"{row['native_optimizer_converged']} | "
            f"{fmt(row['optimality'])} | {row['stop_reason'] or 'continue'} |"
        )
    if result["timing_incomplete"]:
        text += [
            "",
            "Interrupted-window time/calls are unknown; recorded totals are "
            "lower bounds. No interrupted budget is renewed.",
        ]
    public._write(directory / "summary.json", result)
    temporary = directory / "SUMMARY.md.tmp"
    temporary.write_text("\n".join(text) + "\n")
    temporary.replace(directory / "SUMMARY.md")
    return result


def report_convergence(directory: Path) -> dict:
    """Verify and summarize artifacts without starting or completing any optimizer."""
    # Reporting may run while fitting. Do not compete with its publication files.
    with public._lock(directory):
        frozen = _load(directory)
        rows, state, _ = _scan(frozen, directory)
        return _publish_report(frozen, directory, rows, state)
