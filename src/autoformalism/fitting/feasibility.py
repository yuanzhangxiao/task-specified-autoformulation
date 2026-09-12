"""Train-only feasibility screening before derivative-based rollout refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.fitting.directional_poll import poll_fit
from autoformalism.fitting.sensitivity_probe import SymbolicOracle
from autoformalism.fitting.stagnation import instrumented_fit
from autoformalism.rebuttal.fitter_diagnostic import _finite_payload, write_json


class SensitivityUnavailable(RuntimeError):
    """An unavailable augmented rollout must never become a zero Jacobian."""


@dataclass
class EvaluationBudget:
    """One shared budget for screens, local optimization, polling and verification."""

    deadline: float
    maximum: int
    calls: int = 0

    def take(self):
        if self.calls >= self.maximum or monotonic() >= self.deadline:
            raise TimeoutError("shared refinement budget exhausted")
        self.calls += 1


class GuardedOracle(SymbolicOracle):
    """Stop unavailable sensitivity steps; bound each simulation attempt."""

    def __init__(
        self,
        *args,
        budget: EvaluationBudget,
        point_seconds: float,
        fail_fast: bool = True,
        reject_invalid_trials: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.budget, self.point_seconds = budget, point_seconds
        self.fail_fast = fail_fast
        self.reject_invalid_trials = reject_invalid_trials
        self.rejected_trials = 0

    def __call__(self, values):
        self.budget.take()
        self.deadline = min(self.budget.deadline, monotonic() + self.point_seconds)
        before = self.failures.count
        valid_before = self.valid_calls
        try:
            residual = super().__call__(values)
        except TimeoutError as error:
            if monotonic() >= self.budget.deadline:
                raise
            self.failures.record(str(error))
            if self.with_sensitivities:
                self.last_jac = None
                if not (self.reject_invalid_trials and valid_before):
                    raise SensitivityUnavailable(str(error)) from error
            residual = np.full(
                sum(t.number_of_rows for t in self.training.trajectories),
                self.settings.failure_penalty,
            )
        if self.with_sensitivities and self.fail_fast and self.failures.count != before:
            # The legacy oracle records a penalty/zero J internally. Do not let
            # either reach least_squares in this opt-in policy.
            self.last_jac = None
            if self.reject_invalid_trials and valid_before:
                self.rejected_trials += 1
                # TRF rejects nonfinite trial residuals and shrinks its radius.
                # No Jacobian is supplied for this point; a failed first point
                # still exits explicitly rather than pretending to converge.
                return np.full_like(residual, np.inf)
            raise SensitivityUnavailable(
                "augmented integration unavailable; no derivative supplied"
            )
        return residual

    def jacobian(self, values):
        matrix = super().jacobian(values)
        if matrix is None or not np.isfinite(matrix).all():
            raise SensitivityUnavailable("No finite Jacobian at the requested point")
        return matrix


def handoff_starts(oracles, screened: list[dict], ordinary: dict) -> list[dict]:
    """Rank every finite full-training incumbent before older screened starts."""
    candidates = [
        {**o.best, "source": str(o.directory.name)}
        for o in oracles
        if o.best is not None
    ] + screened
    candidates.sort(key=lambda p: p["cost"])
    seen, result = set(), []
    for point in candidates:
        key = tuple(sorted(point["parameters"].items()))
        if key not in seen:
            seen.add(key)
            result.append(point)
    return result or [{"source": "ordinary", "parameters": ordinary, "cost": None}]


def restart_points(initializer, start: dict, design: list[dict]) -> list[dict]:
    """Keep bounded collocation checkpoints, the ordinary start and diverse starts."""
    points = []
    runs = initializer.get("portfolio", [initializer])
    for run in sorted(runs, key=lambda r: r.get("initializer_objective", float("inf"))):
        if run.get("success") and run.get("parameters"):
            points.append(
                {"source": "collocation_converged", "parameters": run["parameters"]}
            )
        saved = (run.get("progress") or {}).get("checkpoints", {})
        for key in ("best_feasible", "least_violation", "latest"):
            if key in saved:
                points.append(
                    {
                        "source": f"collocation_checkpoint:{key}",
                        "parameters": saved[key]["parameters"],
                    }
                )
    points += [{"source": "ordinary", "parameters": start}, *design]
    seen, unique = set(), []
    for point in points:
        key = tuple(sorted(point["parameters"].items()))
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def recover_refinement(
    system,
    training,
    scale,
    settings,
    config,
    directory: Path,
    initializer: dict,
    start: dict,
    design: list[dict],
    identity: str,
    use_poll: bool,
) -> dict:
    """Select on full training rollouts, then refine within the remaining budget."""
    started = monotonic()
    budget = EvaluationBudget(
        started + config.refinement_seconds, config.maximum_function_evaluations
    )
    screen_deadline = started + 0.4 * config.refinement_seconds
    points = restart_points(initializer, start, design)
    screens, stages, oracles, feasible = [], [], [], []

    def oracle(path, sensitivities, point_seconds):
        item = GuardedOracle(
            system,
            training,
            scale,
            settings,
            directory / path,
            budget.deadline,
            sensitivities=sensitivities,
            budget=budget,
            point_seconds=point_seconds,
            reject_invalid_trials=config.sensitivity_invalid_trials == "reject",
        )
        oracles.append(item)
        return item

    primal = oracle("primal_screen", False, config.recovery_probe_seconds)
    for point in points[: config.recovery_max_starts]:
        if (
            budget.calls >= max(1, budget.maximum // 3)
            or monotonic() >= screen_deadline
        ):
            break
        row = {**point, "mode": "primal_only"}
        before, valid_before = primal.calls, primal.valid_calls
        primal.point_seconds = max(
            0.001, min(config.recovery_probe_seconds, screen_deadline - monotonic())
        )
        try:
            residual = primal(primal.vector(point["parameters"]))
            valid = primal.valid_calls > valid_before
            row.update(
                valid=valid, cost=float(0.5 * residual @ residual) if valid else None
            )
            if valid:
                feasible.append({**point, "cost": row["cost"]})
        except (ValueError, TimeoutError) as error:
            row.update(valid=False, error=str(error))
        row["calls"] = primal.calls - before
        if not row["valid"] and primal.failure_evidence:
            row["failure"] = primal.failure_evidence[-1]
        screens.append(row)
        write_json(directory / "feasibility_screen.json", _finite_payload(screens))
        # A successfully converged smooth C start should not pay for irrelevant
        # restarts. Failed/poor-C and piecewise cases retain the bounded screen.
        if row["valid"] and initializer.get("success") and not use_poll:
            break
    feasible.sort(key=lambda point: point["cost"])
    augmented_failed = False
    if feasible and not use_poll:
        for index, point in enumerate(feasible[:2]):
            if budget.calls >= budget.maximum or monotonic() >= budget.deadline:
                break
            if index and config.recovery_handoff == "best_valid":
                point = handoff_starts(oracles, feasible, start)[0]
            augmented = oracle(
                f"augmented_{index}", True, max(30.0, config.recovery_probe_seconds)
            )

            def optimizer(fun, x, _augmented=augmented, **kwargs):
                kwargs.update(jac=_augmented.jacobian, ftol=config.least_squares_ftol)
                return least_squares(fun, x, **kwargs)

            stage_started = monotonic()
            try:
                report = instrumented_fit(
                    augmented,
                    point["parameters"],
                    diff_step=None,
                    max_nfev=max(1, budget.maximum - budget.calls),
                    settings=settings,
                    optimizer=optimizer,
                )
                stages.append(
                    {"mode": "sensitivity", "source": point["source"], "result": report}
                )
                break
            except SensitivityUnavailable as error:
                augmented_failed = True
                stages.append(
                    {
                        "mode": "sensitivity",
                        "source": point["source"],
                        "error": str(error),
                        "failure_evidence": augmented.failure_evidence,
                        "best_evaluated": augmented.best,
                        "actual_residual_calls": augmented.calls,
                        "fit_seconds": monotonic() - stage_started,
                        "optimizer_stationarity_claimed": False,
                    }
                )
    if (
        (use_poll or not feasible or augmented_failed)
        and budget.calls < budget.maximum
        and monotonic() < budget.deadline
    ):
        polling = oracle("poll_calls", False, max(30.0, config.recovery_probe_seconds))
        handoff = (
            handoff_starts(oracles, feasible, start)
            if config.recovery_handoff == "best_valid"
            else feasible or [{"source": "ordinary", "parameters": start}]
        )
        candidates = [point["parameters"] for point in handoff[:2]]
        write_json(directory / "poll_handoff.json", _finite_payload(handoff[:2]))
        report = poll_fit(
            polling,
            candidates,
            scales=np.maximum(np.abs(polling.vector(start)), 1.0),
            max_calls=budget.maximum - budget.calls,
            seconds=max(0.001, budget.deadline - monotonic()),
            checkpoint=directory / "recovery_poll.json",
            identity=identity,
        )
        stages.append({"mode": "directional_poll", "result": report})
    best = min(
        (o.best for o in oracles if o.best is not None),
        key=lambda point: point["cost"],
        default=None,
    )
    result = {
        "method": "feasibility_then_refinement",
        "parameters": best["parameters"] if best else None,
        "cost": best["cost"] if best else None,
        "selection": "best_full_training_evaluation" if best else "no_feasible_rollout",
        "message": "Finite training point retained; independent replay required"
        if best
        else "No feasible full-training rollout after bounded restoration",
        "optimizer_success": False,
        "optimizer_native_success": False,
        "numerical_status": "production_replay_pending"
        if best
        else "no_feasible_rollout",
        "actual_residual_calls": budget.calls,
        "valid_residual_evaluations": sum(o.valid_calls for o in oracles),
        "integration_failures": sum(o.failures.count for o in oracles),
        "rejected_sensitivity_trials": sum(o.rejected_trials for o in oracles),
        "fit_seconds": monotonic() - started,
        "initial_parameters": start,
        "screens": screens,
        "stages": stages,
        "failure_evidence": [e for o in oracles for e in o.failure_evidence],
        "state_integration_succeeded": any(
            o.best is not None and not o.with_sensitivities for o in oracles
        ),
        "augmented_integration_failed": augmented_failed,
        "shared_evaluation_budget": budget.maximum,
        "budget_exhausted": budget.calls >= budget.maximum
        or monotonic() >= budget.deadline,
        "training_only": True,
    }
    write_json(directory / "refinement_result.json", _finite_payload(result))
    return _finite_payload(result)
