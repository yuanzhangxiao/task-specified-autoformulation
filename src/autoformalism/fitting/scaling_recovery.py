"""M26 training-only generic-start portfolio with bounded continuation blocks."""

from pathlib import Path
from time import monotonic, process_time
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import nonlinear_rollout as rollout
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting.affine_propagation import numerically_verified
from autoformalism.fitting.fitting_assessment import assess
from autoformalism.fitting.optimizer_scaling import RelativeScaling
from autoformalism.schemas.base import StrictSchema

METHODS = ("jacobian", "fixed_coordinates", "guarded_relative")


class Policy(StrictSchema):
    """Equal ceilings, reserved restart opportunities and frozen numerical choices."""

    restart_count: Literal[3] = 3
    maximum_blocks: int = Field(default=6, ge=4, le=8)
    maximum_consecutive_blocks: int = Field(default=2, ge=1, le=3)
    block_calls: int = Field(default=40, ge=2, le=120)
    block_seconds: float = Field(default=800, ge=1, le=2400)
    progress_window: int = Field(default=10, ge=2, le=30)
    minimum_relative_progress: float = Field(default=1e-3, gt=0, le=0.1)
    check_seconds: float = Field(default=180, ge=1, le=180)
    sensitivity_seconds: float = Field(default=180, ge=1, le=180)
    replay_seconds: float = Field(default=300, ge=1, le=600)
    target_nmse: float = Field(default=1e-12, gt=0, le=1e-6)
    trajectory_nmse: float = Field(default=1e-11, gt=0, le=1e-5)
    scaling: RelativeScaling = Field(default_factory=RelativeScaling)
    objective_multiplier: float = Field(default=1, ge=1e-3, le=1e6)

    @model_validator(mode="after")
    def valid(self):
        values = self.model_dump(exclude={"scaling"}).values()
        if (
            not np.isfinite(list(values)).all()
            or self.trajectory_nmse < self.target_nmse
        ):
            raise ValueError("invalid finite budget or accuracy policy")
        return self


def continuation_reason(value: dict, policy: Policy) -> str:
    """Use recent training progress, never reference accuracy or validation scores."""
    if value.get("stop_reason") != "budget_limited":
        return "native_stop_or_failure"
    trace = value.get("evaluations", [])
    if len(trace) < policy.progress_window:
        return "insufficient_progress_evidence"
    first = trace[-policy.progress_window]["best_training_nmse"]
    last = trace[-1]["best_training_nmse"]
    relative = (first - last) / max(abs(first), policy.target_nmse)
    return "improving" if relative >= policy.minimum_relative_progress else "plateau"


def validate_starts(oracle, profile, problem, starts: list[dict]) -> None:
    """Check domains and genuinely distinct optimized coordinates before searching."""
    if starts[0] != problem.start or problem.incumbent != problem.start:
        raise ValueError("recovery must begin at the original generic start")
    indices = list(range(len(oracle.names))) if profile is None else list(profile.outer)
    vectors = [oracle.vector(s)[indices] / oracle.units[indices] for s in starts]
    if any(
        np.linalg.norm(a - b) <= 1e-8
        for i, a in enumerate(vectors)
        for b in vectors[:i]
    ):
        raise ValueError("restart optimized coordinates are not distinct")


def run(
    problem: checks.TrainingProblem,
    starts: list[dict],
    policy: Policy,
    method: str,
    folder: Path,
) -> dict:
    """Run one scale policy; no evaluator arrays exist in this function's input."""
    if method not in METHODS or len(starts) != 1 + policy.restart_count:
        raise ValueError("invalid scaling method or complete restart roster")
    identity = {
        "protocol": "generic-scaling-recovery-1",
        "problem_sha256": public.content_sha256(problem.model_dump(mode="json")),
        "starts": starts,
        "method": method,
        "policy": policy.model_dump(mode="json"),
    }
    finished = folder / "finished.json"
    if finished.exists():
        saved = read_seal(finished)
        if saved["identity"] != identity:
            raise ValueError("scaling recovery resume identity differs")
        return saved
    began, cpu = monotonic(), process_time()
    oracle, profile, route = rollout.build(problem)
    validate_starts(oracle, profile, problem, starts)
    setup = folder / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    seal(
        setup / f"{len(list(setup.glob('*.json'))):04d}.json",
        {
            "wall_seconds": monotonic() - began,
            "cpu_seconds": process_time() - cpu,
            "identity": identity,
        },
    )
    operations, candidates, decisions, stops = [], [], [], []
    selected = None

    def operation(name, point, seconds, work):
        op = checks.operation(folder / name, identity | {"point": point}, seconds, work)
        operations.append({"operation": name, **op})
        return op

    def inspect(name, parameters):
        nonlocal selected
        point = operation(
            name,
            parameters,
            policy.check_seconds,
            lambda end, save: rollout.verify(
                oracle, parameters, end, save, timing=True
            ),
        ).get("value")
        if point and (
            not numerical.usable(point)
            or (
                point["training_nmse"] <= 10 * policy.target_nmse
                and not numerically_verified(point, policy.target_nmse)
            )
        ):
            tighter = operation(
                name + "-tight",
                parameters,
                policy.check_seconds,
                lambda end, save: rollout.verify(
                    oracle, parameters, end, save, tight=True
                ),
            ).get("value")
            if numerical.usable(tighter) and (
                not numerical.usable(point)
                or tighter["solver_loss_discrepancy"]
                <= point["solver_loss_discrepancy"]
            ):
                point = tighter
        if point:
            point = point | {"origin": name}
            candidates.append(point)
            if numerical.improves(point, selected):
                selected = point
        return point

    def reached():
        return bool(
            numerical.usable(selected)
            and numerically_verified(selected, policy.target_nmse)
            and max(selected["training_nmse"], selected["alternate_training_nmse"])
            <= policy.target_nmse
            and selected["maximum_trajectory_nmse"] <= policy.trajectory_nmse
        )

    def search_block(name, start):
        def search(initial, end, save, kernel, calls):
            return numerical.search(
                oracle,
                initial,
                end,
                save,
                calls=calls,
                target=policy.target_nmse,
                acceptable=lambda r: max(rollout.trajectory_losses(oracle, r))
                <= policy.trajectory_nmse,
                profile=kernel,
                telemetry=True,
                record_steps=True,
                optimizer_scaling=method,
                scaling_policy=policy.scaling,
                normalized_objective=True,
                objective_multiplier=policy.objective_multiplier,
            )

        op = operation(
            name + "/rollout",
            start,
            policy.block_seconds,
            lambda end, save: search(start, end, save, profile, policy.block_calls),
        )
        value = op.get("value") or {}
        best = value.get("best")
        used = value.get("calls", policy.block_calls)
        remaining = policy.block_seconds - op["budget_charge_seconds"]
        # A failed profile may use a joint fallback, within the same block ceiling.
        if (
            profile is not None
            and value.get("stop_reason") == "numerical_failure"
            and op["accounting_complete"]
            and policy.block_calls - used >= 2
            and remaining > 1
        ):
            fallback = operation(
                name + "/joint-fallback",
                best["parameters"] if best else start,
                remaining,
                lambda end, save: search(
                    best["parameters"] if best else start,
                    end,
                    save,
                    None,
                    policy.block_calls - used,
                ),
            )
            other = fallback.get("value") or {}
            if other.get("best") and (
                best is None or other["best"]["training_nmse"] < best["training_nmse"]
            ):
                best = other["best"]
            # Failure/fallback never supplies evidence to extend a profiled block.
        stops.append(value.get("stop_reason", op["status"]))
        point = inspect(name + "/check", best["parameters"]) if best else None
        return value, point

    inspect("original/check", starts[0])
    blocks = 0
    for index, start in enumerate(starts):
        if reached():
            break
        point = None
        for visit in range(policy.maximum_consecutive_blocks):
            if blocks >= policy.maximum_blocks or reached():
                break
            name = f"start-{index}/block-{visit}"
            value, point = search_block(
                name, start if point is None else point["parameters"]
            )
            blocks += 1
            reason = continuation_reason(value, policy)
            unvisited = len(starts) - index - 1
            if reached():
                reason = "verified_target"
            elif not numerical.usable(point):
                reason = "unverified_branch"
            elif visit + 1 >= policy.maximum_consecutive_blocks:
                reason = "consecutive_block_limit"
            elif policy.maximum_blocks - blocks <= unvisited:
                reason = "reserve_unvisited_starts"
            decisions.append(
                {
                    "block": name,
                    "reason": reason,
                    "continue": reason == "improving",
                    "selected_origin": selected["origin"] if selected else None,
                }
            )
            seal(folder / name / "decision.json", decisions[-1])
            if reason != "improving":
                break
    sensitivity = None
    if selected:

        def derivative(end, save):
            save({"calls": 1})
            _, j = oracle.evaluate(oracle.vector(selected["parameters"]), end)
            return checks.sensitivity(j, oracle.units, oracle.names)

        sensitivity = operation(
            "final-sensitivity",
            selected["parameters"],
            policy.sensitivity_seconds,
            derivative,
        ).get("value")
    assessment = assess(selected, sensitivity, candidates, stops, policy.target_nmse)
    if numerical.usable(selected) and not reached():
        assessment.update(
            fit_quality="above_numerical_target",
            search_status="further_search_needed",
            recommended_action="diversify_or_extend_search",
        )
    searches = [
        o
        for o in operations
        if o["operation"].endswith(("/rollout", "/joint-fallback"))
    ]
    accounted = all(
        o["accounting_complete"]
        and o.get("value") is not None
        and isinstance(o["value"].get("completed_calls"), int)
        for o in searches
    )
    setups = [read_seal(p) for p in sorted(setup.glob("*.json"))]
    result = {
        "identity": identity,
        "status": "complete" if selected else "retained_unverified",
        "selected": selected,
        "retained_parameters": selected["parameters"] if selected else None,
        "assessment": assessment,
        "accuracy_verified": reached(),
        "routing": route,
        "decisions": decisions,
        "blocks_used": blocks,
        "operations": operations,
        "setup": setups,
        "search_calls_started": sum(o["value"]["calls"] for o in searches)
        if accounted
        else None,
        "search_calls_completed": sum(o["value"]["completed_calls"] for o in searches)
        if accounted
        else None,
        "search_call_accounting_complete": accounted,
        "maximum_search_calls": policy.maximum_blocks * policy.block_calls,
        "search_budget_charge_seconds": sum(
            o["budget_charge_seconds"] for o in searches
        ),
        "additional_wall_seconds": sum(
            o["wall_seconds"] or 0 for o in operations + setups
        ),
        "additional_cpu_seconds": sum(
            o["cpu_seconds"] or 0 for o in operations + setups
        ),
        "additional_budget_charge_seconds": sum(
            o["budget_charge_seconds"] for o in operations
        )
        + sum(s["wall_seconds"] for s in setups),
        "cost_complete": all(o["accounting_complete"] for o in operations),
        "validation_used_for_selection": False,
        "reference_values_used": False,
    }
    seal(finished, result)
    return result
