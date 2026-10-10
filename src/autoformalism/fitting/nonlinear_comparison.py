"""M22 conditional trajectories versus strongest applicable rollout recovery."""

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
from autoformalism.fitting.fitting_allocation import (
    MeasuredConditionalEngine,
    precision_reason,
)
from autoformalism.fitting.fitting_assessment import assess
from autoformalism.fitting.trajectory_profile_fit import ConditionalEngine
from autoformalism.schemas.base import StrictSchema

METHODS = ("best_rollout", "conditional_then_best_rollout")


class ComparisonPolicy(StrictSchema):
    """Matched ceilings; conditional computation consumes its arm's search budget."""

    allocation: Literal["legacy", "measured"] = "legacy"
    screening_safety_factor: float = Field(default=2, ge=1.5, le=5)
    minimum_node_seconds: float = Field(default=20, ge=0.1, le=120)
    precision_near_factor: float = Field(default=10, ge=1, le=100)
    warm_seconds: float = Field(default=600, ge=1, le=1200)
    warm_calls: int = Field(default=600, ge=2, le=1200)
    trial_seconds: float = Field(default=200, ge=1, le=600)
    trial_calls: int = Field(default=200, ge=2, le=600)
    portfolio_size: int = Field(default=3, ge=0, le=5)
    continuation_seconds: float = Field(default=600, ge=1, le=1200)
    continuation_calls: int = Field(default=600, ge=2, le=1200)
    check_seconds: float = Field(default=120, ge=1, le=180)
    sensitivity_seconds: float = Field(default=120, ge=1, le=180)
    replay_seconds: float = Field(default=300, ge=1, le=600)
    target_nmse: float = Field(default=1e-12, gt=0, le=1e-6)
    trajectory_nmse: float = Field(default=1e-11, gt=0, le=1e-5)
    conditional_fraction: float = Field(default=0.3, gt=0, le=0.5)
    node_targets: tuple[int, ...] = (36000, 54000)
    penalties: tuple[float, ...] = (100.0, 10000.0)
    minimum_intervals: int = Field(default=16, ge=1, le=128)
    observation_anchors: int = Field(default=8, ge=0, le=32)
    cycles: int = Field(default=4, ge=1, le=20)
    block_iterations: int = Field(default=25, ge=1, le=100)

    @model_validator(mode="after")
    def limits(self):
        if (
            not 1 <= len(self.node_targets) <= 3
            or len(self.penalties) != len(self.node_targets)
            or list(self.node_targets) != sorted(set(self.node_targets))
            or any(n < 1 or n > 60000 for n in self.node_targets)
            or any(not np.isfinite(p) or p <= 0 for p in self.penalties)
            or self.trajectory_nmse < self.target_nmse
        ):
            raise ValueError("invalid conditional mesh/penalty/accuracy policy")
        return self


def run(
    problem: checks.TrainingProblem,
    policy: ComparisonPolicy,
    method: str,
    folder: Path,
    *,
    shared_warm: dict[str, float] | None = None,
    continuation_order: Literal["incumbent_first", "restart_first"] | None = None,
):
    """One training-only strategy; evaluator data cannot enter this interface."""
    if method not in METHODS:
        raise ValueError("unknown nonlinear comparison method")
    if (shared_warm is None) != (continuation_order is None) or (
        continuation_order is not None
        and (
            continuation_order not in {"incumbent_first", "restart_first"}
            or method != METHODS[0]
            or policy.allocation != "measured"
        )
    ):
        raise ValueError("shared warm continuation requires a measured rollout policy")
    problem_sha = public.content_sha256(problem.model_dump(mode="json"))
    identity = {
        "problem_sha256": problem_sha,
        "method": method,
        "policy": policy.model_dump(mode="json"),
        **(
            {"shared_warm": shared_warm, "continuation_order": continuation_order}
            if shared_warm is not None
            else {}
        ),
    }
    finished = folder / "finished.json"
    if finished.exists():
        result = read_seal(finished)
        if result["identity"] != identity:
            raise ValueError("nonlinear comparison resume identity differs")
        return result
    begun, cpu = monotonic(), process_time()
    oracle, profile, route = rollout.build(problem)
    if shared_warm is not None:
        oracle.vector(shared_warm)
    setup = folder / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    seal(
        setup / f"{len(list(setup.glob('*.json'))):04d}.json",
        {
            "wall_seconds": monotonic() - begun,
            "cpu_seconds": process_time() - cpu,
            "routing": route,
            "problem_sha256": problem_sha,
        },
    )
    measured = policy.allocation == "measured"
    engine_type = MeasuredConditionalEngine if measured else ConditionalEngine
    engine = engine_type(oracle, problem, policy) if method != METHODS[0] else None
    operations, candidates, stops = [], [], []
    precision_decisions, precision_cache, screen_costs = [], {}, []
    selected = None

    def operation(name, point, seconds, work):
        record = checks.operation(
            folder / name, identity | {"point": point}, seconds, work
        )
        operations.append({"operation": name, **record})
        return record

    def inspect(name, parameters):
        nonlocal selected
        record = operation(
            name,
            parameters,
            policy.check_seconds,
            lambda end, save: rollout.verify(
                oracle, parameters, end, save, **({"timing": True} if measured else {})
            ),
        )
        point = record.get("value")
        if measured:
            elapsed = (point or {}).get("integrator_seconds", {}).get("DOP853")
            if elapsed is not None and np.isfinite(elapsed) and elapsed > 0:
                screen_costs.append(elapsed)
            reason = precision_reason(point, selected, policy)
            decision = {"origin": name, "reason": reason, "attempts": []}

            def tighter(value):
                if numerically_verified(value, policy.target_nmse):
                    return value
                digest = public.content_sha256(value["parameters"])
                reused = digest in precision_cache
                if not reused:
                    precision_cache[digest] = operation(
                        "precision/" + digest,
                        value["parameters"],
                        policy.check_seconds,
                        lambda end, save: rollout.verify(
                            oracle, value["parameters"], end, save, tight=True
                        ),
                    ).get("value")
                checked = precision_cache[digest]
                decision["attempts"].append(
                    {"parameters_sha256": digest, "reused": reused}
                )
                # A failed or less precise retry cannot replace a useful check.
                if numerical.usable(checked) and (
                    not numerical.usable(value)
                    or checked["solver_loss_discrepancy"]
                    <= value["solver_loss_discrepancy"]
                ):
                    return checked
                return value

            if reason in {
                "ordinary_numerics_unreliable",
                "near_accuracy_target",
                "retention_interval_overlap",
            }:
                point = tighter(point)
                if reason == "retention_interval_overlap":
                    origin = selected["origin"]
                    selected = tighter(selected) | {"origin": origin}
            precision_decisions.append(decision)
        elif point and not numerically_verified(point, 1e-12):
            tighter = operation(
                name + "-tight",
                parameters,
                policy.check_seconds,
                lambda end, save: rollout.verify(
                    oracle, parameters, end, save, tight=True
                ),
            )
            if numerical.usable(tighter.get("value")):
                point = tighter["value"]
        if point:
            point = point | {"origin": name}
            candidates.append(point)
            if numerical.improves(point, selected):
                selected = point
        return point

    def reached(point):
        return bool(
            numerical.usable(point)
            and (not measured or numerically_verified(point, policy.target_nmse))
            and max(point["training_nmse"], point["alternate_training_nmse"])
            <= policy.target_nmse
            and point["maximum_trajectory_nmse"] <= policy.trajectory_nmse
        )

    def stage(name, start, seconds, calls, *, conditional=True):
        point, remaining = start, seconds
        if engine is not None and conditional:
            if measured:
                engine.screen_cost = max(screen_costs) if screen_costs else None
            record = operation(
                name + "/conditional",
                {"start": start, "measured_screen_seconds": engine.screen_cost}
                if measured
                else start,
                seconds * policy.conditional_fraction,
                lambda end, save: engine.propose(
                    start, end, save, folder / name / "nodes"
                ),
            )
            remaining -= record["budget_charge_seconds"]
            if record.get("value"):
                point = record["value"]["parameters"]
                if measured:
                    screen_costs.extend(
                        record["value"].get("complete_screen_seconds", [])
                    )
        if remaining <= 0:
            stops.append("conditional_spent_stage_allowance")
            return None

        def search(initial, end, save, use_profile, limit):
            return numerical.search(
                oracle,
                initial,
                end,
                save,
                calls=limit,
                target=policy.target_nmse,
                profile=use_profile,
                acceptable=lambda r: max(rollout.trajectory_losses(oracle, r))
                <= policy.trajectory_nmse,
                **({"telemetry": True} if shared_warm is not None else {}),
            )

        record = operation(
            name + "/rollout",
            {"start": point, "calls": calls},
            remaining,
            lambda end, save: search(point, end, save, profile, calls),
        )
        value = record.get("value")
        best = value.get("best") if value else None
        stops.append(value["stop_reason"] if value else "operation_unavailable")
        # A mathematically valid profiling block can be numerically singular at
        # one start. A bounded joint fallback keeps that start usable if possible.
        if (
            profile is not None
            and value
            and value["stop_reason"] == "numerical_failure"
            and record["accounting_complete"]
            and calls - value["calls"] >= 2
            and remaining - record["budget_charge_seconds"] > 1
        ):
            fallback_start = best["parameters"] if best else point
            fallback = operation(
                name + "/joint-fallback",
                {"start": fallback_start, "calls": calls - value["calls"]},
                remaining - record["budget_charge_seconds"],
                lambda end, save: search(
                    fallback_start, end, save, None, calls - value["calls"]
                ),
            )
            other = fallback.get("value")
            stops.append(other["stop_reason"] if other else "fallback_unavailable")
            if (
                other
                and other.get("best")
                and (
                    best is None
                    or other["best"]["training_nmse"] < best["training_nmse"]
                )
            ):
                best = other["best"]
        return inspect(name + "/check", best["parameters"]) if best else None

    before = (
        inspect("incumbent", problem.incumbent)
        if shared_warm is None
        else inspect("shared-warm/check", shared_warm)
    )
    if shared_warm is None and not reached(selected):
        stage(
            "warm",
            problem.incumbent,
            policy.warm_seconds,
            policy.warm_calls,
            conditional=not measured,
        )
    after_warm = selected
    incumbent_continuation = (
        continuation_order == "incumbent_first"
        and selected is not None
        and not reached(selected)
    )
    if incumbent_continuation:
        stage(
            "incumbent-continuation",
            selected["parameters"],
            policy.continuation_seconds,
            policy.continuation_calls,
            conditional=False,
        )
    after_incumbent_continuation = selected
    starts = numerical.diverse_starts(
        oracle, problem, policy.portfolio_size, int(problem_sha[:8], 16)
    )
    portfolio = {
        "starts": starts,
        "run": not reached(selected)
        and (shared_warm is None or numerical.usable(selected)),
        "source": "training_and_declared_domains_only",
    }
    seal(folder / "portfolio.json", portfolio)
    trials = []
    if portfolio["run"]:
        for i, start in enumerate(starts):
            if shared_warm is not None and reached(selected):
                break
            point = stage(f"trial-{i}", start, policy.trial_seconds, policy.trial_calls)
            if numerical.usable(point):
                trials.append(point)
    winner = (
        min(trials, key=lambda p: max(p["training_nmse"], p["alternate_training_nmse"]))
        if trials
        else None
    )
    continuation = (
        continuation_order != "incumbent_first"
        and not reached(selected)
        and winner is not None
    )
    seal(folder / "continuation-decision.json", {"run": continuation, "winner": winner})
    if continuation:
        # Keep the incumbent but explore the most promising distinct trial basin.
        stage(
            "continuation",
            winner["parameters"],
            policy.continuation_seconds,
            policy.continuation_calls,
            conditional=False,
        )
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
    if numerical.usable(selected) and not reached(selected):
        assessment.update(
            fit_quality="above_numerical_target", search_status="further_search_needed"
        )
        search_reliable = measured and precision_reason(selected, None, policy) in {
            "ordinary_precision_sufficient_for_search",
            "already_strictly_verified",
        }
        if search_reliable:
            assessment["reasons"].append(
                "Ordinary independent rollouts resolve poor fit; ultimate precision "
                "remains necessary for an eventual accuracy certificate."
            )
        if assessment["numerical_status"] == "strictly_verified" or search_reliable:
            assessment["recommended_action"] = "diversify_or_extend_search"
    trajectory_assessment = {
        "maximum_trajectory_nmse": selected["maximum_trajectory_nmse"]
        if selected
        else None,
        "trajectory_target": policy.trajectory_nmse,
    }
    setups = [read_seal(p) for p in sorted(setup.glob("*.json"))]

    def total(field):
        return sum(r[field] or 0 for r in operations)

    search_operations = [
        o
        for o in operations
        if o["operation"].endswith(("/rollout", "/joint-fallback"))
    ]
    search_accounted = all(
        o["accounting_complete"]
        and o.get("value") is not None
        and isinstance(o["value"].get("calls"), int)
        and isinstance(o["value"].get("completed_calls"), int)
        for o in search_operations
    )

    result = {
        "identity": identity,
        "status": "complete" if selected else "retained_unverified",
        "routing": route,
        "before": before,
        "after_warm": after_warm,
        "selected": selected,
        "retained_parameters": selected["parameters"]
        if selected
        else shared_warm
        if shared_warm is not None
        else problem.incumbent,
        "assessment": assessment,
        "trajectory_assessment": trajectory_assessment,
        "operations": operations,
        "setup": setups,
        "portfolio_triggered": portfolio["run"],
        "trial_count": (
            sum(
                o["operation"].startswith("trial-")
                and o["operation"].endswith("/rollout")
                for o in operations
            )
            if shared_warm is not None
            else len(starts)
            if portfolio["run"]
            else 0
        ),
        "verified_trial_count": len(trials),
        "continuation_run": continuation,
        "additional_wall_seconds": total("wall_seconds")
        + sum(s["wall_seconds"] for s in setups),
        "additional_cpu_seconds": total("cpu_seconds")
        + sum(s["cpu_seconds"] for s in setups),
        "additional_budget_charge_seconds": total("budget_charge_seconds")
        + sum(s["wall_seconds"] for s in setups),
        "cost_complete": all(r["accounting_complete"] for r in operations),
        "rollout_calls_observed": total("calls"),
        "validation_used_for_selection": False,
        "reference_values_used": False,
        **(
            {
                "after_incumbent_continuation": after_incumbent_continuation,
                "incumbent_continuation_run": incumbent_continuation,
                "search_calls_started": sum(
                    o["value"]["calls"] for o in search_operations
                )
                if search_accounted
                else None,
                "search_calls_completed": sum(
                    o["value"]["completed_calls"] for o in search_operations
                )
                if search_accounted
                else None,
                "search_call_accounting_complete": search_accounted,
            }
            if shared_warm is not None
            else {}
        ),
        **({"precision_decisions": precision_decisions} if measured else {}),
    }
    seal(finished, result)
    return result
