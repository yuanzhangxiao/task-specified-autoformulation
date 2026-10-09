"""M23 training-only allocation: protect rollout and fund complete screening."""

from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting.affine_propagation import numerically_verified
from autoformalism.fitting.trajectory_profile import TrajectoryProblem
from autoformalism.fitting.trajectory_profile_fit import ConditionalEngine, optimize


def precision_reason(point: dict | None, incumbent: dict | None, policy) -> str:
    """Extreme precision is for unreliable outputs, near-target fits or ties."""
    if not point:
        return "ordinary_verification_unavailable"
    if not numerical.usable(point):
        return "ordinary_numerics_unreliable"
    if numerically_verified(point, policy.target_nmse) and (
        not incumbent or numerically_verified(incumbent, policy.target_nmse)
    ):
        return "already_strictly_verified"
    upper = max(point["training_nmse"], point["alternate_training_nmse"])
    if (
        upper <= policy.precision_near_factor * policy.target_nmse
        and point["maximum_trajectory_nmse"]
        <= policy.precision_near_factor * policy.trajectory_nmse
    ):
        return "near_accuracy_target"
    if (
        numerical.usable(incumbent)
        and point["parameters"] != incumbent["parameters"]
        and not numerical.improves(point, incumbent)
        and not numerical.improves(incumbent, point)
    ):
        return "retention_interval_overlap"
    return "ordinary_precision_sufficient_for_search"


def screening_plan(seconds: float, measured: float | None, policy) -> dict:
    """Reserve one whole measured rollout before spending on node optimization."""
    if measured is None or not np.isfinite(measured) or measured <= 0:
        return {"status": "budget_skip", "reason": "complete_rollout_cost_unknown"}
    reserve = max(1.0, policy.screening_safety_factor * measured)
    if seconds < reserve + policy.minimum_node_seconds:
        return {
            "status": "budget_skip",
            "reason": "insufficient_solve_and_screen_allowance",
            "screen_reserve_seconds": reserve,
        }
    return {
        "status": "funded",
        "screen_reserve_seconds": reserve,
        "screen_minimum_seconds": max(1.0, measured),
        "node_seconds": seconds - reserve,
    }


class MeasuredConditionalEngine(ConditionalEngine):
    """Explore conditional restarts without repeatedly timing the control vector."""

    screen_cost: float | None = None

    def propose(self, start, deadline, checkpoint, directory: Path):
        begun = monotonic()
        allocation = screening_plan(deadline - begun, self.screen_cost, self.policy)
        records, proposals, candidates, costs = [], [], [], []
        calls = 0
        node_seconds = 0.0
        if allocation["status"] == "funded":
            reserve = allocation["screen_reserve_seconds"]
            node_end = deadline - reserve
            previous, current = None, start
            for level, (target, penalty) in enumerate(
                zip(self.policy.node_targets, self.policy.penalties, strict=True)
            ):
                remaining = node_end - monotonic()
                if remaining < self.policy.minimum_node_seconds:
                    break
                levels_left = len(self.policy.node_targets) - level
                # Do not split a meaningful single solve into tiny mesh attempts.
                count = min(
                    levels_left, int(remaining / self.policy.minimum_node_seconds)
                )
                level_end = monotonic() + remaining / count
                reused, begun_level = level in self.cache, monotonic()
                try:
                    if not reused:
                        self.cache[level] = TrajectoryProblem(
                            self.oracle,
                            self.problem,
                            target,
                            self.policy.minimum_intervals,
                            self.policy.observation_anchors,
                            level_end,
                        )
                    graph_seconds = monotonic() - begun_level
                    native = optimize(
                        self.cache[level],
                        current,
                        penalty,
                        level_end,
                        checkpoint,
                        directory / f"mesh-{level}",
                        f"{directory.name}-mesh-{level}",
                        previous=previous,
                        cycles=self.policy.cycles,
                        block_iterations=self.policy.block_iterations,
                    )
                    previous = native.pop("trajectories")
                    current = native["parameters"]
                    proposals.append((level, current))
                    records.append(
                        {
                            "level": level,
                            "native": native,
                            "graph_reused": reused,
                            "graph_seconds": graph_seconds,
                            "seconds": monotonic() - begun_level,
                        }
                    )
                except (
                    ValueError,
                    RuntimeError,
                    ArithmeticError,
                    TimeoutError,
                ) as error:
                    records.append({"level": level, "error": str(error)[-700:]})
                    break
                checkpoint({"conditional_levels": records})
            node_seconds = monotonic() - begun
            # Finest endpoint first. Each receives a whole reserve, never a
            # fraction of the remaining time divided by untested candidates.
            for level, parameters in reversed(proposals):
                # The reserve includes a safety margin for cooperative solver
                # overrun. Do not reject it because of a millisecond of overhead.
                available = deadline - monotonic()
                if available < allocation["screen_minimum_seconds"]:
                    records.append({"screen_skip": level, "reason": "budget"})
                    continue
                records.append({"screen_point": level, "allowance_seconds": available})
                calls += 1
                checkpoint({"calls": calls})
                screened = monotonic()
                try:
                    r, _ = self.oracle.evaluate(
                        self.oracle.vector(parameters), deadline, jacobian=False
                    )
                    loss = float(np.mean(r**2))
                    if not np.isfinite(loss):
                        raise ValueError("nonfinite conditional screening loss")
                    costs.append(monotonic() - screened)
                    candidates.append(
                        {
                            "parameters": parameters,
                            "training_nmse": loss,
                            "origin": f"mesh-{level}",
                        }
                    )
                except (
                    ValueError,
                    RuntimeError,
                    ArithmeticError,
                    TimeoutError,
                ) as error:
                    records.append({"rollout_point": level, "error": str(error)[-700:]})
                checkpoint({"rollout_candidates": candidates})
        best = min(candidates, key=lambda c: c["training_nmse"]) if candidates else None
        return {
            "parameters": best["parameters"] if best else start,
            "selected": best,
            "candidates": candidates,
            "levels": records,
            "calls": calls,
            "seconds": monotonic() - begun,
            "node_seconds": node_seconds,
            "screening_seconds": max(0.0, monotonic() - begun - node_seconds),
            "complete_screen_seconds": costs,
            "allocation": allocation,
            "original_parameters": start,
            "control_rollout_repeated": False,
            "selection": "best_screened_conditional_restart_else_original",
            "incumbent_replacement_authorized": False,
            "independent_verification_pending": True,
        }
