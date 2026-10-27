"""Matched eager/lazy controls on authenticated full or restricted charts.

Both schedules use the same original-equation residual, coordinate units,
constraints and active derivative columns. A held direction needs an explicit
full-audit record, but need not qualify for the restricted search. Search
stationarity never certifies the full model or its parameter information.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from time import monotonic
from typing import Literal

import numpy as np

from autoformalism.fitting import lazy_rollout_search as backend
from autoformalism.fitting import llm_sr_detention_active_blocks as active
from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_refit as original

PROTOCOL = "llm-sr-detention-lazy-control-numerics-1"
Mode = Literal["original", "reduced_model15_consistent"]
Policy = chart.Policy


def select_active(
    item,
    context,
    training,
    parameters: Mapping[str, float],
    *,
    mode: Mode = "original",
    qualification: dict,
    policy: Policy | None = None,
) -> tuple[str, ...]:
    """Authenticate every full-chart row and return its search-qualified names."""
    if mode not in ("original", "reduced_model15_consistent"):
        raise ValueError("lazy controls require original or consistent reduced chart")
    if not isinstance(qualification, dict) or qualification.get("precision") != "tight":
        raise ValueError("lazy controls require tight complete-chart evidence")
    try:
        chart.validate_qualification(
            item,
            context,
            training,
            parameters,
            qualification.get("directions", []),
            mode=mode,
            qualification=qualification,
            precision="tight",
            policy=policy,
        )
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("malformed full-chart qualification") from error
    names, rows = qualification["coordinate_names"], qualification["rows"]
    if (
        not qualification["complete"]
        or len(rows) != len(names)
        or [row["name"] for row in rows] != names
        or [row["index"] for row in rows] != list(range(len(names)))
    ):
        raise ValueError(
            "lazy controls require all full-chart coordinate audit records"
        )
    return tuple(row["name"] for row in rows if row["search_qualified"])


class ControlOracle(active.ActiveOracle):
    """The identical active oracle for both schedules, retaining finite trials.

    The inherited Jacobian computes only selected columns and leaves held
    columns as NaN sentinels. Saving a residual-only optimizer trial here also
    protects lazy improvements if its subsequently requested derivative fails.
    Finite-difference probes call ``residual`` directly and cannot become best.
    """

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        point = self._checked(vector)
        residual, _ = super().evaluate(point, deadline, jacobian=False, method=method)
        with np.errstate(over="ignore", invalid="ignore"):
            loss = float(np.mean(residual**2))
        if (
            self.retain_base_checkpoints
            and np.isfinite(loss)
            and (
                self.best_residual is None or loss < self.best_residual["training_nmse"]
            )
        ):
            self.best_residual = {
                "parameters": self.parameters(point),
                "training_nmse": loss,
                "origin": "control_search_base_rollout_before_derivative_completion",
                "independent_verification_required": True,
            }
            self.checkpoint({"best_rollout_checkpoint": self.best_residual})
        if not jacobian:
            return residual, None
        return super().evaluate(point, deadline, jacobian=True, method=method)


def fit_control(
    item,
    context,
    training,
    parameters: Mapping[str, float],
    *,
    mode: Mode = "original",
    lazy_jacobian: bool,
    active_names: Sequence[str] | None = None,
    qualification: dict,
    seconds: float = 1800,
    checkpoint: Callable[[dict], None] | None = None,
    policy: Policy | None = None,
) -> dict:
    """Run one bounded matched control, preserving the original full vectors.

    ``None`` requires all chart directions to qualify. An explicit subset must
    follow chart order and contain only qualified directions; held directions
    retain their exact incumbent chart values. The caller owns the durable
    allowance, any later requalification, and independent endpoint verification.
    """
    began, cpu, deadline = original._clock(seconds)
    if not isinstance(lazy_jacobian, bool):
        raise ValueError("lazy_jacobian must be boolean")
    policy = chart.effective_policy(policy, mode=mode)
    eligible = select_active(
        item,
        context,
        training,
        parameters,
        mode=mode,
        qualification=qualification,
        policy=policy,
    )
    names = tuple(qualification["coordinate_names"])
    if active_names is None:
        selected = names
    else:
        if (
            not isinstance(active_names, Sequence)
            or isinstance(active_names, (str, bytes))
            or any(not isinstance(name, str) for name in active_names)
        ):
            raise ValueError("active names must be a coordinate-name sequence")
        selected = tuple(active_names)
        if (
            len(set(selected)) != len(selected)
            or selected != tuple(name for name in names if name in selected)
            or any(name not in eligible for name in selected)
        ):
            raise ValueError("active coordinates must be distinct qualified chart rows")
    held = tuple(name for name in names if name not in selected)
    coordinates = chart.CoordinateChart(
        original.domain_policy(item, context, training), parameters, mode
    )
    anchor = dict(zip(names, coordinates.anchor_vector.tolist(), strict=True))
    binding = {
        "protocol": PROTOCOL,
        "mode": mode,
        "lazy_jacobian": lazy_jacobian,
        "evaluation_policy": (
            "residual_only_trials" if lazy_jacobian else "eager_trials"
        ),
        "qualification_sha256": qualification["qualification_sha256"],
        "active_names": list(selected),
        "held_names": list(held),
        "held_chart_values": {name: anchor[name] for name in held},
        "search_covers_full_chart": not held,
        "temporary_coordinate_restriction": bool(held),
        "active_coordinate_selection": "authenticated_complete_chart_qualification",
        "stationarity_scope": "active_coordinates_only",
        "full_model_convergence_claimed": False,
        "parameter_information_qualified": False,
        "parameter_rank_qualified": False,
        "global_convergence_certified": False,
        "independent_verification_required": True,
        "validation_or_test_access": False,
    }
    state = {}

    def save(value: dict) -> None:
        """Keep the best complete rollout independently of Jacobian completion."""
        state.update(value)
        candidate = value.get("best_rollout_checkpoint") or (
            value.get("search") or {}
        ).get("best")
        if candidate and (
            not state.get("best")
            or candidate["training_nmse"] < state["best"]["training_nmse"]
        ):
            state["best"] = deepcopy(candidate)
        if checkpoint:
            checkpoint(deepcopy(state | binding))

    oracle = ControlOracle(
        item,
        context,
        training,
        parameters,
        mode,
        policy,
        "tight",
        save,
        active_names=selected,
    )
    oracle.retain_base_checkpoints = True
    if not selected or (active_names is None and eligible != names):
        result = {
            "stop_reason": "no_qualified_coordinates"
            if not selected
            else "derivatives_unqualified",
            "calls": 0,
            "best": None,
        }
    else:
        try:
            result = backend.search(
                oracle,
                parameters,
                deadline,
                lambda value: save({"search": value}),
                calls=policy.numerical.maximum_calls,
                target=policy.numerical.target_nmse,
                optimizer_scaling="guarded_relative",
                scaling_policy=policy.numerical.scaling,
                normalized_objective=True,
                poor_fit=policy.numerical.stopping,
                reject_invalid_trials=True,
                telemetry=True,
                record_steps=True,
                record_full_steps=True,
                residual_only_trials=lazy_jacobian,
            )
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
            result = {
                "best": state.get("best"),
                "stop_reason": "budget_limited"
                if isinstance(error, TimeoutError)
                else "numerical_failure",
                "error": f"{type(error).__name__}: {error}",
            }
    if state.get("best") and (
        not result.get("best")
        or state["best"]["training_nmse"] < result["best"]["training_nmse"]
    ):
        result["best"] = state["best"]
    if result.get("best"):
        result["best"] = result["best"] | {"independent_verification_required": True}
    if "stationarity" in result:
        result["active_block_stationarity"] = result.pop("stationarity")
    if result.get("stop_reason") == "stationary_poor_fit":
        result["stop_reason"] = "active_block_stationary_poor_fit"
    return (
        result
        | original._accounting(oracle, began, cpu, "search")
        | binding
        | {
            "chart": oracle.chart.domain,
            "integration_settings": oracle.settings.model_dump(mode="json"),
            "numerical_cost": dict(oracle.cost),
            "wall_seconds": monotonic() - began,
            "budget_exhausted": monotonic() >= deadline,
        }
    )
