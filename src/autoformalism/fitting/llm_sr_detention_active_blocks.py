"""Temporarily optimize reliable coordinates without changing the full model.

The complete corrected-chart audit remains the evidence boundary. Holding an
unqualified coordinate is a search restriction, never a statement that its
value is identified or irrelevant. Every returned checkpoint still requires
independent original-equation verification by the campaign owner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from time import monotonic

import numpy as np

from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_recovery as previous
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import recovery_numerics as recovery

PROTOCOL = "llm-sr-detention-active-block-numerics-1"
MODE = "reduced_model15_consistent"
Policy = chart.Policy


def precision_policy(policy: Policy | None = None) -> Policy:
    """Tighten only the tight integration tolerances for a side diagnostic.

    The same chart, local physical steps, finite-difference stencil and checking
    thresholds are retained. The altered policy has a separate identity and
    cannot be mixed with standard-precision qualification records.
    """
    value = previous._policy(policy)
    numerical = value.numerical.model_copy(
        update={
            "tight_relative_tolerance": value.numerical.tight_relative_tolerance / 100,
            "tight_absolute_tolerance": value.numerical.tight_absolute_tolerance / 100,
        }
    )
    return previous._policy(value.model_copy(update={"numerical": numerical}))


def select_active(
    item,
    context,
    training,
    parameters: Mapping[str, float],
    *,
    qualification: dict,
    policy: Policy | None = None,
) -> tuple[str, ...]:
    """Authenticate a complete eight-coordinate audit and select qualified rows.

    A failed complete audit can support a partial block. Missing or mixed-point
    records cannot: each omitted coordinate must have an explicit audit result.
    This operation performs no rollout and does not relax qualification gates.
    """
    if qualification.get("precision") != "tight":
        raise ValueError("active blocks require tight complete-chart evidence")
    chart.validate_qualification(
        item,
        context,
        training,
        parameters,
        qualification.get("directions", []),
        mode=MODE,
        qualification=qualification,
        precision="tight",
        policy=policy,
    )
    names = qualification["coordinate_names"]
    rows = qualification["rows"]
    if (
        not qualification["complete"]
        or len(names) != 8
        or len(rows) != 8
        or [r["name"] for r in rows] != names
        or [r["index"] for r in rows] != list(range(8))
    ):
        raise ValueError("active blocks require all eight coordinate audit records")
    return tuple(row["name"] for row in rows if row["search_qualified"])


class ActiveOracle(chart.ChartOracle):
    """Evaluate only active derivative columns in one fixed full-coordinate chart.

    The search driver consumes the active column slice. Inactive matrix slots
    are NaN sentinels, not invented zero derivatives, and are never evaluated or
    included in active stationarity/scaling calculations. Full-coordinate
    histories preserve held values and the unchanged chart identity.
    """

    def __init__(self, *args, active_names: Sequence[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.active_indices = tuple(self.names.index(name) for name in active_names)
        self.held_indices = tuple(
            i for i in range(len(self.names)) if i not in self.active_indices
        )

    def _checked(self, vector):
        point = np.asarray(vector, dtype=float)
        if point.shape != self.chart.anchor_vector.shape or any(
            point[i] != self.chart.anchor_vector[i] for i in self.held_indices
        ):
            raise ValueError("temporarily held coordinate differs from incumbent")
        return point

    def parameters(self, vector):
        return super().parameters(self._checked(vector))

    def vector(self, parameters):
        return self._checked(super().vector(parameters))

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        point = self._checked(vector)
        residual = self.residual(point, deadline, method or "Radau")
        if not jacobian:
            return residual, None
        # Keep a useful finite rollout even when a subsequent derivative or
        # deadline fails. Qualification itself does not validate this new point.
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
                "origin": "active_search_base_rollout_before_derivative_completion",
                "independent_verification_required": True,
            }
            self.checkpoint({"best_rollout_checkpoint": self.best_residual})
        key = point.tobytes()
        if key in self.jacobian_cache:
            return residual, self.jacobian_cache[key].copy()
        self.cost["jacobian_calls"] += 1
        matrix = np.full((len(residual), len(self.names)), np.nan)
        for index in self.active_indices:
            matrix[:, index] = previous._column(self, point, index, deadline)[1]
        self.jacobian_cache[key] = matrix.copy()
        if len(self.jacobian_cache) > 8:
            self.jacobian_cache.popitem(last=False)
        return residual, matrix


def fit_active(
    item,
    context,
    training,
    parameters: Mapping[str, float],
    *,
    qualification: dict,
    active_names: Sequence[str],
    seconds: float = 600,
    checkpoint: Callable[[dict], None] | None = None,
    policy: Policy | None = None,
) -> dict:
    """Run a bounded qualified block, retaining full original parameter vectors.

    The caller owns a durable nonrenewable allowance and a recheck/release
    schedule. This primitive does not automatically reset budgets or reuse old
    derivative evidence at the resulting endpoint.
    """
    began, cpu, deadline = original._clock(seconds)
    policy = chart.effective_policy(policy, mode=MODE)
    eligible = select_active(
        item, context, training, parameters, qualification=qualification, policy=policy
    )
    names = tuple(qualification["coordinate_names"])
    if isinstance(active_names, (str, bytes)) or any(
        not isinstance(n, str) for n in active_names
    ):
        raise ValueError("active names must be a coordinate-name sequence")
    selected = tuple(active_names)
    if (
        len(set(selected)) != len(selected)
        or selected != tuple(n for n in names if n in selected)
        or any(n not in eligible for n in selected)
    ):
        raise ValueError("active coordinates must be distinct qualified chart rows")
    held = tuple(n for n in names if n not in selected)
    coordinates = dict(
        zip(
            names,
            chart.CoordinateChart(
                original.domain_policy(item, context, training), parameters, MODE
            ).anchor_vector.tolist(),
            strict=True,
        )
    )
    binding = {
        "protocol": PROTOCOL,
        "mode": MODE,
        "qualification_sha256": qualification["qualification_sha256"],
        "active_names": list(selected),
        "held_names": list(held),
        "held_chart_values": {n: coordinates[n] for n in held},
        "search_covers_full_chart": not held,
        "temporary_coordinate_restriction": bool(held),
        "active_coordinate_selection": (
            "authenticated_complete_chart_search_qualification"
        ),
        "stationarity_scope": "active_coordinates_only",
        "full_model_convergence_claimed": False,
        "parameter_information_qualified": False,
        "global_convergence_certified": False,
        "independent_verification_required": True,
        "validation_or_test_access": False,
    }
    if not selected:
        return binding | {
            "stop_reason": "no_qualified_coordinates",
            "calls": 0,
            "best": None,
        }
    state = {}

    def save(value):
        state.update(value)
        candidate = value.get("best_rollout_checkpoint") or (
            value.get("search") or {}
        ).get("best")
        if candidate and (
            not state.get("best")
            or candidate["training_nmse"] < state["best"]["training_nmse"]
        ):
            state["best"] = candidate
        if checkpoint:
            checkpoint(deepcopy(state | binding))

    oracle = ActiveOracle(
        item,
        context,
        training,
        parameters,
        MODE,
        policy,
        "tight",
        save,
        active_names=selected,
    )
    oracle.retain_base_checkpoints = True
    try:
        result = recovery.search(
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
