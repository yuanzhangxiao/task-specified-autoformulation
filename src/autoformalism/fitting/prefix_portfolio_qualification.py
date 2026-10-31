"""Bounded, independently journaled recovery of unchanged nonlinear starts.

The caller journals each stiff-solver replay and each coordinate separately.
This module never searches, changes a start, loosens a threshold, or diagnoses
a scientific cause from a failed integration. Historical portfolio paths remain
available with their original settings.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import numpy as np

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "portfolio-nonlinear-start-qualification-1"
METHODS = ("Radau", "BDF")


def numerical_policy() -> kernel.NumericalPolicy:
    """Pin the tighter replay settings independently of historical defaults."""
    return kernel.NumericalPolicy(rtol=1e-11, atol=1e-13)


def policy_receipt() -> dict:
    """Expose solver settings and unchanged local qualification thresholds."""
    return {
        "protocol": PROTOCOL,
        "methods": list(METHODS),
        "qualification_method": "Radau",
        "numerical_policy": numerical_policy().model_dump(mode="json"),
        "agreement_tolerance": portfolio.PAIRING_TOLERANCE,
        "derivative_policy": portfolio._policy().model_dump(mode="json"),
        "each_replay_and_coordinate_has_separate_allowance": True,
        "no_search_performed": True,
        "no_new_starts": True,
    }


def verify_start(
    problem: PrefixFitProblem,
    training: PublicSplit,
    method: str,
    seconds: float,
    checkpoint: portfolio.Checkpoint,
) -> dict:
    """Attempt exactly one state-only original-equation replay at its own cost."""
    if method not in METHODS:
        raise ValueError("nonlinear recovery requires Radau or BDF")
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {
        "protocol": PROTOCOL,
        "problem_sha256": public.content_sha256(problem),
        "training_sha256": public.content_sha256(training),
        "method": method,
        "numerical_policy": numerical_policy().model_dump(mode="json"),
        "state_only": True,
        "execution_policy": budget.policy.model_dump(mode="json"),
        "original_equations_verified": False,
        "replay_completed": False,
        "calls_started": 0,
        "calls_completed": 0,
        "failure_cause": "not_established",
        "no_search_performed": True,
    }
    interrupted = None
    try:
        oracle = kernel.TrainingOracle(problem, training, numerical_policy())
        point = oracle.vector(oracle.start)
        result["parameters_sha256"] = public.content_sha256(oracle.start)
        budget.callback(lambda: checkpoint(deepcopy(result)))

        def evaluate(deadline):
            result["calls_started"] += 1
            residual, _ = oracle.evaluate(
                point, deadline, jacobian=False, method=method
            )
            result["calls_completed"] += 1
            return residual

        residual = np.asarray(budget.numerical(evaluate), dtype=float)
        if residual.ndim != 1 or not residual.size or not np.isfinite(residual).all():
            raise ValueError("nonfinite or malformed replay residual")
        loss = float(np.mean(residual**2))
        if not np.isfinite(loss):
            raise ValueError("nonfinite replay loss")
        result.update(
            replay_completed=True,
            residual=residual.tolist(),
            residual_sha256=adequacy._digest(residual),
            training_nmse=loss,
        )
    except Exception as error:
        continuation._invalidate(
            result, error, "verification", preserve_diagnostic=True
        )
    except BaseException as error:
        interrupted = error
        continuation._invalidate(
            result, error, "verification", preserve_diagnostic=True
        )
    result = continuation._finish(
        result,
        budget,
        checkpoint,
        "verification",
        interrupted,
        preserve_diagnostic=True,
    )
    if result["status"] != "complete":
        result["replay_completed"] = False
    return result


class FixedRadauQualifier:
    """Reuse a paired start Jacobian lazily, with all probes fixed to Radau.

    The cache is process-local and holds only the unchanged supplied start;
    restart recomputes it within the next coordinate's reserved allowance.
    """

    def __init__(self, oracle: portfolio.PortfolioOracle):
        self.oracle = oracle
        self._paired_start = None
        self._paired_identity = None
        self.paired_start_cache_hits = 0
        self.failure_diagnostics: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def execution_receipt(self) -> dict:
        """Report paired-start reuse separately from underlying solve counts."""
        return self.oracle.execution_receipt() | {
            "paired_start_cache_hits": self.paired_start_cache_hits,
        }

    def evaluate(self, vector, deadline, *, jacobian=True):
        portfolio._check(deadline)
        at_start = np.array_equal(vector, self.vector(self.start))
        identity = (vector.tobytes(), public.content_sha256(self.original.settings))
        if self._paired_identity != identity and at_start:
            self._paired_start = None
        if at_start and self._paired_start is not None:
            self.paired_start_cache_hits += 1
            residual, derivative = self._paired_start
            return residual.copy(), derivative.copy() if jacobian else None
        try:
            result = self.oracle.evaluate(
                vector, deadline, jacobian=jacobian, method="Radau"
            )
        except Exception as error:
            diagnostic = getattr(error, "diagnostic", None)
            if isinstance(diagnostic, dict) and len(self.failure_diagnostics) < 8:
                self.failure_diagnostics.append(
                    continuation._bounded_diagnostic(diagnostic)
                )
            raise
        if at_start and jacobian:
            self._paired_start = tuple(value.copy() for value in result)
            self._paired_identity = identity
        return result


def make_qualifier(
    problem: PrefixFitProblem, training: PublicSplit
) -> FixedRadauQualifier:
    """Preserve the original certified chart while fixing all solves to Radau."""
    original, oracle = portfolio.make_oracle(problem, training, numerical_policy())
    if original.portfolio_audit["route"] != "original_ODE":
        raise ValueError("nonlinear recovery requires an original ODE route")
    return FixedRadauQualifier(oracle)


def _identity(oracle: FixedRadauQualifier) -> dict:
    """Bind all coordinate receipts to one immutable chart, start and precision."""
    return {
        "protocol": PROTOCOL,
        "portfolio_audit_sha256": public.content_sha256(oracle.portfolio_audit),
        "parameters_sha256": public.content_sha256(oracle.start),
        "coordinate_names": list(oracle.names),
        "lower_bounds_sha256": adequacy._digest(oracle.lower),
        "upper_bounds_sha256": adequacy._digest(oracle.upper),
        "solver_settings_sha256": public.content_sha256(oracle.original.settings),
        "policy": policy_receipt(),
    }


def qualify_coordinate(
    oracle: FixedRadauQualifier,
    index: int,
    seconds: float,
    checkpoint: portfolio.Checkpoint,
) -> dict:
    """Audit one active coordinate; omissions cannot qualify the full vector."""
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(oracle.names)
        or oracle.lower[index] >= oracle.upper[index]
    ):
        raise ValueError("coordinate must be an active index in the frozen chart")
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {
        "identity": _identity(oracle),
        "index": index,
        "name": oracle.names[index],
        "execution_policy": budget.policy.model_dump(mode="json"),
        "passed": False,
        "no_search_performed": True,
    }
    oracle.failure_diagnostics = []
    before = deepcopy(oracle.counts)
    interrupted = None
    try:
        point = oracle.vector(oracle.start)
        budget.callback(lambda: checkpoint(deepcopy(result)))
        check = budget.numerical(
            lambda deadline: adequacy.derivative_check(
                oracle.evaluate,
                point,
                oracle.lower,
                oracle.upper,
                deadline,
                coordinate_units=oracle.units,
                policy=portfolio._policy(),
                directions=[index],
                residual_only=lambda x, d: oracle.evaluate(x, d, jacobian=False)[0],
                local_coordinate_scales=oracle.local_coordinate_scales(point),
            )
        )
        result["derivative_check"] = public._jsonable(check)
        if oracle._paired_start is not None:
            result["base_residual"] = oracle._paired_start[0].tolist()
        result["passed"] = bool(
            check.error is None
            and check.checked_indices == (index,)
            and all(row.precision_status == "passed" for row in check.directions)
        )
    except Exception as error:
        continuation._invalidate(
            result, error, "qualification", preserve_diagnostic=True
        )
    except BaseException as error:
        interrupted = error
        continuation._invalidate(
            result, error, "qualification", preserve_diagnostic=True
        )
    result.update(
        failure_diagnostics=deepcopy(oracle.failure_diagnostics),
        stage_backend_calls={
            key: value - before[key] for key, value in oracle.counts.items()
        },
        portfolio_execution=oracle.execution_receipt(),
    )
    return continuation._finish(
        result,
        budget,
        checkpoint,
        "qualification",
        interrupted,
        preserve_diagnostic=True,
    )


def aggregate_qualification(
    oracle: FixedRadauQualifier, receipts: Iterable[dict]
) -> dict:
    """Require every active coordinate with matching finite local evidence."""
    required = tuple(int(i) for i in np.flatnonzero(oracle.lower < oracle.upper))
    point = oracle.vector(oracle.start)
    indexed, rows, baselines = {}, {}, set()
    identity = _identity(oracle)
    for receipt in receipts:
        index = receipt.get("index")
        if (
            type(index) is not int
            or index not in required
            or receipt.get("identity") != identity
            or receipt.get("name") != oracle.names[index]
        ):
            raise ValueError("coordinate receipt identity differs")
        if index in indexed:
            raise ValueError("duplicate coordinate receipt")
        indexed[index] = receipt
        if "derivative_check" not in receipt:
            continue
        check = adequacy.DerivativeCheck.model_validate(receipt["derivative_check"])
        if (
            check.policy != portfolio._policy()
            or check.reduced_vector != tuple(point)
            or check.coordinate_units != tuple(oracle.units)
            or check.required_indices != required
            or check.lower_bounds_sha256 != identity["lower_bounds_sha256"]
            or check.upper_bounds_sha256 != identity["upper_bounds_sha256"]
            or check.local_coordinate_scales
            != tuple(oracle.local_coordinate_scales(point))
            or any(row.index != index for row in check.directions)
            or any(row.index != index for row in check.local_probe_attempts)
        ):
            raise ValueError("coordinate audit differs from frozen chart or policy")
        if receipt.get("status") != "complete" or check.error:
            continue
        if check.checked_indices != (index,) or len(check.directions) != 1:
            continue
        residual = np.asarray(receipt.get("base_residual"), dtype=float)
        if (
            residual.ndim != 1
            or not residual.size
            or not np.isfinite(residual).all()
            or adequacy._digest(residual) != check.base_residual_sha256
            or check.base_jacobian_sha256 is None
        ):
            raise ValueError("coordinate baseline differs from audit")
        baselines.add((check.base_residual_sha256, check.base_jacobian_sha256))
        if len(baselines) != 1:
            raise ValueError("coordinate baseline or Jacobian hashes differ")
        rows[index] = check.directions[0]
    missing = [index for index in required if index not in rows]
    failed = [
        index for index, row in sorted(rows.items()) if row.precision_status == "failed"
    ]
    inconclusive = [
        index for index, row in sorted(rows.items()) if row.precision_status != "passed"
    ]
    complete = not missing
    search_qualified = bool(
        complete
        and not failed
        and all(
            row.precision_status == "passed"
            or (
                row.precision_status == "inconclusive"
                and row.gradient_error_bound is not None
                and row.gradient_error_bound
                <= portfolio._policy().gradient_tolerance
                * portfolio._policy().precision_gradient_fraction
            )
            for row in rows.values()
        )
    )
    return {
        "identity": identity,
        "status": "failed" if failed else "complete" if complete else "incomplete",
        "complete": complete,
        "required_indices": list(required),
        "checked_indices": sorted(rows),
        "missing_indices": missing,
        "failed_indices": failed,
        "inconclusive_indices": inconclusive,
        "directions": [public._jsonable(row) for _, row in sorted(rows.items())],
        "search_qualified": search_qualified,
        "passed": complete and not inconclusive,
        "parameter_information_qualified": False,
        "no_search_performed": True,
        "scope": (
            "Local derivative evidence only; no convergence or scientific-cause claim."
        ),
    }
