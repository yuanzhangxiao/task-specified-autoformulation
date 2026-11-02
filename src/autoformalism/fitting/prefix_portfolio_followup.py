"""Receipt-preserving precision recovery and explicit numerical call accounting."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_portfolio as numerical
from autoformalism.fitting import prefix_portfolio_nonlinear_search as nonlinear
from autoformalism.fitting import public_fitting as public


def counted(work: Callable, checkpoint: Callable, original, oracle) -> dict:
    """Record requests by oracle view, including cached and failed evaluations.

    Views may call one another. Their counts therefore remain separate and are
    never summed into an invented number of integration solves. Instrumentation
    is task-local and restored even on interruption; no shared class is patched.
    """
    counts, saved = {}, []
    for label, obj in (("original", original), ("optimizer", oracle)):
        if obj is None or any(obj is row[0] for row in saved):
            continue
        method = obj.evaluate
        owned = "evaluate" in vars(obj)
        old = vars(obj).get("evaluate")
        row = dict.fromkeys(("started", "completed", "failed", "jacobian"), 0)
        counts[label] = row

        def wrapped(*args, _evaluate=method, _counts=row, **kwargs):
            _counts["started"] += 1
            _counts["jacobian"] += bool(kwargs.get("jacobian", True))
            try:
                result = _evaluate(*args, **kwargs)
            except BaseException:
                _counts["failed"] += 1
                raise
            _counts["completed"] += 1
            return result

        saved.append((obj, owned, old))
        obj.evaluate = wrapped

    def enrich(value):
        return value | {
            "evaluation_requests_by_view": deepcopy(counts),
            "evaluation_count_scope": "Logical evaluate requests, including cache "
            "hits; views can be nested and must not be summed as physical solves.",
        }

    try:
        return enrich(work(lambda value: checkpoint(enrich(value))))
    except BaseException as error:
        for name in ("execution_receipt", "verification_receipt"):
            value = getattr(error, name, None)
            if isinstance(value, dict):
                setattr(error, name, enrich(value))
        raise
    finally:
        for obj, owned, old in reversed(saved):
            if owned:
                obj.evaluate = old
            else:
                del obj.evaluate


def normalise_counts(value: dict, *, search: bool = False) -> dict:
    """Add prospective aliases without rewriting any historical receipt."""
    result = deepcopy(value)
    prefix = "search" if search else "verification"
    for suffix in ("attempted", "started", "completed", "failed"):
        key = prefix + "_calls_" + suffix
        if key not in result and "calls_" + suffix in result:
            result[key] = result["calls_" + suffix]
    return result


def reuse_precision_probe(oracle, source: dict, key: str, parameters: dict):
    """Reuse only a completed exact-vector tighter-precision primitive."""
    op = source.get(key) or {}
    value = op.get("value") or {}
    if op.get("status") != "complete" or not value.get("passed"):
        return None
    if (
        value.get("identity") != nonlinear._precision_identity(oracle)
        or value.get("kind") != "residual_probe"
        or value.get("method") != nonlinear.METHOD
        or value.get("vector") != parameters
        or value.get("vector_sha256") != public.content_sha256(parameters)
        or value.get("status") != "complete"
    ):
        raise ValueError("source precision probe differs from requested identity")
    nonlinear.prerequisites._array(
        value.get("residual"), digest=value.get("residual_sha256")
    )
    return deepcopy(value)


def meal_precision(oracle, parameters, name, seconds, checkpoint):
    """Smaller *local* steps for a declared meal-rate coordinate; no rank claim."""
    if name not in oracle.names:
        raise ValueError("meal-rate coordinate absent from chart")
    index = oracle.names.index(name)
    budget = execution._budget(seconds, seconds + 60)
    policy = numerical._policy().model_copy(
        update={
            "derivative_relative_step": 1e-5,
            "derivative_absolute_step": 1e-7,
            "local_relative_step_cap": 1e-5,
        }
    )
    point = oracle.vector(parameters)
    result = {
        "parameters_sha256": public.content_sha256(parameters),
        "portfolio_audit_sha256": public.content_sha256(oracle.portfolio_audit),
        "coordinate": name,
        "policy": policy.model_dump(mode="json"),
        "partial_diagnostic_only": True,
        "full_search_qualification": False,
        "passed": False,
    }
    interrupted = None
    try:
        budget.callback(lambda: checkpoint(deepcopy(result)))
        residual, jacobian = budget.numerical(lambda d: oracle.evaluate(point, d))

        def evaluate(vector, deadline):
            numerical._check(deadline)
            if not np.array_equal(vector, point):
                raise ValueError("only the saved meal-direction Jacobian is available")
            return residual.copy(), jacobian.copy()

        def residual_only(vector, deadline):
            if np.array_equal(vector, point):
                return residual.copy()
            return oracle.evaluate(vector, deadline, jacobian=False)[0]

        check = budget.numerical(
            lambda d: adequacy.derivative_check(
                evaluate,
                point,
                oracle.lower,
                oracle.upper,
                d,
                coordinate_units=oracle.units,
                policy=policy,
                directions=[index],
                residual_only=residual_only,
                local_coordinate_scales=oracle.local_coordinate_scales(point),
            )
        )
        result.update(
            derivative_check=public._jsonable(check),
            passed=check.error is None
            and check.checked_indices == (index,)
            and all(row.precision_status == "passed" for row in check.directions),
        )
    except Exception as error:
        continuation._invalidate(result, error, "qualification")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "qualification")
    return continuation._finish(
        result,
        budget,
        checkpoint,
        "qualification",
        interrupted,
        preserve_diagnostic=True,
    )
