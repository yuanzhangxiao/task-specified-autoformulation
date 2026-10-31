"""Journalable BDF prerequisites and local probes for unchanged nonlinear starts.

The owning campaign reserves import, Jacobian and coordinate operations once.
Failed prerequisites block coordinates without another solve. Optional probe
journals are children of a coordinate: their time is included in that parent's
allowance and must not be summed a second time. Source replay costs are inherited
evidence, never charged as new numerical calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from time import monotonic

import numpy as np

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_qualification as previous
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "portfolio-bdf-prerequisites-1"
METHOD = "BDF"
PROBE_SECONDS = 90.0
ProbeRunner = Callable[[str, dict, Callable], dict]


def policy_receipt() -> dict:
    """Record fixed precision and separate prerequisite/probe budgets."""
    return previous.policy_receipt() | {
        "protocol": PROTOCOL,
        "qualification_method": METHOD,
        "baseline_source": "authenticated independent original Radau and BDF replays",
        "sensitivity_prerequisite": "one original-equation BDF full Jacobian",
        "probe_seconds": PROBE_SECONDS,
        "recommended_coordinate_seconds": 240.0,
        "prerequisite_failure_blocks_dependents": True,
        "nested_probe_costs_included_in_coordinate": True,
    }


def make_oracle(problem: PrefixFitProblem, training: PublicSplit):
    """Keep the historical certified chart and tighten only solver precision."""
    original, oracle = portfolio.make_oracle(
        problem, training, previous.numerical_policy()
    )
    if original.portfolio_audit["route"] != "original_ODE":
        raise ValueError("BDF prerequisite recovery requires an original ODE route")
    return oracle


def _identity(oracle) -> dict:
    """Bind primitives to the unchanged data, equations, chart and local policy."""
    settings = public._jsonable(oracle.original.settings) | {
        "integration_method": METHOD
    }
    return previous._identity(oracle) | {
        "protocol": PROTOCOL,
        "policy": policy_receipt(),
        "solver_settings_sha256": public.content_sha256(settings),
    }


def _base(oracle, kind: str, budget) -> dict:
    """Normalize new call counters while retaining zero-cost cache reuse."""
    return {
        "identity": _identity(oracle),
        "kind": kind,
        "method": METHOD,
        "passed": False,
        "no_search_performed": True,
        "verification_calls_attempted": 0,
        "verification_calls_started": 0,
        "verification_calls_completed": 0,
        "verification_calls_failed": 0,
        "execution_policy": budget.policy.model_dump(mode="json"),
    }


def _array(value, *, shape=None, digest=None) -> np.ndarray:
    """Reject malformed, nonfinite or unauthenticated primitive arrays."""
    array = np.asarray(value, dtype=float)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("primitive array must be nonempty and finite")
    if shape is not None and array.shape != shape:
        raise ValueError("primitive array shape differs")
    if digest is None or adequacy._digest(array) != digest:
        raise ValueError("primitive array digest differs")
    return array


def _read_replay(oracle, sealed: Mapping, method: str) -> tuple[dict, np.ndarray]:
    """Authenticate a complete original-start replay before importing its values."""
    operation = sealed.get("value")
    if not isinstance(operation, dict) or public.content_sha256(
        operation
    ) != sealed.get("sha256"):
        raise ValueError("source replay operation seal differs")
    value = operation.get("value") or {}
    audit = oracle.portfolio_audit
    expected = {
        "protocol": previous.PROTOCOL,
        "problem_sha256": audit["problem_sha256"],
        "training_sha256": audit["training_sha256"],
        "parameters_sha256": public.content_sha256(oracle.original.start),
        "numerical_policy": previous.numerical_policy().model_dump(mode="json"),
        "method": method,
        "state_only": True,
        "status": "complete",
        "replay_completed": True,
        "calls_started": 1,
        "calls_completed": 1,
    }
    if (
        operation.get("status") != "complete"
        or value.get("error")
        or any(value.get(key) != wanted for key, wanted in expected.items())
        or operation.get("identity", {})
        .get("campaign", {})
        .get("task", {})
        .get("problem_sha256")
        != audit["problem_sha256"]
    ):
        raise ValueError("source replay identity, precision or completion differs")
    residual = _array(value.get("residual"), digest=value.get("residual_sha256"))
    if residual.ndim != 1:
        raise ValueError("source replay residual must be a vector")
    loss = float(np.mean(residual**2))
    if not np.isfinite(loss) or not np.isclose(
        loss, value.get("training_nmse"), rtol=1e-13, atol=0
    ):
        raise ValueError("source replay loss differs from residual")
    return operation, residual


def _invalidate(result: dict, error: BaseException) -> None:
    """Keep structured failures and explicit new execution counts."""
    result["error_type"] = type(error).__name__
    continuation._invalidate(result, error, "qualification", preserve_diagnostic=True)
    result["verification_calls_failed"] = (
        result["verification_calls_started"] - result["verification_calls_completed"]
    )


def import_baseline(
    oracle, radau_operation, bdf_operation, seconds, checkpoint
) -> dict:
    """Import both sealed state-only references, with no numerical reevaluation."""
    budget = execution._budget(seconds, seconds + 30)
    result = _base(oracle, "imported_baseline", budget)
    interrupted = None
    try:
        radau, reference = _read_replay(oracle, radau_operation, "Radau")
        bdf, residual = _read_replay(oracle, bdf_operation, METHOD)
        if residual.shape != reference.shape:
            raise ValueError("independent replay residual shapes differ")
        discrepancy = float(np.max(np.abs(residual - reference)))
        result.update(
            passed=discrepancy <= portfolio.PAIRING_TOLERANCE,
            residual=residual.tolist(),
            residual_sha256=adequacy._digest(residual),
            reference_residual=reference.tolist(),
            reference_residual_sha256=adequacy._digest(reference),
            solver_prediction_discrepancy=discrepancy,
            source_operation_sha256={
                "Radau": radau_operation["sha256"],
                METHOD: bdf_operation["sha256"],
            },
            inherited_cost={
                "scope": "source costs; excluded from new operation costs",
                "wall_seconds": radau.get("wall_seconds") + bdf.get("wall_seconds"),
                "cpu_seconds": radau.get("cpu_seconds") + bdf.get("cpu_seconds"),
                "verification_calls_started": 2,
                "verification_calls_completed": 2,
            },
        )
    except Exception as error:
        _invalidate(result, error)
    except BaseException as error:
        interrupted = error
        _invalidate(result, error)
    return continuation._finish(
        result,
        budget,
        checkpoint,
        "qualification",
        interrupted,
        preserve_diagnostic=True,
    )


def _baseline(oracle, receipt: dict) -> tuple[np.ndarray, np.ndarray]:
    if (
        receipt.get("identity") != _identity(oracle)
        or receipt.get("kind") != "imported_baseline"
        or receipt.get("status") != "complete"
        or not receipt.get("passed")
        or receipt.get("method") != METHOD
    ):
        raise ValueError("completed matching baseline prerequisite required")
    residual = _array(receipt.get("residual"), digest=receipt.get("residual_sha256"))
    reference = _array(
        receipt.get("reference_residual"),
        shape=residual.shape,
        digest=receipt.get("reference_residual_sha256"),
    )
    if (
        residual.ndim != 1
        or np.max(np.abs(residual - reference)) > portfolio.PAIRING_TOLERANCE
    ):
        raise ValueError(
            "baseline references do not satisfy unchanged agreement policy"
        )
    return residual, reference


def prepare_jacobian(oracle, baseline: dict, seconds: float, checkpoint) -> dict:
    """Perform one separately reserved BDF sensitivity solve and retain its array."""
    budget = execution._budget(seconds, seconds + 60)
    result = _base(oracle, "paired_jacobian", budget)
    interrupted = None
    try:
        residual, reference = _baseline(oracle, baseline)
        result["baseline_sha256"] = public.content_sha256(baseline)
        budget.callback(lambda: checkpoint(deepcopy(result)))
        result["verification_calls_attempted"] = 1

        def evaluate(deadline):
            result["verification_calls_started"] = 1
            value = oracle._call(
                oracle.vector(oracle.start), deadline, jacobian=True, method=METHOD
            )
            result["verification_calls_completed"] = 1
            return value

        augmented, jacobian = budget.numerical(evaluate)
        gap = float(np.max(np.abs(augmented - residual)))
        independent_gap = float(np.max(np.abs(augmented - reference)))
        result.update(
            passed=max(gap, independent_gap) <= portfolio.PAIRING_TOLERANCE,
            pairing_discrepancy=gap,
            original_Radau_discrepancy=independent_gap,
            residual=augmented.tolist(),
            residual_sha256=adequacy._digest(augmented),
            jacobian=jacobian.tolist(),
            jacobian_sha256=adequacy._digest(jacobian),
            baseline_residual_sha256=adequacy._digest(residual),
        )
    except Exception as error:
        _invalidate(result, error)
    except BaseException as error:
        interrupted = error
        _invalidate(result, error)
    return continuation._finish(
        result,
        budget,
        checkpoint,
        "qualification",
        interrupted,
        preserve_diagnostic=True,
    )


def _prepared(oracle, baseline: dict, receipt: dict):
    """Validate both prerequisites without executing an oracle or renewing a budget."""
    residual, reference = _baseline(oracle, baseline)
    if (
        receipt.get("identity") != _identity(oracle)
        or receipt.get("kind") != "paired_jacobian"
        or receipt.get("status") != "complete"
        or not receipt.get("passed")
        or receipt.get("method") != METHOD
        or receipt.get("baseline_sha256") != public.content_sha256(baseline)
        or receipt.get("baseline_residual_sha256") != adequacy._digest(residual)
    ):
        raise ValueError("completed matching Jacobian prerequisite required")
    augmented = _array(
        receipt.get("residual"),
        shape=residual.shape,
        digest=receipt.get("residual_sha256"),
    )
    jacobian = _array(
        receipt.get("jacobian"),
        shape=(len(residual), len(oracle.names)),
        digest=receipt.get("jacobian_sha256"),
    )
    if (
        max(np.max(np.abs(augmented - residual)), np.max(np.abs(augmented - reference)))
        > portfolio.PAIRING_TOLERANCE
    ):
        raise ValueError(
            "Jacobian prerequisite pairing does not satisfy agreement policy"
        )
    return residual, jacobian


def evaluate_probe(
    oracle,
    vector: Mapping[str, float],
    seconds: float,
    checkpoint,
    *,
    parent_deadline: float,
) -> dict:
    """One durable state-only BDF primitive, clipped by its owning coordinate."""
    parent_remaining = parent_deadline - monotonic()
    remaining = min(seconds, parent_remaining)
    if remaining <= 0:
        raise TimeoutError("coordinate allowance exhausted before probe")
    budget = execution._budget(remaining, min(remaining + 30, parent_remaining))
    result = _base(oracle, "residual_probe", budget) | {
        "vector": dict(vector),
        "vector_sha256": public.content_sha256(vector),
    }
    interrupted = None
    try:
        point = oracle.vector(vector)
        if checkpoint is not None:
            budget.callback(lambda: checkpoint(deepcopy(result)))
        result["verification_calls_attempted"] = 1

        def evaluate(deadline):
            result["verification_calls_started"] = 1
            residual, _ = oracle.evaluate(
                point, min(deadline, parent_deadline), jacobian=False, method=METHOD
            )
            result["verification_calls_completed"] = 1
            return residual

        residual = budget.numerical(evaluate)
        result.update(
            passed=True,
            residual=residual.tolist(),
            residual_sha256=adequacy._digest(residual),
        )
    except Exception as error:
        _invalidate(result, error)
    except BaseException as error:
        interrupted = error
        _invalidate(result, error)
    return continuation._finish(
        result,
        budget,
        checkpoint or (lambda _: None),
        "qualification",
        interrupted,
        preserve_diagnostic=True,
    )


def qualify_coordinate(
    oracle,
    baseline: dict,
    jacobian: dict,
    index: int,
    seconds: float,
    checkpoint,
    *,
    probe_runner: ProbeRunner | None = None,
) -> dict:
    """Check one direction using completed shared prerequisites and durable probes.

    ``probe_runner(key, vector, run)`` may own a nested operation and must return
    its unwrapped value; ``run(child_checkpoint=None)`` supplies the bounded
    original-equation evaluation. Keys use deterministic probe order. Reusing a
    child requires a complete, matching vector/settings receipt. The parent
    coordinate owns all child time and may never renew an interrupted allowance.
    """
    if (
        type(index) is not int
        or not 0 <= index < len(oracle.names)
        or oracle.lower[index] >= oracle.upper[index]
    ):
        raise ValueError("coordinate must be active in the frozen chart")
    budget = execution._budget(seconds, seconds + 60)
    result = _base(oracle, "coordinate", budget) | {
        "index": index,
        "name": oracle.names[index],
        "probe_receipts": [],
        "baseline_sha256": public.content_sha256(baseline),
        "jacobian_prerequisite_sha256": public.content_sha256(jacobian),
    }
    interrupted = None
    before = deepcopy(oracle.counts)
    try:
        residual, derivative = _prepared(oracle, baseline, jacobian)
    except (ValueError, TypeError) as error:
        result.update(
            status="blocked", blocked_reason=str(error), prerequisite_failed=True
        )
        return continuation._finish(
            result, budget, checkpoint, "qualification", preserve_diagnostic=True
        )
    point = oracle.vector(oracle.start)

    def evaluate(vector, deadline):
        budget.check()
        if not np.array_equal(vector, point):
            raise ValueError("only the frozen prerequisite Jacobian is available")
        portfolio._check(deadline)
        return residual.copy(), derivative.copy()

    def residual_only(vector, deadline):
        budget.check()
        portfolio._check(deadline)
        if np.array_equal(vector, point):
            return residual.copy()
        parameters = oracle.parameters(vector)
        key = f"probe_{len(result['probe_receipts']):02d}"

        def run(child_checkpoint=None):
            return evaluate_probe(
                oracle,
                parameters,
                PROBE_SECONDS,
                child_checkpoint,
                parent_deadline=min(deadline, budget.deadline()),
            )

        receipt = (
            run()
            if probe_runner is None
            else probe_runner(key, deepcopy(parameters), run)
        )
        if (
            receipt.get("identity") != _identity(oracle)
            or receipt.get("kind") != "residual_probe"
            or receipt.get("method") != METHOD
            or receipt.get("vector") != parameters
            or receipt.get("vector_sha256") != public.content_sha256(parameters)
        ):
            raise ValueError("nested probe identity or vector differs")
        result["probe_receipts"].append(
            {
                "key": key,
                "sha256": public.content_sha256(receipt),
                "value": deepcopy(receipt),
            }
        )
        budget.callback(lambda: checkpoint(deepcopy(result)))
        if receipt.get("status") != "complete" or not receipt.get("passed"):
            if receipt.get("timing", {}).get("callback_failures"):
                raise RuntimeError(
                    "nested probe checkpoint failed; further work blocked"
                )
            if "Timeout" in receipt.get("error_type", "") or receipt.get(
                "timing", {}
            ).get("timeout_reason"):
                raise TimeoutError("nested probe numerical allowance exhausted")
            message = receipt.get("error", "nested probe incomplete")
            if receipt.get("error_type") in {
                "NumericalInvalidTrial",
                "SymbolicRolloutFailure",
            }:
                raise NumericalInvalidTrial(message)
            raise RuntimeError(message)
        return _array(
            receipt.get("residual"),
            shape=residual.shape,
            digest=receipt.get("residual_sha256"),
        )

    try:
        budget.callback(lambda: checkpoint(deepcopy(result)))
        check = budget.numerical(
            lambda _: adequacy.derivative_check(
                evaluate,
                point,
                oracle.lower,
                oracle.upper,
                budget.wall + budget.policy.overall_seconds,
                coordinate_units=oracle.units,
                policy=portfolio._policy(),
                directions=[index],
                residual_only=residual_only,
                local_coordinate_scales=oracle.local_coordinate_scales(point),
            )
        )
        result.update(
            derivative_check=public._jsonable(check),
            base_residual=residual.tolist(),
            passed=check.error is None
            and check.checked_indices == (index,)
            and all(row.precision_status == "passed" for row in check.directions),
        )
    except Exception as error:
        _invalidate(result, error)
    except BaseException as error:
        interrupted = error
        _invalidate(result, error)
    counts = {key: value - before[key] for key, value in oracle.counts.items()}
    result.update(
        stage_backend_calls=counts,
        verification_calls_attempted=counts["residual_solves_started"],
        verification_calls_started=counts["residual_solves_started"],
        verification_calls_completed=counts["residual_solves_completed"],
        verification_calls_failed=counts["residual_solves_started"]
        - counts["residual_solves_completed"],
        cost_scope="nested probes counted once; reused prerequisites add no solves",
    )
    return continuation._finish(
        result,
        budget,
        checkpoint,
        "qualification",
        interrupted,
        preserve_diagnostic=True,
    )


def aggregate_qualification(oracle, receipts: Iterable[dict]) -> dict:
    """Retain the historical full-coverage checks under the explicit BDF identity."""
    receipts = list(receipts)
    for name in ("baseline_sha256", "jacobian_prerequisite_sha256"):
        references = {receipt.get(name) for receipt in receipts}
        if None in references or len(references) > 1:
            raise ValueError("coordinate prerequisite references differ")
    return previous.aggregate_qualification(
        oracle, receipts, identity=_identity(oracle)
    )
