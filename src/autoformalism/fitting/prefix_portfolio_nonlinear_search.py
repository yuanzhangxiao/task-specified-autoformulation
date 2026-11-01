"""Fixed-objective nonlinear search and separately scoped precision diagnostics.

The campaign authenticates source operation envelopes and owns all journals and
nonrenewable allowances. This module rechecks their numerical identities before
reusing arrays. Tighter partial diagnostics never authorize a whole-model search.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from time import monotonic

import numpy as np

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_prerequisites as prerequisites
from autoformalism.fitting import prefix_portfolio_qualification as previous
from autoformalism.fitting import public_fitting as public

PROTOCOL = "portfolio-qualified-bdf-search-1"
PRECISION_PROTOCOL = "portfolio-targeted-bdf-precision-1"
PRECISION_INDICES = (8, 10, 19, 22)
METHOD = "BDF"


class FixedBDFOracle:
    """Freeze the optimizer's residual and sensitivity methods, including defaults."""

    def __init__(self, oracle):
        self.oracle = oracle
        self._paired_start = None
        self._qualified_identity = None
        self._qualification = None
        self.imported_cache_hits = 0

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def execution_receipt(self) -> dict:
        return self.oracle.execution_receipt() | {
            "fixed_search_method": METHOD,
            "imported_paired_start_cache_hits": self.imported_cache_hits,
        }

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        portfolio._check(deadline)
        if method not in (None, METHOD):
            raise ValueError("qualified search requires the fixed BDF objective")
        if self._qualified_identity is not None and (
            prerequisites._identity(self) != self._qualified_identity
        ):
            raise ValueError("qualified search identity changed")
        vector = self.vector(self.parameters(vector))
        if self._paired_start is not None and np.array_equal(
            vector, self.vector(self.start)
        ):
            self.imported_cache_hits += 1
            residual, derivative = self._paired_start
            return residual.copy(), derivative.copy() if jacobian else None
        return self.oracle.evaluate(vector, deadline, jacobian=jacobian, method=METHOD)


def make_oracle(problem, training) -> FixedBDFOracle:
    """Build the unchanged original-ODE chart at its qualified tolerances."""
    return FixedBDFOracle(prerequisites.make_oracle(problem, training))


def authenticate_qualification(
    oracle: FixedBDFOracle,
    baseline: dict,
    jacobian: dict,
    coordinate_receipts: Iterable[dict],
    qualification: dict,
) -> dict:
    """Recompute full coverage and bind every checked column to imported arrays.

    Inputs are unwrapped receipts from campaign-authenticated source operations.
    Array validation alone never substitutes for that outer source authentication.
    """
    residual, derivative = prerequisites._prepared(oracle, baseline, jacobian)
    receipts = list(coordinate_receipts)
    expected = prerequisites.aggregate_qualification(oracle, receipts)
    if qualification != expected or not expected["search_qualified"]:
        raise ValueError("complete matching search qualification required")
    for receipt in receipts:
        check = receipt.get("derivative_check") or {}
        if (
            receipt.get("baseline_sha256") != public.content_sha256(baseline)
            or receipt.get("jacobian_prerequisite_sha256")
            != public.content_sha256(jacobian)
            or check.get("base_residual_sha256") != adequacy._digest(residual)
            or check.get("base_jacobian_sha256") != adequacy._digest(derivative)
        ):
            raise ValueError("qualification arrays differ from shared prerequisites")
    result = {
        "protocol": PROTOCOL,
        "identity": prerequisites._identity(oracle),
        "baseline_sha256": public.content_sha256(baseline),
        "jacobian_sha256": public.content_sha256(jacobian),
        "qualification_sha256": public.content_sha256(qualification),
        "coordinate_receipts_sha256": public.content_sha256(receipts),
        "start_training_nmse": float(np.mean(residual**2)),
        "search_qualified": True,
        "new_numerical_calls": 0,
        "inherited_prerequisite_cost_recharged": False,
    }
    oracle._paired_start = residual.copy(), derivative.copy()
    oracle._qualified_identity = deepcopy(result["identity"])
    oracle._qualification = deepcopy(result)
    return result


def search(
    oracle: FixedBDFOracle, start, seconds, maximum_calls, checkpoint, *, qualification
) -> dict:
    """Run a bounded lazy search only from the independently qualified frozen start."""
    if (
        oracle._qualification is None
        or qualification != oracle._qualification
        or prerequisites._identity(oracle) != qualification["identity"]
        or dict(start) != dict(oracle.start)
    ):
        raise ValueError(
            "authenticated full qualification of this exact start required"
        )

    def save(value):
        checkpoint(value | {"fixed_search_method": METHOD, "protocol": PROTOCOL})

    result = portfolio.search(oracle, start, seconds, maximum_calls, save)
    result.update(
        protocol=PROTOCOL,
        fixed_search_method=METHOD,
        qualification_sha256=public.content_sha256(qualification),
        qualified_start_training_nmse=qualification["start_training_nmse"],
        newly_verified_endpoint=False,
    )
    return result


def verify_endpoint(
    oracle: FixedBDFOracle,
    parameters,
    seconds,
    checkpoint,
    *,
    expected_nmse: float | None = None,
) -> dict:
    """Independently verify the original Radau rollout and any saved search loss."""
    if expected_nmse is not None and (
        not np.isfinite(expected_nmse) or expected_nmse < 0
    ):
        raise ValueError("finite nonnegative expected NMSE required")

    def guarded(value):
        result = deepcopy(value)
        result["methods"] = ["original_BDF", "original_Radau"]
        result["expected_training_nmse"] = expected_nmse
        actual = result.get("training_nmse")
        agreement = (
            actual is not None
            and np.isfinite(actual)
            and (
                expected_nmse is None
                or abs(actual - expected_nmse)
                <= max(1e-12, 1e-8 * max(actual, expected_nmse))
            )
        )
        result["saved_loss_replay_passed"] = bool(agreement)
        if not agreement:
            result["original_equations_verified"] = False
        return result

    value = portfolio.verify_endpoint(
        oracle.original,
        oracle,
        parameters,
        seconds,
        lambda row: checkpoint(guarded(row)),
    )
    return guarded(value)


def precision_policy() -> dict:
    """Keep the local stencil unchanged while tightening integration only."""
    return {
        "protocol": PRECISION_PROTOCOL,
        "numerical_policy": kernel.NumericalPolicy(rtol=1e-12, atol=1e-14).model_dump(
            mode="json"
        ),
        "qualification_method": METHOD,
        "independent_reference_method": "Radau",
        "derivative_policy": portfolio._policy().model_dump(mode="json"),
        "agreement_tolerance": portfolio.PAIRING_TOLERANCE,
        "targeted_indices": list(PRECISION_INDICES),
        "coordinate_seconds": 240.0,
        "probe_seconds": 90.0,
        "full_search_qualification": False,
    }


def make_precision_oracle(problem, training):
    """Use a separate original-ODE instance for the tighter, partial diagnostic."""
    original, oracle = portfolio.make_oracle(
        problem, training, kernel.NumericalPolicy(rtol=1e-12, atol=1e-14)
    )
    if original.portfolio_audit["route"] != "original_ODE":
        raise ValueError("targeted precision diagnostic requires original ODE route")
    return oracle


def _precision_identity(oracle):
    policy = precision_policy()
    if oracle.portfolio_audit.get("numerical_policy") != policy["numerical_policy"]:
        raise ValueError("targeted diagnostic requires its fixed tighter precision")
    return previous._identity(oracle) | {
        "protocol": PRECISION_PROTOCOL,
        "policy": policy,
        "solver_settings_sha256": public.content_sha256(
            public._jsonable(oracle.original.settings) | {"integration_method": METHOD}
        ),
    }


def _precision_base(oracle, kind, budget=None):
    result = {
        "identity": _precision_identity(oracle),
        "kind": kind,
        "method": METHOD,
        "passed": False,
        "no_search_performed": True,
        "full_search_qualification": False,
        "verification_calls_attempted": 0,
        "verification_calls_started": 0,
        "verification_calls_completed": 0,
        "verification_calls_failed": 0,
    }
    if budget is not None:
        result["execution_policy"] = budget.policy.model_dump(mode="json")
    return result


def _precision_solve(
    oracle,
    kind,
    parameters,
    method,
    seconds,
    checkpoint,
    *,
    jacobian=False,
    parent_deadline=None,
):
    """One reserved replay/sensitivity/probe, with no implicit method fallback."""
    if method not in (METHOD, "Radau"):
        raise ValueError("fixed BDF or independent Radau required")
    remaining = (
        seconds
        if parent_deadline is None
        else min(seconds, parent_deadline - monotonic())
    )
    if remaining <= 0:
        raise TimeoutError("targeted coordinate allowance exhausted")
    overall = (
        remaining + 60
        if parent_deadline is None
        else min(remaining + 30, parent_deadline - monotonic())
    )
    budget = execution._budget(remaining, overall)
    result = _precision_base(oracle, kind, budget) | {
        "method": method,
        "vector": dict(parameters),
        "vector_sha256": public.content_sha256(parameters),
    }
    interrupted = None
    try:
        budget.callback(lambda: checkpoint(deepcopy(result)))
        result["verification_calls_attempted"] = 1

        def evaluate(deadline):
            result["verification_calls_started"] = 1
            if kind == "precision_replay":
                lifted = continuation.to_original(oracle.original, oracle, parameters)
                values = oracle.original.evaluate(
                    oracle.original.vector(lifted),
                    deadline,
                    jacobian=False,
                    method=method,
                )
            else:
                values = oracle._call(
                    oracle.vector(parameters),
                    deadline,
                    jacobian=jacobian,
                    method=method,
                )
            result["verification_calls_completed"] = 1
            return values

        residual, derivative = budget.numerical(evaluate)
        residual = np.asarray(residual, dtype=float)
        if residual.ndim != 1 or not residual.size or not np.isfinite(residual).all():
            raise ValueError("malformed tighter replay residual")
        with np.errstate(over="ignore", invalid="ignore"):
            loss = float(np.mean(residual**2))
        if not np.isfinite(loss):
            raise ValueError("nonfinite tighter replay loss")
        result.update(
            passed=True,
            residual=residual.tolist(),
            residual_sha256=adequacy._digest(residual),
            training_nmse=loss,
        )
        if jacobian:
            result.update(
                jacobian=derivative.tolist(),
                jacobian_sha256=adequacy._digest(derivative),
            )
    except Exception as error:
        prerequisites._invalidate(result, error)
    except BaseException as error:
        interrupted = error
        prerequisites._invalidate(result, error)
    return continuation._finish(
        result,
        budget,
        checkpoint,
        "qualification",
        interrupted,
        preserve_diagnostic=True,
    )


def precision_replay(oracle, method, seconds, checkpoint):
    """Reserve one original-equation state-only replay at the unchanged start."""
    return _precision_solve(
        oracle, "precision_replay", oracle.start, method, seconds, checkpoint
    )


def _precision_array(oracle, value, *, kind, method=METHOD):
    if (
        value.get("identity") != _precision_identity(oracle)
        or value.get("kind") != kind
        or value.get("method") != method
        or value.get("status") != "complete"
        or not value.get("passed")
        or value.get("vector") != dict(oracle.start)
        or value.get("vector_sha256") != public.content_sha256(oracle.start)
    ):
        raise ValueError("completed matching tighter prerequisite required")
    return prerequisites._array(
        value.get("residual"), digest=value.get("residual_sha256")
    )


def precision_baseline(oracle, bdf, radau):
    """Pair fresh replays without importing older-precision evidence."""
    residual = _precision_array(oracle, bdf, kind="precision_replay")
    reference = _precision_array(oracle, radau, kind="precision_replay", method="Radau")
    if residual.ndim != 1 or reference.shape != residual.shape:
        raise ValueError("tighter replay residual shape differs")
    gap = float(np.max(np.abs(reference - residual)))
    return _precision_base(oracle, "precision_baseline") | {
        "status": "complete",
        "passed": gap <= portfolio.PAIRING_TOLERANCE,
        "pairing_discrepancy": gap,
        "vector": dict(oracle.start),
        "vector_sha256": public.content_sha256(oracle.start),
        "residual": residual.tolist(),
        "residual_sha256": adequacy._digest(residual),
        "reference_residual": reference.tolist(),
        "reference_residual_sha256": adequacy._digest(reference),
        "source_replay_sha256": [public.content_sha256(v) for v in (bdf, radau)],
    }


def _precision_baseline(oracle, baseline):
    residual = _precision_array(oracle, baseline, kind="precision_baseline")
    reference = prerequisites._array(
        baseline.get("reference_residual"),
        shape=residual.shape,
        digest=baseline.get("reference_residual_sha256"),
    )
    if (
        residual.ndim != 1
        or np.max(np.abs(reference - residual)) > portfolio.PAIRING_TOLERANCE
    ):
        raise ValueError("tighter baseline pairing differs")
    return residual, reference


def precision_jacobian(oracle, baseline, seconds, checkpoint):
    """Compute exactly one fresh full Jacobian and pair it with both new replays."""
    residual, reference = _precision_baseline(oracle, baseline)

    def paired(value):
        value = deepcopy(value) | {"baseline_sha256": public.content_sha256(baseline)}
        if value.get("passed"):
            augmented = prerequisites._array(
                value["residual"], shape=residual.shape, digest=value["residual_sha256"]
            )
            gap = float(
                max(
                    np.max(np.abs(augmented - residual)),
                    np.max(np.abs(augmented - reference)),
                )
            )
            value.update(
                pairing_discrepancy=gap, passed=gap <= portfolio.PAIRING_TOLERANCE
            )
        return value

    result = _precision_solve(
        oracle,
        "precision_jacobian",
        oracle.start,
        METHOD,
        seconds,
        lambda value: checkpoint(paired(value)),
        jacobian=True,
    )
    return paired(result)


def _precision_prepared(oracle, baseline, jacobian):
    residual, reference = _precision_baseline(oracle, baseline)
    augmented = _precision_array(oracle, jacobian, kind="precision_jacobian")
    derivative = prerequisites._array(
        jacobian.get("jacobian"),
        shape=(len(residual), len(oracle.names)),
        digest=jacobian.get("jacobian_sha256"),
    )
    if (
        jacobian.get("baseline_sha256") != public.content_sha256(baseline)
        or augmented.shape != residual.shape
        or max(
            np.max(np.abs(augmented - residual)), np.max(np.abs(augmented - reference))
        )
        > portfolio.PAIRING_TOLERANCE
    ):
        raise ValueError("tighter Jacobian pairing differs")
    return residual, derivative


def precision_coordinate(
    oracle, baseline, jacobian, index, seconds, checkpoint, *, probe_runner=None
):
    """Inspect one of the four unresolved directions, preserving local step policy."""
    if (
        type(index) is not int
        or index not in PRECISION_INDICES
        or index >= len(oracle.names)
    ):
        raise ValueError("only the four declared targeted directions are permitted")
    if seconds > 240 or oracle.lower[index] >= oracle.upper[index]:
        raise ValueError("active targeted direction and at most 240 seconds required")
    residual, derivative = _precision_prepared(oracle, baseline, jacobian)
    budget = execution._budget(seconds, seconds + 60)
    result = _precision_base(oracle, "coordinate", budget) | {
        "index": index,
        "name": oracle.names[index],
        "probe_receipts": [],
        "baseline_sha256": public.content_sha256(baseline),
        "jacobian_prerequisite_sha256": public.content_sha256(jacobian),
    }
    point = oracle.vector(oracle.start)
    before = deepcopy(oracle.counts)
    interrupted = None

    def evaluate(vector, deadline):
        budget.check()
        portfolio._check(deadline)
        if not np.array_equal(vector, point):
            raise ValueError("only the frozen tighter Jacobian is available")
        return residual.copy(), derivative.copy()

    def residual_only(vector, deadline):
        budget.check()
        portfolio._check(deadline)
        if np.array_equal(vector, point):
            return residual.copy()
        parameters = oracle.parameters(vector)
        key = f"probe_{len(result['probe_receipts']):02d}"

        def run(child_checkpoint=None):
            return _precision_solve(
                oracle,
                "residual_probe",
                parameters,
                METHOD,
                90,
                child_checkpoint or (lambda _: None),
                parent_deadline=min(deadline, budget.deadline()),
            )

        receipt = run() if probe_runner is None else probe_runner(key, parameters, run)
        if (
            receipt.get("identity") != _precision_identity(oracle)
            or receipt.get("kind") != "residual_probe"
            or receipt.get("method") != METHOD
            or receipt.get("vector") != parameters
            or receipt.get("vector_sha256") != public.content_sha256(parameters)
        ):
            raise ValueError("tighter nested probe identity differs")
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
                raise prerequisites.NumericalInvalidTrial(message)
            raise RuntimeError(message)
        return prerequisites._array(
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
        prerequisites._invalidate(result, error)
    except BaseException as error:
        interrupted = error
        prerequisites._invalidate(result, error)
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


def precision_summary(oracle, receipts: Iterable[dict]) -> dict:
    """Summarize only targeted directions; never merge older-precision checks."""
    receipts = list(receipts)
    if any(row.get("index") not in PRECISION_INDICES for row in receipts):
        raise ValueError("unexpected targeted precision direction")
    for key in ("baseline_sha256", "jacobian_prerequisite_sha256"):
        values = {row.get(key) for row in receipts}
        if None in values or len(values) > 1:
            raise ValueError("targeted prerequisites differ")
    result = previous.aggregate_qualification(
        oracle, receipts, identity=_precision_identity(oracle)
    )
    checked = set(result["checked_indices"])
    return result | {
        "targeted_indices": list(PRECISION_INDICES),
        "targeted_complete": checked == set(PRECISION_INDICES),
        "targeted_passed": checked == set(PRECISION_INDICES)
        and not result["inconclusive_indices"],
        "partial_diagnostic_only": True,
        "full_search_qualification": False,
        "search_qualified": False,
        "scope": "Four tighter-precision directions only; no full qualification.",
    }
