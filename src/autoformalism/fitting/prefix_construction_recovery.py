"""Opt-in local qualification of unchanged constructed affine models.

Exact training-inactive initializer coordinates are anchored, never estimated or
claimed identifiable. Diagnostic escalation recomputes the complete derivative
check under one allowance; it never relaxes the thresholds or changes equations.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict, replace
from time import monotonic

import numpy as np

from autoformalism.fitting import prefix_affine as affine
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_gain_reduction as gains
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial

PROTOCOL = "construction-derivative-recovery-coordinates-1"
GUARDED_FRACTION = 0.35


def precision_policy(mode: str) -> affine.AffinePrecisionPolicy:
    """Keep the historical guard or explicitly escalate every nonzero generator."""
    policy = affine.AffinePrecisionPolicy()
    if mode == "guarded":
        return policy
    if mode == "multiprecision":
        return replace(policy, dimensionless_norm_threshold=1e-20)
    raise ValueError("unknown construction recovery precision mode")


def policy_receipt() -> dict:
    """Return frozen defaults suitable for an experiment's sealed plan."""
    record = {
        "protocol": PROTOCOL,
        "guarded_budget_fraction": GUARDED_FRACTION,
        "precision_policies": {
            mode: asdict(precision_policy(mode))
            for mode in ("guarded", "multiprecision")
        },
        "inactive_proof": "exact_symbolic_initializers_rhs_and_observations",
        "inactive_scope": "supplied_training_boundaries_only",
        "local_scale_policy": {
            "ordinary_nonsingular": "max(abs(value), optimizer_unit)",
            "sensitive_nonzero": "min(abs(value), optimizer_unit)",
            "sensitive_zero": "optimizer_unit",
            "relative_radius_cap": 1e-4,
        },
        "derivative_policy": _policy(None).model_dump(mode="json"),
        "residual_error_bound": "existing_empirical_float64_envelope_unchanged",
        "original_parameter_identifiability_certified": False,
    }
    return record | {"sha256": public.content_sha256(record)}


def _policy(value):
    policy = value or adequacy.Policy(
        precision_aware=True, local_adaptive=True, local_bounds_aware=True
    )
    if not (
        policy.precision_aware and policy.local_adaptive and policy.local_bounds_aware
    ):
        raise ValueError("bounds-aware local precision policy required")
    return policy


def certify_inactive_initializers(original) -> dict:
    """Prove exact absence after substituting permitted training initial data.

    Every initialized state is checked, including expanded dependencies on other
    initial states. Dynamics and observations are checked at symbolic states,
    inputs, time and parameters. No small-column or fitted-value test is used.
    """
    import casadi as ca

    system = original.system
    theta = ca.SX.sym("recovery_theta", len(system.names))
    x = ca.SX.sym("recovery_x", system.state_count)
    u = ca.SX.sym("recovery_u", len(system.inputs))
    time = ca.SX.sym("recovery_time")
    dynamics = ca.vertcat(
        system.rhs(time, x, theta, u), system.observe(time, x, theta, u)
    )
    initials = [system.initial_symbolic(row, theta) for row in original.rows]
    eligible = set(original.audit["shared_parameter_names"])
    removed, retained = [], []
    for name in original.names:
        if name not in eligible:
            continue
        if name not in system.names:
            raise ValueError(
                "shared initializer coordinate missing from compiled model"
            )
        symbol = theta[system.names.index(name)]
        if ca.depends_on(dynamics, symbol):
            retained.append(
                {"name": name, "reason": "used_in_dynamics_or_observations"}
            )
        elif any(ca.depends_on(expression, symbol) for expression in initials):
            retained.append({"name": name, "reason": "used_in_training_initial_state"})
        else:
            removed.append(name)
    record = {
        "protocol": "training-inactive-initializer-proof-1",
        "source_sha256": gains._source_identity(original),
        "training_sha256": public.content_sha256(original.training),
        "initializer_graph_sha256": public.content_sha256(original.problem.graph),
        "boundary_sha256": public.content_sha256(
            {row.trajectory_id: system._initial_data(row) for row in original.rows}
        ),
        "trajectory_ids": [row.trajectory_id for row in original.rows],
        "removed_names": removed,
        "retained": retained,
        "scope": "Exact symbolic absence for these training boundaries only.",
        "future_boundary_equivalence_certified": False,
        "original_parameters_identified": False,
    }
    return record | {"sha256": public.content_sha256(record)}


def sensitive_parameters(original) -> tuple[str, ...]:
    """Conservatively mark denominator, power and nonlinear-function parameters.

    Dependencies through process definitions are expanded symbolically by the
    restricted AST. Unknown expression nodes retain all descendant dependencies.
    Parameter names or semantic role labels never determine this classification.
    """
    trees = gains._expressions(original)
    processes = {
        key.removeprefix("process:"): value
        for key, value in trees.items()
        if key.startswith("process:")
    }
    parameter_names = set(original.system.names)

    def dependencies(node, active=frozenset()):
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        found = names & parameter_names
        for name in names & processes.keys():
            if name in active:
                raise ValueError("cyclic process expression in local scale analysis")
            found |= dependencies(processes[name], active | {name})
        return found

    sensitive = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Div):
                    sensitive |= dependencies(node.right)
                elif isinstance(node.op, ast.Pow) or not isinstance(
                    node.op, (ast.Add, ast.Sub, ast.Mult)
                ):
                    sensitive |= dependencies(node)
            elif isinstance(node, ast.Call) or (
                isinstance(node, ast.expr)
                and not isinstance(node, (ast.Name, ast.Constant, ast.UnaryOp))
            ):
                sensitive |= dependencies(node)
    return tuple(name for name in original.system.names if name in sensitive)


class ActiveReduction:
    """Compose the exact gain quotient with a fixed training-inactive section."""

    def __init__(self, gain_reduction, proof, anchor):
        self.gain_reduction = gain_reduction
        self.proof = deepcopy(proof)
        self.anchor = gain_reduction.reduce(anchor).copy()
        removed = set(proof["removed_names"])
        if not removed <= set(gain_reduction.names):
            raise ValueError(
                "inactive initialization coordinates overlap gain quotient"
            )
        self.indices = tuple(
            i for i, name in enumerate(gain_reduction.names) if name not in removed
        )
        self.names = tuple(gain_reduction.names[i] for i in self.indices)
        self.lower = gain_reduction.lower[list(self.indices)]
        self.upper = gain_reduction.upper[list(self.indices)]

    @property
    def receipt(self):
        record = {
            "protocol": PROTOCOL,
            "gain_reduction": self.gain_reduction.receipt,
            "inactive_initializer_proof": deepcopy(self.proof),
            "effective_names": list(self.names),
            "anchored_coordinates": {
                name: float(self.anchor[i])
                for i, name in enumerate(self.gain_reduction.names)
                if i not in self.indices
            },
            "lift_projection_identity": "reduce(lift(active)) == active",
            "original_parameter_identifiability_certified": False,
        }
        return record | {"sha256": public.content_sha256(record)}

    def vector(self, parameters):
        return gains._checked_vector(
            gains._mapping_vector(parameters, self.names),
            self.names,
            self.lower,
            self.upper,
        )

    def reduce(self, original_vector):
        return self.gain_reduction.reduce(original_vector)[list(self.indices)]

    def lift(self, active_vector):
        values = gains._checked_vector(
            active_vector, self.names, self.lower, self.upper
        )
        full = self.anchor.copy()
        full[list(self.indices)] = values
        return self.gain_reduction.lift(full)

    def lift_jacobian(self):
        return self.gain_reduction.lift_jacobian()[:, list(self.indices)]


class ActiveTrainingOracle:
    """Unchanged training model in exact gain and active-initializer coordinates."""

    def __init__(self, original, retained, precision_mode):
        self.original = original
        self.precision_mode = precision_mode
        reduced = gains.ReducedTrainingOracle(original)
        proof = certify_inactive_initializers(original)
        self.reduction = ActiveReduction(
            reduced.reduction, proof, original.vector(retained)
        )
        self.names = self.reduction.names
        self.lower, self.upper = self.reduction.lower, self.reduction.upper
        self.start = self.parameters(
            self.reduction.reduce(original.vector(original.start))
        )
        self.units = reduced.units[list(self.reduction.indices)]
        self._lift_jacobian = self.reduction.lift_jacobian()
        sensitive = set(sensitive_parameters(original))
        self.sensitive_names = tuple(
            name
            for name in self.names
            if name in sensitive or name.rsplit(":", 1)[-1] in sensitive
        )

    @property
    def recovery_audit(self):
        return {
            "policy": policy_receipt(),
            "precision_mode": self.precision_mode,
            "coordinate_reduction": self.reduction.receipt,
            "sensitive_names": list(self.sensitive_names),
            "original_coordinates": len(self.original.names),
            "optimizer_coordinates": len(self.names),
            "scope": (
                "Training-local coordinates; no original-parameter confidence claim."
            ),
        }

    def vector(self, parameters):
        return self.reduction.vector(parameters)

    def parameters(self, vector):
        values = gains._checked_vector(vector, self.names, self.lower, self.upper)
        return dict(zip(self.names, values.tolist(), strict=True))

    def local_coordinate_scales(self, vector):
        """Provide physical local scales, distinct from fixed optimizer units.

        Ordinary large coefficients admit small relative perturbations. Sensitive
        denominators and nonlinear expressions retain conservative magnitude caps.
        The derivative checker separately enforces domains and a shrinking ladder.
        """
        values = gains._checked_vector(vector, self.names, self.lower, self.upper)
        scales = np.maximum(np.abs(values), self.units)
        for index, name in enumerate(self.names):
            if name in self.sensitive_names:
                scales[index] = (
                    min(self.units[index], abs(values[index]))
                    if values[index] != 0
                    else self.units[index]
                )
        return scales

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        residual, derivative = self.original.evaluate(
            self.reduction.lift(vector), deadline, jacobian=jacobian, method=method
        )
        return residual, derivative @ self._lift_jacobian if jacobian else None


def make_oracle(problem, training, numerical, retained, *, precision_mode="guarded"):
    """Build the same anchored coordinate chart for qualification/search/diagnosis."""
    original = affine.TrainingOracle(
        problem, training, numerical, precision_policy=precision_policy(precision_mode)
    )
    return original, ActiveTrainingOracle(original, retained, precision_mode)


def _derivative(oracle, vector, deadline, policy):
    return adequacy.derivative_check(
        oracle.evaluate,
        vector,
        oracle.lower,
        oracle.upper,
        deadline,
        coordinate_units=oracle.units,
        policy=policy,
        residual_only=lambda x, d: oracle.evaluate(x, d, jacobian=False)[0],
        local_coordinate_scales=oracle.local_coordinate_scales(vector),
    )


def _retryable_check(check):
    """A returned contract error is not a numerical precision rescue request."""
    if check.error is not None:
        return check.error.startswith(
            ("TimeoutError:", "NumericalInvalidTrial:", "AffinePrecisionUnavailable:")
        )
    return check.status in {"failed", "inconclusive", "incomplete", "unavailable"}


def qualification(
    problem, training, numerical, retained, seconds, checkpoint, *, adequacy_policy=None
):
    """Audit locally; retry numerical uncertainty once at higher precision in budget."""
    policy = _policy(adequacy_policy)
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {
        "passed": False,
        "precision_mode": "guarded",
        "passes": [],
        "policy": policy_receipt(),
        "no_search_performed": True,
    }
    interrupted = None
    try:
        for mode in ("guarded", "multiprecision"):
            budget.check()
            record = {"precision_mode": mode, "status": "started", "passed": False}
            result["passes"].append(record)
            started = monotonic()
            pass_deadline = min(
                budget.deadline(),
                started + seconds * GUARDED_FRACTION
                if mode == "guarded"
                else budget.deadline(),
            )
            original = oracle = None
            numerical_retry = False
            try:
                original, oracle = make_oracle(
                    problem, training, numerical, retained, precision_mode=mode
                )
                budget.check()
                vector = oracle.reduction.reduce(original.vector(retained))
                record.update(recovery_audit=oracle.recovery_audit)
                before, _ = budget.numerical(
                    lambda d, source=original, end=pass_deadline: source.evaluate(
                        source.vector(retained), min(d, end), jacobian=False
                    )
                )
                after, _ = budget.numerical(
                    lambda d,
                    source=oracle,
                    point=vector,
                    end=pass_deadline: source.evaluate(
                        point, min(d, end), jacobian=False
                    )
                )
                gap = float(np.max(np.abs(before - after), initial=0))
                if not np.isfinite(gap):
                    raise NumericalInvalidTrial("nonfinite lift discrepancy")
                record["lift_prediction_discrepancy"] = gap
                check = budget.numerical(
                    lambda d,
                    source=oracle,
                    point=vector,
                    end=pass_deadline: _derivative(source, point, min(d, end), policy)
                )
                record.update(
                    derivative_check=public._jsonable(check),
                    status="complete",
                    passed=bool(gap <= 1e-7 and check.search_qualified),
                )
                # Recompute disagreements too: an inaccurate base derivative may
                # look consistently wrong. The retry still must pass every check.
                numerical_retry = not record["passed"] and _retryable_check(check)
            except (TimeoutError, NumericalInvalidTrial) as error:
                record.update(
                    status="incomplete", error=f"{type(error).__name__}: {error}"
                )
                numerical_retry = True
            except Exception as error:
                record.update(
                    status="incomplete", error=f"{type(error).__name__}: {error}"
                )
                raise
            finally:
                record["elapsed_wall_seconds"] = monotonic() - started
                if original is not None:
                    record["affine_execution"] = deepcopy(original.affine_audit)
            budget.check()
            result.update(precision_mode=mode, passed=record["passed"])
            if oracle is not None:
                result.update(
                    original_coordinates=len(original.names),
                    optimizer_coordinates=len(oracle.names),
                    recovery_audit=oracle.recovery_audit,
                    derivative_check=record.get("derivative_check"),
                    lift_prediction_discrepancy=record.get(
                        "lift_prediction_discrepancy"
                    ),
                )
            budget.callback(
                lambda: checkpoint(deepcopy(result) | {"timing": budget.snapshot()})
            )
            if record["passed"] or not numerical_retry:
                break
    except Exception as error:
        continuation._invalidate(result, error, "qualification")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "qualification")
    return continuation._finish(
        result, budget, checkpoint, "qualification", interrupted
    )


def diagnose(
    original,
    oracle,
    parameters,
    verification,
    search_result,
    seconds,
    checkpoint,
    *,
    adequacy_policy=None,
):
    """Assess in the same frozen chart/precision; removed originals stay unresolved."""
    if oracle.original is not original:
        raise ValueError("diagnosis oracle identity differs")
    policy = _policy(adequacy_policy)
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {
        "assessment": None,
        "recovery_audit": oracle.recovery_audit,
        "original_parameter_identifiability_certified": False,
    }
    interrupted = None
    try:
        vector = oracle.vector(parameters)
        derivative = budget.numerical(lambda d: _derivative(oracle, vector, d, policy))
        result["derivative_check"] = public._jsonable(derivative)
        residual, jacobian = budget.numerical(lambda d: oracle.evaluate(vector, d))
        stop = search_result.get("stop_reason")
        history = [
            {
                key: step[key]
                for key in ("training_nmse", "coordinate_step_norm", "evaluation_count")
            }
            for step in search_result.get("accepted_steps", [])
        ]
        assessment = adequacy.assess(
            vector,
            residual,
            jacobian,
            oracle.lower,
            oracle.upper,
            verification={
                key: verification[key]
                for key in adequacy.OriginalVerification.model_fields
            },
            derivative=derivative,
            stop_reason=stop,
            budget_exhausted=bool(search_result.get("budget_exhausted"))
            or stop in {"budget_limited", "numerical_failure", "interrupted"},
            history=history,
            coordinate_units=oracle.units,
            policy=policy,
        )
        budget.check()
        result["assessment"] = public._jsonable(assessment)
        result["assessment_scope"] = (
            "Remaining active training coordinates only. Anchored initializer "
            "coefficients and removed gain factorizations are not identified."
        )
    except Exception as error:
        continuation._invalidate(result, error, "diagnosis")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "diagnosis")
    result["affine_execution"] = deepcopy(original.affine_audit)
    return continuation._finish(result, budget, checkpoint, "diagnosis", interrupted)
