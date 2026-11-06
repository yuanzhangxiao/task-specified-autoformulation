"""Training-only, signed insulin-feedback fitting with certified gauge removal.

The new feedback coefficient changes the dynamics matrix and remains an outer
optimization coordinate. Only the three forcing gains and trajectory initial
roots are profiled. The gain-one section removes a redundant factorization; it
does not impose stability, biological signs, or parameter identifiability.
"""

from __future__ import annotations

from functools import partial
from time import monotonic

import numpy as np

from autoformalism.fitting import insulin_meal_comparison as comparison
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_gain_reduction as gains
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_meal_coordinates as meal
from autoformalism.fitting import prefix_meal_joint_profile as joint
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_polish as polish
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import prefix_structural_reduction as structural
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.compact_search_checkpoint import SearchCheckpointPolicy
from autoformalism.research import meal_revision_metrics

PROTOCOL = "candidate15-signed-insulin-revision-fit-1"
DELAY = "k_insulin_delay"
GAIN = "af_process_gain_1_0"
AMPLITUDE = "k_insulin_term_1_0_4e94c3cad2"
MEMORY = "X_insulin_action"
Policy = comparison.Policy
allowances = comparison.allowances


def _polynomials(original):
    """Expand restricted executable consumers, including scoped initialization."""
    trees = gains._expressions(original)
    processes = {
        key.removeprefix("process:"): value
        for key, value in trees.items()
        if key.startswith("process:")
    }
    names = set(original.names) | set(original.system.names)
    return {
        key: structural._polynomial(value, processes, names)
        for key, value in trees.items()
        if not key.startswith("process:")
    }


def certify_production_gauge(original):
    """Prove the full family has a gain-one section, including the zero gain."""
    candidate = original.system.model.validated.candidate
    if candidate.constraints or original.problem.base_candidate.constraints:
        raise ValueError("production gauge does not support additional constraints")
    amplitudes = (AMPLITUDE, DELAY)
    for name in (GAIN, *amplitudes):
        if name not in original.names:
            raise ValueError("missing production-gauge coordinate")
    for name in amplitudes:
        i = original.names.index(name)
        if (original.lower[i], original.upper[i]) != (-np.inf, np.inf):
            raise ValueError("production amplitudes require unrestricted real domains")
    gi = original.names.index(GAIN)
    if not original.lower[gi] <= 1 <= original.upper[gi]:
        raise ValueError("production gain domain excludes canonical one")
    consumers, found = [], set()
    for location, polynomial in _polynomials(original).items():
        touched = False
        for powers in polynomial:
            degrees = dict(powers)
            present = set(degrees) & set(amplitudes)
            if GAIN not in degrees and not present:
                continue
            if (
                degrees.get(GAIN) != 1
                or len(present) != 1
                or any(degrees[name] != 1 for name in present)
            ):
                raise ValueError(
                    "production gain or amplitude has an unsupported reuse"
                )
            found.update(present)
            touched = True
        if touched:
            consumers.append(location)
    if found != set(amplitudes):
        raise ValueError("production gauge requires both isolated amplitude products")
    record = {
        "protocol": "shared-insulin-production-gauge-1",
        "problem_sha256": public.content_sha256(original.problem),
        "fixed_coordinate": {GAIN: 1.0},
        "effective_products": {name: f"{GAIN} * {name}" for name in amplitudes},
        "consumers": consumers,
        "proof": "every executable occurrence uses one gain and one real amplitude",
        "zero_gain_included": True,
        "full_model_family_preserved": True,
        "original_factorization_identified": False,
        "stability_restriction_added": False,
    }
    return record | {"sha256": public.content_sha256(record)}


def canonical_parameters(original, parameters):
    """Map a complete original vector onto the certified gain-one section."""
    certify_production_gauge(original)
    original.vector(parameters)
    point = dict(parameters)
    for name in (AMPLITUDE, DELAY):
        point[name] *= point[GAIN]
    point[GAIN] = 1.0
    original.vector(point)
    return point


def _matrix(original, parameters):
    local = original.local_vector(original.vector(parameters), original.rows[0])
    n = original.system.state_count
    matrix = np.asarray(original._propagation.generator(local), dtype=float)[:n, :n]
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite endpoint dynamics matrix")
    return matrix


def _eigenvalues(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return [{"real": float(x.real), "imaginary": float(x.imag)} for x in eigenvalues]


class CenteredMealChart(meal.MealCoordinates):
    """Center shared coordinates and freeze scales from net affine coefficients.

    A cancellation such as ``-I-b*I`` uses the net damping, not ``abs(b)``.
    This is only optimizer preconditioning; feasible domains stay unchanged.
    """

    def __init__(self, oracle, parameters, profiler, feedback_unit):
        self.meal = meal.certify(oracle, parameters)
        original = oracle.original
        vector = oracle.vector(parameters)
        complete = oracle.reduction.lift(vector)
        local = original.local_vector(complete, original.rows[0])
        propagation = original._propagation
        generator = np.asarray(propagation.generator(local), dtype=float)
        derivative = np.asarray(propagation.derivatives(local), dtype=float).reshape(
            (*generator.shape, len(local)), order="F"
        )
        chain = oracle.reduction.lift_jacobian(vector)
        chain = chain[original.indices[original.rows[0].trajectory_id]]
        derivative = np.tensordot(derivative, chain, axes=(2, 0))
        n = original.system.state_count
        m = len(original.system.inputs)
        values = generator[:n, : n + m]
        self.offset = np.zeros(len(oracle.names))
        self.frozen_units = self.meal.units(parameters)
        inventory = {}
        for index in profiler.outer:
            name = oracle.names[index]
            if name in (GAIN, self.meal.rate):
                continue
            d = derivative[:n, : n + m, index]
            nonzero = d != 0
            if not nonzero.any():
                raise ValueError(
                    "outer coordinate has no affine coefficient sensitivity"
                )
            characteristic = np.where(
                values[nonzero] != 0, abs(values[nonzero]), feedback_unit
            )
            unit = float(np.min(characteristic / abs(d[nonzero])))
            if not np.isfinite(unit) or unit <= 0:
                raise ValueError("invalid frozen affine-coordinate scale")
            self.offset[index] = parameters[name]
            self.frozen_units[index] = unit
            inventory[name] = {"center": float(parameters[name]), "unit": unit}
        audit = {
            "protocol": "centered-net-affine-meal-chart-1",
            "meal_chart": self.meal.audit,
            "shared_coordinates": inventory,
            "rule": "min(abs(net coefficient)/abs(coefficient derivative)); "
            "zero coefficient uses inherited self-rate scale",
            "zero_coefficient_reference": float(feedback_unit),
            "frozen_at_start": True,
            "domains_unchanged": True,
            "stability_restriction_added": False,
        }
        object.__setattr__(
            self, "audit", audit | {"sha256": public.content_sha256(audit)}
        )

    def __getattr__(self, name):
        return getattr(self.meal, name)

    def validate_use(self, oracle, fixed):
        self.meal.validate_use(oracle, fixed)

    @property
    def lower(self):
        return self.meal.lower - self.offset

    @property
    def upper(self):
        return self.meal.upper - self.offset

    def encode(self, parameters):
        return self.meal.encode(parameters) - self.offset

    def decode(self, point):
        return self.meal.decode(np.asarray(point) + self.offset)

    def units(self, parameters):
        self.meal.encode(parameters)
        return self.frozen_units.copy()


class InsulinRevisionFit:
    """Fit one admitted revision while preserving its complete nested incumbent."""

    def __init__(
        self,
        problem,
        parent_problem,
        training,
        numerical,
        warm_start_parameters,
        *,
        delay_start=0.0,
    ):
        if training.name != "train" or not np.isfinite(delay_start):
            raise ValueError("requires public training and a finite signed start")
        self.problem, self.parent = problem, parent_problem
        self.training, self.numerical = training, numerical
        self.warm = dict(warm_start_parameters)
        if self.warm.get(DELAY) != 0:
            raise ValueError("nested supplied warm start must set feedback to zero")
        self.parent_original = kernel.TrainingOracle(
            parent_problem, training, numerical
        )
        self.parent_parameters = {k: v for k, v in self.warm.items() if k != DELAY}
        self.parent_original.vector(self.parent_parameters)
        original, _ = polish.make_oracle(problem, training, numerical, self.warm)
        self.gauge = certify_production_gauge(original)
        self.anchor = canonical_parameters(original, self.warm)
        start = self.anchor | {DELAY: float(delay_start)}
        self.original, self.oracle = polish.make_oracle(
            problem, training, numerical, start
        )
        self.start = dict(self.oracle.start)
        if self.start.get(GAIN) != 1 or DELAY not in self.oracle.names:
            raise ValueError("certified gauge coordinates were unexpectedly removed")
        self.fixed = {GAIN: 1.0}
        self.profiler_factory = partial(
            joint.JointMealProfile, additional_forcing="c_I"
        )
        profiler = self.profiler_factory(self.oracle)
        if DELAY not in profiler.audit["outer_names"]:
            raise ValueError(
                "feedback coefficient must remain a matrix/outer coordinate"
            )
        self.profile_audit = profiler.audit
        matrix = _matrix(self.original, self.anchor)
        names = self.original.system.model.state_names
        self.feedback_unit = max(
            abs(matrix[names.index(s), names.index(s)]) for s in ("I", MEMORY)
        )
        if not np.isfinite(self.feedback_unit) or self.feedback_unit <= 0:
            raise ValueError(
                "cannot derive feedback coordinate scale from inherited dynamics"
            )
        self.feedback_unit = float(self.feedback_unit)
        self.chart = CenteredMealChart(
            self.oracle,
            continuation.to_effective(self.original, self.oracle, self.anchor),
            profiler,
            self.feedback_unit,
        )

    def setup_audit(self):
        """Expose frozen chart, scopes and full original-vector inventory."""
        return {
            "protocol": PROTOCOL,
            "passed": True,
            "original_coordinates": len(self.original.names),
            "reduced_coordinates": len(self.oracle.names),
            "active_outer_coordinates": len(self.profile_audit["outer_names"]) - 1,
            "trajectory_roots": len(self.original.roots) * len(self.original.rows),
            "production_gauge": self.gauge,
            "reduction": self.oracle.reduction.receipt,
            "profile": self.profile_audit,
            "meal_chart": self.chart.audit,
            "feedback_outer_coordinate": DELAY,
            "feedback_coordinate_unit": self.feedback_unit,
            "feedback_start": self.start[DELAY],
            "feedback_domain": ["-inf", "+inf"],
            "timescale_chart": (
                "not used: original shared-product coefficient is nonlinear"
            ),
            "positive_meal_rate_stratum": True,
            "global_minimum_certified": False,
        }

    def verify_start(self, expected_nmse, seconds):
        """Check full parent nesting, original replay, and the canonical chart."""
        deadline = monotonic() + seconds
        parent = _polynomials(self.parent_original)
        nested = {
            key: {
                powers: value
                for powers, value in poly.items()
                if DELAY not in dict(powers)
            }
            for key, poly in _polynomials(self.original).items()
        }
        if parent != nested or self.parent.graph != self.problem.graph:
            raise ValueError(
                "zero-feedback executable model is not the complete parent"
            )
        residuals = []
        for oracle, point in (
            (self.parent_original, self.parent_parameters),
            (self.original, self.warm),
            (self.original, self.anchor),
        ):
            residuals.append(
                oracle.evaluate(
                    oracle.vector(point), deadline, jacobian=False, method="Radau"
                )[0]
            )
        effective = continuation.to_effective(self.original, self.oracle, self.anchor)
        chart, _ = self.oracle.evaluate(
            self.oracle.vector(effective), deadline, jacobian=False
        )
        loss = float(np.mean(residuals[1] ** 2))
        gap = max(
            float(np.max(np.abs(r - residuals[0]))) for r in (*residuals[1:], chart)
        )
        chart_loss = float(np.mean(chart**2))
        passed = (
            abs(loss - expected_nmse) <= max(1e-10, 1e-8 * expected_nmse)
            and gap <= 1e-7
            and abs(loss - chart_loss) <= max(1e-12, 1e-8 * loss)
        )
        return {
            "passed": bool(passed),
            "original_equations_verified": bool(passed),
            "training_nmse": loss,
            "parameters": dict(self.anchor),
            "effective_parameters": effective,
            "parent_training_nmse": float(np.mean(residuals[0] ** 2)),
            "expected_parent_training_nmse": expected_nmse,
            "maximum_scaled_discrepancy": gap,
            "solver_prediction_discrepancy": float(
                np.max(np.abs(chart - residuals[1]))
            ),
            "solver_loss_discrepancy": abs(chart_loss - loss),
            "full_executable_nesting_certified": True,
            "initialization_graph_preserved": True,
            "feedback_coefficient": 0.0,
            "setup": self.setup_audit(),
        }

    def qualify(self, seconds, save):
        """Qualify joint bounded forcing/root profiling at this task's start."""
        return roots.root_profile_qualification(
            self.oracle,
            self.start,
            seconds,
            save,
            profiler_factory=self.profiler_factory,
        )

    def search(self, policy, save):
        """Search signed matrix coefficients with bounded, durable improvements."""
        return roots.root_profile_search(
            self.oracle,
            self.start,
            policy.search_seconds,
            policy.maximum_calls,
            save,
            fixed=self.fixed,
            profiler_factory=self.profiler_factory,
            coordinate_chart=self.chart,
            failure_diagnostics=True,
            checkpoint_policy=SearchCheckpointPolicy(
                interval_seconds=policy.checkpoint_interval_seconds
            ),
            overall_wall_seconds=policy.search_seconds
            + policy.persistence_headroom_seconds,
        )

    def verify(self, parameters, expected_nmse, seconds, save):
        """Independently replay the original equations before endpoint retention."""
        result = portfolio.verify_endpoint(
            self.original, self.oracle, parameters, seconds, save
        )
        if result.get("original_equations_verified") and not np.isclose(
            result["training_nmse"], expected_nmse, rtol=1e-8, atol=1e-10
        ):
            result.update(
                original_equations_verified=False,
                error="replay differs from trial loss",
            )
        return result

    def metrics(self, endpoint, seconds):
        """Replay training and report sampled positivity and conditional stability."""
        p = endpoint["parameters"]
        residual, _ = self.original.evaluate(
            self.original.vector(p),
            monotonic() + seconds,
            jacobian=False,
            method="Radau",
        )
        score = float(np.mean(residual**2))
        if not np.isclose(score, endpoint["training_nmse"], atol=1e-10, rtol=1e-8):
            raise ValueError("retained endpoint differs from diagnostic replay")
        channels = self.original.system.channels
        metrics = meal_revision_metrics.summarize(
            self.training, channels, self.original.scales, residual
        )
        values = residual.reshape(-1, len(channels))
        positivity = {}
        for i, name in enumerate(channels):
            observed = np.concatenate(
                [np.asarray(row.targets[name]) for row in self.training.rows]
            )
            predicted = observed + self.original.scales[name] * values[:, i]
            positivity[name] = {
                "minimum": float(np.min(predicted)),
                "negative_samples": int(np.count_nonzero(predicted < 0)),
                "sample_count": len(predicted),
            }
        matrix = _matrix(self.original, p)
        names = self.original.system.model.state_names
        indices = [names.index(name) for name in ("Gp", "I", MEMORY)]
        kappa = float(matrix[names.index("I"), names.index(MEMORY)])
        neutral = [name for i, name in enumerate(names) if not matrix[i].any()]
        return {
            **metrics,
            "training_sha256": public.content_sha256(self.training),
            "metrics": metrics,
            "residual": residual.tolist(),
            "channels": list(channels),
            "scales": dict(self.original.scales),
            "problem_sha256": public.content_sha256(self.problem),
            "verified_training_nmse": score,
            "diagnostics": {
                "declared_feedback_parameter": float(p[DELAY]),
                "production_outer_gain": float(p[GAIN]),
                "effective_kappa": kappa,
                "kappa_sign": "positive"
                if kappa > 0
                else "negative"
                if kappa < 0
                else "zero",
                "sampled_output_positivity": positivity,
                "positivity_scope": (
                    "supplied training observation nodes only; "
                    "not a continuous-time guarantee"
                ),
                "full_state_names": list(names),
                "full_state_matrix": matrix.tolist(),
                "full_state_matrix_eigenvalues": _eigenvalues(matrix),
                "conditional_Gp_I_X_eigenvalues": _eigenvalues(
                    matrix[np.ix_(indices, indices)]
                ),
                "conditional_scope": (
                    "homogeneous Gp/I/X subsystem with other states held fixed"
                ),
                "zero_dynamics_rows": neutral,
                "neutral_mode_caveat": (
                    "Q/Uii has constant or drift dynamics; zero homogeneous mode "
                    "does not certify full asymptotic stability"
                ),
                "stability_constraint_enforced": False,
                "scientific_validity_certified": False,
            },
        }
