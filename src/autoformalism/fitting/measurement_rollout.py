"""Training-only Gaussian observation adapters for the certified rollout portfolio.

Measurement standard deviations, calibration and acquisition masks are frozen
inputs. They never derive from candidate residuals or optimizer coordinates.
"""

from __future__ import annotations

from time import monotonic

import numpy as np

from autoformalism.expressions import compile_candidate
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import fitter
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.affine_profile import AffineProfile
from autoformalism.fitting.affine_propagation import AffineRollouts
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.nonlinear_rollout import TerminalProfile
from autoformalism.fitting.sensitivity_probe import ExplicitInitialization, SymbolicODE
from autoformalism.schemas.candidate import CandidateModel
from autoformalism.schemas.measurement_fitting import (
    MeasurementProblem,
    analysis_layout,
    analysis_training,
)


def _compile(problem: MeasurementProblem):
    """Represent shared initial usage without changing the supplied request."""
    payload = problem.request.base_candidate.model_dump(mode="json")
    payload["initial_conditions"] = [
        {
            "state": rule.state,
            "scope": "global",
            **(
                {"expression": rule.shared_coordinate}
                if rule.mode == "shared_value"
                else {"fixed_value": rule.exact_boundaries[0].value}
            ),
        }
        for rule in problem.initial_rules
    ]
    context = problem.request.context.model_copy(update={"fitted_initialization": True})
    return compile_candidate(CandidateModel.model_validate(payload), context)


class MeasurementRollouts(AffineRollouts):
    """Whiten complete output sensitivities and use explicit physical boundaries."""

    def __init__(
        self, problem: MeasurementProblem, policy: checks.ConfidencePolicy | None = None
    ) -> None:
        policy = policy or checks.ConfidencePolicy()
        self.problem = problem
        # The old lowering binds observed initials to readings. This opt-in
        # adapter compiles the original equations and supplies all boundaries.
        self.model = _compile(problem)
        exact, shared = {}, {}
        for rule in problem.initial_rules:
            if rule.mode == "exact_boundary":
                exact[rule.state] = {
                    entry.trajectory: entry.value for entry in rule.exact_boundaries
                }
            else:
                shared[rule.state] = rule.shared_coordinate
        self.system = SymbolicODE(
            self.model,
            initialization_override=ExplicitInitialization(exact, shared),
        )
        self.names = self.system.names
        self.train = public.unpack_split(analysis_training(problem))
        self.layout = analysis_layout(problem, self.system.channels)
        self.settings = FitConfig(
            integration_method="DOP853",
            relative_tolerance=policy.relative_tolerance,
            absolute_tolerance=policy.absolute_tolerance,
        )
        variables = [
            fitter._parameter_variable(self.model, p, self.settings)
            for p in self.model.validated.candidate.parameters
        ]
        if tuple(v.name for v in variables) != tuple(
            f"parameter:{n}" for n in self.names
        ):
            raise ValueError("measurement parameter order differs from compiled model")
        self.lower = np.asarray([v.lower for v in variables])
        self.upper = np.asarray([v.upper for v in variables])
        self.units = problem.coordinates.arrays("parameters", self.names)[1]
        problem.coordinates.arrays("states", self.model.state_names)
        self.mask = np.asarray(self.layout.included, bool)
        self.sigma = np.asarray(
            [
                s if included else 1.0
                for s, included in zip(self.layout.sigma, self.mask, strict=True)
            ],
            float,
        )
        self.observations = np.asarray(
            [value if value is not None else 0.0 for value in self.layout.values]
        )
        self.likelihood_constant = float(
            np.log(self.sigma[self.mask]).sum()
            + self.mask.sum() * np.log(2 * np.pi) / 2
        )
        self.scales = {}
        for channel in self.system.channels:
            selected = self.mask & (np.asarray(self.layout.channels) == channel)
            self.scales[channel] = max(
                float(np.std(self.observations[selected])), 1e-12
            )
        self._slices = {}
        index = 0
        for row in self.train.trajectories:
            size = len(row.time) * len(self.system.channels)
            self._slices[row.trajectory_id] = slice(index, index + size)
            index += size
        if index != len(self.mask) or not self.mask.any():
            raise ValueError("measurement layout does not cover training outputs")
        self._last_rows: dict[str, np.ndarray] = {}
        self._last_vector: np.ndarray | None = None
        self._last_residual: np.ndarray | None = None
        self.rejected = []
        try:
            self._build_affine_generator()
            self.is_affine = True
        except ValueError as error:
            self.is_affine = False
            self.audit = {"certificate": "restricted-symbolic-ode"}
            self.rejected.append({"kernel": "affine_rollout", "reason": str(error)})
        self.audit |= {
            "observation_objective": "fixed-known-diagonal-Gaussian",
            "initialization": "explicit-all-state-boundaries",
            "reading_count": int(self.mask.sum()),
            "missing_count": int((~self.mask).sum()),
            "stack_order": "trajectory,time,channel; present readings once",
        }
        for point in (problem.start, problem.incumbent):
            self.vector(point)

    def residual_transform(self, row, residual, jacobian):
        """Apply the identical mask and whitening to residuals and Jacobian rows."""
        section = self._slices[row.trajectory_id]
        mask, sigma = self.mask[section], self.sigma[section]
        values = np.asarray(residual).ravel()
        self._last_rows[row.trajectory_id] = values.copy()
        derivative = (
            np.asarray(jacobian).reshape(len(values), len(self.names))[mask]
            / sigma[mask, None]
            if jacobian is not None
            else None
        )
        return values[mask] / sigma[mask], derivative

    def profile_transform(
        self, row, design, offset, d_design, d_offset, *, observations=None
    ):
        """Whiten basis columns, offsets and derivatives before the inner solve."""
        if len(self.system.channels) != 1:
            raise ValueError("measurement profiling requires one identity output")
        section = self._slices[row.trajectory_id]
        mask, sigma = self.mask[section], self.sigma[section]
        if observations is not None:
            offset = offset - observations
        return (
            design[mask] / sigma[mask, None],
            offset[mask] / sigma[mask],
            d_design[mask] / sigma[mask, None, None],
            d_offset[mask] / sigma[mask, None],
        )

    def evaluate(self, x, deadline, *, jacobian=True, method=None):
        """Return standardized readings once; failed trajectories remain unavailable."""
        self._last_vector = None
        self._last_rows.clear()
        if monotonic() >= deadline:
            raise TimeoutError("measurement rollout allowance exhausted")
        kernel = AffineRollouts.evaluate if self.is_affine else checks.Rollouts.evaluate
        residual, derivative = kernel(
            self, x, deadline, jacobian=jacobian, method=method
        )
        if not np.isfinite(residual).all() or (
            derivative is not None and not np.isfinite(derivative).all()
        ):
            raise ValueError("nonfinite measurement residual or sensitivity")
        self._last_vector = np.asarray(x).copy()
        self._last_residual = residual.copy()
        return residual, derivative

    def objective(self, residual: np.ndarray) -> dict:
        """Report the sum likelihood in declared analysis units, including constants."""
        if (
            np.shape(residual) != (int(self.mask.sum()),)
            or not np.isfinite(residual).all()
        ):
            raise ValueError("objective residual layout is invalid")
        quadratic = float(0.5 * np.dot(residual, residual))
        if not np.isfinite(quadratic + self.likelihood_constant):
            raise ValueError("measurement objective is nonfinite")
        return {
            "quadratic_objective": quadratic,
            "gaussian_normalization": self.likelihood_constant,
            "gaussian_nll": quadratic + self.likelihood_constant,
            "reading_count": int(self.mask.sum()),
            "normalization_units": "analysis",
        }

    def evidence(self, x, deadline, *, residual=None) -> dict:
        """Retain raw and analysis residuals, reusing an already evaluated point."""
        if self._last_vector is None or not np.array_equal(x, self._last_vector):
            residual, _ = self.evaluate(x, deadline, jacobian=False)
        elif residual is None:
            residual = self._last_residual
        if not np.array_equal(residual, self._last_residual):
            raise ValueError("evidence residual differs from evaluated point")
        all_errors = np.concatenate(
            [self._last_rows[row.trajectory_id] for row in self.train.trajectories]
        )
        return self._residual_evidence(residual, all_errors) | {
            "parameters": self.parameters(x)
        }

    def residual_evidence(self, residual: np.ndarray) -> dict:
        """Reconstruct present-reading evidence without a new numerical solve."""
        self.objective(residual)
        all_errors = np.zeros(len(self.mask))
        all_errors[self.mask] = residual * self.sigma[self.mask]
        return self._residual_evidence(residual, all_errors, missing_predictions=False)

    def _residual_evidence(self, residual, all_errors, *, missing_predictions=True):
        """Use the immutable layout to decompose one complete Gaussian objective."""
        channel_specs = {c.id: c for c in self.problem.manifest.channels}
        raw_readings = {
            reading.id: reading
            for row in self.problem.training.rows
            for reading in row.readings
        }
        records, nmse, squared_raw = [], [], []
        for i, reading_id in enumerate(self.layout.reading_ids):
            channel = self.layout.channels[i]
            spec, reading = channel_specs[channel], raw_readings[reading_id]
            included = bool(self.mask[i])
            predicted = (
                float(all_errors[i] + self.observations[i])
                if included or missing_predictions
                else None
            )
            error = float(all_errors[i]) if included else None
            raw_error = error / spec.gain if included else None
            if included:
                nmse.append((error / self.scales[channel]) ** 2)
                squared_raw.append(raw_error**2)
            records.append(
                {
                    "reading_id": reading_id,
                    "trajectory_id": self.layout.trajectory_ids[i],
                    "channel": channel,
                    "analysis_time": self.layout.analysis_times[i],
                    "included": included,
                    "raw_value": reading.raw_value,
                    "analysis_value": self.layout.values[i],
                    "prediction_analysis": predicted,
                    "prediction_raw": (predicted - spec.offset) / spec.gain
                    if predicted is not None
                    else None,
                    "residual_analysis": error,
                    "residual_raw": raw_error,
                    "sigma_analysis": self.layout.sigma[i],
                    "sigma_raw": self.layout.sigma[i] / abs(spec.gain)
                    if included
                    else None,
                    "standardized_residual": error / self.sigma[i]
                    if included
                    else None,
                    "quadratic_contribution": 0.5 * (error / self.sigma[i]) ** 2
                    if included
                    else None,
                }
            )
        decomposition = {}
        for dimension in ("channel", "trajectory_id"):
            summaries = []
            for name in dict.fromkeys(record[dimension] for record in records):
                present = [
                    record
                    for record in records
                    if record[dimension] == name and record["included"]
                ]
                summaries.append(
                    {
                        dimension: name,
                        "reading_count": len(present),
                        "quadratic_objective": sum(
                            r["quadratic_contribution"] for r in present
                        ),
                        "analysis_residual_sum_squares": sum(
                            r["residual_analysis"] ** 2 for r in present
                        ),
                        "raw_residual_sum_squares": sum(
                            r["residual_raw"] ** 2 for r in present
                        ),
                    }
                )
            decomposition[dimension] = summaries
        return self.objective(residual) | {
            "reading_audit": records,
            "residual_decomposition": decomposition,
            "training_nmse": float(np.mean(nmse)),
            "raw_residual_sum_squares": float(np.sum(squared_raw)),
            "analysis_residual_sum_squares": float(np.sum(all_errors[self.mask] ** 2)),
            "nmse_denominators": {c: s * s for c, s in self.scales.items()},
            "nmse_convention": (
                "mean squared residual divided by present training channel variance"
            ),
            "confidence_probability": None,
            "statistical_adequacy": "uncalibrated",
        }


def build(problem: MeasurementProblem, policy: checks.ConfidencePolicy | None = None):
    """Retain certified affine/terminal profiles, otherwise use joint rollouts."""
    oracle = MeasurementRollouts(problem, policy)
    rejected = list(oracle.rejected)
    # The active-parameter opt-in permits a purely affine, zero-outer solve.
    active = frozenset(oracle.names)
    choices = ([("affine_profile", AffineProfile)] if oracle.is_affine else []) + [
        ("terminal_output_profile", TerminalProfile)
    ]
    for name, factory in choices:
        try:
            profile = factory(oracle, active_parameters=active)
            return (
                oracle,
                profile,
                {
                    "selected": name,
                    "certificate": profile.audit,
                    "observation_contract": oracle.audit,
                    "inapplicable": rejected,
                },
            )
        except ValueError as error:
            rejected.append({"kernel": name, "reason": str(error)})
    return (
        oracle,
        None,
        {
            "selected": "joint_rollout",
            "observation_contract": oracle.audit,
            "inapplicable": rejected,
            "reason": "No complete-output affine block was certified.",
        },
    )
