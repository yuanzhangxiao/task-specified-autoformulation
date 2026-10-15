"""Shared additive observation drift with exact direct and profiled sensitivities.

Physical integration, explicit preparations, masks and known sigma come from
M34. Drift changes only the observation prediction. No raw data are corrected,
no extra ODE forcing is introduced, and no statistical confidence is inferred.
"""

from __future__ import annotations

from time import monotonic

import numpy as np

from autoformalism.fitting import measurement_rollout, robust_measurement
from autoformalism.schemas.sensor_drift import DriftProblem


class DriftRollouts:
    """Extend a physical rollout by one observation-only parameter and column."""

    def __init__(self, problem: DriftProblem, physical):
        self.problem, self.physical = problem, physical
        drift = problem.drift
        self.names = (*physical.names, drift.name)
        self.lower = np.r_[physical.lower, drift.lower]
        self.upper = np.r_[physical.upper, drift.upper]
        self.units = problem.coordinates.arrays("parameters", self.names)[1]
        self.time_column = np.where(
            np.asarray(physical.layout.channels) == drift.channel,
            np.asarray(physical.layout.analysis_times) - drift.time_origin,
            0.0,
        )
        self.column = self.time_column[physical.mask] / physical.sigma[physical.mask]
        if not np.isfinite(self.column).all():
            raise ValueError("nonfinite shared sensor drift design")
        self.audit = physical.audit | {
            "observation_extension": "shared-additive-linear-sensor-drift-1",
            "observation_nuisance": drift.model_dump(mode="json"),
            "physical_equations_unchanged": True,
        }
        self._last_vector = None
        self._last_residual = None
        for point in (problem.start, problem.incumbent):
            self.vector(point)

    def __getattr__(self, name):
        return getattr(self.physical, name)

    def vector(self, values) -> np.ndarray:
        """Validate the complete physical/nuisance vector against declared bounds."""
        if set(values) != set(self.names):
            raise ValueError("incomplete sensor drift parameter vector")
        vector = np.asarray([values[name] for name in self.names], dtype=float)
        if (
            not np.isfinite(vector).all()
            or np.any(vector < self.lower)
            or np.any(vector > self.upper)
        ):
            raise ValueError("sensor drift parameter vector outside declared domain")
        return vector

    def parameters(self, vector) -> dict:
        """Serialize every physical coefficient and the separate observation slope."""
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (len(self.names),):
            raise ValueError("sensor drift parameter vector has invalid shape")
        values = dict(zip(self.names, vector.tolist(), strict=True))
        self.vector(values)
        return values

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        """Apply the same observation transform after either independent integrator."""
        self._last_vector = None
        self.parameters(vector)
        residual, derivative = self.physical.evaluate(
            np.asarray(vector)[:-1], deadline, jacobian=jacobian, method=method
        )
        residual = residual + self.column * vector[-1]
        derivative = (
            np.column_stack([derivative, self.column])
            if derivative is not None
            else None
        )
        if monotonic() >= deadline:
            raise TimeoutError("sensor drift rollout allowance exhausted")
        if not np.isfinite(residual).all() or (
            derivative is not None and not np.isfinite(derivative).all()
        ):
            raise ValueError("nonfinite sensor drift residual or derivative")
        self._last_vector, self._last_residual = (
            np.asarray(vector).copy(),
            residual.copy(),
        )
        return residual, derivative

    def residual_evidence(self, residual) -> dict:
        """Reconstruct total observation predictions in original raw/analysis units."""
        return self.physical.residual_evidence(residual) | {
            "observation_model": self.audit["observation_extension"],
            "observation_nuisance_definition": self.problem.drift.model_dump(
                mode="json"
            ),
            "raw_readings_modified": False,
            "confidence_probability": None,
        }

    def prediction_evidence(self, residual, parameters) -> dict:
        """Decompose a retained endpoint without integration or mutable cache state."""
        vector = self.vector(parameters)
        evidence = self.residual_evidence(residual)
        channels = {channel.id: channel for channel in self.problem.manifest.channels}
        rows = []
        for index, reading in enumerate(evidence["reading_audit"]):
            prediction = reading["prediction_analysis"]
            contribution = (
                float(self.time_column[index] * vector[-1])
                if prediction is not None
                else None
            )
            physical = prediction - contribution if prediction is not None else None
            channel = channels[reading["channel"]]
            rows.append(
                reading
                | {
                    "drift_contribution_analysis": contribution,
                    "physical_prediction_analysis": physical,
                    "physical_prediction_raw": (physical - channel.offset)
                    / channel.gain
                    if physical is not None
                    else None,
                }
            )
        return evidence | {
            "reading_audit": rows,
            "parameters": dict(parameters),
            "prediction_decomposition": "selected endpoint; total = physical + drift",
        }

    def evidence(self, vector, deadline, *, residual=None) -> dict:
        """Retain complete total and physical prediction evidence without raw edits."""
        if (
            self._last_vector is None
            or not np.array_equal(vector, self._last_vector)
            or not np.array_equal(np.asarray(vector)[:-1], self.physical._last_vector)
        ):
            residual, _ = self.evaluate(vector, deadline, jacobian=False)
        elif residual is None:
            residual = self._last_residual
        if not np.array_equal(residual, self._last_residual):
            raise ValueError(
                "sensor drift evidence residual differs from evaluated point"
            )
        physical_errors = np.concatenate(
            [
                self.physical._last_rows[row.trajectory_id]
                for row in self.train.trajectories
            ]
        )
        complete = physical_errors + self.time_column * vector[-1]
        evidence = self.physical._residual_evidence(residual, complete)
        rows = []
        for index, reading in enumerate(evidence["reading_audit"]):
            contribution = float(self.time_column[index] * vector[-1])
            rows.append(
                reading
                | {
                    "drift_contribution_analysis": contribution,
                    "physical_prediction_analysis": reading["prediction_analysis"]
                    - contribution,
                }
            )
        return evidence | {
            "parameters": self.parameters(vector),
            "reading_audit": rows,
            "observation_model": self.audit["observation_extension"],
            "raw_readings_modified": False,
        }


class DriftProfile:
    """Append the fixed clock column to a certified complete-output affine block."""

    def __init__(self, oracle: DriftRollouts, physical_profile, loss):
        self.oracle, self.physical_profile, self.loss = oracle, physical_profile, loss
        self.outer = tuple(physical_profile.outer)
        physical_inner = (
            physical_profile.inner
            if hasattr(physical_profile, "inner")
            else physical_profile.kernel.gain_indices
        )
        self.inner = (*physical_inner, len(oracle.names) - 1)
        self.audit = physical_profile.audit | {
            "observation_extension": oracle.audit["observation_extension"],
            "gain_names": [oracle.names[i] for i in self.inner],
            "inner_names": [oracle.names[i] for i in self.inner],
            "estimated_unknowns": len(oracle.names),
            "drift_column_independent_of_physical_parameters": True,
        }
        physical_profile.projector = self._project
        self._inner_result = None

    def _project(self, design, offset, d_design, d_offset, lower, upper, *, deadline):
        """Solve the complete physical-plus-drift convex inner problem once."""
        count, _, outer = d_design.shape
        if count != len(self.oracle.column):
            raise ValueError("sensor drift profile reading layout differs")
        result = robust_measurement.project(
            np.column_stack([design, self.oracle.column]),
            offset,
            np.concatenate([d_design, np.zeros((count, 1, outer))], axis=1),
            d_offset,
            np.r_[lower, self.oracle.lower[-1]],
            np.r_[upper, self.oracle.upper[-1]],
            loss=self.loss,
            deadline=deadline,
        )
        self._inner_result = result
        return result | {"gains": result["gains"][:-1]}

    def evaluate(self, vector, deadline):
        """Return fitted physical/nuisance values and complete outer derivatives."""
        self.oracle.parameters(vector)
        self._inner_result = None
        physical, residual, derivative, audit = self.physical_profile.evaluate(
            np.asarray(vector)[:-1], deadline
        )
        if self._inner_result is None:
            raise ValueError("sensor drift inner projection was not evaluated")
        fitted = np.r_[physical, self._inner_result["gains"][-1]]
        self.oracle.parameters(fitted)
        return fitted, residual, derivative, audit


def build(problem: DriftProblem, loss: robust_measurement.Loss) -> tuple:
    """Reuse physical certificates; unsupported profiles retain joint fitting."""
    physical, profile, route = measurement_rollout.build(problem.base)
    oracle = DriftRollouts(problem, physical)
    if profile is not None:
        profile = robust_measurement.RobustProfile(
            DriftProfile(oracle, profile, loss), loss
        )
    route = route | {
        "certificate": profile.audit if profile is not None else None,
        "observation_contract": oracle.audit,
        "loss": loss.model_dump(mode="json"),
        "inner_objective": "convex bounded physical plus sensor drift " + loss.kind,
        "confidence_probability": None,
    }
    return robust_measurement.RobustRollouts(oracle, loss), profile, route
