"""Trajectory simulation and bounded parameter fitting."""

from autoformalism.fitting.fitter import evaluate_fitted_candidate, fit_candidate
from autoformalism.fitting.models import (
    EvaluationMetrics,
    ExactDerivativeFitError,
    FitConfig,
    FitResult,
    OptimizationDiagnostic,
    SimulationResult,
)
from autoformalism.fitting.simulation import simulate_trajectory

__all__ = [
    "EvaluationMetrics",
    "ExactDerivativeFitError",
    "FitConfig",
    "FitResult",
    "OptimizationDiagnostic",
    "SimulationResult",
    "evaluate_fitted_candidate",
    "fit_candidate",
    "simulate_trajectory",
]
