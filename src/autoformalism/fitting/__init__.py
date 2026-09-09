"""Trajectory simulation and bounded parameter fitting."""

from autoformalism.fitting.casadi_initializer import (
    CasadiInitializationResult,
    OptionalFittingDependencyError,
    initialize_parameters_with_multiple_shooting,
)
from autoformalism.fitting.fitter import (
    estimate_profiled_warm_start_from_public_derivatives,
    evaluate_fitted_candidate,
    fit_candidate,
)
from autoformalism.fitting.models import (
    EvaluationMetrics,
    ExactDerivativeFitError,
    FitConfig,
    FitResult,
    InitializationDiagnostic,
    OptimizationDiagnostic,
    SimulationResult,
)
from autoformalism.fitting.simulation import simulate_trajectory

__all__ = [
    "CasadiInitializationResult",
    "EvaluationMetrics",
    "ExactDerivativeFitError",
    "FitConfig",
    "FitResult",
    "InitializationDiagnostic",
    "OptimizationDiagnostic",
    "OptionalFittingDependencyError",
    "SimulationResult",
    "estimate_profiled_warm_start_from_public_derivatives",
    "evaluate_fitted_candidate",
    "fit_candidate",
    "initialize_parameters_with_multiple_shooting",
    "simulate_trajectory",
]
