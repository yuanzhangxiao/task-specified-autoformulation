"""Versioned benchmark-suite design and private-generation contracts."""

from autoformalism.benchmarks.phase_b_gates import (
    MechanismAblationResult,
    MechanismGateDefinition,
    TaskGateReport,
    alien_sensitivity_selected_auxiliaries,
    audit_task_gates,
    mechanism_gate_definition,
)
from autoformalism.benchmarks.phase_b_generation import (
    BasicGateReport,
    PhaseBProtocol,
    PrivateTrajectory,
    audit_basic_gates,
    phase_b_protocols,
    simulate_phase_b,
    write_private_bundle,
)
from autoformalism.benchmarks.phase_b_public import (
    LeakageReport,
    PhaseBPublicSpec,
    PublicChannel,
    audit_public_bundle,
    phase_b_public_spec,
    render_phase_b_prompts,
    write_public_production_bundle,
    write_public_staging_bundle,
)
from autoformalism.benchmarks.suite import BenchmarkSuiteSpec, load_suite_spec

__all__ = [
    "BasicGateReport",
    "BenchmarkSuiteSpec",
    "LeakageReport",
    "MechanismAblationResult",
    "MechanismGateDefinition",
    "PhaseBProtocol",
    "PhaseBPublicSpec",
    "PrivateTrajectory",
    "PublicChannel",
    "TaskGateReport",
    "alien_sensitivity_selected_auxiliaries",
    "audit_basic_gates",
    "audit_public_bundle",
    "audit_task_gates",
    "load_suite_spec",
    "mechanism_gate_definition",
    "phase_b_protocols",
    "phase_b_public_spec",
    "render_phase_b_prompts",
    "simulate_phase_b",
    "write_private_bundle",
    "write_public_production_bundle",
    "write_public_staging_bundle",
]
