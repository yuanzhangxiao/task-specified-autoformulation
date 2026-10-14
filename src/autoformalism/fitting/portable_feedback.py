"""Read-only, portable verification of fixed-vector staged diagnostic evidence.

Hashes and discrete conclusions remain exact. Only named floating-point summaries
are checked numerically, against saved residuals/Jacobians. No model is executed.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import piecewise_assessment as piecewise
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import staged_feedback as staged

# Verification tolerances, not scientific acceptance thresholds. The absolute
# SVD allowance scales with the backward error of the saved matrix; fixed-vector
# derivative summaries allow only arithmetic roundoff at the saved Jacobian scale.
SUMMARY_RTOL = 2e-10
UNIT_ATOL = 1e-14


@dataclass(frozen=True)
class Snapshot:
    """Verified immutable-source provenance and exact reusable experiment inputs."""

    problem: Any
    fit: Any
    backend: dict
    policy: staged.Policy
    packet: staged.FeedbackPacket
    operations: dict
    plan: tuple[piecewise.Perturbation, ...]
    source_sha256: str
    provenance: dict


def numerical_environment() -> dict:
    """Record platform and linked BLAS/LAPACK configuration without secret envs."""
    return {
        "protocol": "diagnostic-numerical-environment-1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_build": np.show_config(mode="dicts"),
        "scipy_build": scipy.__config__.CONFIG,
        "thread_environment": {
            key: os.environ[key]
            for key in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
            if key in os.environ
        },
    }


def save_environment(folder: Path) -> None:
    """Freeze the execution environment once; resume retains original provenance."""
    path = folder / "numerical_environment.json"
    if path.exists():
        read_seal(path)
    else:
        seal(path, numerical_environment())


def _validate_plan(oracle, vector, active, policy, jacobian, plan):
    """Verify all prescribed axes/steps and exact points using the frozen weak axis."""
    weak = next((p.direction_vector for p in plan if p.direction == "weak_joint"), None)
    if policy.include_weak_direction and active and weak is None:
        raise ValueError("frozen perturbation plan is missing its weak direction")
    expected = piecewise.perturbation_plan(
        oracle,
        vector,
        active,
        policy,
        jacobian=jacobian,
        weak_direction=weak,
    )
    if plan != expected:
        raise ValueError(
            "frozen perturbation plan differs from exact prescribed points"
        )
    return plan


def freeze_plan(folder, binding, oracle, vector, active, policy, jacobian):
    """Seal actual directions and points before any call, and reuse them on resume."""
    path = folder / "perturbation_plan.json"
    if path.exists():
        raw = read_seal(path)
        if set(raw) != {"protocol", "identity", "environment_sha256", "points"} or (
            raw["protocol"] != "diagnostic-perturbation-plan-1"
            or raw["identity"] != binding.model_dump(mode="json")
            or raw["environment_sha256"]
            != public.content_sha256(read_seal(folder / "numerical_environment.json"))
        ):
            raise ValueError("frozen perturbation plan identity differs")
        plan = tuple(piecewise.Perturbation.model_validate(p) for p in raw["points"])
        return _validate_plan(oracle, vector, active, policy, jacobian, plan)
    if (folder / "stages/perturb").exists():
        raise ValueError(
            "legacy partial perturbations lack a frozen plan; inspect separately"
        )
    plan = piecewise.perturbation_plan(
        oracle, vector, active, policy, jacobian=jacobian
    )
    seal(
        path,
        {
            "protocol": "diagnostic-perturbation-plan-1",
            "identity": binding.model_dump(mode="json"),
            "environment_sha256": public.content_sha256(
                read_seal(folder / "numerical_environment.json")
            ),
            "points": [p.model_dump(mode="json") for p in plan],
        },
    )
    return plan


def _plan(folder, saved, problem, fit, policy, operations):
    """Recover M32's executed directions; new evidence carries an explicit plan."""
    assessment = saved.diagnostics.piecewise_assessment
    if assessment is None:
        if (folder / "perturbation_plan.json").exists():
            raise ValueError("frozen perturbation plan without an assessment")
        return ()
    primary = staged._value(operations, "primary")
    sensitivity = staged._value(operations, "sensitivity")
    if not primary or not sensitivity:
        raise ValueError("piecewise plan lacks its saved base/Jacobian")
    oracle = staged._proxy(problem, primary["metadata"])
    vector = np.array([fit.parameters[name] for name in oracle.names])
    active = primary["metadata"]["activity"]["active_parameters"]
    jacobian = np.asarray(sensitivity["jacobian"])
    if (folder / "perturbation_plan.json").exists():
        # Existing-plan branch performs only reads and validation.
        plan = freeze_plan(
            folder,
            saved.identity,
            oracle,
            vector,
            active,
            policy.piecewise,
            jacobian,
        )
    else:
        weak = next(
            (
                d.direction_vector
                for d in assessment.directions
                if d.direction == "weak_joint"
            ),
            None,
        )
        if policy.piecewise.include_weak_direction and active and weak is None:
            raise ValueError("legacy packet is missing the executed weak direction")
        plan = piecewise.perturbation_plan(
            oracle,
            vector,
            active,
            policy.piecewise,
            jacobian=jacobian,
            weak_direction=weak,
        )
    return _validate_plan(oracle, vector, active, policy.piecewise, jacobian, plan)


def _numeric(saved, expected, *, atol, label):
    """Compare one explicitly authorized numerical field; reject malformed values."""
    a, b = np.asarray(saved, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"invalid numerical evidence: {label}")
    if not np.all(np.abs(a - b) <= atol + SUMMARY_RTOL * np.abs(b)):
        raise ValueError(f"numerical evidence differs: {label}")


def _sensitivity(saved, expected, operations):
    """Check stable singular data and a nonunique weakest subspace independently."""
    if saved is None or expected is None:
        if saved != expected:
            raise ValueError("active sensitivity availability differs")
        return
    primary = staged._value(operations, "primary")
    metadata = primary["metadata"]
    names = metadata["names"]
    active = metadata["activity"]["active_parameters"]
    indices = [i for i, name in enumerate(names) if name in active]
    jacobian = np.asarray(staged._value(operations, "sensitivity")["jacobian"])
    matrix = jacobian[:, indices] * np.asarray(metadata["units"])[indices]
    matrix /= np.sqrt(len(jacobian))
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite saved sensitivity Jacobian")
    singular = np.linalg.svd(matrix, compute_uv=False)
    atol = max(UNIT_ATOL, 128 * np.finfo(float).eps * max(matrix.shape) * singular[0])
    raw, recomputed = saved.model_dump(mode="json"), expected.model_dump(mode="json")
    for key in ("singular_values", "column_norms"):
        _numeric(raw[key], recomputed[key], atol=atol, label=key)
        raw[key] = recomputed[key]
    values = np.asarray(saved.singular_values)
    if np.any(values < 0) or np.any(np.diff(values) > 0):
        raise ValueError("invalid saved singular value ordering")
    rank = int(np.sum(values > max(values[0] * 1e-8, 1e-15)))
    if rank != saved.rank or saved.rank != expected.rank:
        raise ValueError("saved sensitivity rank differs")
    # Near-zero singular values and their condition ratio are not portable
    # elementwise. The ratio must still be internally consistent with the
    # validated saved spectrum, and a full-rank ratio must match recomputation.
    condition = float(values[0] / values[-1]) if values[-1] > 0 else None
    if condition is None:
        if saved.condition_number is not None:
            raise ValueError("condition number differs from saved spectrum")
    else:
        if saved.condition_number is None:
            raise ValueError("condition number missing for saved spectrum")
        _numeric(
            saved.condition_number, condition, atol=UNIT_ATOL, label="condition_number"
        )
        if rank == saved.parameter_count:
            _numeric(
                condition,
                expected.condition_number,
                atol=UNIT_ATOL,
                label="condition_number",
            )
    if set(saved.weak_direction) != set(saved.parameters):
        raise ValueError("weak direction parameter identities differ")
    direction = np.array([saved.weak_direction[name] for name in saved.parameters])
    piecewise.validate_weak_direction(matrix, direction, reduced_spectrum=True)
    raw["weak_direction"] = recomputed["weak_direction"]
    raw["condition_number"] = recomputed["condition_number"]
    if raw != recomputed:
        raise ValueError("sensitivity discrete evidence differs")


def _direction_roundoff(operations: dict, direction: list[float]) -> float:
    """Bound differences between two reductions of the same saved AD direction.

    Cancellation in large columns can leave a tiny directional derivative. Use
    the magnitude of the summed products, not that tiny result, and cover both
    reductions. This affects numerical verification only, never pass thresholds.
    """
    jacobian = np.asarray(staged._value(operations, "sensitivity")["jacobian"])
    units = np.asarray(staged._value(operations, "primary")["metadata"]["units"])
    products = np.abs(jacobian * units) @ np.abs(np.asarray(direction))
    if not np.isfinite(products).all():
        raise ValueError("nonfinite directional roundoff bound")
    gamma = 64 * np.finfo(float).eps * max(1, len(direction))
    return max(UNIT_ATOL, gamma * piecewise._rms(products))


def verify_packet(saved, recomputed, operations) -> None:
    """Verify a packet against reassembly while retaining strict discrete evidence."""
    _sensitivity(
        saved.diagnostics.active_sensitivity,
        recomputed.diagnostics.active_sensitivity,
        operations,
    )
    original = saved.model_dump(mode="json")
    current = recomputed.model_dump(mode="json")
    original["diagnostics"]["active_sensitivity"] = current["diagnostics"][
        "active_sensitivity"
    ]
    left = original["diagnostics"]["piecewise_assessment"]
    right = current["diagnostics"]["piecewise_assessment"]
    if left is not None and right is not None:
        if len(left["directions"]) != len(right["directions"]):
            raise ValueError("piecewise direction count differs")
        for a, b in zip(left["directions"], right["directions"], strict=True):
            roundoff = _direction_roundoff(operations, b["direction_vector"])
            _numeric(
                a["ad_derivative_rms"],
                b["ad_derivative_rms"],
                atol=roundoff,
                label="ad_derivative_rms",
            )
            a["ad_derivative_rms"] = b["ad_derivative_rms"]
            if len(a["steps"]) != len(b["steps"]):
                raise ValueError("piecewise step count differs")
            for sa, sb in zip(a["steps"], b["steps"], strict=True):
                for key in (
                    "central_error",
                    "forward_error",
                    "backward_error",
                    "one_sided_disagreement",
                    "agreement_tolerance",
                    "derivative_noise_floor",
                    "minimum_signal_to_noise",
                ):
                    allowance = (
                        roundoff
                        if key in {"central_error", "forward_error", "backward_error"}
                        else max(UNIT_ATOL, 0.1 * roundoff)
                        if key == "agreement_tolerance"
                        else UNIT_ATOL
                    )
                    _numeric(sa[key], sb[key], atol=allowance, label=key)
                    sa[key] = sb[key]
    if original != current:
        raise ValueError("staged feedback evidence differs")


def load_snapshot(folder: Path, *, runtime=False) -> Snapshot:
    """Validate sealed context, stage identities, frozen points and derived evidence."""
    folder = Path(folder)
    saved = staged.FeedbackPacket.model_validate(read_seal(folder / "result.json"))
    context = read_seal(folder / "context.json")
    if set(context) != {"problem", "fit", "backend", "policy"}:
        raise ValueError("unexpected diagnostic context fields")
    problem = staged.training_problem(context["problem"])
    fit = staged.adapter.FitResult.model_validate(context["fit"])
    backend, policy = (
        context["backend"],
        staged.Policy.model_validate(context["policy"]),
    )
    staged.legacy.validate_evidence(problem, fit, backend)
    expected = staged._binding(
        problem,
        fit,
        policy,
        source=saved.identity.source_sha256,
        runtime=saved.identity.runtime_sha256,
    )
    identity = read_seal(folder / "identity.json")
    if saved.identity != expected or saved.identity.model_dump(mode="json") != identity:
        raise ValueError("staged feedback identity differs")
    if runtime and (
        saved.identity.source_sha256 != public._source_identity()
        or saved.identity.runtime_sha256 != public.content_sha256(public._runtime())
    ):
        raise ValueError("staged feedback runtime/source differs")
    manifest, operations = read_seal(folder / "stages.json"), {}
    stage_environments = {}
    for key, digest in manifest:
        if key in operations or not all(part.isidentifier() for part in key.split("/")):
            raise ValueError("invalid diagnostic stage key")
        stage_path = folder / "stages" / key
        op = read_seal(stage_path / "record.json")
        if {k: v for k, v in op.items() if k != "progress"} != read_seal(
            stage_path / "result.json"
        ):
            raise ValueError("diagnostic stage progress envelope differs")
        if public.content_sha256(op) != digest or any(
            op["identity"].get(k) != v for k, v in identity.items()
        ):
            raise ValueError("diagnostic stage identity differs")
        if read_seal(stage_path / "started.json") != op["identity"]:
            raise ValueError("diagnostic stage started identity differs")
        operations[key] = op
        if (folder / "numerical_environment.json").exists():
            stage_environments[key] = public.content_sha256(
                read_seal(stage_path / "numerical_environment.json")
            )
    plan = _plan(folder, saved, problem, fit, policy, operations)
    staged._validate_operations(
        problem,
        fit,
        backend,
        policy,
        saved.identity,
        operations,
        perturbation_plan=plan,
    )
    recomputed = staged._assemble(
        problem,
        fit,
        backend,
        policy,
        saved.identity,
        operations,
        perturbation_plan=plan,
    )
    verify_packet(saved, recomputed, operations)
    hashes = {
        name: public.content_sha256(read_seal(folder / name))
        for name in ("identity.json", "context.json", "stages.json", "result.json")
    }
    hashes["stage_environments"] = public.content_sha256(stage_environments)
    environment = None
    for name in ("perturbation_plan.json", "numerical_environment.json"):
        if (folder / name).exists():
            value = read_seal(folder / name)
            hashes[name] = public.content_sha256(value)
            if name == "numerical_environment.json":
                environment = value
    return Snapshot(
        problem,
        fit,
        backend,
        policy,
        saved,
        operations,
        plan,
        public.content_sha256(hashes),
        {
            "protocol": "portable-staged-verification-1",
            "artifact_hashes": hashes,
            "plan_origin": "sealed_plan"
            if "perturbation_plan.json" in hashes
            else "legacy_packet",
            "numerical_environment": environment,
            "stage_environment_hashes": stage_environments,
            "legacy_environment_unrecorded": environment is None,
        },
    )
