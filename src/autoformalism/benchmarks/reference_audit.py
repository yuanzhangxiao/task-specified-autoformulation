"""Checkpointed, evaluator-only numerical audit of every Phase-B family.

No discovery candidates, model scores, fitted parameters or LLMs are used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from autoformalism.benchmarks.phase_b_generation import (
    PhaseBProtocol,
    PrivateTrajectory,
    forcing_boundaries,
    phase_b_protocols,
    simulate_phase_b,
)
from autoformalism.benchmarks.phase_b_public import (
    PhaseBPublicSpec,
    _numeric_payload_sha256,
    _private_array,
    audit_public_bundle,
    phase_b_public_spec,
    write_public_staging_bundle,
)
from autoformalism.benchmarks.suite import load_suite_spec
from autoformalism.rebuttal.dalla_man import DallaManParameters
from autoformalism.reference_integration import REFERENCE_PROTOCOL, ReferenceSolver

PRIMARY = ReferenceSolver()
CHECK = ReferenceSolver(method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.25)
AGREEMENT_LIMIT = 2e-6


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compare_references(a: PrivateTrajectory, b: PrivateTrajectory) -> dict:
    """Compare every state, derived channel and derivative on the same grid."""
    if (
        not np.array_equal(a.time, b.time)
        or not np.array_equal(a.inputs, b.inputs)
        or a.state_names != b.state_names
        or a.input_names != b.input_names
        or set(a.derived) != set(b.derived)
        or set(a.derivatives) != set(b.derivatives)
    ):
        raise ValueError("reference comparison grids/channels/inputs differ")
    rows = {}
    for kind, first, second in (
        (
            "state",
            dict(zip(a.state_names, a.states.T, strict=True)),
            dict(zip(b.state_names, b.states.T, strict=True)),
        ),
        ("derived", a.derived, b.derived),
        ("derivative", a.derivatives, b.derivatives),
    ):
        for name in first:
            difference = float(np.max(np.abs(first[name] - second[name])))
            scale = max(1.0, float(np.max(np.abs(second[name]))))
            rows[f"{kind}:{name}"] = {
                "max_absolute_difference": difference,
                "scaled_max_difference": difference / scale,
            }
    worst = max(value["scaled_max_difference"] for value in rows.values())
    return {
        "passed": worst <= AGREEMENT_LIMIT,
        "limit": AGREEMENT_LIMIT,
        "maximum_scaled_difference": worst,
        "channels": rows,
    }


def check_identities(protocol: PhaseBProtocol, trajectory: PrivateTrajectory) -> dict:
    """Check public quantities against physical balances and event identities."""
    states = dict(zip(trajectory.state_names, trajectory.states.T, strict=True))
    d = trajectory.derivatives
    derived = trajectory.derived
    residuals = {}
    if protocol.family == "dalla_man":
        p = DallaManParameters()
        residuals = {
            "G_equals_Gp_over_VG": derived["G"] - states["Gp"] / p.VG,
            "I_equals_Ip_over_VI": derived["I"] - states["Ip"] / p.VI,
            "U_equals_Uii_plus_Uid": derived["U"] - derived["Uii"] - derived["Uid"],
            "glucose_mass_balance": d["Gp"]
            + d["Gt"]
            - (
                derived["EGP"]
                + derived["Ra"]
                - derived["U"]
                - derived["E"]
                + derived["glucose_forcing"]
            ),
            "gut_mass_between_jumps": d["Qsto1"]
            + d["Qsto2"]
            + d["Qgut"]
            + p.kabs * states["Qgut"],
        }
        exact_q1 = np.zeros_like(trajectory.time)
        meals = protocol.specification.get("meals", [])
        for time, grams in meals:
            exact_q1 += (
                (trajectory.time >= time)
                * grams
                * 1000
                * np.exp(-p.kgri * np.maximum(trajectory.time - time, 0))
            )
        residuals["exact_meal_jump_response"] = (states["Qsto1"] - exact_q1) / max(
            1.0, float(np.max(exact_q1))
        )
        index = trajectory.input_names.index("meal_event_g")
        residuals["exported_event_mass_g"] = np.array(
            [trajectory.inputs[:, index].sum() - sum(grams for _, grams in meals)]
        )
    maxima = {key: float(np.max(np.abs(value))) for key, value in residuals.items()}
    return {
        "passed": all(value <= 2e-6 for value in maxima.values()),
        "maximum_identity_residuals": maxima,
    }


def _save_trajectory(path: Path, trajectory: PrivateTrajectory) -> None:
    with path.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(
            handle,
            time=trajectory.time,
            states=trajectory.states,
            inputs=trajectory.inputs,
            **{f"derived__{k}": v for k, v in trajectory.derived.items()},
            **{f"derivative__{k}": v for k, v in trajectory.derivatives.items()},
        )
    path.with_suffix(".npz.tmp").replace(path)


def _load_trajectory(path: Path, metadata: dict) -> PrivateTrajectory:
    with np.load(path, allow_pickle=False) as arrays:
        return PrivateTrajectory(
            **metadata,
            time=arrays["time"],
            states=arrays["states"],
            inputs=arrays["inputs"],
            derived={
                k.removeprefix("derived__"): arrays[k]
                for k in arrays.files
                if k.startswith("derived__")
            },
            derivatives={
                k.removeprefix("derivative__"): arrays[k]
                for k in arrays.files
                if k.startswith("derivative__")
            },
        )


def audit_one(
    root: Path, protocol: PhaseBProtocol, dynamics: str, data_root: Path, identity: str
) -> tuple[dict, PrivateTrajectory]:
    """Resume only identical source/settings/data; check cached file contents."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{protocol.protocol_id}.json"
    array_path = path.with_suffix(".npz")
    key = _hash([identity, protocol.model_dump(mode="json"), dynamics])
    if path.exists():
        record = json.loads(path.read_text())
        seal = record.pop("record_sha256")
        if (
            record["identity"] != key
            or record["array_sha256"] != _digest(array_path)
            or seal != _hash(record)
        ):
            raise ValueError(
                "reference audit checkpoint changed; use a new output root"
            )
        record["record_sha256"] = seal
        return record, _load_trajectory(array_path, record["trajectory_metadata"])
    primary = simulate_phase_b(
        protocol, dynamics=dynamics, data_root=data_root, solver=PRIMARY
    )
    independent = simulate_phase_b(
        protocol, dynamics=dynamics, data_root=data_root, solver=CHECK
    )
    refined = simulate_phase_b(
        protocol.model_copy(update={"dt": protocol.dt / 2}),
        dynamics=dynamics,
        data_root=data_root,
        solver=PRIMARY,
    )
    sampled = refined.model_copy(
        update={
            "time": refined.time[::2],
            "states": refined.states[::2],
            "inputs": refined.inputs[::2],
            "derived": {k: v[::2] for k, v in refined.derived.items()},
            "derivatives": {k: v[::2] for k, v in refined.derivatives.items()},
        }
    )
    comparison = compare_references(primary, independent)
    grid_comparison = compare_references(primary, sampled)
    identities = check_identities(protocol, primary)
    metadata = primary.model_dump(
        mode="json", exclude={"time", "states", "inputs", "derived", "derivatives"}
    )
    _save_trajectory(array_path, primary)
    record = {
        "identity": key,
        "protocol": protocol.model_dump(mode="json"),
        "dynamics": dynamics,
        "boundaries": list(forcing_boundaries(protocol)),
        "independent_solver": comparison,
        "output_grid_refinement": grid_comparison,
        "identities": identities,
        "trajectory_metadata": metadata,
        "array_sha256": _digest(array_path),
        "passed": comparison["passed"]
        and grid_comparison["passed"]
        and identities["passed"],
    }
    record["record_sha256"] = _hash(record)
    _write(path, record)
    return record, primary


def check_projection(
    path: Path, spec: PhaseBPublicSpec, trajectories: tuple[PrivateTrajectory, ...]
) -> bool:
    """Read exported CSV values back; verify targets and auxiliaries together."""
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        for number, trajectory in enumerate(trajectories):
            split = trajectory.protocol_id.split("_", 1)[0]
            arrays = {
                c.public_name: _private_array(trajectory, c.private_source)
                for c in spec.channels
            }
            for i, time in enumerate(trajectory.time):
                expected = {
                    "trajectory_id": f"{split}_{number:03d}",
                    "t": f"{time:.12g}",
                    **{name: f"{values[i]:.12g}" for name, values in arrays.items()},
                }
                if next(rows, None) != expected:
                    return False
        return next(rows, None) is None


def run_audit(
    root: Path,
    data_root: Path,
    suite_path: Path,
    *,
    include_test_protocols: bool = False,
) -> dict:
    """Audit unique numerical configurations, then all 40 public projections."""
    suite = load_suite_spec(suite_path)
    source_root = Path(__file__).parents[1]
    sources = {
        str(p.relative_to(source_root)): _digest(p)
        for p in [
            source_root / "reference_integration.py",
            source_root / "rebuttal/dalla_man.py",
            *sorted((source_root / "benchmarks").glob("*.py")),
        ]
    }
    plan = {
        "protocol": "phase-b-reference-integrity-audit-1",
        "reference_generation_protocol": REFERENCE_PROTOCOL,
        "primary": PRIMARY.model_dump(),
        "independent": CHECK.model_dump(),
        "agreement_limit": AGREEMENT_LIMIT,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": sources,
        "suite_sha256": _digest(suite_path),
        "private_spec_sha256": {
            str(p.relative_to(data_root)): _digest(p)
            for p in (
                data_root
                / "benchmark5_anonymous_nonlinear_process"
                / "private/system_specification.json",
                data_root / "benchmark6_alien_device/private/selected_system_spec.json",
            )
        },
        "include_test_protocols": include_test_protocols,
        "existing_test_files_opened": False,
        "candidate_models_evaluated": False,
        "private_reference": True,
        "available_to_discovery_methods": False,
    }
    identity = _hash(plan)
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("audit plan changed; use a new output root")
    _write(plan_path, plan)
    records, cells = [], []
    for family in suite.families:
        for task in family.tasks:
            task_argument = task if family.family == "dalla_man" else None
            protocols = tuple(
                p
                for p in phase_b_protocols(family.family, task=task_argument)
                if include_test_protocols or p.split != "test"
            )
            for condition in family.dynamics_conditions:
                dynamics = "canonical" if condition == "not_applicable" else condition
                group = f"{family.family}-{task}-{dynamics}"
                trajectories = []
                for protocol in protocols:
                    record, trajectory = audit_one(
                        root / "private" / group,
                        protocol,
                        dynamics,
                        data_root,
                        identity,
                    )
                    records.append(record)
                    trajectories.append(trajectory)
                for tier in family.tiers:
                    commitments = []
                    for variant in family.semantic_variants:
                        spec = phase_b_public_spec(
                            family.family,
                            tier.name,
                            variant,
                            task=task_argument,
                            dynamics=dynamics,
                            data_root=data_root,
                        )
                        cell_root = root / "public-development" / spec.benchmark_id
                        if not (cell_root / "manifest.json").exists():
                            cell_root.parent.mkdir(parents=True, exist_ok=True)
                            with tempfile.TemporaryDirectory(
                                dir=cell_root.parent
                            ) as temp:
                                staged = Path(temp) / "bundle"
                                write_public_staging_bundle(
                                    staged, spec, tuple(trajectories)
                                )
                                staged.rename(cell_root)
                        public_check = audit_public_bundle(cell_root, spec)
                        manifest = json.loads((cell_root / "manifest.json").read_text())
                        expected = {
                            split: _numeric_payload_sha256(
                                spec,
                                tuple(
                                    t
                                    for t in trajectories
                                    if t.protocol_id.startswith(split + "_")
                                ),
                            )
                            for split in ("train", "validation")
                        }
                        files_match = all(
                            _digest(cell_root / f"{split}.csv") == digest
                            for split, digest in manifest["splits"].items()
                        )
                        passed = (
                            public_check.passed
                            and files_match
                            and manifest["numeric_payload_sha256"] == expected
                            and not manifest["test_sealed"]
                            and all(
                                check_projection(
                                    cell_root / f"{split}.csv",
                                    spec,
                                    tuple(
                                        t
                                        for t in trajectories
                                        if t.protocol_id.startswith(split + "_")
                                    ),
                                )
                                for split in ("train", "validation")
                            )
                        )
                        commitments.append(expected)
                        cells.append(
                            {
                                "benchmark_id": spec.benchmark_id,
                                "passed": passed,
                                "violations": list(public_check.violations),
                            }
                        )
                    if any(item != commitments[0] for item in commitments[1:]):
                        raise ValueError("semantic pair changed numeric content")
                print(f"Audited {group}: {len(protocols)} protocols", flush=True)
    summary = {
        "protocol": plan["protocol"],
        "identity": identity,
        "reference_generation_protocol": REFERENCE_PROTOCOL,
        "numerical_protocols": len(records),
        "numerical_passed": sum(r["passed"] for r in records),
        "public_cells": len(cells),
        "public_cells_passed": sum(c["passed"] for c in cells),
        "maximum_solver_scaled_difference": max(
            r["independent_solver"]["maximum_scaled_difference"] for r in records
        ),
        "maximum_grid_scaled_difference": max(
            r["output_grid_refinement"]["maximum_scaled_difference"] for r in records
        ),
        "numerical_checks_passed": all(r["passed"] for r in records)
        and all(c["passed"] for c in cells),
        "failed_protocols": [
            {"protocol": r["protocol"], "dynamics": r["dynamics"]}
            for r in records
            if not r["passed"]
        ],
        "cells": cells,
        "existing_releases_modified": False,
        "existing_test_files_opened": False,
        "new_private_test_protocols_checked": include_test_protocols,
        "release_ready": False,
        "remaining_contract_issue": (
            "Dalla exact meal jumps versus generic piecewise-linear interpolation, "
            "including half the area at time zero; a new public input/execution "
            "contract is required before claiming equivalence."
        ),
        "limitation": (
            "Numerical consistency of trusted generators and projections, not "
            "independent validation of every physical equation, remote datasets, "
            "identifiability or discovery-method accuracy."
        ),
    }
    _write(root / "summary.json", summary)
    return summary
