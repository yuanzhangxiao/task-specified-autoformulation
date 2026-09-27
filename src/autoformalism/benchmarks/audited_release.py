"""Publish a separate continuous-input release from verified reference checkpoints.

This is evaluator-only packaging, including sealed test generation artifacts.
It performs no model evaluation and never regenerates missing audit records.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from autoformalism.benchmarks import reference_audit as audit
from autoformalism.benchmarks.phase_b_generation import phase_b_protocols
from autoformalism.benchmarks.phase_b_public import (
    audit_public_bundle,
    phase_b_public_spec,
    write_public_production_bundle,
)
from autoformalism.benchmarks.suite import load_suite_spec
from autoformalism.fitting import public_fitting as public


def seal(path: Path, value: dict) -> None:
    """Publish once atomically; a different resume is an error."""
    payload = {"sha256": public.content_sha256(value), "value": value}
    if path.exists():
        if public._read(path) != payload:
            raise ValueError(f"sealed artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        public._write(path, payload)


def read_seal(path: Path) -> dict:
    """Verify a diagnostic artifact before consuming it."""
    payload = public._read(path)
    if public.content_sha256(payload["value"]) != payload["sha256"]:
        raise ValueError(f"sealed artifact digest differs: {path}")
    return payload["value"]


def publish_audited_release(
    audit_root: Path, data_root: Path, suite_path: Path, output: Path
) -> dict:
    """Verify the completed audit, then atomically publish/resume each public cell."""
    plan = public._read(audit_root / "plan.json")
    summary = public._read(audit_root / "summary.json")
    identity = audit._hash(plan)  # Historical audit's hash convention is preserved.
    if (
        plan["protocol"] != "phase-b-reference-integrity-audit-1"
        or plan["input_contract"] != "continuous-rates-1"
        or not plan["include_test_protocols"]
        or plan["suite_sha256"] != audit._digest(suite_path)
        or summary["identity"] != identity
        or not summary["numerical_checks_passed"]
    ):
        raise ValueError("a complete matching continuous-input audit is required")
    for name, digest in plan["private_spec_sha256"].items():
        path = data_root / name
        if not path.resolve().is_relative_to(data_root.resolve()):
            raise ValueError("private specification path escapes data root")
        if audit._digest(path) != digest:
            raise ValueError("private specification differs from audited reference")
    suite = load_suite_spec(suite_path)
    release = output / "phase_b_continuous_inputs_v1"
    with public._lock(release):
        frozen = {
            "protocol": "audited-continuous-release-1",
            "audit_identity": identity,
            "audit_summary_sha256": public.content_sha256(summary),
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "suite_sha256": audit._digest(suite_path),
            "private_spec_sha256": plan["private_spec_sha256"],
            "input_contract": "continuous-rates-1",
            "model_test_evaluation_performed": False,
            "scientific_attainability": "not_established_by_numerical_audit",
        }
        seal(release / "release_plan.json", frozen)
        records, numerical_count = [], 0
        for family in suite.families:
            for task in family.tasks:
                task_arg = task if family.family == "dalla_man" else None
                protocols = phase_b_protocols(
                    family.family, task=task_arg, input_contract="continuous-rates-1"
                )
                for condition in family.dynamics_conditions:
                    dynamics = (
                        "canonical" if condition == "not_applicable" else condition
                    )
                    group = (
                        audit_root / "private" / f"{family.family}-{task}-{dynamics}"
                    )
                    trajectories = []
                    for protocol in protocols:
                        path = group / f"{protocol.protocol_id}.json"
                        # Never let audit_one regenerate a missing checkpoint.
                        if not path.is_file() or not path.with_suffix(".npz").is_file():
                            raise ValueError(f"missing audited trajectory: {path}")
                        record, trajectory = audit.audit_one(
                            group, protocol, dynamics, data_root, identity
                        )
                        if not record["passed"] or record[
                            "protocol"
                        ] != protocol.model_dump(mode="json"):
                            raise ValueError(f"unqualified reference record: {path}")
                        trajectories.append(trajectory)
                        numerical_count += 1
                    for tier in family.tiers:
                        commitments = []
                        for variant in family.semantic_variants:
                            spec = phase_b_public_spec(
                                family.family,
                                tier.name,
                                variant,
                                task=task_arg,
                                dynamics=dynamics,
                                data_root=data_root,
                                input_contract="continuous-rates-1",
                            )
                            cell = release / spec.benchmark_id
                            # Render even on resume: detects changed prompts, CSVs,
                            # manifests and unexpected extra files without overwriting.
                            with tempfile.TemporaryDirectory(dir=release) as temporary:
                                staged = Path(temporary) / "cell"
                                write_public_production_bundle(
                                    staged, spec, tuple(trajectories)
                                )
                                files = {
                                    p.name: audit._digest(p) for p in staged.iterdir()
                                }
                                if cell.exists():
                                    actual = {
                                        p.name: audit._digest(p) for p in cell.iterdir()
                                    }
                                    if files != actual:
                                        raise ValueError(
                                            f"published cell differs: {cell}"
                                        )
                                else:
                                    staged.rename(cell)
                            manifest = public._read(cell / "manifest.json")
                            check = audit_public_bundle(cell, spec)
                            if not check.passed or not manifest["test_sealed"]:
                                raise ValueError(f"public release audit failed: {cell}")
                            commitments.append(manifest["numeric_payload_sha256"])
                            records.append(
                                {"benchmark_id": spec.benchmark_id, "files": files}
                            )
                        if any(c != commitments[0] for c in commitments[1:]):
                            raise ValueError(
                                "semantic variants have different numerical data"
                            )
        if (
            len(records) != suite.number_of_cells
            or numerical_count != summary["numerical_protocols"]
        ):
            raise ValueError("release coverage differs from audited suite")
        result = {
            **frozen,
            "status": "complete",
            "released_cells": len(records),
            "numerical_protocols": numerical_count,
            "cells": records,
            "test_payloads_packaged": True,
            "historical_releases_modified": False,
        }
        seal(release / "release_audit.json", result)
        return result
