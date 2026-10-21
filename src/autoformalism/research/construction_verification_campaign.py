"""Verify saved construction parameters without repeating optimization or selection."""

from __future__ import annotations

import gzip
import hashlib
import io
import math
import shutil
import tarfile
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as original
from autoformalism.research import construction_fit_verification as numerical
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-saved-verification-campaign-1"
ENTRYPOINTS = (
    "scripts/verify_construction_parameters.py",
    "scripts/hpc/run_construction_verification_aces.sh",
    "scripts/hpc/start_construction_verification_aces.sh",
    "scripts/submit_construction_fit_aces.py",
    "scripts/submit_review_continuation.py",
)


class Policy(StrictSchema):
    """Separate recorded verification budget; all historical endpoints retained."""

    protocol: Literal["construction-saved-verification-policy-1"] = (
        "construction-saved-verification-policy-1"
    )
    execution: verification.ExecutionPolicy = Field(
        default_factory=lambda: verification.ExecutionPolicy(
            numerical_seconds=300, overall_seconds=360
        )
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entrypoints() -> dict:
    repo = Path(__file__).resolve().parents[3]
    return {name: _hash(repo / name) for name in ENTRYPOINTS}


def _source_files(source: Path) -> list[Path]:
    """Snapshot only public campaign files, never traverse arbitrary data roots."""
    paths = [source / "plan.json", source / "inputs.json"]
    for name in ("results", "report", "logs", "submissions"):
        directory = source / name
        if directory.is_symlink():
            raise ValueError("source symlink directory prohibited")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("source symlink prohibited")
            if path.is_file() and not path.name.startswith("."):
                paths.append(path)
    for name in ("SUMMARY.md", "launcher_execution.json"):
        if (source / name).exists():
            paths.append(source / name)
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("source path escapes campaign")
    return sorted(paths)


def _source(source: Path) -> tuple[dict, dict, list[dict]]:
    """Check seals, operation provenance and exact membership of saved bests."""
    plan, data = original.verify(source, runtime=False)
    policy = original.Policy.model_validate(plan["policy"])
    candidates = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = source / "results" / task["task_id"]
        saved = read_seal(folder / "result.json")
        if saved["identity"] != identity:
            raise ValueError("source task identity differs")
        operations = original._journal_snapshot(folder, identity, data, policy)
        reconstructed = original._assemble(identity, data, operations, policy)
        if (
            saved["cost"] != reconstructed["cost"]
            or saved["training_status"] != reconstructed["training_status"]
            or not _same_derived_score(
                saved["validation_score"], reconstructed["validation_score"]
            )
        ):
            raise ValueError("source task summary differs from numerical receipts")
        operation = operations.get("training", {})
        if operation.get("status") != "complete":
            raise ValueError("terminal training operation required")
        fit = operation["value"]
        best = fit.get("search", {}).get("best")
        if not best or not isinstance(best.get("parameters"), dict):
            raise ValueError("saved best parameters required; refitting prohibited")
        digest = public.content_sha256(best["parameters"])
        accepted = any(
            e.get("parameters_sha256") == digest
            and e.get("training_nmse") == best["training_nmse"]
            for e in fit["search"]["evaluations"]
        )
        if not accepted:
            raise ValueError("saved best has no completed search evaluation")
        if (
            fit["problem_sha256"] != task["problem_sha256"]
            or fit["training_sha256"] != public.content_sha256(data["training"])
            or fit["policy"] != plan["policy"]["training"]
        ):
            raise ValueError(
                "training receipt input identity or numerical policy differs"
            )
        candidates.append(
            {
                "task": task,
                "parameters": best["parameters"],
                "parameters_sha256": digest,
                "search_training_nmse": best["training_nmse"],
                "original_verification_passed": fit["fit_success"],
                "original_training_status": saved["training_status"],
                "historical_validation_score": saved["validation_score"],
                "historical_cost": saved["cost"],
            }
        )
    return plan, data, candidates


def _same_derived_score(saved: object, rebuilt: object) -> bool:
    """Allow cross-platform summation roundoff only in historical score views.

    Source bytes and parameter/evaluation membership are still checked exactly.
    This does not change a numerical gate or any historical score.
    """
    if isinstance(saved, dict) and isinstance(rebuilt, dict):
        return saved.keys() == rebuilt.keys() and all(
            _same_derived_score(saved[key], rebuilt[key]) for key in saved
        )
    if isinstance(saved, float) and isinstance(rebuilt, float):
        return math.isclose(saved, rebuilt, rel_tol=1e-12, abs_tol=1e-12)
    return saved == rebuilt


def prepare(root: Path, source: Path, policy: Policy | None = None) -> dict:
    """Freeze all source endpoints in a separate immutable public snapshot."""
    root, source = root.resolve(), source.resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("source and supplemental root must be separate")
    policy = policy or Policy()
    source_plan, _, candidates = _source(source)
    paths = _source_files(source)
    manifest = {str(p.relative_to(source)): _hash(p) for p in paths}
    plan = {
        "protocol": PROTOCOL,
        "source_plan_sha256": public.content_sha256(source_plan),
        "source_files": manifest,
        "candidates": candidates,
        "policy": policy.model_dump(mode="json"),
        "numerical_policy": source_plan["policy"]["training"],
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "entrypoints": _entrypoints(),
        "optimizer_calls": 0,
        "llm_calls": 0,
        "test_data_opened": False,
        "new_validation_inference": False,
        "automatic_promotion": False,
    }
    with public._lock(root):
        if (root / "plan.json").exists() and read_seal(root / "plan.json") != plan:
            raise ValueError("supplemental identity differs; use a new root")
        snapshot = root / "source_snapshot"
        for path in paths:
            relative = path.relative_to(source)
            target = snapshot / relative
            if target.is_symlink() or target.resolve() != target.absolute():
                raise ValueError("snapshot symlink prohibited")
            if target.exists():
                if _hash(target) != manifest[str(relative)]:
                    raise ValueError("snapshot differs")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            if _hash(target) != manifest[str(relative)]:
                raise ValueError("source changed while copying")
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(candidates)}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Verify source bytes, semantic provenance and optionally execution identity."""
    root = root.resolve()
    plan = read_seal(root / "plan.json")
    if plan["protocol"] != PROTOCOL:
        raise ValueError("unsupported supplemental protocol")
    Policy.model_validate(plan["policy"])
    snapshot = root / "source_snapshot"
    if snapshot.is_symlink():
        raise ValueError("source snapshot symlink prohibited")
    for name, digest in plan["source_files"].items():
        path = snapshot / name
        if (
            path.is_symlink()
            or path.resolve() != path.absolute()
            or not path.resolve().is_relative_to(snapshot.resolve())
            or _hash(path) != digest
        ):
            raise ValueError("source snapshot differs")
    source_plan, data, candidates = _source(snapshot)
    if (
        public.content_sha256(source_plan) != plan["source_plan_sha256"]
        or candidates != plan["candidates"]
        or source_plan["policy"]["training"] != plan["numerical_policy"]
    ):
        raise ValueError("saved endpoint provenance differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["entrypoints"] != _entrypoints()
    ):
        raise ValueError("supplement source/runtime/entrypoints differ")
    return plan, data


def _identity(plan: dict, candidate: dict) -> dict:
    return {
        "plan_sha256": public.content_sha256(plan),
        "task": candidate["task"],
        "parameters_sha256": candidate["parameters_sha256"],
    }


def run_task(root: Path, index: int) -> dict:
    """Verify each frozen candidate once; an interruption cannot reset its budget."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["candidates"]):
        raise ValueError("unknown task index")
    candidate = plan["candidates"][index]
    policy = Policy.model_validate(plan["policy"])
    task_id = candidate["task"]["task_id"]
    problem = PrefixFitProblem.model_validate(data["records"][task_id]["problem"])
    training = PublicSplit.model_validate(data["training"])
    settings = kernel.NumericalPolicy.model_validate(plan["numerical_policy"])

    def work(save):
        try:
            return numerical.verify_saved_parameters(
                problem,
                training,
                candidate["parameters"],
                settings,
                policy.execution,
                checkpoint=save,
            )
        except Exception as error:
            receipt = getattr(error, "verification_receipt", None)
            if receipt is None:
                raise
            save(receipt)
            return receipt

    return original._operation(
        root / "results" / task_id,
        _identity(plan, candidate),
        "output_verification",
        policy.execution.overall_seconds,
        work,
    )


def report(root: Path) -> dict:
    """Report verification and original fit costs separately; never infer initials."""
    plan, _ = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows, operations = [], {}
    terminal_count = 0
    for candidate in plan["candidates"]:
        task_id = candidate["task"]["task_id"]
        folder = root / "results" / task_id / "operations/output_verification"
        identity = {
            "campaign": _identity(plan, candidate),
            "stage": "output_verification",
            "allowance_seconds": policy.execution.overall_seconds,
        }
        op = None
        if (folder / "result.json").exists():
            op = original._read_operation(folder, identity)
            operations[task_id] = op
            terminal_count += 1
        elif (folder / "started.json").exists():
            if read_seal(folder / "started.json") != identity:
                raise ValueError("verification reservation differs")
            operations[task_id] = {
                "identity": identity,
                "status": "running_or_interrupted",
                "value": None,
                "observed_progress": public._read(folder / "progress.json")
                if (folder / "progress.json").exists()
                else {},
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": policy.execution.overall_seconds,
                "accounting_complete": False,
            }
        elif (folder / "progress.json").exists():
            raise ValueError("verification progress lacks reservation")
        value = (op or {}).get("value") or {}
        success = value.get("success", False)
        rows.append(
            {
                "task_id": task_id,
                "status": "verified"
                if success
                else (
                    "verification_failed"
                    if op
                    else "running_or_interrupted"
                    if (folder / "started.json").exists()
                    else "pending"
                ),
                "source_training_status": candidate["original_training_status"],
                "retained_search_nmse": candidate["search_training_nmse"],
                "normalized_mse": value.get("normalized_mse") if success else None,
                "per_target_normalized_mse": value.get("per_target_normalized_mse")
                if success
                else None,
                "verification_error": value.get("verification_error")
                or (op or {}).get("error"),
                "solver_difference": value.get("solver_difference"),
                "historical_validation_score": candidate["historical_validation_score"],
                "new_validation_score": None,
            }
        )
    values = [r["normalized_mse"] for r in rows if r["status"] == "verified"]
    median = float(np.median(values)) if values else None
    output = {
        "protocol": PROTOCOL,
        "model_count": len(rows),
        "completed_tasks": terminal_count,
        "verified_models": len(values),
        "records": rows,
        "training_nmse": {
            "available_models": len(values),
            "total_models": len(rows),
            "median": median,
            "mad": float(np.median(np.abs(np.asarray(values) - median)))
            if values
            else None,
        },
        "new_cost": original._cost(operations),
        "historical_cost_by_model": {
            c["task"]["task_id"]: c["historical_cost"] for c in plan["candidates"]
        },
        "optimizer_calls": 0,
        "llm_calls": 0,
        "test_data_opened": False,
        "new_validation_inference": False,
        "automatic_promotion": False,
        "rank_assessed": False,
    }
    if any(r["status"] == "running_or_interrupted" for r in rows):
        output["new_cost"]["accounting_complete"] = False
    lines = [
        "# Saved-parameter output verification",
        "",
        f"Terminal records: {terminal_count}/{len(rows)}; verified: {len(values)}.",
        "No optimization, new validation inference, rank gate, or model promotion.",
        "Historical results and charges remain unchanged.",
        "",
        "| Model | Status | Retained search NMSE | Verified training NMSE |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        shown = row["normalized_mse"]
        lines.append(
            f"| {row['task_id']} | {row['status']} | "
            f"{row['retained_search_nmse']:.6g} | "
            f"{shown if shown is not None else 'unavailable'} |"
        )
    lines += [
        "",
        f"Median / unscaled MAD: {median} / "
        f"{output['training_nmse']['mad']} ({len(values)}/{len(rows)} available).",
        "",
        "Numerical verification establishes agreement, not scientific "
        "correctness or the best attainable fit.",
    ]
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", output)
        (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return output


def inspect(root: Path) -> dict:
    """Package preserved source and new receipts without numerical execution."""
    summary = report(root)
    destination = root / "inspection.tar.gz"
    temporary = root / "inspection.tar.gz.tmp"
    paths = [root / "plan.json", root / "SUMMARY.md"]
    for name in ("source_snapshot", "results", "report", "submissions", "logs"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file())
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(paths):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("inspection path escapes root")
            if path.name.startswith("."):
                continue
            value = path.read_bytes()
            entry = tarfile.TarInfo(path.relative_to(root).as_posix())
            entry.size, entry.mode = len(value), 0o644
            archive.addfile(entry, io.BytesIO(value))
    temporary.replace(destination)
    return {"summary": summary, "inspection_archive": str(destination.resolve())}
