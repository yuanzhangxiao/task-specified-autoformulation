#!/usr/bin/env python3
"""Replay saved round checkpoints in a separate snapshot, never rerun fitting.

Run this standalone script with the ORIGINAL campaign's Python and PYTHONPATH.
It deliberately lives outside src so its addition cannot change a frozen fitter.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.data import TrainingScaler
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.fitting.fitter import _training_variables
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitResult
from autoformalism.search.residual_evidence import build_residual_evidence

PROTOCOL = "review-quota-checkpoint-recovery-1"
SPLIT_SECONDS = 300
WORKER_SECONDS = 720
MANIFEST = "quota_recovery.json"


def digest(path: Path) -> str:
    """Hash bytes without interpreting possibly incomplete JSON."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(source: Path) -> dict[str, str]:
    """Copy development provenance only; never enumerate evaluation data."""
    files = [source / "plan.json"]
    plan = sealed_read(files[0])
    for cell, spec in plan["cells"].items():
        files.extend(source / "public/phase_b_v1" / cell / n for n in spec["assets"])
    for name in ("results", "imports"):
        for directory, folders, names in os.walk(source / name, followlinks=False):
            folders[:] = sorted(n for n in folders if n != "__pycache__")
            if any((Path(directory) / n).is_symlink() for n in folders):
                raise ValueError("snapshot refuses directory symlinks")
            files.extend(
                Path(directory) / n
                for n in sorted(names)
                if not (n.endswith((".tmp", ".pyc", ".lock")) or n == ".lock")
            )
    if any(p.is_symlink() for p in files):
        raise ValueError("snapshot refuses file symlinks")
    return {str(p.relative_to(source)): digest(p) for p in sorted(files)}


def point_candidates(fit: Path) -> tuple[list[dict], list[dict]]:
    """Only full-training rollout checkpoints qualify, never collocation nodes."""
    _, request, training, _ = sibling_fit._load(fit)
    model, _, _ = public._lower(request)
    bounds = _training_variables(model, public.unpack_split(training), FitConfig())
    names = {v.name.removeprefix("parameter:") for v in bounds}
    if names != set(model.parameter_names):
        raise ValueError("recovery requires lowered global initial parameters")
    candidates, rejected = [], []
    for path in sorted((fit / "backend/recovery").glob("*/best_evaluated.json")):
        if path.parent.name not in {"poll_calls", "primal_screen"} and not (
            path.parent.name.startswith("augmented_")
            and path.parent.name.removeprefix("augmented_").isdigit()
        ):
            continue
        record = {"path": str(path.relative_to(fit)), "sha256": digest(path)}
        try:
            value = public._read(path)
            params, cost = value["parameters"], value["cost"]
            if set(params) != names or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in params.values()
            ):
                raise ValueError("incomplete or nonfinite parameter vector")
            if (
                isinstance(cost, bool)
                or not isinstance(cost, (int, float))
                or not (math.isfinite(cost) and cost >= 0)
            ):
                raise ValueError("invalid training cost")
            for variable in bounds:
                if (
                    not variable.lower
                    <= params[variable.name.removeprefix("parameter:")]
                    <= variable.upper
                ):
                    raise ValueError("parameter outside frozen domain")
            candidates.append({**record, "parameters": params, "cost": cost})
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({**record, "error": str(error)})
    return sorted(candidates, key=lambda p: (p["cost"], p["path"])), rejected


def validate_attempt(root: Path, plan: dict, task: dict, index: int) -> tuple:
    """Check request/data/parent bindings before trusting an unsealed checkpoint."""
    directory = io.round_path(root, task, index)
    parent = io.read_round(root, task, index - 1)
    proposal = sealed_read(directory / "proposal.json")
    if parent is None or parent["selected"] is None:
        raise ValueError("recovery requires the retained preceding model")
    if proposal["parent_sha256"] != parent["artifact_sha256"]:
        raise ValueError("proposal parent differs")
    frozen, request, training, validation = sibling_fit._load(directory / "fit")
    fallback = proposal["status"] in {
        "revision_failed",
        "no_change",
        "residual_evidence_unavailable",
    }
    bundle = parent["selected"]["bundle"] if fallback else proposal["bundle"]
    expected = pipeline.request_for(bundle, plan, task, index)
    cell = plan["cells"][task["cell"]]
    if (
        request != expected
        or request.profile != "collocation-single-target-v2"
        or training.model_dump(mode="json") != cell["training"]
        or validation.model_dump(mode="json") != cell["validation"]
        or frozen["parent_parameters"] != parent["selected"]["fit"]["parameters"]
        or frozen["parent_request"] != parent["selected"]["request"]
        or frozen["lineage"]
        != {
            "campaign": plan["artifact_sha256"],
            "proposal": proposal["artifact_sha256"],
        }
        or public._read(directory / "fit/started.json")["identity"]
        != frozen["identity"]
    ):
        raise ValueError("checkpoint attempt differs from campaign lineage")
    return frozen, request, training, validation, parent, proposal, bundle, fallback


def prepare(source: Path, root: Path, index: int, indices: list[int]) -> dict:
    """Snapshot quiescent inputs to group scratch, preserving original artifacts."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("source and recovery must be disjoint")
    with (source / "execution.lock").open("r") as lease, public._lock(root):
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        io.require_open(source)
        plan = io.verify(source)
        if not 1 <= index < plan["config"]["rounds"]:
            raise ValueError("invalid recovery round")
        missing = [
            i
            for i, t in enumerate(plan["tasks"])
            if io.read_round(source, t, index) is None
        ]
        if sorted(set(indices)) != missing:
            raise ValueError(f"missing task indices differ: {missing}")
        for future in range(index + 1, plan["config"]["rounds"]):
            if (source / f"submission-round-{future}.json").exists() or (
                source / "submission-intent" / f"round-{future}"
            ).exists():
                raise ValueError(
                    "later round submission exists; inspect scheduler first"
                )
        for task in plan["tasks"]:
            if io.read_round(source, task, index - 1) is None:
                raise ValueError("previous round incomplete")
            for future in range(index + 1, plan["config"]["rounds"]):
                if io.round_path(source, task, future).exists():
                    raise ValueError("later round already began; refuse fork")
        files = inventory(source)
        entries = []
        for ti in missing:
            task = plan["tasks"][ti]
            directory = io.round_path(source, task, index)
            validate_attempt(source, plan, task, index)
            if any(
                (directory / "fit" / n).exists()
                for n in ("result.json", "backend_result.json")
            ):
                raise ValueError("a final fit exists; use result-publication recovery")
            points, rejected = point_candidates(directory / "fit")
            entries.append(
                {
                    "index": ti,
                    "task_id": task["task_id"],
                    "point": points[0] if points else None,
                    "candidates": points,
                    "rejected": rejected,
                }
            )
        body = {
            "protocol": PROTOCOL,
            "source": str(source),
            "phase_round": index,
            "global_round": index + plan.get("continuation", {}).get("source_round", 0),
            "plan_sha256": plan["artifact_sha256"],
            "files": files,
            "entries": entries,
            "script_sha256": digest(Path(__file__)),
            "checkpoint_selection": "minimum_saved_full_training_cost_then_path",
            "split_seconds": SPLIT_SECONDS,
            "worker_seconds": WORKER_SECONDS,
            "optimizer_calls": 0,
            "llm_calls": 0,
            "test_data_opened": False,
        }
        if (root / MANIFEST).exists():
            old = sealed_read(root / MANIFEST)
            if {k: v for k, v in old.items() if k != "artifact_sha256"} != body:
                raise ValueError("source or recovery policy changed")
            verify_snapshot(root)
            return old
        intent = root / "snapshot_intent.json"
        sealed_write(intent, body)
        for name, sha in files.items():
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if digest(dest) != sha:
                    raise ValueError(f"partial snapshot differs: {name}")
            else:
                tmp = dest.with_name(dest.name + ".copy.tmp")
                shutil.copyfile(source / name, tmp)
                if digest(tmp) != sha:
                    raise ValueError(f"source changed during snapshot: {name}")
                tmp.replace(dest)
        if inventory(source) != files:
            raise ValueError("source changed during snapshot")
        return sealed_write(root / MANIFEST, body)


def verify_snapshot(root: Path) -> tuple[dict, dict]:
    """Every original copied file remains immutable, including failed attempts."""
    manifest = sealed_read(root / MANIFEST)
    if manifest["protocol"] != PROTOCOL or manifest["script_sha256"] != digest(
        Path(__file__)
    ):
        raise ValueError("recovery code differs")
    io.require_open(root)
    plan = io.verify(root)
    if manifest["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("recovery plan differs")
    for name, sha in manifest["files"].items():
        if digest(root / name) != sha:
            raise ValueError(f"copied source changed: {name}")
    return manifest, plan


def fixed_replay(model, split, parameters: dict, scales: dict) -> dict:
    """One production rollout at a fixed vector; never optimize initial values."""
    settings = FitConfig(
        integration_method="Radau",
        relative_tolerance=1e-7,
        absolute_tolerance=1e-9,
        allow_derivative_regression=False,
        maximum_wall_time_seconds=SPLIT_SECONDS,
    )
    started, predictions, squares = monotonic(), {}, {k: [] for k in scales}
    for row in split.trajectories:
        sim = simulate_trajectory(
            model,
            row,
            parameters,
            {},
            settings,
            deadline=started + SPLIT_SECONDS,
            reset_observed_states=False,
        )
        if not sim.success:
            raise ValueError(f"{split.name.value}/{row.trajectory_id}: {sim.message}")
        predictions[row.trajectory_id] = {
            "time": row.time.tolist(),
            "predictions": {k: sim.predictions[k].tolist() for k in scales},
        }
        for target, scale in scales.items():
            residual = (sim.predictions[target] - row.targets[target]) / scale
            if (
                not np.isfinite(residual).all()
                or np.max(np.abs(residual)) >= settings.failure_penalty
            ):
                raise ValueError("invalid or clipped production residual")
            squares[target].extend((residual**2).tolist())
    combined = np.concatenate([np.asarray(v) for v in squares.values()])
    return {
        "metrics": {
            "normalized_mse": float(combined.mean()),
            "per_target_normalized_mse": {
                k: float(np.mean(v)) for k, v in squares.items()
            },
            "failed_trajectories": [],
        },
        "cost": float(0.5 * combined.sum()),
        "predictions": predictions,
        "seconds": monotonic() - started,
    }


def replay_inner(root: Path, slot: int) -> dict:
    """Replay the preselected point; validation cannot choose another checkpoint."""
    manifest, plan = verify_snapshot(root)
    entry = manifest["entries"][slot]
    task, index = plan["tasks"][entry["index"]], manifest["phase_round"]
    frozen, request, training, validation, _, _, bundle, _ = validate_attempt(
        root, plan, task, index
    )
    point = entry["point"]
    if point is None:
        return {"status": "no_saved_point"}
    model, _, _ = public._lower(request)
    train = public.unpack_split(training)
    scalers = TrainingScaler().fit(train).scales
    scales = {
        k: scalers[f"target:{k}"].standard_deviation for k in request.context.targets
    }
    tr = fixed_replay(model, train, point["parameters"], scales)
    # The existing residual-packet replay uses this same score agreement tolerance.
    if not math.isclose(tr["cost"], point["cost"], rel_tol=1e-5, abs_tol=1e-8):
        return {
            "status": "training_cost_disagrees",
            "saved_cost": point["cost"],
            "replayed_cost": tr["cost"],
            "training_seconds": tr["seconds"],
        }
    va = fixed_replay(
        model, public.unpack_split(validation), point["parameters"], scales
    )
    raw = {
        "parameters": point["parameters"],
        "training": tr["metrics"],
        "validation": va["metrics"],
        "refinement": {},
        "recovery": {
            "protocol": PROTOCOL,
            "manifest_sha256": manifest["artifact_sha256"],
            "checkpoint": point,
            "optimizer_calls": 0,
            "original_optimizer_status": "unknown_after_quota_interruption",
        },
    }
    result = PublicFitResult(
        **public._result_base(
            {**frozen["public_fit"], "identity": frozen["identity"]}, request
        ),
        **public._evidence(request, raw),
        status="complete",
        backend_result_sha256=public.content_sha256(raw),
        message=(
            "Fixed checkpoint replay after quota interruption; no new optimization. "
            "Original stopping status and evaluation count are unavailable."
        ),
    )
    packet = build_residual_evidence(
        train,
        model.validated.context,
        tr["predictions"],
        candidate_sha256=public.content_sha256(model.validated.candidate),
        parameters=point["parameters"],
        numerical_status={"feedback_status": "numerical_failure_unresolved"},
    ).model_dump(mode="json")
    payload = result.model_dump(mode="json")
    envelope = {"result": payload, "sha256": public.content_sha256(payload)}
    trial = {
        "bundle": bundle,
        "request": request.model_dump(mode="json"),
        "certificate": pipeline.certificates(bundle, plan["cells"][task["cell"]], task),
        "fit": payload,
        "packet": packet,
        "origin_task": task["task_id"],
        "origin_round": manifest["global_round"],
        "fit_result_sha256": public.content_sha256(envelope),
    }
    return {
        "status": "replayed",
        "trial": trial,
        "fit_envelope": envelope,
        "backend": raw,
        "saved_cost": point["cost"],
        "replayed_cost": tr["cost"],
        "training_seconds": tr["seconds"],
        "validation_seconds": va["seconds"],
    }


def worker(root: Path, slot: int) -> dict:
    """One bounded replay attempt; a started attempt is never silently rerun."""
    manifest, _ = verify_snapshot(root)
    entry = manifest["entries"][slot]
    directory = root / "quota_replays" / str(entry["index"])
    with public._lock(directory):
        result = directory / "result.json"
        identity = {"manifest_sha256": manifest["artifact_sha256"], "entry": entry}
        if result.exists():
            saved = sealed_read(result)
            if saved["identity"] != identity:
                raise ValueError("replay result identity differs")
            return saved
        if (directory / "started.json").exists():
            if sealed_read(directory / "started.json")["identity"] != identity:
                raise ValueError("replay start identity differs")
            value = (
                read_inner(directory, identity)
                if (directory / "inner.json").exists()
                else {"status": "replay_interrupted", "fresh_budget_on_resume": False}
            )
        elif entry["point"] is None:
            value = {"status": "no_saved_point"}
        else:
            sealed_write(directory / "started.json", {"identity": identity})
            with (directory / "worker.log").open("w") as log:
                try:
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "inner",
                            "--root",
                            str(root),
                            "--slot",
                            str(slot),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=WORKER_SECONDS,
                        check=False,
                    )
                    if process.returncode == 0 and (directory / "inner.json").exists():
                        value = read_inner(directory, identity)
                    else:
                        value = {
                            "status": "replay_failed",
                            "exit_code": process.returncode,
                        }
                except subprocess.TimeoutExpired:
                    value = (
                        read_inner(directory, identity)
                        if (directory / "inner.json").exists()
                        else {"status": "replay_timeout"}
                    )
        return sealed_write(
            result,
            {
                **value,
                "identity": identity,
                "optimizer_calls": 0,
                "test_data_opened": False,
            },
        )


def read_inner(directory: Path, identity: dict) -> dict:
    """Keep a completed child replay even if its supervisor was interrupted."""
    value = sealed_read(directory / "inner.json")
    if value["identity"] != identity:
        raise ValueError("child replay identity differs")
    return {k: v for k, v in value.items() if k != "artifact_sha256"}


def finalize(root: Path) -> dict:
    """Publish only the missing visits, preserving successful results and costs."""
    with io.execution_lease(root, exclusive=True):
        manifest, plan = verify_snapshot(root)
        receipts = []
        for entry in manifest["entries"]:
            receipt = sealed_read(
                root / "quota_replays" / str(entry["index"]) / "result.json"
            )
            if receipt["identity"] != {
                "manifest_sha256": manifest["artifact_sha256"],
                "entry": entry,
            }:
                raise ValueError("replay receipt identity differs")
            receipts.append((entry, receipt))
        rows = []
        for entry, receipt in receipts:
            task, index = plan["tasks"][entry["index"]], manifest["phase_round"]
            directory = io.round_path(root, task, index)
            _, _, _, _, parent, proposal, _, fallback = validate_attempt(
                root, plan, task, index
            )
            trial = receipt.get("trial") if receipt["status"] == "replayed" else None
            selected = parent["selected"]
            if pipeline.selection_key(trial) < pipeline.selection_key(selected):
                selected = trial
            if trial is not None:
                for name, value in (
                    ("fit/backend_result.json", receipt["backend"]),
                    ("fit/result.json", receipt["fit_envelope"]),
                ):
                    path = directory / name
                    if path.exists() and public._read(path) != value:
                        raise ValueError("recovery publication differs")
                    if not path.exists():
                        public._write(path, value)
            recovery = {
                "manifest_sha256": manifest["artifact_sha256"],
                "receipt_sha256": receipt["artifact_sha256"],
                "status": receipt["status"],
                "optimizer_calls": 0,
                "original_failure": "disk_quota_interruption",
            }
            sealed_write(
                directory / "result.json",
                {
                    "task": task,
                    "round": index,
                    "status": "complete" if trial is not None else "worker_interrupted",
                    "trial": trial,
                    "selected": selected,
                    "closed": selected is None,
                    "parent_sha256": parent["artifact_sha256"],
                    "proposal_sha256": proposal["artifact_sha256"],
                    "selection_uses": "validation_only_then_complexity",
                    "cost": proposal["cost"],
                    "proposal_status": proposal["status"],
                    "fit_trigger": "incumbent_fallback"
                    if fallback
                    else "proposed_or_control",
                    "citation_audit": (proposal.get("decision") or {})
                    .get("provenance", {})
                    .get("citation_audit"),
                    "selected_new_trial": trial is not None and selected is trial,
                    "recovery": recovery,
                    "test_data_opened": False,
                },
            )
            rows.append(
                {
                    "index": entry["index"],
                    "task": task["task_id"],
                    "recovery_status": receipt["status"],
                    "selected_new_trial": trial is not None and selected is trial,
                    "trial_training": (trial or {}).get("fit", {}).get("training"),
                    "trial_validation": (trial or {}).get("fit", {}).get("validation"),
                }
            )
        reporting.report(root)
        return sealed_write(
            root / "quota_recovery_summary.json",
            {
                "manifest_sha256": manifest["artifact_sha256"],
                "rows": rows,
                "next_global_round": manifest["global_round"] + 1,
                "optimizer_calls": 0,
                "test_data_opened": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "worker", "inner", "finalize", "verify")
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--indices", default="26,33,34,36,38,39,41,43")
    parser.add_argument("--slot", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        value = prepare(
            args.source, root, args.round, [int(v) for v in args.indices.split(",")]
        )
        value = {"identity": value["artifact_sha256"], "entries": value["entries"]}
    elif args.mode == "worker":
        value = worker(root, args.slot)
        value = {"status": value["status"], "identity": value["identity"]}
    elif args.mode == "inner":
        manifest = sealed_read(root / MANIFEST)
        entry = manifest["entries"][args.slot]
        index = entry["index"]
        value = replay_inner(root, args.slot)
        sealed_write(
            root / "quota_replays" / str(index) / "inner.json",
            {
                **value,
                "identity": {
                    "manifest_sha256": manifest["artifact_sha256"],
                    "entry": entry,
                },
            },
        )
        value = {"status": value["status"]}
    elif args.mode == "finalize":
        value = finalize(root)
    else:
        manifest, _ = verify_snapshot(root)
        value = {"identity": manifest["artifact_sha256"], "verified": True}
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
