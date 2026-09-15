"""Standard-library provenance and reporting for saved scaled-fitter recovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path


def read(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def digest(value: object) -> str:
    # Match staged_topology.content_hash, including its default separators.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: object, *, immutable: bool = False) -> None:
    """Atomic local/persistent publication; never replace a different frozen item."""
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def checked(path: Path, identity: str) -> dict:
    value = read(path)
    if value.get("identity") != identity:
        raise ValueError(f"identity differs: {path}")
    return value


def self_checked(path: Path) -> dict:
    value = read(path)
    if value.get("identity") != digest(
        {k: v for k, v in value.items() if k != "identity"}
    ):
        raise ValueError(f"content identity differs: {path}")
    return value


def child_path(root: Path, name: str) -> Path:
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"unsafe artifact path: {name}")
    return path


def source_file(source: Path, target: Path, expected: str | None = None) -> str:
    """Copy one small frozen asset, validating the bytes actually copied."""
    if target.exists():
        actual = sha(target)
        if actual != (expected or sha(source)):
            raise ValueError(f"frozen asset differs: {target}")
        return actual
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    actual = sha(target)
    if expected is not None and actual != expected:
        raise ValueError(f"asset digest differs: {source}")
    return actual


def code_files(repo: Path) -> dict[str, str]:
    paths = sorted((repo / "src/autoformalism").rglob("*.py"))
    paths += [
        repo / "scripts" / p
        for p in (
            "run_scaled_recovery.py",
            "scaled_recovery_io.py",
            "stage_scaled_recovery_runtime.py",
            "smoke_scaled_recovery.py",
            "hpc/scaled_recovery_delta.slurm",
            "hpc/submit_scaled_recovery_delta.sh",
        )
    ]
    return {str(p.relative_to(repo)): sha(p) for p in paths}


def prepare(source: Path, output: Path, repo: Path) -> dict:
    """Freeze an explicitly authorized recovery; never reopen collocation."""
    if source.resolve().is_relative_to(
        output.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("source and recovery output must be separate")
    old = self_checked(source / "freeze.json")
    if (
        old["tasks"] != ["joint", "alternating"]
        or old["plan"]["protocol"] != "scaled-alternating-1"
    ):
        raise ValueError("requires the original scaled joint/alternating pair")
    if old.get("test_data_opened", False) or old.get("proposer_access", False):
        raise ValueError("requires isolated diagnostic source")
    gate = checked(source / "gate/result.json", old["identity"])
    common = self_checked(source / "common.json")
    if not gate.get("pass") or gate["common_identity"] != common["identity"]:
        raise ValueError("original gate/common identity differs")
    if sha(source / "nodes.npy") != gate["nodes_sha256"]:
        raise ValueError("original frozen nodes differ")
    problem = read(source / "problem.json")
    if set(problem["splits"]) != {"train", "val"}:
        raise ValueError("only training and validation splits are allowed")
    if len(common["parameter_names"]) != len(common["ordinary_parameters"]):
        raise ValueError("ordinary parameter length differs")
    # Keep the standard-library tools compatible with Python 3.9 diagnostics too.
    ordinary = dict(zip(common["parameter_names"], common["ordinary_parameters"]))  # noqa: B905
    if ordinary != problem["start"] or not all(
        math.isfinite(v) for v in ordinary.values()
    ):
        raise ValueError("ordinary start differs or is not finite")
    assets = {}

    def copy(relative, destination=None, expected=None):
        name = destination or relative
        assets[name] = source_file(
            child_path(source, relative), output / name, expected
        )

    copy("problem.json", expected=old["assets"]["problem.json"])
    copy("common.json")
    copy("freeze.json", "provenance/source_freeze.json")
    copy("gate/result.json", "provenance/source_gate.json")
    tasks = []
    for index, arm in enumerate(old["tasks"]):
        identity = digest({"freeze": old["identity"], "arm": arm})
        relative = f"results/task_{index:03d}"
        root = source / relative
        screen = checked(root / "screen/result.json", identity)
        refinement = checked(root / "refinement/result.json", identity)
        if (
            screen["status"] != "interrupted"
            or screen.get("best")
            or (root / "screen/calls").exists()
            or refinement["status"] != "no_feasible_screen"
            or refinement.get("calls") != 0
            or refinement.get("best")
        ):
            raise ValueError(
                "recovery requires screening not begun and zero refinement calls"
            )
        for stage in ("initializer", "screen", "refinement"):
            checked(root / stage / "result.json", identity)
            copy(f"{relative}/{stage}/result.json", f"provenance/{arm}/{stage}.json")
        checkpoint = f"{relative}/initializer/native/checkpoint.json"
        record = read(source / checkpoint)
        # Array/parameter validation belongs to the independent checkpoint worker.
        # A bad array must never prevent the ordinary point from being evaluated.
        copy(checkpoint, f"points/{arm}/checkpoint.json")
        array_error = None
        try:
            name = record["array"]
            if Path(name).name != name:
                raise ValueError("unsafe checkpoint array name")
            copy(f"{relative}/initializer/native/{name}", f"points/{arm}/{name}")
        except (OSError, ValueError, KeyError) as error:
            array_error = f"{type(error).__name__}: {error}"
        tasks.append(
            {"arm": arm, "source_task_identity": identity, "array_error": array_error}
        )
    frozen = {
        "protocol": "scaled-checkpoint-recovery-1",
        "source_identity": old["identity"],
        "plan": old["plan"],
        "runtime": old["runtime"],
        "tasks": tasks,
        "assets": assets,
        "code": code_files(repo),
        "startup_seconds": 180,
        "publish_seconds": 60,
        "authorization": (
            "explicit recovery of screening never begun and unused refinement; "
            "no initializer rerun"
        ),
        "source_charged_budgets_retained": True,
        "test_data_opened": False,
        "llm_calls": 0,
        "collocation_reruns": 0,
    }
    frozen["identity"] = digest(frozen)
    write(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify(output: Path, repo: Path) -> dict:
    """Run once on node-local inputs, before starting numerical stage budgets."""
    frozen = self_checked(output / "freeze.json")
    if frozen["protocol"] != "scaled-checkpoint-recovery-1":
        raise ValueError("unknown recovery protocol")
    for name, expected in frozen["assets"].items():
        if sha(child_path(output, name)) != expected:
            raise ValueError(f"frozen asset differs: {name}")
    if code_files(repo) != frozen["code"]:
        raise ValueError("recovery source code differs; use pinned checkout")
    return frozen


def publish(source: Path, target: Path) -> None:
    """Mirror atomically; callers supervise this separately from numerical work."""
    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(
            p.startswith(".") for p in path.relative_to(source).parts
        ):
            continue
        if path.name.endswith(".lock"):
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        descriptor, temporary = tempfile.mkstemp(
            prefix=".publish-", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)


def report(output: Path) -> dict:
    """Keep execution completion, physical feasibility and accuracy separate."""
    frozen = self_checked(output / "freeze.json")
    rows = []
    for index, task in enumerate(frozen["tasks"]):
        path = output / f"results/task_{index:03d}/result.json"
        rows.append(
            checked(path, frozen["identity"])
            if path.exists()
            else {"arm": task["arm"], "status": "missing"}
        )
    lines = [
        "# Saved-checkpoint execution recovery",
        "",
        "Original checkpoints; no collocation rerun. "
        "Screening is an explicitly authorized recovery attempt.",
        "Startup/publication times are separate from numerical budgets. "
        "Outage/recovery timings are not a controlled speed comparison.",
        "NMSE scores v01 only. Replay agreement is numerical consistency, "
        "not prediction accuracy.",
        "",
        "| Arm | Status | Selected | Train NMSE | Validation NMSE | "
        "Replay | Practical |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(k))
                for k in (
                    "arm",
                    "status",
                    "selected_source",
                    "train_nmse",
                    "validation_nmse",
                    "replay_agreement",
                    "practical",
                )
            )
            + " |"
        )
        for stage, value in row.get("stages", {}).items():
            lines.append(
                f"\n{row['arm']} / {stage}: {value.get('status')}; "
                f"calls={value.get('calls')}; startup={value.get('startup_seconds')}; "
                f"numerical={value.get('numerical_seconds')}"
            )
    result = {"identity": frozen["identity"], "rows": rows, "collocation_reruns": 0}
    write(output / "summary.json", result)
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return result
