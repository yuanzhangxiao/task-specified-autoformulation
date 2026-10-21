#!/usr/bin/env python3
"""Validate M47 tasks independently, then report their sealed reporting snapshots.

Run outside the frozen bundle, with its src on PYTHONPATH and AF_REPO_ROOT set.
Only the original campaign verifies results and computes reporting statistics.
Checkpoints attest to inputs at validation time, not perpetual operation-file
immutability. Reduction rehashes their dependencies without repeating semantics.
Reporting timings are separate from the original, unchanged numerical ledger.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, process_time

if __name__ != "__main__":
    from autoformalism.fitting import prefix_noise_reliability as campaign

OUTPUT = "report-sharded-v1"
PROTOCOL = "prefix-noise-sharded-report-1"


def file_digest(path: Path) -> str:
    """Hash immutable file bytes without changing the original artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _same(left: object, right: object) -> bool:
    """Retain JSON numeric and boolean types when comparing sealed snapshots."""
    return json.dumps(
        left, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) == json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_identity() -> dict:
    """Require imports from the checksum-verified original source bundle."""
    repo = Path(os.environ["AF_REPO_ROOT"]).resolve(strict=True)
    manifest = repo / "SHA256SUMS"
    raw = manifest.read_bytes()
    inventory = {}
    for line in raw.decode().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match is None:
            raise ValueError("invalid original source manifest")
        path = repo / match[2]
        relative = (
            str(path.resolve(strict=True).relative_to(repo))
            if (path.resolve(strict=True).is_relative_to(repo))
            else None
        )
        if (
            path.is_symlink()
            or relative is None
            or relative in inventory
            or file_digest(path) != match[1]
        ):
            raise ValueError("original source inventory differs")
        inventory[relative] = match[1]
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if (name == "autoformalism" or name.startswith("autoformalism.")) and path:
            resolved = Path(path).resolve(strict=True)
            if (
                not resolved.is_relative_to(repo)
                or str(resolved.relative_to(repo)) not in inventory
            ):
                raise ValueError("campaign import is outside frozen source inventory")
    if not inventory or manifest.read_bytes() != raw:
        raise ValueError("original source manifest changed during verification")
    return {
        "repo": str(repo),
        "frozen_source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_inventory_sha256": campaign.public.content_sha256(inventory),
    }


def _task_id(task: dict) -> str:
    """Reject path traversal before constructing any task artifact path."""
    name = task.get("task_id")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("unsafe task identity")
    return name


def inventory(root: Path, task: dict) -> dict:
    """Hash result, selections and every operation file, including progress."""
    folder = root / "results" / _task_id(task)
    files = [folder / "result.json", folder / "selections.json"]
    operations = folder / "operations"
    if folder.is_symlink() or operations.is_symlink():
        raise ValueError("symlink in original task inventory")
    for path in sorted(operations.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in original task inventory")
        if path.is_file():
            files.append(path)
    result = {}
    for path in files:
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("unsafe original task inventory")
        result[str(path.relative_to(root))] = file_digest(path)
    return result


def _identity(root: Path, source: dict | None = None) -> dict:
    """Bind every reporting snapshot to exact plan, input, helper and code bytes."""
    if source is None:
        source = source_identity()
    elif (
        "repo" in source
        and file_digest(Path(source["repo"]) / "SHA256SUMS")
        != (source["frozen_source_manifest_sha256"])
    ):
        raise ValueError("frozen source manifest changed during reporting")
    return {
        "protocol": PROTOCOL,
        "plan_file_sha256": file_digest(root / "plan.json"),
        "inputs_file_sha256": file_digest(root / "inputs.json"),
        "helper_sha256": file_digest(Path(__file__)),
        "source": source,
        "reporting_runtime": campaign.public._runtime(),
    }


def _verified(root: Path) -> tuple[dict, dict, dict]:
    identity = _identity(root)
    plan, data = campaign.verify(root, runtime=False)
    ids = [_task_id(task) for task in plan["tasks"]]
    if not ids or len(set(ids)) != len(ids) or set(data["records"]) != set(ids):
        raise ValueError("reporting task roster differs")
    if identity != _identity(root, identity["source"]):
        raise ValueError("reporting source changed during verification")
    return plan, data, identity


@contextmanager
def _step(directory: Path, operation: str, name: str):
    """Flush progress and persist report-only CPU/wall timings, including errors."""
    wall, cpu = monotonic(), process_time()
    event = {"operation": operation, "step": name, "status": "started"}
    path = directory / f"{operation}.progress.jsonl"

    def emit(value: dict) -> None:
        text = json.dumps(value, sort_keys=True, allow_nan=False)
        try:
            print(text, flush=True)
            with path.open("a") as stream:
                stream.write(text + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            # Logging failure after atomic publication cannot undo a valid summary.
            print(
                f"Reporting progress unavailable: {error}", file=sys.stderr, flush=True
            )

    emit(event)
    try:
        yield
    except BaseException:
        event["status"] = "failed"
        raise
    else:
        event["status"] = "completed"
    finally:
        emit(
            event
            | {
                "reporting_wall_seconds": monotonic() - wall,
                "reporting_cpu_seconds": process_time() - cpu,
                "included_in_fit_cost": False,
            }
        )


def _checkpoint_identity(binding: dict, plan: dict, data: dict, index: int) -> dict:
    task = plan["tasks"][index]
    return binding | {
        "plan_sha256": campaign.public.content_sha256(plan),
        "inputs_sha256": campaign.public.content_sha256(data),
        "index": index,
        "task": task,
        "input_record_sha256": campaign.public.content_sha256(
            data["records"][_task_id(task)]
        ),
    }


def _read_checkpoint(path: Path, identity: dict, root: Path) -> dict:
    value = campaign.read_seal(path)
    task_identity = {"plan_sha256": identity["plan_sha256"], "task": identity["task"]}
    terminal = root / "results" / _task_id(identity["task"]) / "result.json"
    if (
        not _same(value["identity"], identity)
        or not _same(value["result"]["identity"], task_identity)
        or value["inventory_sha256"]
        != campaign.public.content_sha256(value["inventory"])
        or value["inventory"].get(str(terminal.relative_to(root)))
        != file_digest(terminal)
        or not _same(value["result"], campaign.read_seal(terminal))
    ):
        raise ValueError("reporting checkpoint identity or terminal result differs")
    return value


def check(root: Path, index: int) -> dict:
    """Call original completion verification once, or reuse an unchanged snapshot."""
    root = root.resolve(strict=True)
    if type(index) is not int or index < 0:
        raise ValueError("report task index out of range")
    directory = root / OUTPUT
    with campaign.public._lock(directory / f"check-{index:03d}"):
        with _step(directory, f"check-{index:03d}", "verify_plan_and_source"):
            plan, data, binding = _verified(root)
            if index >= len(plan["tasks"]):
                raise ValueError("report task index out of range")
            identity = _checkpoint_identity(binding, plan, data, index)
        task = plan["tasks"][index]
        name = _task_id(task)
        path = directory / "checkpoints" / f"{name}.json"
        with _step(directory, f"check-{index:03d}", "inventory_before"):
            before = inventory(root, task)
        if path.exists():
            with _step(directory, f"check-{index:03d}", "resume_checkpoint"):
                value = _read_checkpoint(path, identity, root)
                if value["inventory"] != before:
                    raise ValueError("original task inventory changed since checkpoint")
        else:
            with _step(directory, f"check-{index:03d}", "original_completed"):
                result = campaign._completed(
                    root / "results" / name,
                    {"plan_sha256": identity["plan_sha256"], "task": task},
                    data["records"][name],
                    campaign.Policy.model_validate(plan["policy"]),
                )
                value = {
                    "identity": identity,
                    "inventory": before,
                    "inventory_sha256": campaign.public.content_sha256(before),
                    "result": result,
                }
        with _step(directory, f"check-{index:03d}", "inventory_after_and_seal"):
            if before != inventory(root, task) or binding != _identity(
                root, binding["source"]
            ):
                raise ValueError("original reporting inputs changed during validation")
            campaign.seal(path, value)
        return value


def reduce(root: Path) -> dict:
    """Use original scoring with a guarded snapshot resolver, restored in finally.

    A guarded atomic writer verifies the complete visited roster before publishing,
    so a missing file, changed input or unexpected report path preserves summary.
    """
    root = root.resolve(strict=True)
    directory = root / OUTPUT
    with campaign.public._lock(directory / "reduce"):
        with _step(directory, "reduce", "verify_plan_and_source"):
            plan, data, binding = _verified(root)
        checkpoints, identities = {}, {}
        with _step(directory, "reduce", "verify_all_checkpoints"):
            paths = directory / "checkpoints"
            expected = {f"{_task_id(t)}.json" for t in plan["tasks"]}
            if {p.name for p in paths.glob("*.json")} != expected:
                raise ValueError("incomplete or unexpected reporting checkpoint roster")
            for index, task in enumerate(plan["tasks"]):
                name = _task_id(task)
                identities[name] = _checkpoint_identity(binding, plan, data, index)
                with _step(directory, "reduce", f"checkpoint_{name}"):
                    checkpoints[name] = _read_checkpoint(
                        paths / f"{name}.json", identities[name], root
                    )
        visited = set()
        original_completed, original_write = campaign._completed, campaign.public._write

        def cached(folder, identity, record, policy):
            name = _task_id(identity["task"])
            expected_identity = identities.get(name)
            if (
                expected_identity is None
                or name in visited
                or folder != root / "results" / name
                or not _same(
                    identity,
                    {
                        "plan_sha256": expected_identity["plan_sha256"],
                        "task": expected_identity["task"],
                    },
                )
                or campaign.public.content_sha256(record)
                != expected_identity["input_record_sha256"]
                or policy != campaign.Policy.model_validate(plan["policy"])
            ):
                raise ValueError("original report requested an unexpected task")
            visited.add(name)
            return checkpoints[name]["result"]

        def guarded_write(path, value):
            if path != root / "summary.json" or visited != set(checkpoints):
                raise ValueError("original report did not consume the full task roster")
            with _step(directory, "reduce", "verify_and_publish_summary"):
                if binding != _identity(root, binding["source"]):
                    raise ValueError("reporting source changed during reduction")
                for name, checkpoint in checkpoints.items():
                    with _step(directory, "reduce", f"inventory_{name}"):
                        if not _same(
                            _read_checkpoint(
                                paths / f"{name}.json", identities[name], root
                            ),
                            checkpoint,
                        ) or checkpoint["inventory"] != inventory(
                            root, identities[name]["task"]
                        ):
                            raise ValueError(
                                "reporting inputs changed during reduction"
                            )
                original_write(path, value)

        with _step(directory, "reduce", "original_report_from_snapshots"):
            try:
                campaign._completed, campaign.public._write = cached, guarded_write
                return campaign.report(root)
            finally:
                campaign._completed, campaign.public._write = (
                    original_completed,
                    original_write,
                )


def main() -> None:
    """Expose report-only modes and periodic tracebacks for stalled filesystem work."""
    global campaign
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    print('{"step":"load_original_campaign","status":"started"}', flush=True)
    from autoformalism.fitting import prefix_noise_reliability as campaign

    print('{"step":"load_original_campaign","status":"completed"}', flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "reduce"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, type=Path)
        if name == "check":
            command.add_argument("--index", required=True, type=int)
    args = parser.parse_args()
    try:
        result = (
            check(args.root, args.index)
            if args.command == "check"
            else reduce(args.root)
        )
        print(
            json.dumps(
                {
                    "command": args.command,
                    "status": result.get("status", "validated"),
                    "artifact": str(
                        args.root
                        / (OUTPUT if args.command == "check" else "summary.json")
                    ),
                }
            ),
            flush=True,
        )
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
