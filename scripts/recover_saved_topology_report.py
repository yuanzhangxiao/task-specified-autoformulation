#!/usr/bin/env python3
"""Resume the unchanged 3404d29 campaign with a report-only Counter correction.

Copy this file outside the delivered bundle. Run with its existing Python venv.
The run command stays in the foreground; use nohup for a detached worker.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

PROTOCOL = "saved-topology-report-recovery-1"
SOURCE_IDENTITY = "5a6e2da9899e20ab3c22d5471a148d5d42781901bedc8b9f06b30091b7b6640d"
PLAN_IDENTITY = "bc43e95ff2c7094599f261ce4a14d4de088525a8bfed8725b5bdd1054f401295"
MANIFEST_SHA256 = "56486581cca71b2b4e6c19a5e2cc1618f6e05eb10fdaa7a197b65653f861dfe2"
REPORT_SHA256 = "e432e777b5f65920dc2e8c6703b87bf19ebeb27424248bd419af37beb1c0ca6d"
REPORT_MODULE = "src/autoformalism/research/saved_topology_repair.py"
CONTAINER = "af-construction-vllm"
RECEIPT = "report-recovery.json"


def sha256(path: Path) -> str:
    """Hash bytes without loading large bundle files into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(bundle: Path) -> None:
    """Reject altered or substituted delivered files before importing any code."""
    manifest = bundle / "MANIFEST.sha256"
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("original bundle manifest differs; do not edit the bundle")
    seen = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = (bundle / name).resolve()
        if name in seen or not path.is_relative_to(bundle) or sha256(path) != digest:
            raise ValueError(f"original bundle file differs: {name}")
        seen.add(name)
    if sha256(bundle / REPORT_MODULE) != REPORT_SHA256:
        raise ValueError("original report source differs")


def load_legacy(bundle: Path) -> SimpleNamespace:
    """Import only the original package and entry points, without bytecode writes."""
    verify_bundle(bundle)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(bundle), str(bundle / "src"), str(bundle / "scripts")]
    names = {
        "campaign": "autoformalism.research.saved_topology_repair",
        "cli": "scripts.saved_topology_repair",
        "server": "scripts.jetstream_server_record",
    }
    # Refuse an embedding interpreter that already imported a different checkout.
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"autoformalism", "scripts"}:
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(bundle):
                raise ValueError(f"non-bundle module already loaded: {name}")
    loaded = {key: importlib.import_module(name) for key, name in names.items()}
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"autoformalism", "scripts"}:
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(bundle):
                raise ValueError(f"loaded module is outside original bundle: {name}")
    return SimpleNamespace(**loaded)


def verify_campaign(legacy: SimpleNamespace, bundle: Path, root: Path) -> dict:
    """Require the exact existing input, campaign plan, and execution provenance."""
    plan = legacy.campaign.verify(bundle / "inputs/saved-repair")
    if (
        plan["artifact_sha256"] != PLAN_IDENTITY
        or plan["source_identity"] != SOURCE_IDENTITY
        or legacy.campaign.source_identity() != SOURCE_IDENTITY
    ):
        raise ValueError("pinned saved-repair identity differs")
    if legacy.campaign.verify(root) != plan:
        raise ValueError("existing campaign plan differs from delivered inputs")
    execution = legacy.campaign.sealed_read(root / "execution.json")
    if (
        execution["plan_sha256"] != PLAN_IDENTITY
        or execution["source_identity"] != SOURCE_IDENTITY
        or execution["protocol"] != plan["protocol"]
        or execution["test_data_opened"] is not False
    ):
        raise ValueError("existing campaign execution identity differs")
    return plan


def report_counter(iterable=None, **kwargs) -> Counter:
    """Keep all counts, using a string only for the unassessed report key."""
    counts = Counter(iterable, **kwargs)
    if None in counts:
        counts["not_assessed"] += counts.pop(None)
    return counts


@contextmanager
def corrected_report(campaign):
    """Adapt Counter strictly inside legacy report calls, restoring both names."""
    original_report = campaign.report

    def report(*args, **kwargs):
        original_counter = campaign.Counter
        campaign.Counter = report_counter
        try:
            return original_report(*args, **kwargs)
        finally:
            campaign.Counter = original_counter

    campaign.report = report
    try:
        yield
    finally:
        campaign.report = original_report


@contextmanager
def worker_lock(path: Path):
    """Use the same nonblocking advisory lock as the original shell worker."""
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"worker is still running: {path}") from error
        yield


def record_recovery(legacy: SimpleNamespace, bundle: Path, root: Path) -> dict:
    """Seal a separate receipt; original plans and execution records stay intact."""
    return legacy.campaign.sealed_write(
        root / RECEIPT,
        {
            "protocol": PROTOCOL,
            "plan_sha256": PLAN_IDENTITY,
            "source_identity": SOURCE_IDENTITY,
            "bundle_manifest_sha256": sha256(bundle / "MANIFEST.sha256"),
            "original_report_source_sha256": sha256(bundle / REPORT_MODULE),
            "original_plan_file_sha256": sha256(root / "plan.json"),
            "original_execution_file_sha256": sha256(root / "execution.json"),
            "helper_sha256": sha256(Path(__file__)),
            "correction": "report Counter key None becomes not_assessed",
            "scientific_runtime_modified": False,
            "budgets_reset": False,
        },
    )


def log(root: Path, message: str) -> None:
    """Keep a small recovery log, including controlled failure diagnostics."""
    with (root / "recovery.log").open("a") as stream:
        stream.write(f"{datetime.now(UTC).isoformat()} {message}\n")


def refresh_server(legacy: SimpleNamespace, work: Path, root: Path) -> None:
    """Reuse the original owner/configuration checks and metadata generation."""
    result = subprocess.run(
        ["sudo", "-n", "docker", "inspect", CONTAINER],
        check=True,
        text=True,
        capture_output=True,
    )
    inspection = json.loads(result.stdout)[0]
    if inspection["Config"].get("Labels", {}).get("autoformalism.role") != (
        "construction-compatibility"
    ):
        raise ValueError("the expected owned model server is absent")
    if inspection["State"]["Running"] is not True:
        raise ValueError("server is stopped; restart the original compatibility server")
    value = legacy.server.server_record(inspection, work)
    value["gpu_driver"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    (work / "server.json").write_text(json.dumps(value, indent=2) + "\n")
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True)
    for source, name in (
        (work / "server.json", "server.json"),
        (work / "weights.json", "weights.json"),
        (work / "probe/plan.json", "compatibility-plan.json"),
    ):
        shutil.copyfile(source, runtime / name)
    # The unchanged CLI then rechecks exact compatibility and sealed execution.


def package(root: Path) -> Path:
    """Archive the original inspection file set and separate recovery evidence."""
    names = [
        "plan.json",
        "execution.json",
        "summary.json",
        "SUMMARY.md",
        "REPORT.html",
        "results",
        "runtime",
        "exit-code.txt",
        RECEIPT,
        "recovery.log",
    ]
    paths = [root / name for name in names if (root / name).exists()]
    paths += sorted(root.glob("launcher-*.log"))
    paths += sorted(root.glob("recovery-*.log"))
    temporary = root / "inspection.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.name)
    output = root / "inspection.tar.gz"
    temporary.replace(output)
    return output


def execute(command: str, bundle: Path, root: Path, work: Path) -> dict:
    """Run only the pinned recovery under existing worker and campaign locks."""
    bundle, root, work = (path.resolve() for path in (bundle, root, work))
    if Path(__file__).resolve().is_relative_to(bundle):
        raise ValueError("copy this recovery helper outside the original bundle")
    if root.is_relative_to(bundle) or work.is_relative_to(bundle):
        raise ValueError("campaign and server work must be outside the original bundle")
    if not root.is_dir() or not work.is_dir():
        raise ValueError("use the existing campaign root and server work directories")
    if command not in {"run", "report", "inspect"}:
        raise ValueError("use run, report, or inspect")
    legacy = load_legacy(bundle)
    with ExitStack() as stack:
        if command == "run":
            stack.enter_context(worker_lock(work / "worker.lock"))
        stack.enter_context(worker_lock(root / "worker.lock"))
        stack.enter_context(legacy.cli._lock(root / "campaign-lock"))
        plan = verify_campaign(legacy, bundle, root)
        record_recovery(legacy, bundle, root)
        stack.enter_context(corrected_report(legacy.campaign))
        log(root, f"{command} started; original plan {PLAN_IDENTITY}")
        try:
            if command == "run":
                (root / "exit-code.txt").write_text("running\n")
                code = 1
                try:
                    refresh_server(legacy, work, root)
                    if shutil.disk_usage(root).free < 2 * 1024**3:
                        raise ValueError("need 2 GiB free before resuming")
                    stats = os.statvfs(root)
                    if stats.f_files and stats.f_favail < 10000:
                        raise ValueError("need 10,000 free filesystem entries")
                    value = legacy.cli.run(
                        bundle / "inputs/saved-repair",
                        root,
                        work,
                        bundle / "probe-requests.json",
                    )
                    code = 0
                finally:
                    (root / "exit-code.txt").write_text(f"{code}\n")
            else:
                value = legacy.campaign.report(root, plan)
            log(root, f"{command} finished")
            if command == "inspect":
                value = {"archive": str(package(root)), "summary": value}
            return value
        except BaseException:
            log(root, traceback.format_exc())
            raise


def main(argv: list[str] | None = None) -> int:
    """Expose a standalone recovery CLI without changing the original launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report", "inspect"))
    for name in ("bundle", "root", "work"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = execute(args.command, args.bundle, args.root, args.work)
    except Exception as error:
        print(f"Recovery stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
