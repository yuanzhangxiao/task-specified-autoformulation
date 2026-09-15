#!/usr/bin/env python3
"""Build one relocatable CPU runtime bundle before launching recovery workers."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from time import monotonic, time

from scaled_recovery_io import digest, prepare, publish, read, sha, write


def audit_numerical_sources(repo: Path, source: Path) -> dict:
    """Bind the old source hash to Git and require unchanged numerical modules."""
    old = read(source / "freeze.json")
    commit = read(source / "submission.json")["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid original commit")
    scripts = [
        "scripts/" + name
        for name in (
            "run_scaled_alternating.py",
            "scaled_alternating_report.py",
            "hpc/scaled_alternating_delta.slurm",
            "hpc/submit_scaled_alternating_delta.sh",
        )
    ]
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", commit, "src/autoformalism", *scripts],
        timeout=120,
    )
    old_files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle:
            if member.isfile() and (
                member.name.endswith(".py") or member.name in scripts
            ):
                data = bundle.extractfile(member).read()
                old_files[member.name] = hashlib.sha256(data).hexdigest()
    if digest(old_files) != old["code"]:
        raise ValueError("original commit does not match frozen source identity")
    prefixes = (
        "src/autoformalism/fitting/",
        "src/autoformalism/expressions/",
        "src/autoformalism/data/",
        "src/autoformalism/schemas/",
    )
    helpers = (
        "src/autoformalism/rebuttal/scaled_alternating.py",
        "src/autoformalism/rebuttal/piecewise_campaign.py",
    )
    audited = {
        p: value
        for p, value in old_files.items()
        if p.startswith(prefixes) or p in helpers
    }
    if any(sha(repo / name) != value for name, value in audited.items()):
        raise ValueError("numerical source changed; not an execution-only recovery")
    return {
        "source_commit": commit,
        "source_code_identity": old["code"],
        "unchanged_numerical_files": audited,
    }


def command(args: list[str], root: Path, label: str, *, env=None, seconds=600) -> str:
    started = monotonic()
    print(json.dumps({"event": label + "_begin", "utc_seconds": time()}), flush=True)
    with (root / (label + ".log")).open("w") as stream:
        process = subprocess.run(
            args,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=seconds,
            check=False,
        )
    elapsed = monotonic() - started
    write(
        root / (label + ".json"), {"seconds": elapsed, "exit_code": process.returncode}
    )
    print(
        json.dumps(
            {
                "event": label + "_end",
                "seconds": elapsed,
                "exit_code": process.returncode,
            }
        ),
        flush=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"{label} failed: {(root / (label + '.log')).read_text()[-3000:]}"
        )
    return (root / (label + ".log")).read_text()


def build(
    repo: Path, python: Path, extras: Path, source: Path, output: Path, work: Path
):
    payload, logs = work / "payload", work / "preparation"
    logs.mkdir(parents=True, exist_ok=True)
    write(logs / "numerical_source_audit.json", audit_numerical_sources(repo, source))
    runtime = payload / "runtime"
    (payload / "repo").mkdir(parents=True, exist_ok=True)
    discovery = command(
        [
            str(python),
            "-S",
            "-c",
            "import json,sys,sysconfig; print(json.dumps({'base':sys.base_prefix,"
            "'stdlib':sysconfig.get_path('stdlib'),'version':sys.version_info[:2]}))",
        ],
        logs,
        "discover_runtime",
        seconds=180,
    )
    info = json.loads(discovery)
    version = ".".join(map(str, info["version"]))
    stdlib = Path(info["stdlib"])
    purelib = python.parent.parent / "lib" / ("python" + version) / "site-packages"
    if not purelib.is_dir() or not extras.is_dir() or not stdlib.is_dir():
        raise ValueError(
            "missing venv site-packages, CasADi directory, or standard library"
        )
    (runtime / "python/bin").mkdir(parents=True)
    (runtime / "python/lib").mkdir(parents=True)
    shutil.copy2(python.resolve(), runtime / "python/bin/python3")
    command(
        [
            "rsync",
            "-aL",
            "--exclude=site-packages",
            "--exclude=__pycache__",
            str(stdlib) + "/",
            str(runtime / "python/lib" / ("python" + version)) + "/",
        ],
        logs,
        "copy_stdlib",
    )
    for library in (Path(info["base"]) / "lib").glob("*.so*"):
        if library.is_file():
            shutil.copy2(
                library, runtime / "python/lib" / library.name, follow_symlinks=True
            )
    for label, src in (("site", purelib), ("extras", extras)):
        command(
            [
                "rsync",
                "-aL",
                "--exclude=__pycache__",
                str(src) + "/",
                str(runtime / label) + "/",
            ],
            logs,
            "copy_" + label,
        )
    for name in ("src", "scripts"):
        command(
            [
                "rsync",
                "-aL",
                "--exclude=__pycache__",
                str(repo / name) + "/",
                str(payload / "repo" / name) + "/",
            ],
            logs,
            "copy_code_" + name,
        )
    local_python = runtime / "python/bin/python3"
    env = local_environment(payload)
    versions = command(
        [
            str(local_python),
            "-S",
            "-c",
            "import sys,json,numpy,scipy,casadi,pandas,pydantic,platform; "
            "from importlib.metadata import version; "
            "print(json.dumps({'python':platform.python_version(),"
            "'packages':{p:version(p) for p in "
            "['numpy','pandas','pydantic','scipy']},'casadi':version('casadi'),"
            "'origins':{m.__name__:m.__file__ for m in "
            "[numpy,scipy,casadi,pandas,pydantic]},'base_prefix':sys.base_prefix}))",
        ],
        logs,
        "local_import_audit",
        env=env,
        seconds=180,
    )
    observed = json.loads(versions)
    expected = read(source / "freeze.json")["runtime"]
    if any(observed[k] != expected[k] for k in ("python", "packages", "casadi")):
        raise ValueError("copied runtime differs from original numerical versions")
    if any(
        not Path(p).resolve().is_relative_to(runtime.resolve())
        for p in [*observed["origins"].values(), observed["base_prefix"]]
    ):
        raise ValueError("numerical imports escaped the node-local runtime")
    frozen = prepare(source, payload / "campaign", payload / "repo")
    command(
        [
            str(local_python),
            "-S",
            str(payload / "repo/scripts/smoke_scaled_recovery.py"),
            "--output",
            str(work / "smoke"),
        ],
        logs,
        "recovery_smoke",
        env=env,
        seconds=300,
    )
    write(logs / "runtime.json", observed)
    # No Python, package, or repository discovery is needed in the fit array.
    command(
        ["tar", "-cf", str(work / "bundle.tar"), "-C", str(payload), "."],
        logs,
        "pack_bundle",
    )
    bundle_hash = sha(work / "bundle.tar")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "bundle.json").exists():
        raise ValueError("existing prepared bundle; do not create a fresh recovery")
    publish(payload / "campaign", output)
    publish(logs, output / "preparation")
    shutil.copyfile(work / "bundle.tar", output / "bundle.tar.tmp")
    if sha(output / "bundle.tar.tmp") != bundle_hash:
        raise ValueError("published runtime bundle digest differs")
    (output / "bundle.tar.tmp").replace(output / "bundle.tar")
    write(
        output / "bundle.json",
        {
            "identity": frozen["identity"],
            "sha256": bundle_hash,
            "smoke_passed": True,
            "runtime_versions": {
                k: observed[k] for k in ("python", "packages", "casadi")
            },
        },
        immutable=True,
    )
    print(json.dumps({"prepared": True, "identity": frozen["identity"]}), flush=True)


def local_environment(payload: Path) -> dict:
    runtime = payload / "runtime"
    return {
        **os.environ,
        "PYTHONHOME": str(runtime / "python"),
        "PYTHONPATH": os.pathsep.join(
            str(p) for p in (payload / "repo/src", runtime / "extras", runtime / "site")
        ),
        "LD_LIBRARY_PATH": str(runtime / "python/lib")
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def restore(output: Path, work: Path, index: int):
    logs = work / "staging"
    logs.mkdir(parents=True, exist_ok=True)
    manifest = read(output / "bundle.json")
    command(
        ["cp", str(output / "bundle.tar"), str(work / "bundle.tar")],
        logs,
        "copy_bundle",
    )
    if sha(work / "bundle.tar") != manifest["sha256"]:
        raise ValueError("runtime bundle digest differs")
    payload = work / "payload"
    payload.mkdir()
    command(
        ["tar", "-xf", str(work / "bundle.tar"), "-C", str(payload)],
        logs,
        "unpack_bundle",
    )
    # The archive is our own hash-checked bundle, never a proposer-produced file.
    env = local_environment(payload)
    publish(logs, output / f"staging/task_{index:03d}")
    return subprocess.call(
        [
            str(payload / "runtime/python/bin/python3"),
            "-S",
            str(payload / "repo/scripts/run_scaled_recovery.py"),
            "run",
            "--output",
            str(payload / "campaign"),
            "--persistent",
            str(output),
            "--task-index",
            str(index),
        ],
        env=env,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("build", "restore"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--extras", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "build":
        build(args.repo, args.python, args.extras, args.source, args.output, args.work)
    else:
        raise SystemExit(restore(args.output, args.work, args.task_index))


if __name__ == "__main__":
    main()
