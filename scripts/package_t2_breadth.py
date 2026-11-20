"""Build the ACES delivery from a Git commit, the pinned intake and Harmony asset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import harmony_vocab, t2_breadth

NAME = "phase-c-t2-breadth-v2"
FITTED_NAME = "phase-c-t2-fitted-revisions-v1"
FRESH_20B_NAME = "phase-c-t2-fresh-20b-v1"
FITTED_20B_NAME = "phase-c-t2-fitted-breadth-20b-v1"


def package(
    destination: Path,
    intake: Path,
    harmony: Path,
    *,
    fresh_20b: bool = False,
    fitted_20b: bool = False,
) -> Path:
    """Include committed runtime files and exactly the two reviewed external inputs."""
    reviewed = sealed_read(intake)
    fitted_run = reviewed.get("protocol") == t2_breadth.fitted.INPUT_PROTOCOL
    name = FITTED_NAME if fitted_run else NAME
    input_name = "inputs/fitted-targets.json" if fitted_run else "inputs/intake.json"
    if fresh_20b:
        if fitted_run:
            raise ValueError("Use the public training intake for the fresh 20B retest")
        name, input_name = FRESH_20B_NAME, "inputs/fresh-20b.json"
    if fitted_20b:
        if not fitted_run or fresh_20b:
            raise ValueError("Use the two fitted parents for independent 20B proposals")
        name = FITTED_20B_NAME
    expected = t2_breadth.fitted.INTAKE if fitted_run else t2_breadth.INTAKE
    if reviewed["artifact_sha256"] != expected:
        raise ValueError("Different T2 training intake")
    harmony_vocab.verify(harmony)
    repo = t2_breadth.REPO
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    tracked = [
        "src",
        "pyproject.toml",
        "docs/PHASE_C_T2_BREADTH.md",
        *t2_breadth.ENTRYPOINTS,
    ]
    if fitted_run:
        tracked.append("scripts/hpc/start_t2_fitted_aces.sh")
    if fresh_20b:
        tracked.append("scripts/hpc/start_t2_fresh_20b_aces.sh")
    if fitted_20b:
        tracked.append("scripts/hpc/start_t2_fitted_20b_aces.sh")
    source = subprocess.check_output(
        ["git", "archive", "--format=tar", commit.strip(), *tracked], cwd=repo
    )
    files = {}
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        for member in archive:
            if member.isfile():
                files[member.name] = archive.extractfile(member).read(), member.mode
    for target, path in [
        (input_name, intake),
        ("assets/harmony/o200k_base.tiktoken", harmony / "o200k_base.tiktoken"),
    ]:
        files[target] = path.read_bytes(), 0o644
    files["SOURCE_COMMIT"] = commit.encode(), 0o644
    checks = [
        f"{hashlib.sha256(data).hexdigest()}  {path}\n"
        for path, (data, _) in sorted(files.items())
    ]
    files["SHA256SUMS"] = "".join(checks).encode(), 0o644
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (name + ".tar.gz")
    temporary = target.with_suffix(".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as stream,
        tarfile.open(fileobj=stream, mode="w") as archive,
    ):
        for path, (data, mode) in sorted(files.items()):
            member = tarfile.TarInfo(f"{name}/{path}")
            member.mode, member.size = mode, len(data)
            archive.addfile(member, io.BytesIO(data))
    temporary.replace(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--intake", required=True, type=Path)
    parser.add_argument("--harmony", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fresh-20b", action="store_true")
    modes.add_argument("--fitted-20b", action="store_true")
    args = parser.parse_args()
    result = package(
        args.out,
        args.intake,
        args.harmony,
        fresh_20b=args.fresh_20b,
        fitted_20b=args.fitted_20b,
    )
    print(
        json.dumps(
            {
                "archive": str(result.resolve()),
                "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                "bytes": result.stat().st_size,
            },
            indent=2,
        )
    )
