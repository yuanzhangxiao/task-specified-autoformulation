#!/usr/bin/env python3
"""Select installed runtime dependency files without importing numerical packages."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import re
import sys
from pathlib import Path, PurePosixPath

from scaled_recovery_io import write

# Runtime imports include the project configuration/providers, but no dev,
# plotting, Torch, or symbolic-regression extras. Versions come from the source
# environment; this performs no installation or dependency upgrade.
ROOTS = (
    "numpy",
    "scipy",
    "pandas",
    "pydantic",
    "pydantic-settings",
    "python-dotenv",
    "PyYAML",
    "openai",
    "casadi",
)


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_parser():
    """Use the environment's metadata parser, with pip's vendor as a fallback."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
        except ImportError as error:
            raise RuntimeError(
                "Dependency discovery requires packaging or pip in the source "
                "environment; no packages were installed or copied."
            ) from error
    return Requirement


def select(roots: list[Path], requested: tuple[str, ...] = ROOTS) -> dict:
    """Resolve mandatory dependencies in import-path order and retain RECORD files."""
    Requirement = requirement_parser()
    pending = [(name, "runtime") for name in requested]
    chosen, files = {}, {str(root): set() for root in roots}
    while pending:
        text, parent = pending.pop(0)
        req = Requirement(text)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        if req.extras or req.url:
            raise ValueError(f"unexpected extra/URL dependency: {text}")
        key = normalized(req.name)
        if key in chosen:
            version = chosen[key]["version"]
            if req.specifier and version not in req.specifier:
                raise ValueError(f"installed {key} {version} does not satisfy {text}")
            continue
        # Match PYTHONPATH: extras can intentionally override the venv version.
        found = None
        for root in roots:
            matches = list(
                metadata.Distribution.discover(name=req.name, path=[str(root)])
            )
            if len(matches) > 1:
                raise ValueError(
                    f"ambiguous installed distribution: {req.name} in {root}"
                )
            if matches:
                found = (root, matches[0])
                break
        if found is None:
            raise ValueError(
                f"missing installed dependency {text}, required by {parent}"
            )
        root, dist = found
        if req.specifier and dist.version not in req.specifier:
            raise ValueError(f"installed {key} {dist.version} does not satisfy {text}")
        records = dist.files
        if not records:
            raise ValueError(
                f"no installed file inventory for {key}; refusing whole-site copy"
            )
        retained, skipped = [], []
        known_bytes = 0
        for entry in records:
            path = PurePosixPath(str(entry))
            if path.is_absolute() or "\n" in str(path) or "\x00" in str(path):
                raise ValueError(f"invalid installed file inventory for {key}: {path}")
            # Wheel console scripts and data outside site-packages are not used
            # by Python -S workers. Do not follow RECORD traversal outside it.
            if (
                ".." in path.parts
                or "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pth"}
            ):
                skipped.append(str(path))
                continue
            retained.append(str(path))
            known_bytes += entry.size or 0
        files[str(root)].update(retained)
        chosen[key] = {
            "name": dist.metadata["Name"],
            "version": dist.version,
            "root": str(root),
            "file_count": len(retained),
            "recorded_bytes": known_bytes,
            "skipped_files": skipped,
        }
        print(
            json.dumps(
                {
                    "event": "dependency_selected",
                    "package": key,
                    "version": dist.version,
                    "files": len(retained),
                    "recorded_bytes": known_bytes,
                }
            ),
            flush=True,
        )
        pending.extend((value, key) for value in dist.requires or [])
    return {
        "requested": list(requested),
        "packages": chosen,
        "files": {key: sorted(value) for key, value in files.items()},
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # -S deliberately disables editable .pth files and user site discovery.
    sys.path[0:0] = [str(root) for root in args.root]
    manifest = select(args.root)
    write(args.output / "dependencies.json", manifest)
    for index, root in enumerate(args.root):
        (args.output / f"dependency-files-{index}.txt").write_text(
            "\n".join(manifest["files"][str(root)]) + "\n"
        )


if __name__ == "__main__":
    main()
