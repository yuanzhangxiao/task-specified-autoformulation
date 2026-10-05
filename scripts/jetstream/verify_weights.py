#!/usr/bin/env python3
"""Check downloaded model weights against the hashes Hugging Face publishes.

Every file Hugging Face lists for the pinned revision, apart from the skipped
folders, must be in the local snapshot and carry the published hash: SHA-256
for large (LFS) files, the git blob hash for the rest. Nothing else may be
there. The verified listing is written beside the weights as a record of what
the model server loads.

Standard library only, so it runs before any environment is installed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

#: Folders vLLM does not load: the original checkpoint and a Metal build.
SKIPPED = ("original/*", "metal/*")


def published(repo: str, revision: str) -> dict[str, dict]:
    """What Hugging Face lists for one revision, as path -> expected hash."""
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        value = json.load(response)
    if value.get("sha") != revision:
        raise ValueError(f"{repo} at {revision} resolved to {value.get('sha')}")
    return expected_hashes(value["siblings"])


def expected_hashes(siblings: list[dict]) -> dict[str, dict]:
    """The hash each listed file must have, leaving out the skipped folders."""
    expected: dict[str, dict] = {}
    for item in siblings:
        name = item["rfilename"]
        if any(fnmatch.fnmatch(name, pattern) for pattern in SKIPPED):
            continue
        lfs = item.get("lfs")
        if lfs:
            expected[name] = {"sha256": lfs["sha256"], "size": lfs["size"]}
        else:
            expected[name] = {"git_blob": item["blobId"], "size": item.get("size")}
    return expected


def local_hashes(path: Path) -> dict:
    """SHA-256 and git blob hash of one file, in a single read."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    # git names a blob by SHA-1 of a header and the content; this is not security.
    blob = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 24):
            sha256.update(chunk)
            blob.update(chunk)
    return {"sha256": sha256.hexdigest(), "git_blob": blob.hexdigest(), "size": size}


def compare(snapshot: Path, expected: dict[str, dict]) -> list[str]:
    """Every way the snapshot differs from the published listing."""
    present = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    problems = [f"unexpected file {name}" for name in sorted(present - set(expected))]
    for name, wanted in sorted(expected.items()):
        if name not in present:
            problems.append(f"missing {name}")
            continue
        kind = "sha256" if "sha256" in wanted else "git_blob"
        found = local_hashes(snapshot / name)[kind]
        if found != wanted[kind]:
            problems.append(
                f"{name}: {kind} {found} is not the published {wanted[kind]}"
            )
    return problems


def snapshot_path(hf_home: Path, repo: str, revision: str) -> Path:
    """Where huggingface_hub keeps one revision under HF_HOME."""
    folder = "models--" + repo.replace("/", "--")
    return hf_home / "hub" / folder / "snapshots" / revision


def main() -> None:
    """Verify one snapshot and write the listing it matched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    snapshot = snapshot_path(args.hf_home, args.repo, args.revision)
    if not snapshot.is_dir():
        sys.exit(f"no snapshot of {args.repo} at {args.revision} in {snapshot}")
    expected = published(args.repo, args.revision)
    problems = compare(snapshot, expected)
    if problems:
        heading = f"{args.repo} at {args.revision} does not verify:"
        sys.exit("\n".join([heading, *problems]))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "repo": args.repo,
                "revision": args.revision,
                "skipped": list(SKIPPED),
                "files": expected,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.repo} at {args.revision}: {len(expected)} files match "
        "the published hashes"
    )


if __name__ == "__main__":
    main()
