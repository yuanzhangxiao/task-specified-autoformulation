"""Build a versioned, public-only ACES/Delta delivery from committed source."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import harmony_vocab
from scripts import iclr_targeted_perturbations as campaign

REPO = Path(__file__).resolve().parents[1]
ENTRIES = (
    "iclr_targeted_perturbations",
    "iclr_perturbed_cstr_refits",
    "iclr_aces_pilot",
    "iclr_controlled_search",
    "iclr_stress_resources",
    "harmony_vocab",
    "package_iclr_targeted_perturbations",
)
SHELLS = (
    "scripts/hpc/iclr_targeted_perturbations_aces.sh",
    "scripts/hpc/iclr_targeted_refits.sh",
    "scripts/hpc/run_iclr_controlled_search_aces.sh",
)


def script_dependencies() -> list[str]:
    """Include transitive local script imports without unrelated experiment files."""
    pending, found = list(ENTRIES), set()
    while pending:
        module = pending.pop()
        if module in found:
            continue
        path = REPO / "scripts" / (module.replace(".", "/") + ".py")
        if not path.is_file():
            raise ValueError(f"missing script dependency: {module}")
        found.add(module)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    package = ["scripts", *module.split(".")[:-1]]
                    base = package[: len(package) - node.level + 1]
                    imports = (
                        [node.module] if node.module else [a.name for a in node.names]
                    )
                    for name in imports:
                        dependency = ".".join([*base, name])
                        if dependency.startswith("scripts."):
                            pending.append(dependency.removeprefix("scripts."))
                elif node.module == "scripts":
                    pending.extend(alias.name for alias in node.names)
                elif node.module and node.module.startswith("scripts."):
                    pending.append(node.module.removeprefix("scripts."))
            elif isinstance(node, ast.Import):
                pending.extend(
                    alias.name.removeprefix("scripts.")
                    for alias in node.names
                    if alias.name.startswith("scripts.")
                )
    return sorted("scripts/" + name.replace(".", "/") + ".py" for name in found)


def build(output: Path, handoff: Path, original_t1: Path, vocabulary: Path) -> dict:
    """Pin code and public bytes, then verify the complete transferable archive."""
    paths = [
        "src",
        "configs",
        "pyproject.toml",
        *script_dependencies(),
        *SHELLS,
        "docs/PROJECT_CONTEXT.md",
        "docs/ICLR_TARGETED_PERTURBATIONS_20261117.md",
    ]
    if (REPO / "scripts/__init__.py").exists():
        paths.append("scripts/__init__.py")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=REPO, check=True
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    name = f"iclr-targeted-20b-{revision[:7]}"
    output.mkdir(parents=True, exist_ok=True)
    final = output / (name + ".tar.gz")
    harmony_vocab.verify(vocabulary)
    with TemporaryDirectory(dir=output) as temporary:
        base = Path(temporary)
        raw = base / "source.tar"
        subprocess.run(
            ["git", "archive", "HEAD", "--format=tar", f"--output={raw}", *paths],
            cwd=REPO,
            check=True,
        )
        root = base / name
        root.mkdir()
        with tarfile.open(raw) as source:
            source.extractall(root, filter="data")
        campaign.stage(handoff, original_t1, root / "public")
        (root / "SOURCE_COMMIT").write_text(revision + "\n")
        assets = root / "assets/harmony"
        assets.mkdir(parents=True)
        shutil.copyfile(
            vocabulary / harmony_vocab.VOCAB_FILENAME,
            assets / harmony_vocab.VOCAB_FILENAME,
        )
        for entry, script in (
            ("run_aces.sh", SHELLS[0]),
            ("run_refits.sh", SHELLS[1]),
        ):
            (root / entry).write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                'bundle=$(cd "$(dirname "$0")" && pwd -P)\n'
                f'exec bash "$bundle/{script}" "$@"\n'
            )
        files = sorted(p for p in root.rglob("*") if p.is_file())
        (root / "SHA256SUMS").write_text(
            "".join(
                f"{campaign.search.file_digest(p)}  {p.relative_to(root)}\n"
                for p in files
            )
        )
        archive = base / "delivery.tar.gz"

        def canonical_metadata(info: tarfile.TarInfo) -> tarfile.TarInfo:
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if info.isdir() else 0o644
            info.pax_headers = {}
            return info

        with (
            archive.open("wb") as stream,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=stream, mtime=0
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as target,
        ):
            target.add(root, arcname=name, filter=canonical_metadata)
        with tarfile.open(archive) as target:
            if any(p.issym() or p.islnk() for p in target.getmembers()):
                raise ValueError("delivery may not contain links")
        if final.exists() and campaign.search.file_digest(
            final
        ) != campaign.search.file_digest(archive):
            raise ValueError(
                f"preserve existing delivery; choose a fresh output directory: {final}"
            )
        archive.replace(final)
    return {
        "archive": str(final),
        "source_commit": revision,
        "files": len(files) + 1,
        "sha256": campaign.search.file_digest(final),
        "test_data_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "handoff", "original-t1", "vocabulary"):
        parser.add_argument("--" + name, required=True, type=Path)
    a = parser.parse_args()
    print(json.dumps(build(a.output, a.handoff, a.original_t1, a.vocabulary), indent=2))


if __name__ == "__main__":
    main()
