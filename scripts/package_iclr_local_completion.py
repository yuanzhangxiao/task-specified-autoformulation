"""Build the internal analyst delivery from frozen historical and local evidence."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.iclr_submission_evidence import REPO, digest
from scripts.package_iclr_dynamics_evidence import pack

EVIDENCE_HASH = "7e1a64fe6ef8e9c93f3521f8e513edc2aac14f2b8144bcb081f2d53dca315663"
TRACE_HASH = "3092a860152247773e74a6086e18f95f5e4c64145a8db51dce65b1dbcea9b98a"
EARLIER_HASH = "fed42997c2fa33e55a1f1f2c7319d1ae3bc868447e4a6be48cb84ca571aa3092"
ARCHIVES = {
    "iclr-nonbaseline-evidence.zip": EVIDENCE_HASH,
    "iclr-nonbaseline-fidelity-traces.zip": TRACE_HASH,
}
DOCUMENTS = (
    "ACES_EXECUTION",
    "LATENT_REVIEW",
    "HISTORICAL_FOLLOWUP",
    "REVIEWER_RESPONSES",
)


def completed_files(root: Path) -> dict[str, Path]:
    """Validate receipts and require complete local study denominators."""
    files = {}
    for directory, summary_name, digests in (
        ("latent", "summary.json", "table_sha256"),
        ("historical", "historical_summary.json", "table_digests"),
    ):
        folder = root / directory
        summary = sealed_read(folder / summary_name)
        for key, sha in summary[digests].items():
            if digest(folder / (key + ".csv")) != sha:
                raise ValueError("table differs from sealed local analysis: " + key)
        for f in folder.iterdir():
            if f.is_file() and f.suffix in (".csv", ".json", ".pdf", ".png"):
                files[f"local/{directory}/{f.name}"] = f
    transfer = root / "historical"
    plan = sealed_read(transfer / "transfer_plan.json")
    if (
        sealed_read(transfer / "transfer_summary.json")["plan"]
        != plan["artifact_sha256"]
    ):
        raise ValueError("transfer results use another plan")
    fits = sorted((transfer / "transfer-fits").glob("*/*/fit.json"))
    if len(fits) != 6:
        raise ValueError("expected six sensitivity fits")
    for f in fits:
        if sealed_read(f)["plan"] != plan["artifact_sha256"]:
            raise ValueError("unexpected fit")
        files["local/historical/" + f.relative_to(transfer).as_posix()] = f
    clamp = root / "clamp-final"
    plan = sealed_read(clamp / "plan.json")
    result = sealed_read(clamp / "summary.json")
    if (
        plan["mode"] != "development"
        or result["plan"] != plan["artifact_sha256"]
        or len(result["rows"]) != 80
    ):
        raise ValueError("incomplete retained T2 clamp assay")
    jobs = sorted((clamp / "jobs").glob("*.json"))
    if len(jobs) != 32 or {sealed_read(f)["artifact_sha256"] for f in jobs} != set(
        result["job_seals"]
    ):
        raise ValueError("clamp jobs differ from the frozen result")
    for f in [
        *(
            clamp / n
            for n in (
                "plan.json",
                "summary.json",
                "latent_clamp.csv",
                "paired-glucose-clamps.pdf",
            )
        ),
        *jobs,
    ]:
        files["local/clamp/" + f.relative_to(clamp).as_posix()] = f
    stress = root / "stress-coverage"
    coverage = json.loads((stress / "summary.json").read_text())
    if coverage["planned"] != 126 or coverage["complete"] != 0:
        raise ValueError("expected the explicitly unevaluated prospective roster")
    for f in stress.iterdir():
        if f.suffix in (".csv", ".json"):
            files[f"local/stress-coverage/{f.name}"] = f
    verification = root / "verification"
    files["verification/local/aces_launch.json"] = (
        root / "aces/launch_verification.json"
    )
    files["verification/local/cstr_development.json"] = (
        verification / "cstr-loader.json"
    )
    synthetic = verification / "receipt-export-synthetic"
    for f in synthetic.iterdir():
        if f.suffix in (".csv", ".json"):
            files[f"verification/synthetic-receipt-export/{f.name}"] = f
    for name in ("pytest.txt", "ruff.txt"):
        if (verification / name).is_file():
            files[f"verification/local/{name}"] = verification / name
    return files


def earlier_t1(archive: Path, stage: Path) -> dict[str, Path]:
    """Recover internal free-rollout and equation-clamp rows from the round-12 assay."""
    if digest(archive) != EARLIER_HASH:
        raise ValueError("earlier T1 archive identity differs")
    files = {}
    with ZipFile(archive) as z:
        for member, name in (
            ("comparisons/paired_responses.csv", "paired_responses.csv"),
            ("meal-appearance/scores.csv", "meal_flux.csv"),
        ):
            with z.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                rows = [
                    r for r in reader if r["method"] in ("full", "no_latent", "no_spec")
                ]
            if not rows:
                raise ValueError("historical internal results absent")
            path = stage / name
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            files["local/earlier-t1/" + name] = path
    return files


def package(output: Path, local: Path, dest: Path):
    """Keep the original ACES scoring package immutable and add a new analyst ZIP."""
    files = completed_files(local)
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        files.update(earlier_t1(output / "iclr-2027-rebuttal-evidence.zip", base))
        for archive, sha in ARCHIVES.items():
            path = output / archive
            if digest(path) != sha:
                raise ValueError("source ZIP hash differs: " + archive)
            with ZipFile(path) as z:
                # Both ZIP identities are fixed above. Never overwrite a member
                # from the earlier delivery when adding a companion.
                for item in z.namelist():
                    if (
                        item in files
                        or Path(item).is_absolute()
                        or ".." in Path(item).parts
                    ):
                        raise ValueError("duplicate or unsafe archive member")
                    dst = base / archive / item
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(z.read(item))
                    target = {
                        "README.md": "ORIGINAL_REPORT.md",
                        "SHA256SUMS": "original-evidence-checksums.txt",
                        "docs/ICLR_2027_REVIEWER_RESPONSES.md": (
                            "docs/ORIGINAL_REVIEWER_RESPONSES.md"
                        ),
                    }.get(item, item)
                    files[target] = dst
        commit = base / "commit.txt"
        commit.write_text(
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO)
        )
        files["local/source-commit.txt"] = commit
        files["README.md"] = REPO / "docs/ICLR_2027_LOCAL_RESULTS.md"
        for name in DOCUMENTS:
            rel = f"docs/ICLR_2027_{name}.md"
            files[rel] = REPO / rel
        pack(files, dest)
    print(f"{dest} ({len(files)} checksummed files), SHA256 {digest(dest)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=REPO / "output")
    p.add_argument("--local", type=Path, default=REPO / "output/iclr-local-completion")
    p.add_argument(
        "--destination",
        type=Path,
        default=REPO / "output/iclr-local-completion-results.zip",
    )
    a = p.parse_args()
    package(a.inputs, a.local, a.destination)
